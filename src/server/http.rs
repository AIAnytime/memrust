//! HTTP API for services and SDKs.
//!
//! POST /v1/remember       RememberRequest  -> MemoryRecord
//! POST /v1/remember_batch { items }        -> { records } (one embed round-trip)
//! POST /v1/checkpoint                      -> persist state, truncate WAL
//! POST /v1/recall         RecallRequest    -> { hits: [RecallHit] }
//! POST /v1/forget         { id }           -> { forgotten: bool }
//! POST /v1/lifecycle/run                   -> { report: LifecycleReport }
//! POST /v1/snapshot       { session_id? }  -> { snapshot: Snapshot }
//! POST /v1/restore        { records }      -> { restored: usize }
//! GET  /v1/memories?offset&limit           -> { total, records } (newest first)
//! GET  /v1/entities?limit                  -> { entities: [{name, count}] }
//! GET  /v1/namespaces                      -> { namespaces } (admin key)
//! POST /v1/namespaces/drop { namespace }   -> { dropped } (admin key)
//! GET  /health                             -> EngineStats
//! GET  /                                   -> embedded web dashboard
//!
//! Every request selects a namespace with the `X-Memrust-Namespace` header
//! (default: `default`) and, when keys are configured, presents one with
//! `Authorization: Bearer <key>` or `X-API-Key`.

use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::server::tenancy::{extract_key, extract_namespace, ApiKey, Auth, Namespace, Registry};
use crate::types::{MemoryRecord, RecallRequest, RememberRequest};

/// API responses omit raw embeddings; they're an internal representation.
fn without_embedding(mut record: MemoryRecord) -> MemoryRecord {
    record.embedding = None;
    record
}

/// Writes run in two phases — persist under a *read* lock (so readers are not
/// blocked for the fsync), then apply under the write lock. The per-namespace
/// commit mutex serializes writers across both phases. Without it a checkpoint
/// could land between them, saving state that lacks the record and truncating
/// the WAL entry that describes it, which would lose an acknowledged write.
#[derive(Clone)]
pub struct AppState {
    registry: Arc<Registry>,
    auth: Auth,
}

type Rejection = (StatusCode, String);

impl AppState {
    /// Authenticate, then resolve the namespace the caller asked for. Both
    /// failures are refusals, so they answer the same way a caller can act
    /// on: what was wrong and what to send instead.
    fn resolve(&self, headers: &HeaderMap) -> Result<(Namespace, String), Rejection> {
        let namespace = extract_namespace(headers);
        if self.auth.enabled() {
            let presented = extract_key(headers).ok_or((
                StatusCode::UNAUTHORIZED,
                "missing API key — send 'Authorization: Bearer <key>' or 'X-API-Key: <key>'"
                    .to_string(),
            ))?;
            let key = self
                .auth
                .authenticate(&presented)
                .ok_or((StatusCode::UNAUTHORIZED, "invalid API key".to_string()))?;
            if !key.may_access(&namespace) {
                return Err((
                    StatusCode::FORBIDDEN,
                    format!("this key is not scoped to namespace '{namespace}'"),
                ));
            }
        }
        let ns = self
            .registry
            .get_or_create(&namespace)
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
        Ok((ns, namespace))
    }

    /// Administrative routes need a key with unrestricted scope. With auth
    /// off, the server is already open and there is nothing to gate.
    fn require_admin(&self, headers: &HeaderMap) -> Result<(), Rejection> {
        if !self.auth.enabled() {
            return Ok(());
        }
        let presented = extract_key(headers)
            .ok_or((StatusCode::UNAUTHORIZED, "missing API key".to_string()))?;
        let key: &ApiKey = self
            .auth
            .authenticate(&presented)
            .ok_or((StatusCode::UNAUTHORIZED, "invalid API key".to_string()))?;
        if !key.is_admin() {
            return Err((
                StatusCode::FORBIDDEN,
                "this operation needs a key with access to all namespaces".to_string(),
            ));
        }
        Ok(())
    }
}

/// Bulk ingest sends raw embeddings, so request bodies are large by nature:
/// 10k memories at 1024 dims is ~125 MB of JSON. Axum defaults to 2 MB,
/// which rejects any real batch, so raise it well past that while keeping a
/// bound (an unbounded body is a denial-of-service invitation).
pub const MAX_BODY_BYTES: usize = 256 * 1024 * 1024;

pub fn router(registry: Arc<Registry>, auth: Auth) -> Router {
    Router::new()
        .route("/", get(dashboard))
        .route("/dashboard", get(dashboard))
        .route("/health", get(health))
        .route("/v1/memories", get(list_memories))
        .route("/v1/entities", get(entities))
        .route("/v1/remember", post(remember))
        .route("/v1/remember_batch", post(remember_batch))
        .route("/v1/checkpoint", post(checkpoint))
        .route("/v1/recall", post(recall))
        .route("/v1/forget", post(forget))
        .route("/v1/lifecycle/run", post(lifecycle_run))
        .route("/v1/snapshot", post(snapshot))
        .route("/v1/restore", post(restore))
        .route("/v1/namespaces", get(list_namespaces))
        .route("/v1/namespaces/drop", post(drop_namespace))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(AppState { registry, auth })
}

pub async fn serve(registry: Arc<Registry>, auth: Auth, addr: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("memrust listening on http://{addr}");
    if auth.enabled() {
        println!("authentication: enabled");
    } else {
        println!("authentication: DISABLED — anyone who can reach this port can read and write every memory");
        let public = !addr.starts_with("127.") && !addr.starts_with("localhost");
        if public {
            eprintln!(
                "warning: {addr} is not loopback and no --api-key was given; \
                 anyone who can reach it has full access"
            );
        }
    }
    axum::serve(listener, router(registry, auth)).await?;
    Ok(())
}

/// The embedded web UI (single self-contained HTML file, no external assets).
async fn dashboard() -> Html<&'static str> {
    Html(include_str!("dashboard.html"))
}

#[derive(Deserialize)]
struct ListQuery {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_page")]
    limit: usize,
}

fn default_page() -> usize {
    50
}

async fn list_memories(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<ListQuery>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let (ns, _) = state.resolve(&headers)?;
    let (total, records) = ns
        .engine
        .read()
        .unwrap()
        .list_memories(q.offset, q.limit.min(200));
    let records: Vec<_> = records.into_iter().map(without_embedding).collect();
    Ok(Json(json!({ "total": total, "records": records })))
}

#[derive(Deserialize)]
struct EntitiesQuery {
    #[serde(default = "default_entities")]
    limit: usize,
}

fn default_entities() -> usize {
    40
}

async fn entities(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<EntitiesQuery>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let (ns, _) = state.resolve(&headers)?;
    let entities: Vec<serde_json::Value> = ns
        .engine
        .read()
        .unwrap()
        .top_entities(q.limit.min(200))
        .into_iter()
        .map(|(name, count)| json!({ "name": name, "count": count }))
        .collect();
    Ok(Json(json!({ "entities": entities })))
}

async fn health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, Rejection> {
    let (ns, name) = state.resolve(&headers)?;
    let stats = ns.engine.read().unwrap().stats();
    Ok(Json(
        json!({ "status": "ok", "namespace": name, "stats": stats }),
    ))
}

// remember/recall may call a remote embedding API (blocking I/O), so they
// run on the blocking pool instead of stalling tokio workers.

async fn remember(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RememberRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let record = tokio::task::spawn_blocking(move || -> anyhow::Result<MemoryRecord> {
        let _commit = ns.commit.lock().expect("commit lock");
        // Persist under a read lock: concurrent recalls keep running while
        // this write waits on the disk.
        let record = ns.engine.read().unwrap().stage(req)?;
        // Make it visible: in-memory only, so the exclusive lock is brief.
        ns.engine
            .write()
            .unwrap()
            .apply_staged(std::iter::once(record.clone()));
        Ok(record)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(json!({ "record": without_embedding(record) })))
}

async fn recall(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RecallRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let mut hits = tokio::task::spawn_blocking(move || ns.engine.read().unwrap().recall(&req))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    for hit in &mut hits {
        hit.record.embedding = None;
    }
    Ok(Json(json!({ "hits": hits })))
}

#[derive(Deserialize)]
struct RememberBatchBody {
    items: Vec<RememberRequest>,
}

async fn remember_batch(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<RememberBatchBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let records = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryRecord>> {
        let _commit = ns.commit.lock().expect("commit lock");
        let records = ns.engine.read().unwrap().stage_batch(body.items)?;
        ns.engine.write().unwrap().apply_staged(records.clone());
        Ok(records)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let records: Vec<_> = records.into_iter().map(without_embedding).collect();
    Ok(Json(json!({ "records": records })))
}

async fn checkpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    tokio::task::spawn_blocking(move || {
        let _commit = ns.commit.lock().expect("commit lock");
        ns.engine.write().unwrap().checkpoint()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "checkpointed": true })))
}

async fn list_namespaces(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, Rejection> {
    state.require_admin(&headers)?;
    Ok(Json(json!({ "namespaces": state.registry.list() })))
}

#[derive(Deserialize)]
struct DropBody {
    namespace: String,
}

async fn drop_namespace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<DropBody>,
) -> Result<Json<serde_json::Value>, Rejection> {
    state.require_admin(&headers)?;
    let dropped = state
        .registry
        .drop_namespace(&body.namespace)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(json!({ "dropped": dropped })))
}

// Lifecycle runs the summarizer and embedder (possibly remote), so it goes
// on the blocking pool like remember/recall.
async fn lifecycle_run(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let report = tokio::task::spawn_blocking(move || {
        let _commit = ns.commit.lock().expect("commit lock");
        ns.engine.write().unwrap().run_lifecycle()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "report": report })))
}

#[derive(Deserialize)]
struct SnapshotBody {
    #[serde(default)]
    session_id: Option<String>,
}

async fn snapshot(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SnapshotBody>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let (ns, _) = state.resolve(&headers)?;
    let snapshot = ns
        .engine
        .read()
        .unwrap()
        .snapshot(body.session_id.as_deref());
    Ok(Json(json!({ "snapshot": snapshot })))
}

#[derive(Deserialize)]
struct RestoreBody {
    records: Vec<MemoryRecord>,
}

async fn restore(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<RestoreBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let restored = tokio::task::spawn_blocking(move || {
        let _commit = ns.commit.lock().expect("commit lock");
        ns.engine.write().unwrap().restore(body.records)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    Ok(Json(json!({ "restored": restored })))
}

#[derive(Deserialize)]
struct ForgetBody {
    id: Uuid,
}

async fn forget(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<ForgetBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (ns, _) = state.resolve(&headers)?;
    let forgotten = {
        let _commit = ns.commit.lock().expect("commit lock");
        ns.engine
            .write()
            .unwrap()
            .forget(body.id)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    Ok(Json(json!({ "forgotten": forgotten })))
}
