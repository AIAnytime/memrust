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
//! GET  /health                             -> EngineStats
//! GET  /                                   -> embedded web dashboard

use std::sync::{Arc, Mutex, RwLock};

use axum::extract::{DefaultBodyLimit, Query, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use uuid::Uuid;

use crate::engine::MemoryEngine;
use crate::types::{MemoryRecord, RecallRequest, RememberRequest};

/// API responses omit raw embeddings; they're an internal representation.
fn without_embedding(mut record: MemoryRecord) -> MemoryRecord {
    record.embedding = None;
    record
}

type Shared = Arc<RwLock<MemoryEngine>>;

/// Writes run in two phases — persist under a *read* lock (so readers are not
/// blocked for the fsync), then apply under the write lock. This mutex
/// serializes writers across both phases. Without it a checkpoint could land
/// between them, saving state that lacks the record and truncating the WAL
/// entry that describes it, which would lose an acknowledged write.
#[derive(Clone)]
struct AppState {
    engine: Shared,
    commit: Arc<Mutex<()>>,
}

impl AppState {
    fn new(engine: Shared) -> Self {
        Self {
            engine,
            commit: Arc::new(Mutex::new(())),
        }
    }
}

/// Bulk ingest sends raw embeddings, so request bodies are large by nature:
/// 10k memories at 1024 dims is ~125 MB of JSON. Axum defaults to 2 MB,
/// which rejects any real batch, so raise it well past that while keeping a
/// bound (an unbounded body is a denial-of-service invitation).
pub const MAX_BODY_BYTES: usize = 256 * 1024 * 1024;

pub fn router(engine: Shared) -> Router {
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
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(AppState::new(engine))
}

pub async fn serve(engine: Shared, addr: &str) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("memrust listening on http://{addr}");
    axum::serve(listener, router(engine)).await?;
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
    Query(q): Query<ListQuery>,
) -> Json<serde_json::Value> {
    let (total, records) = state
        .engine
        .read()
        .unwrap()
        .list_memories(q.offset, q.limit.min(200));
    let records: Vec<_> = records.into_iter().map(without_embedding).collect();
    Json(json!({ "total": total, "records": records }))
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
    Query(q): Query<EntitiesQuery>,
) -> Json<serde_json::Value> {
    let entities: Vec<serde_json::Value> = state
        .engine
        .read()
        .unwrap()
        .top_entities(q.limit.min(200))
        .into_iter()
        .map(|(name, count)| json!({ "name": name, "count": count }))
        .collect();
    Json(json!({ "entities": entities }))
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let stats = state.engine.read().unwrap().stats();
    Json(json!({ "status": "ok", "stats": stats }))
}

// remember/recall may call a remote embedding API (blocking I/O), so they
// run on the blocking pool instead of stalling tokio workers.

async fn remember(
    State(state): State<AppState>,
    Json(req): Json<RememberRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let record = tokio::task::spawn_blocking(move || -> anyhow::Result<MemoryRecord> {
        let _commit = state.commit.lock().expect("commit lock");
        // Persist under a read lock: concurrent recalls keep running while
        // this write waits on the disk.
        let record = state.engine.read().unwrap().stage(req)?;
        // Make it visible: in-memory only, so the exclusive lock is brief.
        state
            .engine
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
    Json(req): Json<RecallRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut hits = tokio::task::spawn_blocking(move || state.engine.read().unwrap().recall(&req))
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
    Json(body): Json<RememberBatchBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let records = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<MemoryRecord>> {
        let _commit = state.commit.lock().expect("commit lock");
        let records = state.engine.read().unwrap().stage_batch(body.items)?;
        state.engine.write().unwrap().apply_staged(records.clone());
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
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || {
        let _commit = state.commit.lock().expect("commit lock");
        state.engine.write().unwrap().checkpoint()
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(json!({ "checkpointed": true })))
}

// Lifecycle runs the summarizer and embedder (possibly remote), so it goes
// on the blocking pool like remember/recall.
async fn lifecycle_run(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let report = tokio::task::spawn_blocking(move || {
        let _commit = state.commit.lock().expect("commit lock");
        state.engine.write().unwrap().run_lifecycle()
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
    Json(body): Json<SnapshotBody>,
) -> Json<serde_json::Value> {
    let snapshot = state
        .engine
        .read()
        .unwrap()
        .snapshot(body.session_id.as_deref());
    Json(json!({ "snapshot": snapshot }))
}

#[derive(Deserialize)]
struct RestoreBody {
    records: Vec<MemoryRecord>,
}

async fn restore(
    State(state): State<AppState>,
    Json(body): Json<RestoreBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let restored = tokio::task::spawn_blocking(move || {
        let _commit = state.commit.lock().expect("commit lock");
        state.engine.write().unwrap().restore(body.records)
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
    Json(body): Json<ForgetBody>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let forgotten = {
        let _commit = state.commit.lock().expect("commit lock");
        state
            .engine
            .write()
            .unwrap()
            .forget(body.id)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    };
    Ok(Json(json!({ "forgotten": forgotten })))
}
