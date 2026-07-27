use anyhow::Result;
use memrust::engine::MemoryEngine;
use memrust::rerank::Reranker;
use memrust::types::{MemoryFilter, MemoryKind, RecallRequest, RecallStrategy, RememberRequest};

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-v04-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn remember_in(engine: &mut MemoryEngine, text: &str, session: Option<&str>) {
    engine
        .remember(RememberRequest {
            text: text.into(),
            session_id: session.map(String::from),
            ..Default::default()
        })
        .unwrap();
}

/// The pre-filtering fix: a selective filter must yield a full result set,
/// not the leftovers of filtering a global candidate pool. 300 on-topic
/// memories in session "big" would previously crowd session "small" out of
/// the top-100 candidates entirely.
#[test]
fn selective_filters_get_full_results() {
    let dir = tmp_dir("prefilter");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    for i in 0..300 {
        remember_in(
            &mut engine,
            &format!("database migration progress update number {i}"),
            Some("big"),
        );
    }
    for i in 0..5 {
        remember_in(
            &mut engine,
            &format!("database migration note {i} for the side project"),
            Some("small"),
        );
    }

    let hits = engine.recall(&RecallRequest {
        query: "database migration".into(),
        top_k: Some(5),
        filter: MemoryFilter {
            session_id: Some("small".into()),
            ..Default::default()
        },
        ..Default::default()
    });
    assert_eq!(
        hits.len(),
        5,
        "pre-filtering must find all 5 session-small memories"
    );
    assert!(hits
        .iter()
        .all(|h| h.record.session_id.as_deref() == Some("small")));
    std::fs::remove_dir_all(&dir).ok();
}

/// Graph leg: a query naming an entity finds directly-mentioning records,
/// and 1-hop co-occurrence reaches records with *no* textual overlap with
/// the query at all.
#[test]
fn graph_signal_finds_related_not_just_similar() {
    let dir = tmp_dir("graph");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    remember_in(
        &mut engine,
        "Project Phoenix depends on the Billing Service",
        None,
    );
    remember_in(&mut engine, "Dana Whitfield leads Project Phoenix", None);
    // Related only through the graph: mentions Billing Service, never Phoenix.
    remember_in(
        &mut engine,
        "the Billing Service rate limits at 100 requests per second",
        None,
    );
    // Noise.
    for i in 0..20 {
        remember_in(
            &mut engine,
            &format!("unrelated standup note number {i}"),
            None,
        );
    }

    let hits = engine.recall(&RecallRequest {
        query: "tell me about Project Phoenix".into(),
        top_k: Some(3),
        strategy: RecallStrategy::Relational,
        ..Default::default()
    });
    let texts: Vec<&str> = hits.iter().map(|h| h.record.text.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("depends on the Billing")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("Dana Whitfield")),
        "{texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("rate limits")),
        "1-hop entity expansion should reach the billing-only record: {texts:?}"
    );
    assert!(hits.iter().any(|h| h.signals.graph > 0.0));
    std::fs::remove_dir_all(&dir).ok();
}

/// Entities persist through WAL replay and the graph survives restart.
#[test]
fn graph_survives_restart_and_forget() {
    let dir = tmp_dir("graph-durable");
    {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        remember_in(&mut engine, "Atlas Cluster hosts the search index", None);
        assert!(engine.stats().entities >= 1);
    }
    {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        assert!(engine.stats().entities >= 1, "graph rebuilt after restart");
        let hits = engine.recall(&RecallRequest {
            query: "what runs on Atlas Cluster".into(),
            strategy: RecallStrategy::Relational,
            ..Default::default()
        });
        assert!(hits[0].signals.graph > 0.0);
        let id = hits[0].record.id;
        engine.forget(id).unwrap();
        assert_eq!(
            engine.stats().entities,
            0,
            "forget must remove graph entries"
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn procedural_kind_roundtrips() {
    let dir = tmp_dir("procedural");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let rec = engine
        .remember(RememberRequest {
            text: "to deploy: run make release, then trigger the pipeline".into(),
            kind: MemoryKind::Procedural,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rec.kind, MemoryKind::Procedural);
    let hits = engine.recall(&RecallRequest {
        query: "how do I deploy".into(),
        filter: MemoryFilter {
            kinds: vec![MemoryKind::Procedural],
            ..Default::default()
        },
        ..Default::default()
    });
    assert_eq!(hits.len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

/// A reranker reorders the fused pool; scores land in signals.rerank while
/// `score` keeps the fused value for explainability.
struct KeywordReranker(&'static str);
impl Reranker for KeywordReranker {
    fn rerank(&self, _query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        Ok(docs
            .iter()
            .map(|d| if d.contains(self.0) { 1.0 } else { 0.1 })
            .collect())
    }
}

#[test]
fn reranker_reorders_hits() {
    let dir = tmp_dir("rerank");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    remember_in(&mut engine, "notes about search latency in general", None);
    remember_in(
        &mut engine,
        "search latency doc mentioning the magic keyword",
        None,
    );
    remember_in(&mut engine, "another search latency status update", None);
    engine.set_reranker(Box::new(KeywordReranker("magic")));

    let hits = engine.recall(&RecallRequest {
        query: "search latency".into(),
        top_k: Some(3),
        ..Default::default()
    });
    assert!(
        hits[0].record.text.contains("magic"),
        "reranker should promote the magic doc"
    );
    assert!((hits[0].signals.rerank - 1.0).abs() < 1e-6);

    // Per-request opt-out returns fused order untouched.
    let hits = engine.recall(&RecallRequest {
        query: "search latency".into(),
        top_k: Some(3),
        rerank: Some(false),
        ..Default::default()
    });
    assert!(hits.iter().all(|h| h.signals.rerank == 0.0));
    std::fs::remove_dir_all(&dir).ok();
}
