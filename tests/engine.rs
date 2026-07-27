use memrust::engine::MemoryEngine;
use memrust::types::{MemoryFilter, MemoryKind, RecallRequest, RecallStrategy, RememberRequest};

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-test-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn remember(engine: &mut MemoryEngine, text: &str, kind: MemoryKind) -> uuid::Uuid {
    engine
        .remember(RememberRequest {
            text: text.to_string(),
            kind,
            ..Default::default()
        })
        .unwrap()
        .id
}

fn recall_texts(engine: &MemoryEngine, query: &str) -> Vec<String> {
    engine
        .recall(&RecallRequest {
            query: query.to_string(),
            ..Default::default()
        })
        .into_iter()
        .map(|h| h.record.text)
        .collect()
}

#[test]
fn remember_then_recall() {
    let dir = tmp_dir("roundtrip");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    remember(
        &mut engine,
        "the sky was cloudy over the data center",
        MemoryKind::Episodic,
    );
    remember(
        &mut engine,
        "pricing for the enterprise tier is $99 per seat",
        MemoryKind::Semantic,
    );
    remember(
        &mut engine,
        "customer asked for a discount on enterprise pricing",
        MemoryKind::Episodic,
    );

    let texts = recall_texts(&engine, "enterprise pricing discussion");
    assert!(
        texts[0].contains("pricing") || texts[0].contains("enterprise"),
        "got: {texts:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn survives_restart() {
    let dir = tmp_dir("restart");
    let id = {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        remember(
            &mut engine,
            "the wal replay should restore this memory",
            MemoryKind::Semantic,
        )
    };
    let engine = MemoryEngine::open(&dir).unwrap();
    assert!(engine.get(&id).is_some());
    assert_eq!(engine.stats().total_memories, 1);
    let texts = recall_texts(&engine, "wal replay restore");
    assert_eq!(texts.len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn forget_is_durable_and_compaction_works() {
    let dir = tmp_dir("forget");
    let (keep, drop) = {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        let keep = remember(&mut engine, "keep this fact around", MemoryKind::Semantic);
        let drop = remember(
            &mut engine,
            "sensitive thing to forget",
            MemoryKind::Episodic,
        );
        assert!(engine.forget(drop).unwrap());
        assert!(!engine.forget(drop).unwrap());
        (keep, drop)
    };
    let mut engine = MemoryEngine::open(&dir).unwrap();
    assert!(engine.get(&keep).is_some());
    assert!(engine.get(&drop).is_none());
    assert!(recall_texts(&engine, "sensitive forget")
        .iter()
        .all(|t| !t.contains("sensitive")));

    engine.compact().unwrap();
    let engine = MemoryEngine::open(&dir).unwrap();
    assert_eq!(engine.stats().total_memories, 1);
    assert!(engine.get(&keep).is_some());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn filters_scope_recall() {
    let dir = tmp_dir("filters");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine
        .remember(RememberRequest {
            text: "session alpha: user wants dark mode".into(),
            session_id: Some("alpha".into()),
            ..Default::default()
        })
        .unwrap();
    engine
        .remember(RememberRequest {
            text: "session beta: user wants light mode".into(),
            session_id: Some("beta".into()),
            ..Default::default()
        })
        .unwrap();

    let hits = engine.recall(&RecallRequest {
        query: "what mode does the user want".into(),
        filter: MemoryFilter {
            session_id: Some("alpha".into()),
            ..Default::default()
        },
        ..Default::default()
    });
    assert_eq!(hits.len(), 1);
    assert!(hits[0].record.text.contains("alpha"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn lexical_strategy_finds_exact_identifiers() {
    let dir = tmp_dir("lexical");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    remember(
        &mut engine,
        "incident INC-90312 was caused by a bad rollout",
        MemoryKind::Episodic,
    );
    for i in 0..20 {
        remember(
            &mut engine,
            &format!("routine status update number {i}"),
            MemoryKind::Episodic,
        );
    }
    let hits = engine.recall(&RecallRequest {
        query: "INC-90312".into(),
        strategy: RecallStrategy::Lexical,
        top_k: Some(1),
        ..Default::default()
    });
    assert!(hits[0].record.text.contains("INC-90312"));
    std::fs::remove_dir_all(&dir).ok();
}
