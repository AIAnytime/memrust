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

// ---------------------------------------------------------------------------
// created_at: importing history with the times it actually happened.
// ---------------------------------------------------------------------------

const DAY_MS: i64 = 24 * 3600 * 1000;

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

#[test]
fn created_at_is_honoured_and_defaults_to_now() {
    let dir = tmp_dir("created-at");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let backdated = now_ms() - 30 * DAY_MS;

    let old = engine
        .remember(RememberRequest {
            text: "the Q2 planning call happened".into(),
            created_at: Some(backdated),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(old.created_at, backdated);

    let fresh = engine
        .remember(RememberRequest {
            text: "the Q3 planning call happened".into(),
            ..Default::default()
        })
        .unwrap();
    assert!(
        (now_ms() - fresh.created_at).abs() < 5_000,
        "unset created_at should mean now, got {}",
        fresh.created_at
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn backdated_memories_decay_in_recall() {
    // The whole point of the field: an imported memory must not look as fresh
    // as one written today, or the recency signal is decoration.
    let dir = tmp_dir("created-at-decay");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine
        .remember(RememberRequest {
            text: "deployment freeze agreed for the release window".into(),
            created_at: Some(now_ms() - 60 * DAY_MS),
            ..Default::default()
        })
        .unwrap();
    engine
        .remember(RememberRequest {
            text: "deployment freeze lifted for the release window".into(),
            ..Default::default()
        })
        .unwrap();

    let hits = engine.recall(&RecallRequest {
        query: "deployment freeze release window".into(),
        strategy: RecallStrategy::Recent,
        ..Default::default()
    });
    assert_eq!(hits.len(), 2);
    let old = hits.iter().find(|h| h.record.text.contains("agreed")).unwrap();
    let new = hits.iter().find(|h| h.record.text.contains("lifted")).unwrap();
    assert!(
        old.signals.recency < new.signals.recency,
        "60-day-old memory scored recency {} vs fresh {}",
        old.signals.recency,
        new.signals.recency
    );
    // Two months is ~8.6 half-lives, so decay should be heavy, not marginal.
    assert!(old.signals.recency < 0.05, "got {}", old.signals.recency);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ttl_counts_from_created_at_not_from_the_write() {
    // Re-importing a snapshot must not keep resurrecting memories that were
    // already dead. An hour-old memory with a 30-minute TTL is expired on
    // arrival, however late it is written.
    let dir = tmp_dir("created-at-ttl");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let rec = engine
        .remember(RememberRequest {
            text: "scratch state from an hour ago".into(),
            created_at: Some(now_ms() - 3600 * 1000),
            ttl_seconds: Some(1800),
            ..Default::default()
        })
        .unwrap();
    assert!(rec.expires_at.unwrap() < now_ms(), "should be expired already");
    let hits = engine.recall(&RecallRequest {
        query: "scratch state".into(),
        ..Default::default()
    });
    assert!(hits.is_empty(), "expired memory must not be recallable");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn working_memories_inherit_the_default_ttl_from_created_at() {
    let dir = tmp_dir("created-at-working");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let backdated = now_ms() - 3 * DAY_MS;
    let rec = engine
        .remember(RememberRequest {
            text: "half-finished draft from three days ago".into(),
            kind: MemoryKind::Working,
            created_at: Some(backdated),
            ..Default::default()
        })
        .unwrap();
    // Default working TTL is one day, so three days back is long gone.
    assert_eq!(rec.expires_at, Some(backdated + DAY_MS));
    assert!(rec.expires_at.unwrap() < now_ms());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn seconds_where_milliseconds_belong_is_rejected() {
    // The silent-failure case: accepted, the memory is dated 1970 and its
    // recency is pinned at zero with nothing to show you why.
    let dir = tmp_dir("created-at-units");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let err = engine
        .remember(RememberRequest {
            text: "written with a seconds timestamp".into(),
            created_at: Some(1_785_306_704),
            ..Default::default()
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("seconds"), "unhelpful error: {err}");
    assert!(err.contains("1785306704000"), "should suggest the fix: {err}");

    for bad in [0, -1] {
        assert!(engine
            .remember(RememberRequest {
                text: "nonsense timestamp".into(),
                created_at: Some(bad),
                ..Default::default()
            })
            .is_err());
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn batch_ingest_honours_per_item_created_at() {
    let dir = tmp_dir("created-at-batch");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let base = now_ms() - 10 * DAY_MS;
    let records = engine
        .remember_batch(
            (0..5)
                .map(|i| RememberRequest {
                    text: format!("imported transcript turn {i}"),
                    created_at: Some(base + i * 1000),
                    ..Default::default()
                })
                .collect(),
        )
        .unwrap();
    for (i, r) in records.iter().enumerate() {
        assert_eq!(r.created_at, base + i as i64 * 1000);
    }
    // A bad item must fail the whole batch rather than land half of it.
    assert!(engine
        .remember_batch(vec![
            RememberRequest {
                text: "good".into(),
                created_at: Some(base),
                ..Default::default()
            },
            RememberRequest {
                text: "bad".into(),
                created_at: Some(42),
                ..Default::default()
            },
        ])
        .is_err());
    assert_eq!(engine.stats().total_memories, 5, "partial batch was applied");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn created_at_survives_restart() {
    let dir = tmp_dir("created-at-recovery");
    let backdated = now_ms() - 45 * DAY_MS;
    let id = {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        engine
            .remember(RememberRequest {
                text: "an imported memory that must keep its date".into(),
                created_at: Some(backdated),
                ..Default::default()
            })
            .unwrap()
            .id
    };
    let engine = MemoryEngine::open(&dir).unwrap();
    assert_eq!(engine.get(&id).unwrap().created_at, backdated);
    std::fs::remove_dir_all(&dir).ok();
}
