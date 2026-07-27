use std::path::{Path, PathBuf};

use memrust::embed::HashEmbedder;
use memrust::engine::MemoryEngine;
use memrust::summarize::ExtractiveSummarizer;
use memrust::types::{LifecycleConfig, MemoryKind, RecallRequest, RememberRequest};

fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-lc-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn open_with(dir: &Path, cfg: LifecycleConfig) -> MemoryEngine {
    MemoryEngine::open_full(
        dir,
        Box::new(HashEmbedder::new(256)),
        Box::new(ExtractiveSummarizer::default()),
        cfg,
    )
    .unwrap()
}

fn remember_kind(engine: &mut MemoryEngine, text: &str, kind: MemoryKind, session: Option<&str>) {
    engine
        .remember(RememberRequest {
            text: text.into(),
            kind,
            session_id: session.map(String::from),
            ..Default::default()
        })
        .unwrap();
}

#[test]
fn working_memory_expires_and_is_swept() {
    let dir = tmp_dir("ttl");
    {
        let mut engine = open_with(&dir, LifecycleConfig::default());
        // ttl 0 => expired the moment it lands.
        engine
            .remember(RememberRequest {
                text: "scratch state for the current step".into(),
                kind: MemoryKind::Working,
                ttl_seconds: Some(0),
                ..Default::default()
            })
            .unwrap();
        // Default working TTL applies when unset.
        let rec = engine
            .remember(RememberRequest {
                text: "still-fresh working note".into(),
                kind: MemoryKind::Working,
                ..Default::default()
            })
            .unwrap();
        assert!(rec.expires_at.is_some());

        // Expired memory is invisible to recall even before the sweep.
        let hits = engine.recall(&RecallRequest {
            query: "scratch state current step".into(),
            ..Default::default()
        });
        assert!(hits.iter().all(|h| !h.record.text.contains("scratch")));

        let report = engine.run_lifecycle().unwrap();
        assert_eq!(report.expired_swept, 1);
        assert_eq!(engine.stats().total_memories, 1);
    }
    // The sweep is durable.
    let engine = open_with(&dir, LifecycleConfig::default());
    assert_eq!(engine.stats().total_memories, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn old_episodic_memories_consolidate_into_semantic_summary() {
    let dir = tmp_dir("consolidate");
    let cfg = LifecycleConfig {
        consolidate_after_secs: 0, // everything is immediately "old"
        min_batch: 3,
        max_batch: 10,
        ..LifecycleConfig::default()
    };
    {
        let mut engine = open_with(&dir, cfg.clone());
        for text in [
            "deploy 41 failed on cluster prod-west with timeout",
            "deploy 42 failed on cluster prod-west with timeout",
            "deploy 43 rolled back after the prod-west timeout",
            "deploy 44 succeeded once the prod-west timeout was fixed",
            "postmortem: prod-west timeouts came from a stale dns cache",
        ] {
            remember_kind(&mut engine, text, MemoryKind::Episodic, Some("ops"));
        }
        // Below min_batch: left alone.
        remember_kind(
            &mut engine,
            "one-off note in another session",
            MemoryKind::Episodic,
            Some("misc"),
        );
        // Non-episodic: never consolidated.
        remember_kind(
            &mut engine,
            "the oncall rotation is weekly",
            MemoryKind::Semantic,
            Some("ops"),
        );

        let report = engine.run_lifecycle().unwrap();
        assert_eq!(report.batches_consolidated, 1);
        assert_eq!(report.summaries.len(), 1);

        // 5 episodic replaced by 1 summary; the other two survive.
        assert_eq!(engine.stats().total_memories, 3);
        let summary = engine.get(&report.summaries[0]).unwrap();
        assert_eq!(summary.kind, MemoryKind::Semantic);
        assert_eq!(summary.sources.len(), 5);
        assert_eq!(summary.session_id.as_deref(), Some("ops"));
        for id in &summary.sources {
            assert!(
                engine.get(id).is_none(),
                "source memory should be forgotten"
            );
        }

        // The summary is retrievable in place of its sources.
        let hits = engine.recall(&RecallRequest {
            query: "what happened with prod-west deploys".into(),
            top_k: Some(3),
            ..Default::default()
        });
        assert!(hits.iter().any(|h| h.record.id == summary.id));
    }
    // Consolidation is durable across restart.
    let engine = open_with(&dir, cfg);
    assert_eq!(engine.stats().total_memories, 3);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn snapshot_restore_roundtrip_is_idempotent() {
    let dir_a = tmp_dir("snap-a");
    let dir_b = tmp_dir("snap-b");
    let mut a = open_with(&dir_a, LifecycleConfig::default());
    for text in ["alpha fact one", "alpha fact two", "alpha fact three"] {
        remember_kind(&mut a, text, MemoryKind::Semantic, Some("alpha"));
    }
    remember_kind(&mut a, "beta fact", MemoryKind::Semantic, Some("beta"));

    let snap = a.snapshot(Some("alpha"));
    assert_eq!(snap.records.len(), 3);

    let mut b = open_with(&dir_b, LifecycleConfig::default());
    assert_eq!(b.restore(snap.records.clone()).unwrap(), 3);
    assert_eq!(
        b.restore(snap.records.clone()).unwrap(),
        0,
        "restore must be idempotent"
    );
    assert_eq!(b.stats().total_memories, 3);

    // Ids survive the round-trip.
    assert!(b.get(&snap.records[0].id).is_some());
    let hits = b.recall(&RecallRequest {
        query: "alpha fact".into(),
        ..Default::default()
    });
    assert!(!hits.is_empty());
    std::fs::remove_dir_all(&dir_a).ok();
    std::fs::remove_dir_all(&dir_b).ok();
}
