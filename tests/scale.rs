use std::path::{Path, PathBuf};

use memrust::embed::HashEmbedder;
use memrust::engine::MemoryEngine;
use memrust::index::vector::HnswConfig;
use memrust::summarize::ExtractiveSummarizer;
use memrust::types::{LifecycleConfig, MemoryKind, RecallRequest, RememberRequest};

fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-scale-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn open(dir: &Path, quantize: bool) -> MemoryEngine {
    MemoryEngine::open_with_options(
        dir,
        Box::new(HashEmbedder::new(256)),
        Box::new(ExtractiveSummarizer::default()),
        LifecycleConfig::default(),
        HnswConfig {
            quantize,
            ..HnswConfig::default()
        },
    )
    .unwrap()
}

fn wal_lines(dir: &Path) -> usize {
    std::fs::read_to_string(dir.join("memory.wal"))
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
        .unwrap_or(0)
}

#[test]
fn checkpoint_plus_tail_recovery() {
    let dir = tmp_dir("checkpoint");
    {
        let mut engine = open(&dir, false);
        for i in 0..5 {
            engine
                .remember(RememberRequest {
                    text: format!("pre-checkpoint fact number {i}"),
                    ..Default::default()
                })
                .unwrap();
        }
        engine.checkpoint().unwrap();
        assert_eq!(wal_lines(&dir), 0, "checkpoint must truncate the WAL");
        assert_eq!(engine.stats().wal_tail_ops, 0);

        // Post-checkpoint ops form the tail.
        engine
            .remember(RememberRequest {
                text: "tail fact after the checkpoint".into(),
                ..Default::default()
            })
            .unwrap();
        engine
            .remember(RememberRequest {
                text: "second tail fact".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(wal_lines(&dir), 2);
    }
    // Reopen: checkpoint + 2-op tail, not a 7-op replay.
    let engine = open(&dir, false);
    assert_eq!(engine.stats().total_memories, 7);
    let hits = engine.recall(&RecallRequest {
        query: "tail fact checkpoint".into(),
        ..Default::default()
    });
    assert!(hits.iter().any(|h| h.record.text.contains("tail fact")));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn forgetting_survives_checkpoint_and_tail() {
    let dir = tmp_dir("forget-tail");
    let victim = {
        let mut engine = open(&dir, false);
        let keep = engine
            .remember(RememberRequest {
                text: "durable fact".into(),
                ..Default::default()
            })
            .unwrap();
        let victim = engine
            .remember(RememberRequest {
                text: "doomed fact".into(),
                ..Default::default()
            })
            .unwrap();
        engine.checkpoint().unwrap();
        // Forget lands in the tail, after the checkpoint captured the record.
        assert!(engine.forget(victim.id).unwrap());
        drop(keep);
        victim.id
    };
    let engine = open(&dir, false);
    assert!(
        engine.get(&victim).is_none(),
        "tail forget must apply over checkpoint state"
    );
    assert_eq!(engine.stats().total_memories, 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn quantized_engine_end_to_end() {
    let dir = tmp_dir("quantized");
    {
        let mut engine = open(&dir, true);
        assert!(engine.stats().quantized);
        for text in [
            "the billing service uses stripe for payments",
            "deploys run through github actions",
            "the search cluster lives in us-east-1",
        ] {
            engine
                .remember(RememberRequest {
                    text: text.into(),
                    kind: MemoryKind::Semantic,
                    ..Default::default()
                })
                .unwrap();
        }
        let hits = engine.recall(&RecallRequest {
            query: "how do payments work".into(),
            top_k: Some(1),
            ..Default::default()
        });
        assert!(hits[0].record.text.contains("stripe"));
        engine.checkpoint().unwrap();
    }
    // Quantized index round-trips through the checkpoint.
    let engine = open(&dir, true);
    assert!(engine.stats().quantized);
    let hits = engine.recall(&RecallRequest {
        query: "payments provider".into(),
        top_k: Some(1),
        ..Default::default()
    });
    assert!(hits[0].record.text.contains("stripe"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn remember_batch_ingests_and_recalls() {
    let dir = tmp_dir("batch");
    let mut engine = open(&dir, false);
    let reqs: Vec<RememberRequest> = (0..10)
        .map(|i| RememberRequest {
            text: format!("batch document {i} about vector quantization"),
            ..Default::default()
        })
        .collect();
    let records = engine.remember_batch(reqs).unwrap();
    assert_eq!(records.len(), 10);
    assert_eq!(engine.stats().total_memories, 10);
    let hits = engine.recall(&RecallRequest {
        query: "vector quantization documents".into(),
        ..Default::default()
    });
    assert!(!hits.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
