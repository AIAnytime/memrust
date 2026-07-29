//! The optional extraction layer, end to end against a scripted extractor.
//!
//! These use a fake `Extractor` rather than a mock HTTP server: the wire
//! format is already covered by unit tests in `src/extract.rs`, and what
//! matters here is the engine's behaviour around whatever a model returns —
//! what gets stored, what gets deduplicated, what provenance is recorded, and
//! what happens when the model is wrong.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use memrust::engine::MemoryEngine;
use memrust::extract::{Candidate, Extractor, Turn};
use memrust::types::{IngestRequest, MemoryKind, RecallRequest};

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-extract-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

/// Returns a scripted list of facts, and counts how often it was called.
struct Scripted {
    facts: Mutex<Vec<Vec<&'static str>>>,
    calls: AtomicUsize,
    supersede: Mutex<Vec<usize>>,
    supersede_calls: AtomicUsize,
}

impl Scripted {
    fn new(batches: Vec<Vec<&'static str>>) -> Arc<Self> {
        Arc::new(Self {
            facts: Mutex::new(batches),
            calls: AtomicUsize::new(0),
            supersede: Mutex::new(Vec::new()),
            supersede_calls: AtomicUsize::new(0),
        })
    }
    fn superseding(batches: Vec<Vec<&'static str>>, indices: Vec<usize>) -> Arc<Self> {
        let s = Self::new(batches);
        *s.supersede.lock().unwrap() = indices;
        s
    }
}

impl Extractor for Scripted {
    fn extract(&self, _turns: &[Turn], _known: &[String]) -> anyhow::Result<Vec<Candidate>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut batches = self.facts.lock().unwrap();
        let batch = if batches.is_empty() {
            Vec::new()
        } else {
            batches.remove(0)
        };
        Ok(batch
            .into_iter()
            .map(|text| Candidate {
                text: text.to_string(),
                kind: MemoryKind::Semantic,
                importance: 0.7,
                tags: Vec::new(),
                embedding: None,
            })
            .collect())
    }

    fn superseded_by(&self, _fact: &str, existing: &[String]) -> anyhow::Result<Vec<usize>> {
        self.supersede_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .supersede
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|&i| i < existing.len())
            .collect())
    }
}

fn turns(lines: &[(&str, &str)]) -> Vec<Turn> {
    lines
        .iter()
        .map(|(role, content)| Turn {
            role: role.to_string(),
            content: content.to_string(),
        })
        .collect()
}

fn ingest(turns: Vec<Turn>) -> IngestRequest {
    IngestRequest {
        turns,
        store_raw: true,
        extract: true,
        ..Default::default()
    }
}

#[test]
fn without_an_extractor_ingest_is_a_verbatim_write() {
    // The default posture: no model configured means no model involved, and
    // the exchange is still stored.
    let dir = tmp_dir("no-extractor");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let report = engine
        .ingest(ingest(turns(&[
            ("user", "cap the Redis pool at 64"),
            ("assistant", "done"),
        ])))
        .unwrap();

    assert_eq!(report.raw.len(), 2);
    assert!(report.extracted.is_empty());
    assert!(!report.extraction_ran);
    assert_eq!(engine.stats().total_memories, 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn extraction_stores_facts_alongside_the_raw_turns() {
    // The design claim: memrust keeps both, where an extract-only system keeps
    // only the distilled fact and cannot recover from getting it wrong.
    let dir = tmp_dir("both");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::new(vec![vec![
        "The team capped the Redis connection pool at 64.",
    ]]));

    let report = engine
        .ingest(ingest(turns(&[
            ("user", "we should cap the Redis pool at 64"),
            ("assistant", "agreed, capping it"),
        ])))
        .unwrap();

    assert_eq!(report.raw.len(), 2, "raw turns must survive extraction");
    assert_eq!(report.extracted.len(), 1);
    assert_eq!(report.proposed, 1);
    assert!(report.extraction_ran);
    assert_eq!(engine.stats().total_memories, 3);

    let fact = engine.get(&report.extracted[0]).unwrap();
    assert_eq!(fact.kind, MemoryKind::Semantic);
    // Provenance: the fact points back at the turns it came from, which is
    // what makes a bad extraction auditable rather than merely wrong.
    assert_eq!(fact.sources, report.raw);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn store_raw_false_drops_the_turns_but_keeps_the_facts() {
    let dir = tmp_dir("facts-only");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::new(vec![vec!["Martin prefers terse answers."]]));

    let report = engine
        .ingest(IngestRequest {
            turns: turns(&[("user", "keep it short please")]),
            store_raw: false,
            extract: true,
            ..Default::default()
        })
        .unwrap();

    assert!(report.raw.is_empty());
    assert_eq!(report.extracted.len(), 1);
    assert_eq!(engine.stats().total_memories, 1);
    // No raw turns means no provenance to record.
    assert!(engine.get(&report.extracted[0]).unwrap().sources.is_empty());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_repeated_fact_is_deduplicated_without_a_second_model_call() {
    let dir = tmp_dir("dedup");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let scripted = Scripted::new(vec![
        vec!["Martin Mark works at Huaxin Consulting."],
        vec!["Martin Mark works at Huaxin Consulting."],
    ]);
    engine.set_extractor(scripted.clone());

    let first = engine.ingest(ingest(turns(&[("user", "I work at Huaxin")]))).unwrap();
    assert_eq!(first.extracted.len(), 1);
    assert_eq!(first.duplicates, 0);

    let second = engine
        .ingest(ingest(turns(&[("user", "as I said, Huaxin")])))
        .unwrap();
    assert!(second.extracted.is_empty(), "identical fact stored twice");
    assert_eq!(second.duplicates, 1);
    assert_eq!(second.proposed, 1);

    // Two ingests, two extraction calls — and no routing call per candidate.
    assert_eq!(scripted.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        scripted.supersede_calls.load(Ordering::SeqCst),
        0,
        "supersede is off by default and must not call the model"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn duplicates_within_one_exchange_are_caught_too() {
    let dir = tmp_dir("dedup-self");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::new(vec![vec![
        "The deploy window is Thursday.",
        "The deploy window is Thursday.",
        "The rollback owner is Dana.",
    ]]));

    let report = engine.ingest(ingest(turns(&[("user", "deploy thursday")]))).unwrap();
    assert_eq!(report.proposed, 3);
    assert_eq!(report.duplicates, 1);
    assert_eq!(report.extracted.len(), 2);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn supersede_is_off_by_default_and_keeps_both_memories() {
    // The safety default. An LLM that routes updates wrongly should cost a
    // duplicate, not a deletion.
    let dir = tmp_dir("no-supersede");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::superseding(
        vec![
            vec!["Martin's deploy window is Thursday."],
            vec!["Martin's deploy window is Monday."],
        ],
        vec![0],
    ));

    engine.ingest(ingest(turns(&[("user", "thursday works")]))).unwrap();
    let second = engine.ingest(ingest(turns(&[("user", "actually monday")]))).unwrap();

    assert!(second.superseded.is_empty(), "deleted a memory without being asked");
    let hits = engine.recall(&RecallRequest {
        query: "deploy window".into(),
        ..Default::default()
    });
    let texts: Vec<&str> = hits.iter().map(|h| h.record.text.as_str()).collect();
    assert!(texts.iter().any(|t| t.contains("Thursday")));
    assert!(texts.iter().any(|t| t.contains("Monday")));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn supersede_when_asked_deletes_the_old_fact_and_keeps_its_provenance() {
    let dir = tmp_dir("supersede");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::superseding(
        vec![
            vec!["Martin's deploy window is Thursday."],
            vec!["Martin's deploy window is Monday."],
        ],
        vec![0],
    ));

    let first = engine
        .ingest(IngestRequest {
            turns: turns(&[("user", "thursday works")]),
            store_raw: false,
            extract: true,
            ..Default::default()
        })
        .unwrap();
    let old_id = first.extracted[0];

    let second = engine
        .ingest(IngestRequest {
            turns: turns(&[("user", "actually monday")]),
            store_raw: false,
            extract: true,
            supersede: true,
            ..Default::default()
        })
        .unwrap();

    assert_eq!(second.superseded, vec![old_id]);
    assert!(engine.get(&old_id).is_none(), "superseded memory still live");
    // The replacement records what it replaced, so the deletion is traceable.
    let new = engine.get(&second.extracted[0]).unwrap();
    assert!(new.sources.contains(&old_id));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_extractor_returning_nothing_is_not_an_error() {
    let dir = tmp_dir("empty");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::new(vec![vec![]]));

    let report = engine.ingest(ingest(turns(&[("user", "thanks!")]))).unwrap();
    assert_eq!(report.proposed, 0);
    assert!(report.extracted.is_empty());
    assert_eq!(report.raw.len(), 1, "the turn is still stored verbatim");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn extract_false_skips_the_model_entirely() {
    let dir = tmp_dir("opt-out");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let scripted = Scripted::new(vec![vec!["should never be reached"]]);
    engine.set_extractor(scripted.clone());

    let report = engine
        .ingest(IngestRequest {
            turns: turns(&[("user", "hello")]),
            store_raw: true,
            extract: false,
            ..Default::default()
        })
        .unwrap();

    assert_eq!(scripted.calls.load(Ordering::SeqCst), 0);
    assert!(!report.extraction_ran);
    assert_eq!(report.raw.len(), 1);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn extracted_facts_are_recallable_and_survive_restart() {
    let dir = tmp_dir("durable");
    let fact_id;
    {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        engine.set_extractor(Scripted::new(vec![vec![
            "Globex renewed at $120k ARR for 2027.",
        ]]));
        let report = engine
            .ingest(ingest(turns(&[("user", "globex renewed, 120k")])))
            .unwrap();
        fact_id = report.extracted[0];
    }
    let engine = MemoryEngine::open(&dir).unwrap();
    let stored = engine.get(&fact_id).expect("extracted fact lost on restart");
    assert!(stored.text.contains("120k"));
    // Provenance must survive too — it is written through a separate WAL op.
    assert_eq!(stored.sources.len(), 1);

    let hits = engine.recall(&RecallRequest {
        query: "Globex renewal".into(),
        ..Default::default()
    });
    assert!(hits.iter().any(|h| h.record.id == fact_id));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn extraction_respects_agent_ownership() {
    let dir = tmp_dir("agent");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    engine.set_extractor(Scripted::new(vec![vec!["The planner chose route B."]]));

    let report = engine
        .ingest(IngestRequest {
            turns: turns(&[("user", "take route B")]),
            agent_id: Some("planner".into()),
            store_raw: false,
            extract: true,
            ..Default::default()
        })
        .unwrap();

    let fact = engine.get(&report.extracted[0]).unwrap();
    assert_eq!(fact.agent_id.as_deref(), Some("planner"));
    // agent_id set with no explicit visibility means private, same as remember.
    assert_eq!(fact.visibility, memrust::types::Visibility::Private);

    let other = engine.recall(&RecallRequest {
        query: "route B".into(),
        as_agent: Some("writer".into()),
        ..Default::default()
    });
    assert!(other.is_empty(), "another agent recalled a private extracted fact");
    std::fs::remove_dir_all(&dir).ok();
}
