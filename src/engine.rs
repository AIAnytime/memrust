//! The memory engine: an agent-native facade (`remember` / `recall` /
//! `forget`) over hybrid retrieval. Recall fans out to the HNSW vector index
//! and the BM25 lexical index, fuses both rankings with reciprocal-rank
//! fusion, then applies recency decay and importance boosts. Scores are
//! returned decomposed per signal so callers can see *why* something
//! surfaced.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::embed::{Embedder, HashEmbedder};
use crate::index::graph::{extract_entities, GraphIndex};
use crate::index::text::Bm25Index;
use crate::index::vector::{normalize, Hnsw, HnswConfig};
use crate::rerank::Reranker;
use crate::store::{load_checkpoint, save_checkpoint, Wal, WalOp};
use crate::summarize::{ExtractiveSummarizer, Summarizer};
use crate::types::*;

const RRF_K: f32 = 60.0;
const CANDIDATES: usize = 100;
const DEFAULT_TOP_K: usize = 10;
/// Recency half-life: one week.
const HALF_LIFE_MS: f64 = 7.0 * 24.0 * 3600.0 * 1000.0;
const IMPORTANCE_WEIGHT: f32 = 0.005;
/// Auto-checkpoint (during the lifecycle pass) once the WAL tail grows past
/// this many ops, bounding restart replay time.
const CHECKPOINT_AFTER_OPS: usize = 1000;

fn is_expired(record: &MemoryRecord, now: i64) -> bool {
    record.expires_at.is_some_and(|t| t <= now)
}

/// Multi-agent visibility: an unscoped recall sees everything; a recall
/// `as_agent` sees shared memories, unowned memories, and its own.
fn visible_to(record: &MemoryRecord, as_agent: Option<&str>) -> bool {
    match as_agent {
        None => true,
        Some(agent) => {
            record.visibility == Visibility::Shared
                || record.agent_id.is_none()
                || record.agent_id.as_deref() == Some(agent)
        }
    }
}

/// Serialized derived state: records plus both *built* indexes, so a restart
/// deserializes instead of re-running index construction.
#[derive(Serialize)]
struct CheckpointRef<'a> {
    records: &'a HashMap<Uuid, MemoryRecord>,
    vec_index: &'a Hnsw,
    text_index: &'a Bm25Index,
    vec_ids: &'a Vec<Uuid>,
    text_ids: &'a Vec<Uuid>,
    loc: &'a HashMap<Uuid, (Option<usize>, Option<usize>)>,
    graph: &'a GraphIndex,
    index_dim: Option<usize>,
}

#[derive(Deserialize)]
struct CheckpointOwned {
    records: HashMap<Uuid, MemoryRecord>,
    vec_index: Hnsw,
    text_index: Bm25Index,
    vec_ids: Vec<Uuid>,
    text_ids: Vec<Uuid>,
    loc: HashMap<Uuid, (Option<usize>, Option<usize>)>,
    /// Absent in pre-v0.4 checkpoints; rebuilt from records when empty.
    #[serde(default)]
    graph: GraphIndex,
    index_dim: Option<usize>,
}

pub struct MemoryEngine {
    /// Behind its own lock: an fsync is milliseconds, and holding the
    /// engine's exclusive lock for that long blocks every reader.
    wal: Mutex<Wal>,
    embedder: Box<dyn Embedder>,
    summarizer: Box<dyn Summarizer>,
    cfg: LifecycleConfig,
    index_cfg: HnswConfig,
    dir: std::path::PathBuf,
    records: HashMap<Uuid, MemoryRecord>,
    vec_index: Hnsw,
    text_index: Bm25Index,
    /// internal index ids -> record id
    vec_ids: Vec<Uuid>,
    text_ids: Vec<Uuid>,
    /// record id -> (vec internal id, text internal id)
    loc: HashMap<Uuid, (Option<usize>, Option<usize>)>,
    /// Dimension of the vector index, fixed by the first vector added.
    /// Mixing embedding models in one collection is not meaningful.
    index_dim: Option<usize>,
    graph: GraphIndex,
    reranker: Option<Box<dyn Reranker>>,
    /// Live records carrying an `expires_at`. When zero, recall can skip the
    /// per-candidate expiry check.
    expiring: usize,
}

impl MemoryEngine {
    pub fn open(dir: &Path) -> Result<Self> {
        Self::open_with_embedder(dir, Box::new(HashEmbedder::new(256)))
    }

    pub fn open_with_embedder(dir: &Path, embedder: Box<dyn Embedder>) -> Result<Self> {
        Self::open_full(
            dir,
            embedder,
            Box::new(ExtractiveSummarizer::default()),
            LifecycleConfig::default(),
        )
    }

    pub fn open_full(
        dir: &Path,
        embedder: Box<dyn Embedder>,
        summarizer: Box<dyn Summarizer>,
        cfg: LifecycleConfig,
    ) -> Result<Self> {
        Self::open_with_options(dir, embedder, summarizer, cfg, HnswConfig::default())
    }

    pub fn open_with_options(
        dir: &Path,
        embedder: Box<dyn Embedder>,
        summarizer: Box<dyn Summarizer>,
        cfg: LifecycleConfig,
        index_cfg: HnswConfig,
    ) -> Result<Self> {
        let (wal, ops) = Wal::open(dir)?;
        let mut engine = Self {
            wal: Mutex::new(wal),
            embedder,
            summarizer,
            cfg,
            dir: dir.to_path_buf(),
            records: HashMap::new(),
            vec_index: Hnsw::new(index_cfg.clone()),
            text_index: Bm25Index::new(),
            vec_ids: Vec::new(),
            text_ids: Vec::new(),
            loc: HashMap::new(),
            index_dim: None,
            index_cfg,
            graph: GraphIndex::default(),
            reranker: None,
            expiring: 0,
        };
        // Checkpoint + tail recovery: load the last serialized state, then
        // replay only the WAL ops that landed after it. Replay is idempotent
        // (a crash between checkpoint write and WAL truncation leaves
        // already-applied ops in the tail).
        if let Some(state) = load_checkpoint::<CheckpointOwned>(dir)? {
            engine.records = state.records;
            engine.vec_index = state.vec_index;
            engine.text_index = state.text_index;
            engine.vec_ids = state.vec_ids;
            engine.text_ids = state.text_ids;
            engine.loc = state.loc;
            engine.graph = state.graph;
            engine.index_dim = state.index_dim;
            engine.expiring = engine
                .records
                .values()
                .filter(|r| r.expires_at.is_some())
                .count();
            // Pre-v0.4 checkpoints have no graph; rebuild it from records.
            if engine.graph.is_empty() && !engine.records.is_empty() {
                let ids: Vec<Uuid> = engine.records.keys().copied().collect();
                for id in ids {
                    let rec = engine.records.get_mut(&id).unwrap();
                    if rec.entities.is_empty() {
                        rec.entities = extract_entities(&rec.text, &rec.tags);
                    }
                    let ents = rec.entities.clone();
                    engine.graph.add(id, &ents);
                }
            }
        }
        for op in ops {
            match op {
                WalOp::Remember { record } => {
                    if !engine.records.contains_key(&record.id) {
                        engine.index_record(*record);
                    }
                }
                WalOp::Forget { id } => {
                    engine.drop_record(&id);
                }
            }
        }
        Ok(engine)
    }

    /// Persist the full derived state and truncate the WAL. Atomic: the
    /// checkpoint is temp-written and renamed before the log is emptied.
    pub fn checkpoint(&mut self) -> Result<()> {
        save_checkpoint(
            &self.dir,
            &CheckpointRef {
                records: &self.records,
                vec_index: &self.vec_index,
                text_index: &self.text_index,
                vec_ids: &self.vec_ids,
                text_ids: &self.text_ids,
                loc: &self.loc,
                graph: &self.graph,
                index_dim: self.index_dim,
            },
        )?;
        self.wal.lock().expect("wal lock").truncate()
    }

    fn index_record(&mut self, mut record: MemoryRecord) {
        if record.expires_at.is_some() {
            self.expiring += 1;
        }
        if record.entities.is_empty() {
            record.entities = extract_entities(&record.text, &record.tags);
        }
        self.graph.add(record.id, &record.entities);
        // Vectors from different embedding models are not comparable; the
        // first vector fixes the index dimension and mismatches stay
        // lexical-only (happens after switching embedders on an existing
        // data dir).
        let vec_id = record.embedding.as_ref().and_then(|e| {
            match self.index_dim {
                None => self.index_dim = Some(e.len()),
                Some(dim) if dim != e.len() => {
                    eprintln!(
                        "memrust: memory {} has embedding dim {} but the index is dim {}; vector search disabled for it (re-ingest to fix)",
                        record.id,
                        e.len(),
                        dim
                    );
                    return None;
                }
                Some(_) => {}
            }
            let id = self.vec_index.add(e.clone());
            self.vec_ids.push(record.id);
            Some(id)
        });
        let text_id = self.text_index.add(&record.text);
        self.text_ids.push(record.id);
        self.loc.insert(record.id, (vec_id, Some(text_id)));
        self.records.insert(record.id, record);
    }

    fn drop_record(&mut self, id: &Uuid) -> bool {
        let Some((vec_id, text_id)) = self.loc.remove(id) else {
            return false;
        };
        if let Some(rec) = self.records.get(id) {
            if rec.expires_at.is_some() {
                self.expiring = self.expiring.saturating_sub(1);
            }
            let ents = rec.entities.clone();
            self.graph.remove(*id, &ents);
        }
        if let Some(v) = vec_id {
            self.vec_index.remove(v);
        }
        if let Some(t) = text_id {
            self.text_index.remove(t);
        }
        self.records.remove(id).is_some()
    }

    fn prepare_supplied(&self, mut e: Vec<f32>) -> Result<Vec<f32>> {
        if let Some(dim) = self.index_dim {
            if e.len() != dim {
                bail!(
                    "supplied embedding has dim {} but this collection's vector index is dim {}; \
                     use one embedding model consistently",
                    e.len(),
                    dim
                );
            }
        }
        normalize(&mut e);
        Ok(e)
    }

    pub fn remember(&mut self, req: RememberRequest) -> Result<MemoryRecord> {
        let record = self.stage(req)?;
        self.apply_staged(std::iter::once(record.clone()));
        Ok(record)
    }

    /// Phase one of a write: build the record and get it on disk. Takes
    /// `&self`, so a caller holding only a *read* lock can run it — which
    /// means concurrent readers are not blocked for the fsync. The record is
    /// durable when this returns but not yet visible to recall; phase two
    /// (`apply_staged`) makes it visible and needs exclusive access.
    ///
    /// Callers must hold a commit lock across both phases: a checkpoint
    /// landing in between would persist state without this record and
    /// truncate the WAL entry that describes it, losing the write.
    pub fn stage(&self, mut req: RememberRequest) -> Result<MemoryRecord> {
        if req.text.trim().is_empty() {
            bail!("memory text must not be empty");
        }
        let embedding = match req.embedding.take() {
            Some(e) => self.prepare_supplied(e)?,
            None => self.embedder.embed(&req.text)?,
        };
        let record = self.build_record(req, embedding);
        self.wal
            .lock()
            .expect("wal lock")
            .append(&WalOp::Remember {
                record: Box::new(record.clone()),
            })?;
        Ok(record)
    }

    /// Phase one for a batch: one fsync covers the whole group.
    pub fn stage_batch(&self, mut reqs: Vec<RememberRequest>) -> Result<Vec<MemoryRecord>> {
        let mut embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(reqs.len());
        for req in &mut reqs {
            if req.text.trim().is_empty() {
                bail!("memory text must not be empty");
            }
            embeddings.push(match req.embedding.take() {
                Some(e) => Some(self.prepare_supplied(e)?),
                None => None,
            });
        }
        let missing: Vec<usize> = (0..reqs.len())
            .filter(|&i| embeddings[i].is_none())
            .collect();
        if !missing.is_empty() {
            let texts: Vec<&str> = missing.iter().map(|&i| reqs[i].text.as_str()).collect();
            for (&i, e) in missing.iter().zip(self.embedder.embed_batch(&texts)?) {
                embeddings[i] = Some(e);
            }
        }
        let records: Vec<MemoryRecord> = reqs
            .into_iter()
            .zip(embeddings)
            .map(|(req, e)| self.build_record(req, e.expect("embedding filled above")))
            .collect();
        let ops: Vec<WalOp> = records
            .iter()
            .map(|r| WalOp::Remember {
                record: Box::new(r.clone()),
            })
            .collect();
        self.wal.lock().expect("wal lock").append_batch(&ops)?;
        Ok(records)
    }

    /// Phase two: make staged records visible. In-memory only — no disk, so
    /// the exclusive lock is held for microseconds instead of an fsync.
    pub fn apply_staged(&mut self, records: impl IntoIterator<Item = MemoryRecord>) {
        for record in records {
            self.index_record(record);
        }
    }

    /// Bulk ingestion: texts without supplied vectors are embedded in one
    /// `embed_batch` call, and the whole group is persisted behind a single
    /// fsync before any of it becomes visible.
    pub fn remember_batch(&mut self, reqs: Vec<RememberRequest>) -> Result<Vec<MemoryRecord>> {
        let records = self.stage_batch(reqs)?;
        self.apply_staged(records.clone());
        Ok(records)
    }

    /// Build a record without persisting it, so batch ingest can group the
    /// WAL writes behind one fsync.
    fn build_record(&self, req: RememberRequest, embedding: Vec<f32>) -> MemoryRecord {
        let now = now_ms();
        let expires_at = match req.ttl_seconds {
            Some(secs) => Some(now + secs as i64 * 1000),
            None if req.kind == MemoryKind::Working => {
                Some(now + self.cfg.working_ttl_secs as i64 * 1000)
            }
            None => None,
        };
        let entities = extract_entities(&req.text, &req.tags);
        let visibility = req.visibility.unwrap_or(if req.agent_id.is_some() {
            Visibility::Private
        } else {
            Visibility::Shared
        });
        MemoryRecord {
            id: Uuid::new_v4(),
            kind: req.kind,
            text: req.text,
            created_at: now,
            importance: req.importance.unwrap_or(0.5).clamp(0.0, 1.0),
            expires_at,
            sources: Vec::new(),
            entities,
            tags: req.tags,
            session_id: req.session_id,
            agent_id: req.agent_id,
            visibility,
            metadata: req.metadata,
            embedding: Some(embedding),
        }
    }

    pub fn forget(&mut self, id: Uuid) -> Result<bool> {
        if !self.records.contains_key(&id) {
            return Ok(false);
        }
        self.wal
            .lock()
            .expect("wal lock")
            .append(&WalOp::Forget { id })?;
        Ok(self.drop_record(&id))
    }

    pub fn get(&self, id: &Uuid) -> Option<&MemoryRecord> {
        self.records.get(id)
    }

    pub fn recall(&self, req: &RecallRequest) -> Vec<RecallHit> {
        let top_k = req.top_k.unwrap_or(DEFAULT_TOP_K);
        let (w_vec, w_text, w_graph, w_rec) = req.strategy.weights();
        let now = now_ms();

        // Pre-filtering: the filter (and expiry) is applied *inside* each
        // index search, so selective filters get full result sets instead of
        // whatever survives filtering a global top-100.
        //
        // The predicate costs a hash lookup and a dynamic call on every node
        // the traversal visits, so when there is nothing to filter — no
        // filter, no agent scoping, nothing expiring — it is skipped
        // entirely and the index uses its own cheap tombstone check.
        let needs_filter = !req.filter.is_empty() || req.as_agent.is_some() || self.expiring > 0;
        let passes = |id: &Uuid| -> bool {
            self.records
                .get(id)
                .map(|r| {
                    req.filter.matches(r)
                        && !is_expired(r, now)
                        && visible_to(r, req.as_agent.as_deref())
                })
                .unwrap_or(false)
        };

        // Signal 1: semantic neighbors from HNSW. A failed or
        // dimension-mismatched query embedding degrades recall to
        // lexical+recency instead of erroring: agents would rather get
        // keyword results than none.
        let query_vec = match &req.query_embedding {
            Some(q) => {
                let mut q = q.clone();
                normalize(&mut q);
                Some(q)
            }
            None => match self.embedder.embed_query(&req.query) {
                Ok(q) => Some(q),
                Err(e) => {
                    eprintln!("memrust: query embedding failed ({e}); lexical-only recall");
                    None
                }
            },
        };
        let vec_pred = |i: usize| passes(&self.vec_ids[i]);
        let vec_filter: Option<&dyn Fn(usize) -> bool> =
            if needs_filter { Some(&vec_pred) } else { None };
        let vec_hits = match query_vec {
            Some(q) if Some(q.len()) == self.index_dim => {
                self.vec_index
                    .search_filtered(&q, CANDIDATES, vec_filter, req.ef_search)
            }
            Some(q) => {
                if self.index_dim.is_some() {
                    eprintln!(
                        "memrust: query embedding dim {} != index dim {:?}; lexical-only recall",
                        q.len(),
                        self.index_dim
                    );
                }
                Vec::new()
            }
            None => Vec::new(),
        };
        // Signal 2: lexical matches from BM25.
        let text_pred = |i: usize| passes(&self.text_ids[i]);
        let text_filter: Option<&dyn Fn(usize) -> bool> =
            if needs_filter { Some(&text_pred) } else { None };
        let text_hits = self
            .text_index
            .search_filtered(&req.query, CANDIDATES, text_filter);
        // Signal 3: graph traversal — records sharing entities with the
        // query, or reachable through co-occurring entities (1 hop).
        let query_entities = extract_entities(&req.query, &[]);
        let graph_hits: Vec<(Uuid, f32)> = self
            .graph
            .related(&query_entities, CANDIDATES)
            .into_iter()
            .filter(|(id, _)| passes(id))
            .collect();

        const ZERO: RecallSignals = RecallSignals {
            vector: 0.0,
            lexical: 0.0,
            graph: 0.0,
            recency: 0.0,
            importance: 0.0,
            rerank: 0.0,
        };
        let mut fused: HashMap<Uuid, RecallSignals> = HashMap::new();
        for (rank, (internal, _)) in vec_hits.iter().enumerate() {
            let id = self.vec_ids[*internal];
            fused.entry(id).or_insert(ZERO).vector = 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, (internal, _)) in text_hits.iter().enumerate() {
            let id = self.text_ids[*internal];
            fused.entry(id).or_insert(ZERO).lexical = 1.0 / (RRF_K + rank as f32 + 1.0);
        }
        for (rank, (id, _)) in graph_hits.iter().enumerate() {
            fused.entry(*id).or_insert(ZERO).graph = 1.0 / (RRF_K + rank as f32 + 1.0);
        }

        // Score against record *references*: cloning every fused candidate
        // here would copy a few hundred records (embeddings included) just to
        // discard all but top_k a few lines later.
        let mut scored: Vec<(Uuid, f32, RecallSignals)> = fused
            .into_iter()
            .filter_map(|(id, mut signals)| {
                let record = self.records.get(&id)?;
                if !req.filter.matches(record) || is_expired(record, now) {
                    return None;
                }
                let age_ms = (now - record.created_at).max(0) as f64;
                signals.recency = (-age_ms * std::f64::consts::LN_2 / HALF_LIFE_MS).exp() as f32;
                signals.importance = record.importance * IMPORTANCE_WEIGHT;
                // Recency scaled into RRF range so it modulates rather than
                // dominates rank order (1/RRF_K is the max RRF contribution).
                let score = w_vec * signals.vector
                    + w_text * signals.lexical
                    + w_graph * signals.graph
                    + w_rec * signals.recency / RRF_K
                    + signals.importance;
                Some((id, score, signals))
            })
            .collect();

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        // A reranker reorders a wider pool, so materialize that much.
        let keep = if self.reranker.is_some() && req.rerank != Some(false) {
            (top_k * 3).min(scored.len())
        } else {
            top_k.min(scored.len())
        };
        scored.truncate(keep);

        let mut hits: Vec<RecallHit> = scored
            .into_iter()
            .map(|(id, score, signals)| RecallHit {
                record: self.records[&id].clone(),
                score,
                signals,
            })
            .collect();

        // Optional rerank stage: score the fused top pool with the
        // configured reranker and order by relevance. `score` keeps the
        // fused value; `signals.rerank` carries the reranker's verdict.
        // Failures fall back to fused order — degraded recall beats none.
        if let Some(reranker) = &self.reranker {
            if req.rerank != Some(false) && !hits.is_empty() {
                let pool = hits.len().min(top_k * 3);
                let docs: Vec<&str> = hits[..pool]
                    .iter()
                    .map(|h| h.record.text.as_str())
                    .collect();
                match reranker.rerank(&req.query, &docs) {
                    Ok(scores) if scores.len() == pool => {
                        for (hit, s) in hits[..pool].iter_mut().zip(&scores) {
                            hit.signals.rerank = *s;
                        }
                        hits[..pool].sort_by(|a, b| {
                            b.signals
                                .rerank
                                .total_cmp(&a.signals.rerank)
                                .then(b.score.total_cmp(&a.score))
                        });
                    }
                    Ok(scores) => eprintln!(
                        "memrust: reranker returned {} scores for {pool} docs; keeping fused order",
                        scores.len()
                    ),
                    Err(e) => eprintln!("memrust: rerank failed ({e}); keeping fused order"),
                }
            }
        }

        hits.truncate(top_k);
        hits
    }

    /// Install a reranker; recall applies it unless a request opts out.
    pub fn set_reranker(&mut self, reranker: Box<dyn Reranker>) {
        self.reranker = Some(reranker);
    }

    /// Rebuild indexes from live records only (dropping tombstones), then
    /// checkpoint the rebuilt state to disk.
    pub fn compact(&mut self) -> Result<()> {
        let records: Vec<MemoryRecord> = self.records.values().cloned().collect();
        self.vec_index = Hnsw::new(self.index_cfg.clone());
        self.text_index = Bm25Index::new();
        self.vec_ids.clear();
        self.text_ids.clear();
        self.loc.clear();
        self.records.clear();
        self.index_dim = None;
        self.expiring = 0;
        for r in records {
            self.index_record(r);
        }
        self.checkpoint()
    }

    /// One lifecycle pass: durably forget expired memories, then fold old
    /// episodic memories into semantic summaries (grouped per session,
    /// chronological batches, provenance kept in `sources`). Every mutation
    /// goes through the WAL, so a crash mid-pass loses nothing.
    pub fn run_lifecycle(&mut self) -> Result<LifecycleReport> {
        let mut report = LifecycleReport::default();
        let now = now_ms();

        let expired: Vec<Uuid> = self
            .records
            .values()
            .filter(|r| is_expired(r, now))
            .map(|r| r.id)
            .collect();
        for id in expired {
            if self.forget(id)? {
                report.expired_swept += 1;
            }
        }

        let cutoff = now - self.cfg.consolidate_after_secs as i64 * 1000;
        let mut by_session: BTreeMap<Option<String>, Vec<MemoryRecord>> = BTreeMap::new();
        for r in self.records.values() {
            if r.kind == MemoryKind::Episodic && r.created_at <= cutoff {
                by_session
                    .entry(r.session_id.clone())
                    .or_default()
                    .push(r.clone());
            }
        }

        for (session_id, mut group) in by_session {
            group.sort_by_key(|r| r.created_at);
            for batch in group.chunks(self.cfg.max_batch) {
                // Undersized trailing batches wait for more history.
                if batch.len() < self.cfg.min_batch {
                    continue;
                }
                let texts: Vec<&str> = batch.iter().map(|r| r.text.as_str()).collect();
                let summary_text = self.summarizer.summarize(&texts)?;
                // In a BYO-embedding collection the engine's embedder can't
                // produce dimension-compatible vectors; the centroid of the
                // source embeddings is both compatible and semantically the
                // consolidation of those memories.
                let engine_dim = self.embedder.dim();
                let embedding = match self.index_dim {
                    Some(dim) if dim != engine_dim => {
                        let mut acc = vec![0.0f32; dim];
                        let mut n = 0usize;
                        for r in batch {
                            if let Some(e) = &r.embedding {
                                if e.len() == dim {
                                    for (a, x) in acc.iter_mut().zip(e) {
                                        *a += x;
                                    }
                                    n += 1;
                                }
                            }
                        }
                        if n > 0 {
                            normalize(&mut acc);
                            acc
                        } else {
                            self.embedder.embed(&summary_text)?
                        }
                    }
                    _ => self.embedder.embed(&summary_text)?,
                };

                let mut tags: Vec<String> = Vec::new();
                for r in batch {
                    for t in &r.tags {
                        if !tags.contains(t) && tags.len() < 8 {
                            tags.push(t.clone());
                        }
                    }
                }
                let summary = MemoryRecord {
                    id: Uuid::new_v4(),
                    kind: MemoryKind::Semantic,
                    text: summary_text,
                    created_at: now,
                    importance: batch.iter().map(|r| r.importance).fold(0.0f32, f32::max),
                    expires_at: None,
                    sources: batch.iter().map(|r| r.id).collect(),
                    entities: Vec::new(),
                    tags,
                    session_id: session_id.clone(),
                    agent_id: batch[0].agent_id.clone(),
                    visibility: if batch.iter().all(|r| r.visibility == Visibility::Shared) {
                        Visibility::Shared
                    } else {
                        Visibility::Private
                    },
                    metadata: Some(json!({
                        "consolidated": {
                            "count": batch.len(),
                            "from": batch[0].created_at,
                            "to": batch[batch.len() - 1].created_at,
                        }
                    })),
                    embedding: Some(embedding),
                };
                self.wal
                    .lock()
                    .expect("wal lock")
                    .append(&WalOp::Remember {
                        record: Box::new(summary.clone()),
                    })?;
                report.summaries.push(summary.id);
                self.index_record(summary);
                for r in batch {
                    self.forget(r.id)?;
                }
                report.batches_consolidated += 1;
            }
        }

        // Bound restart replay time: once the WAL tail is long enough,
        // serialize the built state and truncate the log.
        if self
            .wal
            .lock()
            .expect("wal lock")
            .appends_since_checkpoint()
            >= CHECKPOINT_AFTER_OPS
        {
            self.checkpoint()?;
            report.checkpointed = true;
        }
        Ok(report)
    }

    /// Export all live memories (optionally scoped to a session), oldest
    /// first. The result round-trips through `restore`.
    pub fn snapshot(&self, session_id: Option<&str>) -> Snapshot {
        let now = now_ms();
        let mut records: Vec<MemoryRecord> = self
            .records
            .values()
            .filter(|r| !is_expired(r, now))
            .filter(|r| session_id.is_none() || r.session_id.as_deref() == session_id)
            .cloned()
            .collect();
        records.sort_by_key(|r| r.created_at);
        Snapshot {
            created_at: now,
            session_id: session_id.map(String::from),
            records,
        }
    }

    /// Import snapshot records, preserving ids and timestamps. Records whose
    /// id already exists are skipped, so restoring twice is a no-op. Returns
    /// how many records were added.
    pub fn restore(&mut self, records: Vec<MemoryRecord>) -> Result<usize> {
        let mut added = 0;
        for mut record in records {
            if self.records.contains_key(&record.id) {
                continue;
            }
            if record.embedding.is_none() {
                record.embedding = Some(self.embedder.embed(&record.text)?);
            }
            self.wal
                .lock()
                .expect("wal lock")
                .append(&WalOp::Remember {
                    record: Box::new(record.clone()),
                })?;
            self.index_record(record);
            added += 1;
        }
        Ok(added)
    }

    /// Newest-first page of live (non-expired) memories, for browsing UIs.
    /// Returns (total live, page).
    pub fn list_memories(&self, offset: usize, limit: usize) -> (usize, Vec<MemoryRecord>) {
        let now = now_ms();
        let mut live: Vec<&MemoryRecord> = self
            .records
            .values()
            .filter(|r| !is_expired(r, now))
            .collect();
        live.sort_by_key(|r| std::cmp::Reverse(r.created_at));
        let total = live.len();
        let page = live.into_iter().skip(offset).take(limit).cloned().collect();
        (total, page)
    }

    /// Most-mentioned entities in the graph index.
    pub fn top_entities(&self, limit: usize) -> Vec<(String, usize)> {
        self.graph.top_entities(limit)
    }

    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_memories: self.records.len(),
            vector_indexed: self.vec_index.len(),
            lexical_indexed: self.text_index.len(),
            embedding_dim: self.embedder.dim(),
            vector_dim: self.index_dim,
            entities: self.graph.entity_count(),
            quantized: self.vec_index.is_quantized(),
            wal_tail_ops: self
                .wal
                .lock()
                .expect("wal lock")
                .appends_since_checkpoint(),
        }
    }
}
