use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Agent-native memory categories, mirroring how agent frameworks reason
/// about memory rather than how databases store rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Things that happened: events, conversation turns, actions taken.
    #[default]
    Episodic,
    /// Distilled facts and knowledge, independent of when they were learned.
    Semantic,
    /// Short-lived scratch state for the current task/session.
    Working,
    /// The agent's own conclusions about its behavior or the task.
    Reflection,
    /// Indexed history of tool invocations and their results.
    ToolCall,
    /// How-to knowledge: workflows, tool usage patterns, few-shot examples.
    /// Agents that remember *how* they did something are the ones that improve.
    Procedural,
}

/// Who can recall a memory in a multi-agent deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Recallable by every agent (and by unscoped recalls).
    #[default]
    Shared,
    /// Recallable only by the owning `agent_id` (and unscoped recalls).
    Private,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub kind: MemoryKind,
    pub text: String,
    /// Unix milliseconds.
    pub created_at: i64,
    /// 0.0..=1.0, used as a retrieval boost.
    pub importance: f32,
    /// Unix milliseconds; the memory is invisible to recall after this and
    /// durably removed by the next lifecycle sweep.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    /// For consolidated summaries: ids of the memories this was distilled
    /// from (which were forgotten in the same lifecycle pass).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<Uuid>,
    /// Entities extracted at ingestion (names, identifiers, tags); the
    /// graph index links memories that share or co-mention them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// Defaults at ingest: `private` when `agent_id` is set, else `shared`.
    /// (Pre-v0.5 records deserialize as `shared`, preserving old behavior.)
    #[serde(default)]
    pub visibility: Visibility,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// L2-normalized embedding. Present after ingestion (auto-embedded when
    /// the caller does not supply one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RememberRequest {
    pub text: String,
    #[serde(default)]
    pub kind: MemoryKind,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Defaults to 0.5 when unset.
    #[serde(default)]
    pub importance: Option<f32>,
    /// Time-to-live in seconds. Unset: `working` memories get the engine's
    /// default working TTL; other kinds never expire.
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    /// Unset: `private` when `agent_id` is set, else `shared`.
    #[serde(default)]
    pub visibility: Option<Visibility>,
    /// Bring-your-own embedding; auto-embedded otherwise.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    /// When this memory happened, in Unix **milliseconds**. Unset means now.
    ///
    /// Set it when the memory is older than the write: importing history from
    /// another system, replaying a transcript, or ingesting dated documents.
    /// Without it every imported memory looks equally recent and the recency
    /// signal — which exists to answer "what did we decide lately" — is
    /// flattened into a constant.
    ///
    /// `ttl_seconds` counts from this instant, not from the write, so an
    /// import is idempotent in time: re-importing the same records twice
    /// yields the same expiry rather than extending it.
    #[serde(default)]
    pub created_at: Option<i64>,
}

/// How recall should weight its retrieval signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RecallStrategy {
    /// Even blend of semantic and lexical signals with mild recency.
    #[default]
    Balanced,
    /// Meaning-first: lean on the vector index.
    Semantic,
    /// Exact-term-first: lean on BM25.
    Lexical,
    /// Recency-first: what happened lately matters most.
    Recent,
    /// Relation-first: follow entity links — "things connected to X",
    /// not just "things similar to X".
    Relational,
}

impl RecallStrategy {
    /// (vector_weight, lexical_weight, graph_weight, recency_weight)
    pub fn weights(self) -> (f32, f32, f32, f32) {
        match self {
            RecallStrategy::Balanced => (1.0, 1.0, 0.7, 0.3),
            RecallStrategy::Semantic => (1.0, 0.3, 0.5, 0.1),
            RecallStrategy::Lexical => (0.3, 1.0, 0.5, 0.1),
            RecallStrategy::Recent => (0.5, 0.5, 0.3, 1.2),
            RecallStrategy::Relational => (0.4, 0.4, 1.2, 0.2),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MemoryFilter {
    #[serde(default)]
    pub kinds: Vec<MemoryKind>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    /// Unix milliseconds, inclusive bounds.
    #[serde(default)]
    pub since: Option<i64>,
    #[serde(default)]
    pub until: Option<i64>,
}

impl MemoryFilter {
    /// True when the filter constrains nothing, letting recall skip the
    /// per-candidate predicate entirely.
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
            && self.tags.is_empty()
            && self.session_id.is_none()
            && self.agent_id.is_none()
            && self.since.is_none()
            && self.until.is_none()
    }

    pub fn matches(&self, rec: &MemoryRecord) -> bool {
        if !self.kinds.is_empty() && !self.kinds.contains(&rec.kind) {
            return false;
        }
        if let Some(s) = &self.session_id {
            if rec.session_id.as_deref() != Some(s.as_str()) {
                return false;
            }
        }
        if let Some(a) = &self.agent_id {
            if rec.agent_id.as_deref() != Some(a.as_str()) {
                return false;
            }
        }
        if !self.tags.is_empty() && !self.tags.iter().all(|t| rec.tags.contains(t)) {
            return false;
        }
        if let Some(since) = self.since {
            if rec.created_at < since {
                return false;
            }
        }
        if let Some(until) = self.until {
            if rec.created_at > until {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub strategy: RecallStrategy,
    #[serde(default)]
    pub filter: MemoryFilter,
    /// Bring-your-own query vector (must match the collection's embedding
    /// model); when unset, the engine embeds `query` itself.
    #[serde(default)]
    pub query_embedding: Option<Vec<f32>>,
    /// Set false to skip the configured reranker for this request.
    #[serde(default)]
    pub rerank: Option<bool>,
    /// Recall *on behalf of* this agent: sees shared memories, unowned
    /// memories, and its own private ones. Unset = unscoped (sees all —
    /// single-agent mode or an operator view). Distinct from
    /// `filter.agent_id`, which narrows to exactly one agent's records
    /// within the visible set.
    #[serde(default)]
    pub as_agent: Option<String>,
    /// Per-query HNSW beam width. Higher searches harder: more accurate,
    /// slower. Unset uses the index default (100).
    #[serde(default)]
    pub ef_search: Option<usize>,
}

/// Per-signal contributions, so agents (and humans) can see *why* a memory
/// was recalled instead of getting an opaque score.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct RecallSignals {
    /// Reciprocal-rank contribution from the vector index (0 if not matched).
    pub vector: f32,
    /// Reciprocal-rank contribution from BM25 (0 if not matched).
    pub lexical: f32,
    /// Reciprocal-rank contribution from entity-graph traversal (0 if not
    /// matched via shared or co-occurring entities).
    pub graph: f32,
    /// Exponential time-decay factor in 0..=1.
    pub recency: f32,
    /// Importance boost contribution.
    pub importance: f32,
    /// Reranker relevance in 0..=1 (0 when no reranker ran). When set, hits
    /// are ordered by this; `score` still reports the fused retrieval score.
    pub rerank: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RecallHit {
    pub record: MemoryRecord,
    pub score: f32,
    pub signals: RecallSignals,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub total_memories: usize,
    pub vector_indexed: usize,
    pub lexical_indexed: usize,
    /// Dimension of the engine's own embedder (unused when callers supply
    /// their own vectors).
    pub embedding_dim: usize,
    /// Dimension of the vector index itself — set by the first vector stored,
    /// so it reflects bring-your-own embeddings. None until one is stored.
    pub vector_dim: Option<usize>,
    /// Distinct entities tracked by the graph index.
    pub entities: usize,
    /// Whether the vector index stores SQ8 codes instead of f32.
    pub quantized: bool,
    /// WAL ops a restart would replay (resets to 0 at each checkpoint).
    pub wal_tail_ops: usize,
}

/// Policy knobs for the memory lifecycle pass.
#[derive(Debug, Clone)]
pub struct LifecycleConfig {
    /// Default TTL applied to `working` memories that don't set one. 1 day.
    pub working_ttl_secs: u64,
    /// Episodic memories older than this become consolidation candidates. 7 days.
    pub consolidate_after_secs: u64,
    /// Don't summarize groups smaller than this; they wait for more context.
    pub min_batch: usize,
    /// Cap on how many memories fold into one summary.
    pub max_batch: usize,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            working_ttl_secs: 24 * 3600,
            consolidate_after_secs: 7 * 24 * 3600,
            min_batch: 4,
            max_batch: 12,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LifecycleReport {
    /// Expired memories durably forgotten this pass.
    pub expired_swept: usize,
    /// Episodic batches folded into semantic summaries.
    pub batches_consolidated: usize,
    /// Ids of the summary memories created.
    pub summaries: Vec<Uuid>,
    /// Whether this pass also checkpointed state and truncated the WAL.
    pub checkpointed: bool,
}

/// A portable export of (a session's) memory, restorable into any engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub created_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub records: Vec<MemoryRecord>,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
