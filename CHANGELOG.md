# Changelog

## 0.5.2 — 2026-07-27

- **SQ8 quantization is now the default for vectors >= 1024 dims.** At that
  width and above, quantized search measures as fast as (or faster than) f32
  because memory bandwidth dominates the extra decode arithmetic — so the
  4x memory saving is effectively free. Below 1024 dims f32 is faster and
  stays the default. The storage mode is settled by the first vector stored,
  since that is when the width becomes known.
- `--quantize` / `--no-quantize` force the mode; `HnswConfig.quantize` is now
  `Quantization::{Auto, Always, Never}` (was a bool). Checkpoints written
  with the old bool form still load — `false` maps to `Never`, so an existing
  collection never silently changes representation.

## 0.5.1 — 2026-07-27

Fixes found by writing the Colab notebooks — the notebooks are the first
consumer of the bring-your-own-embeddings path end to end.

- **Consolidation summaries are now recallable by vector in BYO-embedding
  collections.** Summaries were embedded with the engine's own embedder,
  which cannot produce dimension-compatible vectors when callers supply
  their own — the summary silently degraded to lexical-only. Summaries now
  use the normalized centroid of their source embeddings, which is
  dimension-correct by construction and semantically the consolidation of
  those memories.
- **`vector_dim` in engine stats** — the vector index's actual dimension
  (set by the first stored vector), distinct from `embedding_dim` (the
  engine's own embedder, unused under BYO). The dashboard shows whichever
  governs search.
- Three Colab notebooks in `notebooks/`, all pip-installable.

## 0.5.0 — 2026-07-27

The launch release: engine, lifecycle, scale features, four-signal retrieval,
multi-agent memory, and SDKs.

### v0.5 — multi-agent + SDKs
- `visibility: private | shared` on memories; agent-owned memories default
  private, unowned default shared
- Recall `as_agent` scoping, enforced inside the index pre-filters; unscoped
  recall is the operator view
- Consolidation summaries are shared only if every source memory was shared
- MCP `--agent-id` stamps remembers and scopes recalls per mount
- Zero-dependency Python SDK (`sdks/python`) and TypeScript SDK
  (`sdks/typescript`), each with a live E2E test

### v0.4 — retrieval quality
- Entity graph as a third retrieval leg: heuristic extraction at ingest,
  co-occurrence edges, rarity-weighted 1-hop traversal, `relational` strategy
- Pre-filtering inside HNSW (predicate-aware beam search) and BM25
- Optional `Reranker` stage (`--reranker openai`), degrading to fused order
  on failure
- `procedural` memory kind; E5-style query/passage embedding prefixes

### v0.3 — scale
- Checkpoint + WAL-tail recovery: built indexes serialize to disk, restarts
  replay only the tail; auto-checkpoint bounds tail length
- SQ8 scalar quantization (`--quantize`): 1 byte/dim, distances on codes
- Contiguous flat vector storage, multi-accumulator dot kernels
- HNSW diversity-based neighbor selection (fixed recall at scale)
- `remember_batch` / `embed_batch` (one API round-trip), `bench` subcommand

### v0.2 — memory lifecycle
- Working-memory TTLs with durable sweeps
- Automatic consolidation of old episodic memories into semantic summaries
  with `sources` provenance (extractive default, optional LLM summarizer)
- Session snapshot/restore (id-preserving, idempotent)
- Background lifecycle pass in `serve`/`mcp`

### v0.1 — core engine
- WAL-first storage, fsync on append
- In-crate HNSW and BM25, reciprocal-rank fusion, recency decay, importance
- Per-signal explained recall scores
- Agent-native API (`remember`/`recall`/`forget`), memory kinds, filters
- HTTP API (axum) and MCP server (stdio)
- Pluggable embeddings: OpenAI-compatible protocol, Gemini, BYO vectors,
  offline hash embedder default
