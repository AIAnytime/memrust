# Changelog

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
