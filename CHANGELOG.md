# Changelog

## Unreleased

- **Namespaces.** `X-Memrust-Namespace` selects an isolated store. Each
  namespace is a separate engine with its own indexes, WAL, checkpoint,
  embedding dimension and directory — not a filter over a shared index — so
  tenants can't slow each other down and dropping one is a directory delete.
  Data directories written before namespaces existed are adopted as `default`
  in place, so upgrading loses nothing.
- **API keys.** `--api-key KEY` requires authentication;
  `--api-key KEY:ns1,ns2` scopes a key to specific namespaces. Keys arrive as
  `Authorization: Bearer` or `X-API-Key`, comparison is constant-time, and
  admin routes need an unscoped key. With no keys configured the server stays
  open — and warns when that happens on a non-loopback address.
- `GET /v1/namespaces` and `POST /v1/namespaces/drop` for administration; both
  SDKs take `api_key` / `namespace`.
- The background lifecycle pass now sweeps every open namespace.

## 0.5.4 — 2026-07-28

- **Writes no longer block readers for the duration of an fsync.** The write
  path is split: `stage` builds the record and gets it on disk while holding
  only a read lock, then `apply_staged` takes the exclusive lock briefly to
  make it visible. Reads under a concurrent writer went from 467 to 957 QPS
  with p50 halved (8.52 ms -> 4.78 ms). A commit lock brackets the two phases
  so a checkpoint cannot land between them and truncate the WAL entry for a
  record the saved state lacks — verified with concurrent writers, a
  checkpoint loop, and `kill -9`: no acknowledged write lost.

- **Recall is ~3x faster and recall@10 is now 1.000** (was 1.92 ms / 0.985 at
  20k vectors). Two causes, both found by profiling rather than guessing: the
  filter predicate ran on every visited node even when nothing needed
  filtering, and it silently engaged a traversal visit cap that cost recall.
  Recall now skips the predicate entirely when there is no filter, no agent
  scoping and nothing expiring.
- Recall no longer clones every fused candidate (hundreds of records,
  embeddings included) before discarding all but top-k; records are
  materialized after ranking.
- `ef_search` is settable per recall request, trading latency for accuracy
  without a server restart.
- `benches/concurrency.py` measures read scaling and write interference, and
  `memrust bench --engine` measures end-to-end recall latency without HTTP.
- Both SDKs expose `ef_search` / `efSearch` on recall.

- **Bulk ingest no longer fails on real batches.** The HTTP body limit was
  axum's 2 MB default, which rejected any batch bigger than ~400 memories at
  384 dims (413 Payload Too Large). Raised to 256 MB.
- **Batch ingest is ~4.5x faster** (221 -> 997 records/s measured): the WAL now
  group-commits, so one fsync covers a whole batch instead of one per record.
  Durability is unchanged — the call still returns only after data is on disk.
- Cross-engine benchmarks in `benches/` comparing memrust with Qdrant, Chroma,
  FAISS, pgvector and LanceDB, plus a comparison section in the README.

## 0.5.3 — 2026-07-27

- Dashboard supports light and dark themes with a header toggle. It follows
  the system preference by default and remembers an explicit choice
  (localStorage), applied before first paint so there is no flash. Each
  theme's signal colors are selected against their own surface rather than
  auto-inverted, and both palettes pass the categorical color checks.

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
