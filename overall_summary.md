# memrust — overall summary

*As of v0.6.1, 2026-07-28.*

## What it is

Memory infrastructure for AI agents: one Rust binary that stores what an agent
learns and answers *"what do I know about this?"* — not *"which vectors are
nearest?"*

The API is `remember` / `recall` / `forget`. Every recall runs four retrieval
signals over the same memories and fuses their rankings:

| Signal | Index | Answers |
|---|---|---|
| Vector | in-crate HNSW (M=16, ef 100/200) | meaning, paraphrase |
| Lexical | in-crate BM25 (k1=1.2, b=0.75) | exact identifiers, error codes, names |
| Graph | entity extraction + co-occurrence edges, 1-hop | *related*, not merely *similar* |
| Recency | exponential decay, 1-week half-life | "what did we decide lately?" |

Fusion is reciprocal-rank (K=60) plus an importance boost, optionally reranked
by an LLM. Every hit returns its per-signal breakdown, so an agent can tell a
strong semantic match from a lucky keyword hit.

Nothing is a dependency — the HNSW, the BM25 index, the graph, the WAL, the
quantizer and the Prometheus exposition are all in-crate. The whole engine is
~5,100 lines of Rust with four runtime crates (serde, tokio, axum, ureq).

## The finding that defines the product

Benchmarked against FAISS, Chroma, Qdrant, pgvector and LanceDB on 500
memories with identical all-MiniLM-L6-v2 embeddings, asking two question
shapes an agent actually asks (hit@5):

| Engine | `"INC-90312"` (identifier) | paraphrase question |
|---|---|---|
| Exact brute-force vector search *(the ceiling)* | 27% | 90% |
| FAISS / Chroma / Qdrant | 27% | 70–90% |
| **memrust (hybrid, default)** | **75%** | 80% |

**Exact, brute-force vector search also scores 27%.** This is not an
index-quality problem that a better ANN implementation fixes — the embedding
cannot separate `INC-90312` from `INC-90319`, because they are near-identical
strings with near-identical vectors. Every pure vector store inherits that
ceiling. Architecture is the only way past it.

That reframed the whole positioning: memrust does not claim to be faster than
Qdrant. It claims that speed is not the binding constraint on agent memory.

Honest counterpart: on pure vector search memrust is *competitive, not
fastest*. 0.64 ms p50 (second-best among servers), recall@10 = 1.000, but
ingest at 988/s is an order of magnitude behind — memrust fsyncs every write.
If raw ANN throughput on a static corpus is the problem, use FAISS.

## Architecture decisions worth remembering

- **WAL-first, indexes derived.** Every mutation is one fsynced JSON line;
  recovery is checkpoint + WAL tail. JSON not bincode, because records use
  `skip_serializing_if` and that needs a self-describing format.
- **Two-phase writes.** `stage()` persists the record (fsync included) under a
  *read* lock; `apply_staged()` takes the exclusive lock just long enough to
  make it visible. Reads under a concurrent writer went 467 → 957 QPS. A
  commit mutex brackets both phases, otherwise a checkpoint landing between
  them truncates the WAL entry for a record state doesn't yet contain — an
  acknowledged write, lost. Verified with four writers, a checkpoint loop and
  `kill -9`: 1,000 acknowledged, 1,000 recovered.
- **Pre-filtering inside the index**, not after it. The predicate is passed
  into HNSW beam search (visit cap `ef*32`) and applied at BM25 accumulation.
  The recall hot path skips the predicate entirely when nothing needs
  filtering, and scores against references before cloning survivors.
- **Namespaces are separate engines**, not a filter column — own indexes, WAL,
  checkpoint, embedding dimension, directory. One tenant's ingest can't slow
  another's recall, and dropping a namespace is a directory delete.
- **SQ8 quantization is automatic at ≥1024 dims.** At that width it's *faster*
  than f32 (memory bandwidth dominates the decode arithmetic) at 0.99 recall
  and a quarter of the memory, so it is free. Below 1024, f32 stays.
- **Everything external is a trait.** `Embedder`, `Summarizer`, `Reranker` all
  ship an offline default plus an OpenAI-compatible remote. A dead remote
  embedder degrades recall to lexical+recency instead of failing.

## Version arc

| | Shipped |
|---|---|
| v0.1 | WAL + HNSW + BM25 + fusion, HTTP + MCP |
| v0.2 | Working-memory TTLs, episodic→semantic consolidation, session snapshots |
| v0.3 | Checkpoint + WAL-tail recovery, SQ8, contiguous storage, diversity-heuristic neighbor selection, batch ingest |
| v0.4 | Entity graph as a third leg, pre-filtering inside indexes, LLM reranking, asymmetric query/passage prefixes |
| v0.5 | Multi-agent private/shared visibility, Python + TypeScript SDKs, per-query `ef_search`, two-phase writes |
| v0.6 | Namespaces, API keys with per-namespace scopes, Prometheus `/metrics`, JSON logs, `/healthz`, official Docker image |

## Distribution

| Channel | Name | Version |
|---|---|---|
| crates.io | `memrust` | 0.6.1 |
| PyPI | `memrust` | 0.6.1 |
| npm | `memrust-client` | 0.6.1 |
| Docker Hub | `aianytime/memrust` | 0.6.1, `latest` — multi-arch amd64+arm64, 36 MB |
| GitHub | `AIAnytime/memrust` | release v0.6.1 with three binaries |

Plus an embedded web dashboard at `http://127.0.0.1:7700/`, three runnable
Colab notebooks in `notebooks/`, a landing page in `docs/`, and reproducible
cross-engine benchmark scripts in `benches/`.

49 tests: engine, lifecycle, scale/recovery, graph recall, multi-agent
visibility, tenancy.

## The business shape

Open core. The engine stays Apache-2.0. The product is the multi-tenant
control plane — hosted memory with per-tenant isolation, which is exactly what
namespaces + scoped API keys + metrics were built to make possible.

Deliberately **not** built yet: distribution. Zero users; a single node covers
the addressable market; multi-tenancy is not the same problem as distribution;
and a hosted offering can run per-tenant single-node instances. Read replicas
via WAL streaming happen when a user asks, not before.

## Open work

- Scheduled backup/restore to object storage — the next production item.
- The single-writer ceiling. HNSW insertion (~2 ms) still holds the exclusive
  lock, so a writer costs about half of read throughput. Lifting it means a
  concurrent or segmented index.
- LLM-based entity extraction (current extraction is heuristic: capitalized
  runs, identifiers with digits, short all-caps, tags).
- Hierarchical (RAPTOR-style) summarization.
- mmap'd zero-copy segments and a binary checkpoint format.

## Outstanding, needs a human

- **GitHub Actions is billing-locked.** Every run fails instantly with
  "account locked due to billing issue", so the repo shows red checks and CI
  has never actually built the Linux binaries. Clear the billing issue, then
  `gh run rerun`.
- **Rotate the Docker Hub PAT** — it appeared in a chat transcript.
- **Enable GitHub Pages** on `main` / `/docs` to publish the landing page.
- The landing page contact is a personal Gmail; a branded address would read
  better to the investors and Product Hunt / Indie Hackers / Reddit audiences
  it's aimed at.
- `docs/index.html` predates v0.6 and still argues the pre-production story:
  no namespaces, no API keys, no metrics, and its install tab shows
  `docker build` rather than `docker run aianytime/memrust`.
