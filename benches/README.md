# Benchmarks

Two scripts, both reproducible. Numbers in the project README came from these
on an M-series MacBook — re-run them on your hardware before trusting any of it.

## compare.py — vector search speed and recall

```bash
pip install numpy faiss-cpu chromadb lancedb "psycopg[binary]" qdrant-client

docker run -d -p 6444:6333 qdrant/qdrant:v1.12.4
docker run -d -p 5544:5432 -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench pgvector/pgvector:pg16
memrust serve --addr 127.0.0.1:7900 --data-dir /tmp/memrust-bench

python benches/compare.py --n 20000 --dim 384 --queries 200
```

Every engine gets identical vectors, identical queries and — where the knob
exists — identical HNSW settings (M=16, ef_construction=200, ef_search=100).
Recall is measured against exact brute-force ground truth from numpy.

Engines that are not running are skipped, not faked. Use `--only faiss memrust`
to run a subset.

## agent_recall.py — retrieval quality on agent questions

```bash
pip install sentence-transformers
memrust serve --addr 127.0.0.1:7901 --data-dir /tmp/memrust-quality

python benches/agent_recall.py      # first run encodes and caches embeddings
python benches/agent_recall.py      # second run benchmarks
```

It runs twice by design: FAISS and PyTorch both bundle OpenMP and segfault if
loaded into one process on macOS, so embedding is a separate phase that caches
to `bench-agent-embeddings.npz`.

Every engine receives the same all-MiniLM-L6-v2 embeddings, so semantics are
held constant and the only variable is what each engine does with them. The
`Exact vector search (ceiling)` row is brute-force cosine — it bounds what any
pure vector index can score, which is what makes the identifier result
meaningful rather than a story about index tuning.

## Fairness notes

- **In-process vs server.** FAISS, Chroma and LanceDB are libraries; memrust,
  Qdrant and pgvector are servers paying network and serialization costs per
  query. The tables label which is which.
- **memrust fsyncs every write.** That is a durability choice the in-process
  libraries do not make, and it dominates the ingest column.
- **FAISS HNSW uses L2**, not `METRIC_INNER_PRODUCT`. For normalized vectors
  the ranking is identical, and inner product measurably degrades FAISS's
  results — using it would have made FAISS look worse than it is.
- **memrust strategies reweight signals, they never disable them**, so the
  "semantic-weighted" row still carries some lexical and graph contribution.
  It is not a vector-only mode.
