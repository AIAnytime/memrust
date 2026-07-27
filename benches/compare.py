"""Cross-engine retrieval benchmark: memrust vs Qdrant, Chroma, FAISS,
pgvector and LanceDB.

Every engine gets the same vectors, the same queries, the same top-k and —
where the knob exists — the same HNSW parameters (M=16, ef_construction=200,
ef_search=100). Recall is measured against exact brute-force ground truth
computed with numpy, so it is comparable across engines.

Run:  python benches/compare.py [--n 20000] [--dim 384] [--queries 200]

Servers the script expects (start them yourself so nothing is hidden):
  docker run -d -p 6444:6333 qdrant/qdrant:v1.12.4
  docker run -d -p 5544:5432 -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench \
      pgvector/pgvector:pg16
  memrust serve --addr 127.0.0.1:7900 --data-dir /tmp/memrust-bench
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

M, EF_CONSTRUCTION, EF_SEARCH, TOP_K = 16, 200, 100, 10


def clustered_dataset(n: int, dim: int, queries: int, seed: int = 42):
    """Mixture-of-centers vectors: the shape real embeddings have. Uniform
    random vectors are the worst case for every ANN method and would flatter
    nobody fairly."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(n // 100, 10), dim)).astype(np.float32)
    pick = rng.integers(0, len(centers), size=n)
    data = centers[pick] + 0.15 * rng.normal(size=(n, dim)).astype(np.float32)
    qpick = rng.integers(0, len(centers), size=queries)
    qs = centers[qpick] + 0.15 * rng.normal(size=(queries, dim)).astype(np.float32)
    norm = lambda a: (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)
    return norm(data), norm(qs)


def ground_truth(data: np.ndarray, qs: np.ndarray, k: int) -> list[set[int]]:
    sims = qs @ data.T
    return [set(np.argpartition(-row, k)[:k].tolist()) for row in sims]


def recall_at_k(got: list[list[int]], truth: list[set[int]], k: int) -> float:
    hit = sum(len(set(g[:k]) & t) for g, t in zip(got, truth))
    return hit / (len(truth) * k)


class Result(dict):
    pass


def measure(name: str, ingest_fn, query_fn, data, qs, truth) -> Result:
    t0 = time.perf_counter()
    ingest_fn(data)
    ingest_s = time.perf_counter() - t0

    query_fn(qs[0])  # warm up caches / JIT / connections
    lat, got = [], []
    for q in qs:
        t = time.perf_counter()
        ids = query_fn(q)
        lat.append((time.perf_counter() - t) * 1000)
        got.append(ids)

    lat.sort()
    return Result(
        engine=name,
        ingest_per_s=len(data) / ingest_s,
        p50_ms=statistics.median(lat),
        p95_ms=lat[int(len(lat) * 0.95)],
        qps=1000.0 / statistics.median(lat),
        recall=recall_at_k(got, truth, TOP_K),
    )


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------
def bench_faiss(data, qs, truth):
    import faiss

    idx = faiss.IndexHNSWFlat(data.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch = EF_SEARCH
    return measure(
        "FAISS (HNSW, in-process)",
        lambda d: idx.add(d),
        lambda q: idx.search(q.reshape(1, -1), TOP_K)[1][0].tolist(),
        data, qs, truth,
    )


def bench_chroma(data, qs, truth):
    import chromadb

    tmp = tempfile.mkdtemp(prefix="chroma-bench-")
    client = chromadb.PersistentClient(path=tmp)
    coll = client.create_collection(
        "bench",
        metadata={"hnsw:space": "cosine", "hnsw:M": M,
                  "hnsw:construction_ef": EF_CONSTRUCTION, "hnsw:search_ef": EF_SEARCH},
    )
    ids = [str(i) for i in range(len(data))]

    def ingest(d):
        for i in range(0, len(d), 5000):
            coll.add(ids=ids[i:i + 5000], embeddings=d[i:i + 5000].tolist())

    def query(q):
        r = coll.query(query_embeddings=[q.tolist()], n_results=TOP_K)
        return [int(x) for x in r["ids"][0]]

    try:
        return measure("Chroma (in-process)", ingest, query, data, qs, truth)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_lancedb(data, qs, truth):
    import lancedb
    import pyarrow as pa

    tmp = tempfile.mkdtemp(prefix="lance-bench-")
    db = lancedb.connect(tmp)
    tbl = {}

    def ingest(d):
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), d.shape[1])),
        ])
        t = db.create_table(
            "bench",
            data=pa.table({"id": pa.array(range(len(d))),
                           "vector": pa.FixedSizeListArray.from_arrays(
                               pa.array(d.reshape(-1).tolist(), pa.float32()), d.shape[1])},
                          schema=schema),
        )
        tbl["t"] = t

    def query(q):
        rows = tbl["t"].search(q.tolist()).metric("cosine").limit(TOP_K).to_list()
        return [int(r["id"]) for r in rows]

    try:
        return measure("LanceDB (embedded)", ingest, query, data, qs, truth)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bench_qdrant(data, qs, truth, url="http://127.0.0.1:6444"):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

    client = QdrantClient(url=url)
    name = "memrust_bench"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config=VectorParams(size=data.shape[1], distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=M, ef_construct=EF_CONSTRUCTION),
    )

    def ingest(d):
        for i in range(0, len(d), 1000):
            client.upsert(name, points=[
                PointStruct(id=int(i + j), vector=v.tolist())
                for j, v in enumerate(d[i:i + 1000])
            ], wait=True)

    def query(q):
        res = client.query_points(name, query=q.tolist(), limit=TOP_K,
                                  search_params={"hnsw_ef": EF_SEARCH}).points
        return [int(p.id) for p in res]

    try:
        return measure("Qdrant (server, HTTP)", ingest, query, data, qs, truth)
    finally:
        client.delete_collection(name)


def bench_pgvector(data, qs, truth, dsn="postgresql://postgres:bench@127.0.0.1:5544/bench"):
    import psycopg

    conn = psycopg.connect(dsn, autocommit=True)
    dim = data.shape[1]
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS bench")
        cur.execute(f"CREATE TABLE bench (id int primary key, v vector({dim}))")

    def ingest(d):
        with conn.cursor() as cur:
            with cur.copy("COPY bench (id, v) FROM STDIN") as cp:
                for i, vec in enumerate(d):
                    cp.write_row((i, "[" + ",".join(f"{x:.6f}" for x in vec) + "]"))
            # pgvector's HNSW takes the same knobs; build after load, as docs advise.
            cur.execute(f"CREATE INDEX ON bench USING hnsw (v vector_cosine_ops) "
                        f"WITH (m = {M}, ef_construction = {EF_CONSTRUCTION})")
            cur.execute(f"SET hnsw.ef_search = {EF_SEARCH}")

    def query(q):
        vec = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM bench ORDER BY v <=> %s::vector LIMIT %s", (vec, TOP_K))
            return [r[0] for r in cur.fetchall()]

    try:
        return measure("pgvector (server, SQL)", ingest, query, data, qs, truth)
    finally:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS bench")
        conn.close()


def bench_memrust(data, qs, truth, url="http://127.0.0.1:7900"):
    def post(path, payload):
        req = urllib.request.Request(
            url + path, method="POST", data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    order: list[str] = []

    def ingest(d):
        for i in range(0, len(d), 1000):
            recs = post("/v1/remember_batch", {"items": [
                {"text": f"item {i + j}", "embedding": v.tolist()}
                for j, v in enumerate(d[i:i + 1000])]})["records"]
            order.extend(r["id"] for r in recs)

    index_of = {}

    def query(q):
        if not index_of:
            index_of.update({rid: i for i, rid in enumerate(order)})
        # Empty query text on purpose: every other engine here does pure
        # vector search, so memrust's lexical and graph legs are given
        # nothing to match and this measures the same work. (With text, BM25
        # would also score documents and fusion would reorder them — that is
        # memrust's real behavior, and agent_recall.py is where it is
        # measured. Comparing it against vector-only ground truth would
        # penalize memrust for answering a different, better question.)
        hits = post("/v1/recall", {"query": "", "query_embedding": q.tolist(),
                                   "top_k": TOP_K, "strategy": "semantic"})["hits"]
        return [index_of[h["record"]["id"]] for h in hits]

    return measure("memrust (server, HTTP)", ingest, query, data, qs, truth)


ENGINES = {
    "faiss": bench_faiss,
    "chroma": bench_chroma,
    "lancedb": bench_lancedb,
    "qdrant": bench_qdrant,
    "pgvector": bench_pgvector,
    "memrust": bench_memrust,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=list(ENGINES))
    args = ap.parse_args()

    print(f"dataset: n={args.n} dim={args.dim} queries={args.queries} "
          f"(HNSW M={M} ef_c={EF_CONSTRUCTION} ef_s={EF_SEARCH}, top-{TOP_K})")
    data, qs = clustered_dataset(args.n, args.dim, args.queries)
    truth = ground_truth(data, qs, TOP_K)

    rows = []
    for key in args.only:
        try:
            r = ENGINES[key](data, qs, truth)
            rows.append(r)
            print(f"  {r['engine']:<26} ingest {r['ingest_per_s']:>8.0f}/s  "
                  f"p50 {r['p50_ms']:>6.2f}ms  p95 {r['p95_ms']:>6.2f}ms  "
                  f"recall@10 {r['recall']:.3f}")
        except Exception as e:  # a missing server should not kill the run
            print(f"  {key:<26} SKIPPED: {type(e).__name__}: {str(e)[:90]}")

    Path("bench-results.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote bench-results.json")


if __name__ == "__main__":
    main()
