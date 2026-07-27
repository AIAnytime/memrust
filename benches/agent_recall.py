"""Agent-memory retrieval quality: can the engine find the memory you meant?

Vector search is only one of the ways an agent asks for a memory. This
benchmark separates the two question shapes that matter in practice:

  paraphrase  "how quickly do feature flags reach users?"   -> semantics
  identifier  "INC-90312"                                    -> exact terms

Every engine receives the *same* all-MiniLM-L6-v2 embeddings, so semantics
are held constant. The difference is architectural: a pure vector store can
only compare embeddings, while memrust also runs BM25 and an entity graph
over the same memories and fuses the rankings.

Metric is hit@5 — is the one correct memory in the top 5?

Run:  python benches/agent_recall.py
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import urllib.request

import numpy as np

TOP_K = 5
# FAISS and PyTorch both ship OpenMP runtimes and crash if loaded in one
# process on macOS, so embedding runs as its own phase and caches to disk.
CACHE = "bench-agent-embeddings.npz"
SERVICES = ["Orion", "Meridian", "Atlas", "Vertex", "Halcyon", "Cobalt"]
TEAMS = ["billing", "search", "mobile", "platform", "growth"]


def corpus(seed: int = 7):
    """Realistic agent memories. Some carry unique identifiers — the kind of
    token an agent later asks for verbatim."""
    rng = random.Random(seed)
    memories, probes = [], []

    for i in range(60):
        svc, team = rng.choice(SERVICES), rng.choice(TEAMS)
        ticket = f"INC-{90000 + i * 7}"
        text = (f"Incident {ticket}: the {svc} service returned stale results "
                f"for the {team} team after a cache TTL change")
        probes.append({"query": ticket, "answer": len(memories), "kind": "identifier"})
        memories.append(text)

    paraphrases = [
        ("Feature flags are evaluated at the edge and cached for 30 seconds",
         "how quickly do toggles reach users?"),
        ("Rollbacks are performed with helm rollback and complete in about five minutes",
         "what is the procedure for reverting a bad deploy?"),
        ("The mobile clients reach the platform through the GraphQL gateway",
         "how do phone apps talk to our backend?"),
        ("Enterprise customers are guaranteed 99.9 percent uptime in their contracts",
         "what availability did we promise big accounts?"),
        ("Postmortems must be filed within 48 hours of an incident being resolved",
         "how long do engineers have to write up an outage?"),
        ("The nightly export writes parquet files to the analytics bucket at 02:00 UTC",
         "when does data land in the warehouse?"),
        ("On-call handoff happens every Monday at 10:00 UTC in the ops channel",
         "when does the pager change hands?"),
        ("Search relevance is tuned with a learning-to-rank model retrained weekly",
         "how is result ordering improved?"),
        ("Customer refunds above 500 dollars require a manager approval step",
         "when does a money-back request need a supervisor?"),
        ("The staging environment is reset from a production snapshot every Sunday",
         "how often is the test environment refreshed?"),
    ]
    for text, question in paraphrases:
        probes.append({"query": question, "answer": len(memories), "kind": "paraphrase"})
        memories.append(text)

    # Filler so the corpus is not trivially small.
    for i in range(430):
        svc, team = rng.choice(SERVICES), rng.choice(TEAMS)
        memories.append(f"Routine note {i}: the {team} team reviewed {svc} dashboards "
                        f"and found no anomalies worth escalating")
    return memories, probes


def hit_rate(results: dict[str, list[int]], probes, kind: str) -> float:
    subset = [p for p in probes if p["kind"] == kind]
    hits = sum(1 for p in subset if p["answer"] in results[p["query"]][:TOP_K])
    return hits / len(subset)


# --------------------------------------------------------------------------
def run_vector_engine(name, build, search, memories, probes, vecs, qvecs):
    build(vecs)
    out = {p["query"]: search(qvecs[i]) for i, p in enumerate(probes)}
    return {"engine": name,
            "identifier": hit_rate(out, probes, "identifier"),
            "paraphrase": hit_rate(out, probes, "paraphrase")}


def faiss_engine(memories, probes, vecs, qvecs):
    import faiss
    # HNSW in FAISS is built for L2; with normalized vectors L2 ranking is
    # identical to cosine, and METRIC_INNER_PRODUCT measurably degrades it.
    idx = faiss.IndexHNSWFlat(vecs.shape[1], 16)
    idx.hnsw.efConstruction, idx.hnsw.efSearch = 200, 100
    return run_vector_engine("FAISS (vector only)", lambda v: idx.add(v),
                             lambda q: idx.search(q.reshape(1, -1), TOP_K)[1][0].tolist(),
                             memories, probes, vecs, qvecs)


def exact_engine(memories, probes, vecs, qvecs):
    """Brute-force cosine: the best any pure vector search can do. Anything
    an approximate index scores is bounded by this."""
    def search(q):
        return np.argpartition(-(vecs @ q), TOP_K)[:TOP_K][
            np.argsort(-(vecs @ q)[np.argpartition(-(vecs @ q), TOP_K)[:TOP_K]])].tolist()

    return run_vector_engine("Exact vector search (ceiling)", lambda v: None,
                             search, memories, probes, vecs, qvecs)


def chroma_engine(memories, probes, vecs, qvecs):
    import chromadb
    tmp = tempfile.mkdtemp(prefix="chroma-quality-")
    coll = chromadb.PersistentClient(path=tmp).create_collection(
        "quality", metadata={"hnsw:space": "cosine"})
    try:
        return run_vector_engine(
            "Chroma (vector only)",
            lambda v: coll.add(ids=[str(i) for i in range(len(v))], embeddings=v.tolist()),
            lambda q: [int(x) for x in coll.query(query_embeddings=[q.tolist()],
                                                  n_results=TOP_K)["ids"][0]],
            memories, probes, vecs, qvecs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def qdrant_engine(memories, probes, vecs, qvecs, url="http://127.0.0.1:6444"):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    client = QdrantClient(url=url)
    name = "memrust_quality"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(
        size=vecs.shape[1], distance=Distance.COSINE))

    def build(v):
        client.upsert(name, points=[PointStruct(id=i, vector=x.tolist())
                                    for i, x in enumerate(v)], wait=True)

    try:
        return run_vector_engine(
            "Qdrant (vector only)", build,
            lambda q: [int(p.id) for p in client.query_points(
                name, query=q.tolist(), limit=TOP_K).points],
            memories, probes, vecs, qvecs)
    finally:
        client.delete_collection(name)


def memrust_engine(memories, probes, vecs, qvecs, url="http://127.0.0.1:7901"):
    def post(path, payload):
        req = urllib.request.Request(url + path, method="POST",
                                     data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    order = []
    for i in range(0, len(memories), 500):
        chunk = memories[i:i + 500]
        recs = post("/v1/remember_batch", {"items": [
            {"text": t, "embedding": v.tolist()}
            for t, v in zip(chunk, vecs[i:i + 500])]})["records"]
        order.extend(r["id"] for r in recs)
    pos = {rid: i for i, rid in enumerate(order)}

    def search(query, qvec, strategy):
        hits = post("/v1/recall", {"query": query, "query_embedding": qvec.tolist(),
                                   "top_k": TOP_K, "strategy": strategy})["hits"]
        return [pos[h["record"]["id"]] for h in hits]

    rows = []
    # memrust strategies reweight the four signals; none turns a leg off, so
    # even "semantic" keeps some lexical and graph contribution.
    for label, strategy in [("memrust (semantic-weighted)", "semantic"),
                            ("memrust (hybrid, default)", "balanced")]:
        out = {p["query"]: search(p["query"], qvecs[i], strategy)
               for i, p in enumerate(probes)}
        rows.append({"engine": label,
                     "identifier": hit_rate(out, probes, "identifier"),
                     "paraphrase": hit_rate(out, probes, "paraphrase")})
    return rows


def encode_phase():
    from sentence_transformers import SentenceTransformer

    memories, probes = corpus()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(memories, normalize_embeddings=True).astype(np.float32)
    qvecs = model.encode([p["query"] for p in probes],
                         normalize_embeddings=True).astype(np.float32)
    np.savez(CACHE, vecs=vecs, qvecs=qvecs)
    print(f"encoded {len(memories)} memories + {len(probes)} probes -> {CACHE}")


def main():
    import os

    if not os.path.exists(CACHE):
        encode_phase()
        print("re-run to benchmark (embedding and FAISS cannot share a process)")
        return
    memories, probes = corpus()
    cached = np.load(CACHE)
    vecs, qvecs = cached["vecs"], cached["qvecs"]
    print(f"corpus: {len(memories)} memories | probes: "
          f"{sum(1 for p in probes if p['kind']=='identifier')} identifier, "
          f"{sum(1 for p in probes if p['kind']=='paraphrase')} paraphrase | hit@{TOP_K}\n")

    rows = []
    for fn in (exact_engine, faiss_engine, chroma_engine, qdrant_engine):
        try:
            rows.append(fn(memories, probes, vecs, qvecs))
        except Exception as e:
            print(f"  skipped {fn.__name__}: {type(e).__name__}: {str(e)[:70]}")
    try:
        rows.extend(memrust_engine(memories, probes, vecs, qvecs))
    except Exception as e:
        print(f"  skipped memrust: {type(e).__name__}: {str(e)[:70]}")

    print(f"{'engine':<28}{'identifier':>12}{'paraphrase':>12}")
    for r in rows:
        print(f"{r['engine']:<28}{r['identifier']:>11.0%}{r['paraphrase']:>12.0%}")
    with open("bench-agent-recall.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
