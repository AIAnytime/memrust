"""Does memrust scale across concurrent clients, and what do writes cost?

The engine sits behind a single RwLock: readers share it, a writer excludes
everyone. That is fine until it isn't, and the only way to know is to measure.

Two questions:
  1. read scaling   — does aggregate QPS grow with concurrent readers?
  2. write impact   — what happens to read latency while a writer is active?

Run:  python benches/concurrency.py            (expects memrust on :7940)
"""

from __future__ import annotations

import http.client
import json
import statistics
import threading
import time

import numpy as np

HOST, PORT, DIM, N, DURATION = "127.0.0.1", 7940, 384, 20000, 3.0


def conn():
    return http.client.HTTPConnection(HOST, PORT)


def post(c, path, payload):
    c.request("POST", path, body=json.dumps(payload),
              headers={"content-type": "application/json"})
    return json.loads(c.getresponse().read())


def ingest(vecs):
    c = conn()
    for i in range(0, len(vecs), 1000):
        post(c, "/v1/remember_batch", {"items": [
            {"text": f"m{i + j}", "embedding": v.tolist()}
            for j, v in enumerate(vecs[i:i + 1000])]})
    c.close()


def reader(qs, stop, out):
    """Hammer recall on a private connection until told to stop."""
    c, lat = conn(), []
    i = 0
    while not stop.is_set():
        t = time.perf_counter()
        post(c, "/v1/recall", {"query": "", "query_embedding": qs[i % len(qs)].tolist(),
                               "top_k": 10, "strategy": "semantic"})
        lat.append((time.perf_counter() - t) * 1000)
        i += 1
    c.close()
    out.append(lat)


def writer(vecs, stop, out):
    """Continuous single-record writes — each takes the exclusive lock."""
    c, count = conn(), 0
    while not stop.is_set():
        post(c, "/v1/remember", {"text": f"w{count}",
                                 "embedding": vecs[count % len(vecs)].tolist()})
        count += 1
    c.close()
    out.append(count)


def run(threads: int, with_writer: bool, qs, vecs):
    stop, lats, writes = threading.Event(), [], []
    workers = [threading.Thread(target=reader, args=(qs, stop, lats)) for _ in range(threads)]
    if with_writer:
        workers.append(threading.Thread(target=writer, args=(vecs, stop, writes)))
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    time.sleep(DURATION)
    stop.set()
    for w in workers:
        w.join()
    elapsed = time.perf_counter() - t0

    flat = [x for l in lats for x in l]
    flat.sort()
    return {
        "readers": threads,
        "writer": with_writer,
        "qps": len(flat) / elapsed,
        "p50": statistics.median(flat),
        "p95": flat[int(len(flat) * 0.95)],
        "writes": writes[0] if writes else 0,
    }


def main():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(N, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    qs = vecs[:200]

    print(f"ingesting {N} vectors...")
    ingest(vecs)

    print(f"\n{'readers':>8}{'writer':>8}{'QPS':>10}{'p50 ms':>9}{'p95 ms':>9}{'scaling':>9}")
    base = None
    for threads in (1, 2, 4, 8):
        r = run(threads, False, qs, vecs)
        base = base or r["qps"]
        print(f"{r['readers']:>8}{'-':>8}{r['qps']:>10.0f}{r['p50']:>9.2f}"
              f"{r['p95']:>9.2f}{r['qps'] / base:>8.1f}x")

    r = run(4, True, qs, vecs)
    print(f"{r['readers']:>8}{'yes':>8}{r['qps']:>10.0f}{r['p50']:>9.2f}{r['p95']:>9.2f}"
          f"{'':>9}   ({r['writes']} writes landed concurrently)")


if __name__ == "__main__":
    main()
