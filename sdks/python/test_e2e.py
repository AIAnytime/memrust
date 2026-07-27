"""End-to-end test for the Python SDK. Needs a running server:
MEMRUST_URL=http://127.0.0.1:7700 python3 sdks/python/test_e2e.py
Uses a throwaway data dir server — the test writes and deletes memories."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memrust import MemrustClient

BASE = os.environ.get("MEMRUST_URL", "http://127.0.0.1:7700")

planner = MemrustClient(BASE, agent_id="planner")
researcher = MemrustClient(BASE, agent_id="researcher")
operator = MemrustClient(BASE)  # unscoped

# Planner keeps a private note; researcher publishes a shared finding.
planner.remember("draft strategy: undercut Initech pricing by 20 percent", kind="working")
researcher.remember(
    "Initech announced their Q3 pricing at $50 per seat",
    kind="semantic",
    visibility="shared",
    tags=["pricing"],
)

# Researcher must NOT see the planner's private draft.
seen = [h["record"]["text"] for h in researcher.recall("Initech pricing strategy")]
assert not any("undercut" in t for t in seen), f"privacy leak: {seen}"
assert any("Q3 pricing" in t for t in seen), seen

# Planner sees both its own private note and the shared finding.
seen = [h["record"]["text"] for h in planner.recall("Initech pricing strategy")]
assert any("undercut" in t for t in seen), seen
assert any("Q3 pricing" in t for t in seen), seen

# Unscoped operator sees everything; signals are exposed.
hits = operator.recall("Initech pricing", top_k=5)
assert len(hits) == 2, [h["record"]["text"] for h in hits]
assert all("signals" in h and "graph" in h["signals"] for h in hits)

# Batch + lifecycle + snapshot round-trip.
records = operator.remember_batch(
    [{"text": f"log line {i} about deploys", "session_id": "ops"} for i in range(3)]
)
assert len(records) == 3
snap = operator.snapshot("ops")
assert len(snap["records"]) == 3
assert operator.restore(snap["records"]) == 0  # idempotent: already present
stats = operator.health()
assert stats["total_memories"] == 5, stats
operator.checkpoint()
assert operator.health()["wal_tail_ops"] == 0

# Forget works and is visible immediately.
victim = records[0]["id"]
assert operator.forget(victim) is True
assert operator.health()["total_memories"] == 4

print("python SDK: all multi-agent + lifecycle checks passed")
