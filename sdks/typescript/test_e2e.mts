// End-to-end test for the TypeScript SDK. Needs a running server:
// MEMRUST_URL=http://127.0.0.1:7700 node --experimental-strip-types test_e2e.mts
import { MemrustClient } from "./src/index.ts";

const BASE = process.env.MEMRUST_URL ?? "http://127.0.0.1:7700";
const assert = (cond: boolean, msg: string) => {
  if (!cond) throw new Error(msg);
};

const coder = new MemrustClient(BASE, { agentId: "coder" });

// Private by default for an agent-owned memory.
const rec = await coder.remember("coder private scratch: refactor auth module next", {
  kind: "working",
});
assert(rec.visibility === "private", `expected private, got ${rec.visibility}`);

// Another agent cannot see it; the coder can.
const reviewer = new MemrustClient(BASE, { agentId: "reviewer" });
const reviewerSees = await reviewer.recall("refactor auth module");
assert(
  !reviewerSees.some((h) => h.record.text.includes("scratch")),
  "privacy leak across agents",
);
const coderSees = await coder.recall("refactor auth module");
assert(
  coderSees.some((h) => h.record.text.includes("scratch")),
  "owner cannot see own memory",
);

// Shared memories flow across agents, with typed per-signal scores.
await coder.remember("Auth Service owns the login flow", {
  kind: "semantic",
  visibility: "shared",
});
const hits = await reviewer.recall("who owns login", { strategy: "relational" });
assert(hits.length > 0, "shared memory not visible");
assert(typeof hits[0].signals.graph === "number", "signals missing");

const stats = await coder.health();
assert(stats.total_memories > 0, "health failed");

await coder.forget(rec.id);
console.log("typescript SDK: all multi-agent checks passed");
