/**
 * memrust TypeScript SDK — zero-dependency client for the memrust agent
 * memory engine. Uses fetch; works in Node 18+, Bun, Deno and browsers.
 *
 *   import { MemrustClient } from "memrust-client";
 *
 *   const memory = new MemrustClient("http://127.0.0.1:7700", { agentId: "planner" });
 *   await memory.remember("user prefers concise answers", { kind: "semantic" });
 *   const hits = await memory.recall("what does the user prefer?");
 *
 * Passing `agentId` scopes the client: remembers are stamped with that
 * identity (private by default) and recalls run as that agent — it sees
 * shared memories, unowned memories, and its own private ones.
 */

export type MemoryKind =
  | "episodic"
  | "semantic"
  | "working"
  | "reflection"
  | "tool_call"
  | "procedural";

export type RecallStrategy = "balanced" | "semantic" | "lexical" | "recent" | "relational";

export type Visibility = "private" | "shared";

export interface MemoryRecord {
  id: string;
  kind: MemoryKind;
  text: string;
  created_at: number;
  importance: number;
  expires_at?: number;
  sources?: string[];
  entities?: string[];
  tags?: string[];
  session_id?: string;
  agent_id?: string;
  visibility: Visibility;
  metadata?: unknown;
}

export interface RecallSignals {
  vector: number;
  lexical: number;
  graph: number;
  recency: number;
  importance: number;
  rerank: number;
}

export interface RecallHit {
  record: MemoryRecord;
  score: number;
  signals: RecallSignals;
}

export interface RememberOptions {
  kind?: MemoryKind;
  tags?: string[];
  sessionId?: string;
  agentId?: string;
  importance?: number;
  ttlSeconds?: number;
  visibility?: Visibility;
  metadata?: unknown;
  embedding?: number[];
}

export interface RecallOptions {
  topK?: number;
  strategy?: RecallStrategy;
  asAgent?: string;
  kinds?: MemoryKind[];
  tags?: string[];
  sessionId?: string;
  since?: number;
  until?: number;
  rerank?: boolean;
  queryEmbedding?: number[];
}

export interface EngineStats {
  total_memories: number;
  vector_indexed: number;
  lexical_indexed: number;
  embedding_dim: number;
  vector_dim: number | null;
  entities: number;
  quantized: boolean;
  wal_tail_ops: number;
}

export interface LifecycleReport {
  expired_swept: number;
  batches_consolidated: number;
  summaries: string[];
  checkpointed: boolean;
}

export interface Snapshot {
  created_at: number;
  session_id?: string;
  records: MemoryRecord[];
}

export class MemrustError extends Error {}

function clean(obj: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined));
}

export class MemrustClient {
  readonly baseUrl: string;
  readonly agentId?: string;

  constructor(baseUrl = "http://127.0.0.1:7700", opts: { agentId?: string } = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.agentId = opts.agentId;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const resp = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new MemrustError(`${method} ${path} -> ${resp.status}: ${await resp.text()}`);
    }
    return (await resp.json()) as T;
  }

  /** Engine stats: memory counts, index sizes, WAL tail length. */
  async health(): Promise<EngineStats> {
    const r = await this.request<{ stats: EngineStats }>("GET", "/health");
    return r.stats;
  }

  /** Store one memory; returns the stored record. */
  async remember(text: string, opts: RememberOptions = {}): Promise<MemoryRecord> {
    const r = await this.request<{ record: MemoryRecord }>(
      "POST",
      "/v1/remember",
      clean({
        text,
        kind: opts.kind,
        tags: opts.tags,
        session_id: opts.sessionId,
        agent_id: opts.agentId ?? this.agentId,
        importance: opts.importance,
        ttl_seconds: opts.ttlSeconds,
        visibility: opts.visibility,
        metadata: opts.metadata,
        embedding: opts.embedding,
      }),
    );
    return r.record;
  }

  /** Bulk ingest; texts are embedded in one round-trip server-side. */
  async rememberBatch(
    items: Array<{ text: string } & RememberOptions>,
  ): Promise<MemoryRecord[]> {
    const r = await this.request<{ records: MemoryRecord[] }>("POST", "/v1/remember_batch", {
      items: items.map((item) =>
        clean({
          text: item.text,
          kind: item.kind,
          tags: item.tags,
          session_id: item.sessionId,
          agent_id: item.agentId ?? this.agentId,
          importance: item.importance,
          ttl_seconds: item.ttlSeconds,
          visibility: item.visibility,
          metadata: item.metadata,
          embedding: item.embedding,
        }),
      ),
    });
    return r.records;
  }

  /**
   * Hybrid recall (vector + BM25 + entity graph + recency). Each hit carries
   * the per-signal score breakdown, so callers can see *why* it surfaced.
   */
  async recall(query: string, opts: RecallOptions = {}): Promise<RecallHit[]> {
    const filter = clean({
      kinds: opts.kinds,
      tags: opts.tags,
      session_id: opts.sessionId,
      since: opts.since,
      until: opts.until,
    });
    const r = await this.request<{ hits: RecallHit[] }>(
      "POST",
      "/v1/recall",
      clean({
        query,
        top_k: opts.topK,
        strategy: opts.strategy,
        as_agent: opts.asAgent ?? this.agentId,
        rerank: opts.rerank,
        query_embedding: opts.queryEmbedding,
        filter: Object.keys(filter).length > 0 ? filter : undefined,
      }),
    );
    return r.hits;
  }

  /** Durably delete a memory by id. */
  async forget(memoryId: string): Promise<boolean> {
    const r = await this.request<{ forgotten: boolean }>("POST", "/v1/forget", { id: memoryId });
    return r.forgotten;
  }

  /** Sweep expired memories and consolidate old episodic ones. */
  async runLifecycle(): Promise<LifecycleReport> {
    const r = await this.request<{ report: LifecycleReport }>("POST", "/v1/lifecycle/run");
    return r.report;
  }

  /** Export live memories (optionally one session), restorable anywhere. */
  async snapshot(sessionId?: string): Promise<Snapshot> {
    const r = await this.request<{ snapshot: Snapshot }>(
      "POST",
      "/v1/snapshot",
      clean({ session_id: sessionId }),
    );
    return r.snapshot;
  }

  /** Import snapshot records (id-preserving, idempotent). */
  async restore(records: MemoryRecord[]): Promise<number> {
    const r = await this.request<{ restored: number }>("POST", "/v1/restore", { records });
    return r.restored;
  }

  /** Persist index state and truncate the WAL. */
  async checkpoint(): Promise<void> {
    await this.request("POST", "/v1/checkpoint");
  }
}
