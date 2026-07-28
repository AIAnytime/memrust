"""memrust Python SDK — zero-dependency client for the memrust agent memory engine.

Talks to a running `memrust serve` over HTTP using only the standard library.

    from memrust import MemrustClient

    memory = MemrustClient("http://127.0.0.1:7700", agent_id="planner")
    memory.remember("user prefers concise answers", kind="semantic")
    for hit in memory.recall("what does the user prefer?"):
        print(hit["score"], hit["record"]["text"], hit["signals"])

Passing `agent_id` scopes the client: remembers are stamped with that
identity (private by default) and recalls run as that agent — it sees shared
memories, unowned memories, and its own private ones.

`namespace` selects an isolated store (its own indexes and data directory);
`api_key` authenticates when the server was started with --api-key.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional

__all__ = ["MemrustClient", "MemrustError"]
__version__ = "0.5.4"


class MemrustError(RuntimeError):
    """Raised when the memrust server returns an error."""


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


class MemrustClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7700",
        *,
        agent_id: Optional[str] = None,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.api_key = api_key
        self.namespace = namespace
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.namespace:
            headers["X-Memrust-Namespace"] = self.namespace
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise MemrustError(f"{method} {path} -> {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise MemrustError(f"cannot reach memrust at {self.base_url}: {e.reason}") from e

    # -- core API ----------------------------------------------------------

    def health(self) -> dict:
        """Engine stats: memory counts, index sizes, WAL tail length."""
        return self._request("GET", "/health")["stats"]

    def remember(
        self,
        text: str,
        *,
        kind: Optional[str] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        importance: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        visibility: Optional[str] = None,
        metadata: Optional[dict] = None,
        embedding: Optional[list[float]] = None,
    ) -> dict:
        """Store one memory; returns the stored record (id, entities, ...).

        kind: episodic | semantic | working | reflection | tool_call | procedural
        visibility: private | shared (default: private when an agent owns it)
        """
        body = _clean(
            {
                "text": text,
                "kind": kind,
                "tags": tags,
                "session_id": session_id,
                "agent_id": agent_id or self.agent_id,
                "importance": importance,
                "ttl_seconds": ttl_seconds,
                "visibility": visibility,
                "metadata": metadata,
                "embedding": embedding,
            }
        )
        return self._request("POST", "/v1/remember", body)["record"]

    def remember_batch(self, items: list[dict]) -> list[dict]:
        """Bulk ingest; texts are embedded in one round-trip server-side."""
        stamped = []
        for item in items:
            item = dict(item)
            item.setdefault("agent_id", self.agent_id)
            stamped.append(_clean(item))
        return self._request("POST", "/v1/remember_batch", {"items": stamped})["records"]

    def recall(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        as_agent: Optional[str] = None,
        kinds: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        rerank: Optional[bool] = None,
        query_embedding: Optional[list[float]] = None,
        ef_search: Optional[int] = None,
    ) -> list[dict]:
        """Hybrid recall (vector + BM25 + entity graph + recency).

        Returns hits as {"record": ..., "score": ..., "signals": {...}} with
        the per-signal score breakdown.
        strategy: balanced | semantic | lexical | recent | relational
        ef_search: widen the HNSW beam for this query — more accurate,
            slower. Unset uses the index default (100).
        """
        body = _clean(
            {
                "query": query,
                "top_k": top_k,
                "strategy": strategy,
                "as_agent": as_agent or self.agent_id,
                "rerank": rerank,
                "query_embedding": query_embedding,
                "ef_search": ef_search,
                "filter": _clean(
                    {
                        "kinds": kinds,
                        "tags": tags,
                        "session_id": session_id,
                        "since": since,
                        "until": until,
                    }
                )
                or None,
            }
        )
        return self._request("POST", "/v1/recall", body)["hits"]

    def forget(self, memory_id: str) -> bool:
        """Durably delete a memory by id."""
        return self._request("POST", "/v1/forget", {"id": memory_id})["forgotten"]

    # -- lifecycle ---------------------------------------------------------

    def run_lifecycle(self) -> dict:
        """Sweep expired memories and consolidate old episodic ones."""
        return self._request("POST", "/v1/lifecycle/run")["report"]

    def snapshot(self, session_id: Optional[str] = None) -> dict:
        """Export live memories (optionally one session), restorable anywhere."""
        return self._request("POST", "/v1/snapshot", _clean({"session_id": session_id}))["snapshot"]

    def restore(self, records: list[dict]) -> int:
        """Import snapshot records (id-preserving, idempotent)."""
        return self._request("POST", "/v1/restore", {"records": records})["restored"]

    def checkpoint(self) -> None:
        """Persist index state and truncate the WAL."""
        self._request("POST", "/v1/checkpoint")

    # -- namespaces --------------------------------------------------------

    def namespaces(self) -> list[str]:
        """Every namespace on the server. Needs a key with full access."""
        return self._request("GET", "/v1/namespaces")["namespaces"]

    def drop_namespace(self, namespace: str) -> bool:
        """Delete a namespace and everything in it. Irreversible."""
        return self._request("POST", "/v1/namespaces/drop", {"namespace": namespace})["dropped"]
