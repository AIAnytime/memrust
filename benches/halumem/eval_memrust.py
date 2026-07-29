"""HaluMem adapter for memrust — memory question answering task.

Scope, stated plainly: this runs the **memory QA** task only. memrust has no
LLM in its write path, so it does not attempt the extraction or updating
operations HaluMem also scores; there is nothing to measure there and
reporting a zero would be an artifact of scope, not of quality.

What it does, per user:

  1. Walk the user's sessions in order.
  2. Store every dialogue turn verbatim, one memory per turn.
  3. After ingesting session N, answer the questions attached to session N.

Step 3 is the part that matters for honesty: questions only ever see memories
from sessions up to and including their own, exactly as eval_memzero.py does.
Ingesting everything first and then asking would leak future memories into
past questions and inflate the score.

Parity with the Mem0 adapter is deliberate:
  * top_k = 20 retrieved memories
  * PROMPT_MEMZERO, unmodified, as the answering prompt
  * the same "{timestamp}: {memory}" context lines
  * the same OpenAI model, via the benchmark's own llms.llm_request

Timestamps are the conversation's own, passed through `created_at`, so the
recency signal decays from when a turn happened rather than from when this
script wrote it. The timestamp is *also* written into the indexed text, which
is what the reference adapter puts in front of the model — keeping that makes
the answering step identical rather than quietly advantaged.

(Earlier revisions of this harness could not do the first of those: memrust had
no way to backdate a memory, so every imported turn looked equally recent and
recency was inert. Running this benchmark is what surfaced the gap.)
"""

import os
import re
import copy
import json
import time
import argparse
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llms import llm_request
from prompts import PROMPT_MEMZERO


MEMRUST_URL = os.getenv("MEMRUST_URL", "http://127.0.0.1:7801")

# Same shape as eval_memzero.TEMPLATE_MEM0, so the answering prompt sees the
# same thing structurally regardless of which system produced the memories.
TEMPLATE_MEMRUST = """Memories for user {user_id}:

    {memories}
"""


# ---------------------------------------------------------------- transport


def _post(path: str, body: dict, namespace: str, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{MEMRUST_URL}{path}",
        method="POST",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "X-Memrust-Namespace": namespace,
        },
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"POST {path} failed after retries: {last}")


DATE_FORMAT = "%b %d, %Y, %H:%M:%S"


def to_unix_ms(stamp: str):
    """HaluMem's 'Dec 15, 2025, 08:41:23' -> Unix milliseconds, or None."""
    if not stamp:
        return None
    try:
        dt = datetime.strptime(stamp, DATE_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def add_dialogue(namespace: str, turns: list, session_idx: int) -> float:
    """Store one session's turns verbatim. One batch = one embedding round-trip."""
    items = [
        {
            "text": f"[{t.get('timestamp', '')}] {t['role']}: {t['content']}",
            "kind": "episodic",
            "session_id": f"s{session_idx}",
            # The conversation's own time, so recency decays from when the turn
            # happened rather than from when this import ran.
            "created_at": to_unix_ms(t.get("timestamp", "")),
            "metadata": {
                "timestamp": t.get("timestamp", ""),
                "role": t["role"],
                "turn": t.get("dialogue_turn"),
            },
        }
        for t in turns
    ]
    # created_at is dropped when unparseable rather than sent as null, so the
    # engine falls back to "now" for that turn instead of rejecting the batch.
    items = [{k: v for k, v in it.items() if v is not None} for it in items]
    if not items:
        return 0.0
    start = time.time()
    _post("/v1/remember_batch", {"items": items}, namespace)
    return (time.time() - start) * 1000


def search_memory(namespace: str, user_name: str, query: str, top_k: int = 20):
    start = time.time()
    hits = _post(
        "/v1/recall",
        {"query": query, "top_k": top_k, "strategy": "balanced"},
        namespace,
    )["hits"]
    duration_ms = (time.time() - start) * 1000

    memories = [h["record"]["text"] for h in hits]
    context = TEMPLATE_MEMRUST.format(
        user_id=user_name,
        memories=json.dumps(memories, indent=4),
    )
    return context, memories, duration_ms


# ---------------------------------------------------------------- the run


def extract_user_name(persona_info: str) -> str:
    match = re.search(r"Name:\s*(.*?); Gender:", persona_info)
    if not match:
        raise ValueError("No name found.")
    return match.group(1).strip()


def process_user(user_data: dict, top_k: int, save_path: str) -> dict:
    user_name = extract_user_name(user_data["persona_info"])
    namespace = "u" + user_data["uuid"].replace("-", "")[:24]

    tmp_dir = os.path.join(save_path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(tmp_dir, f"{user_data['uuid']}.json")

    new_user_data = {
        "uuid": user_data["uuid"],
        "user_name": user_name,
        "sessions": [],
    }

    try:
        sessions = user_data["sessions"]
        for sid, session in enumerate(
            tqdm(sessions, desc=f"{user_name[:18]:<18}", leave=False)
        ):
            new_session = {}

            add_ms = add_dialogue(namespace, session.get("dialogue", []), sid)
            new_session["add_dialogue_duration_ms"] = add_ms

            if not session.get("questions"):
                new_user_data["sessions"].append(new_session)
                continue

            new_session["questions"] = []
            for qa in session["questions"]:
                context, _, search_ms = search_memory(
                    namespace, user_name, qa["question"], top_k=top_k
                )
                new_qa = copy.deepcopy(qa)
                new_qa["context"] = context
                new_qa["search_duration_ms"] = search_ms

                prompt = PROMPT_MEMZERO.format(
                    context=context, question=qa["question"]
                )
                start = time.time()
                new_qa["system_response"] = llm_request(prompt)
                new_qa["response_duration_ms"] = (time.time() - start) * 1000

                new_session["questions"].append(new_qa)

            new_user_data["sessions"].append(new_session)

        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(new_user_data, f, ensure_ascii=False)
        return {"uuid": user_data["uuid"], "status": "ok", "path": tmp_file}

    except Exception as e:
        with open(os.path.join(tmp_dir, f"{user_data['uuid']}_error.log"), "w") as f:
            f.write(traceback.format_exc())
        print(f"error on {user_name}: {e}")
        return {"uuid": user_data["uuid"], "status": "error"}


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(data_path: str, version: str, top_k: int, max_workers: int, limit: int):
    frame = "memrust"
    save_path = f"results/{frame}-{version}/"
    os.makedirs(os.path.join(save_path, "tmp"), exist_ok=True)

    users = list(iter_jsonl(data_path))
    if limit:
        users = users[:limit]

    print(f"{len(users)} users, top_k={top_k}, workers={max_workers} -> {save_path}")
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_user, u, top_k, save_path): u["uuid"] for u in users}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"[{i}/{len(users)}] {r['status']} {r['uuid']}")

    out = os.path.join(save_path, f"{frame}_eval_results.jsonl")
    tmp_dir = os.path.join(save_path, "tmp")
    with open(out, "w", encoding="utf-8") as f_out:
        for name in sorted(os.listdir(tmp_dir)):
            if name.endswith(".json"):
                with open(os.path.join(tmp_dir, name), encoding="utf-8") as f_in:
                    f_out.write(json.dumps(json.load(f_in), ensure_ascii=False) + "\n")

    print(f"done in {time.time() - start:.1f}s -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="../data/HaluMem-Medium.jsonl")
    p.add_argument("--version", default="medium")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first N users only")
    a = p.parse_args()
    main(a.data_path, a.version, a.top_k, a.max_workers, a.limit)
