"""Score the memory-QA task only, using HaluMem's own judge.

Why this exists rather than `evaluation.py --frame memrust`: the stock scorer
aggregates all three tasks in one pass and divides by the extraction-record
count, so a QA-only artifact makes it raise ZeroDivisionError before it reaches
the QA section. Nothing about the QA scoring itself is changed here —
`evaluation_for_question` is imported from `eval_tools` and used as-is, with
the benchmark's own prompt, its own model and its own Correct / Hallucination /
Omission taxonomy. The three ratios below are computed with the same formulas
as `aggregate_eval_results`.
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from eval_tools import evaluation_for_question


def collect(results_path: str):
    qas = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            user = json.loads(line)
            for sid, session in enumerate(user.get("sessions", [])):
                for qa in session.get("questions", []):
                    qa["uuid"] = user["uuid"]
                    qa["session_id"] = sid
                    qas.append(qa)
    return qas


def judge(qa: dict) -> dict:
    result = evaluation_for_question(
        qa["question"],
        qa["answer"],
        "\n".join(i["memory_content"] for i in qa.get("evidence", [])),
        qa["system_response"],
    )
    out = dict(qa)
    if isinstance(result, dict):
        out["reasoning"] = result.get("reasoning")
        out["result_type"] = result.get("evaluation_result")
    else:
        out["result_type"] = None
    return out


def main(results_path: str, out_path: str, workers: int):
    qas = collect(results_path)
    print(f"{len(qas)} questions to judge")

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(judge, qa) for qa in qas]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judging"):
            records.append(fut.result())

    # Same accounting as aggregate_eval_results: (all) counts every question,
    # (valid) counts only those the judge returned a usable verdict for.
    n = len(records)
    valid = [r for r in records if r["result_type"] in
             ("Correct", "Hallucination", "Omission")]
    counts = Counter(r["result_type"] for r in valid)

    def ratio(k):
        return {
            "all": counts[k] / n if n else 0.0,
            "valid": counts[k] / len(valid) if valid else 0.0,
        }

    by_type = defaultdict(Counter)
    by_diff = defaultdict(Counter)
    for r in valid:
        by_type[r.get("question_type", "?")][r["result_type"]] += 1
        by_diff[r.get("difficulty", "?")][r["result_type"]] += 1

    def breakdown(d):
        out = {}
        for k, c in sorted(d.items()):
            tot = sum(c.values())
            out[k] = {
                "n": tot,
                "correct": round(c["Correct"] / tot, 4),
                "hallucination": round(c["Hallucination"] / tot, 4),
                "omission": round(c["Omission"] / tot, 4),
            }
        return out

    summary = {
        "frame": "memrust",
        "qa_num": n,
        "qa_valid_num": len(valid),
        "correct_qa_ratio": ratio("Correct"),
        "hallucination_qa_ratio": ratio("Hallucination"),
        "omission_qa_ratio": ratio("Omission"),
        "by_question_type": breakdown(by_type),
        "by_difficulty": breakdown(by_diff),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"overall_score": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/memrust-medium/memrust_eval_results.jsonl")
    p.add_argument("--out", default="results/memrust-medium/qa_score.json")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    main(a.results, a.out, a.workers)
