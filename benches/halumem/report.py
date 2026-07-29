"""Print the memrust QA result next to HaluMem's published Medium leaderboard."""

import json
import sys

# Transcribed from the HaluMem README leaderboard, HaluMem-Medium,
# Question Answering columns: Correct / Hallucination / Omission.
PUBLISHED = [
    ("MemOS",       67.23, 15.17, 17.59, 1028.84),
    ("Zep",         55.47, 21.92, 22.62, None),
    ("Mem0-Graph",  54.66, 19.28, 26.06, 2840.07),
    ("Supermemory", 54.07, 22.24, 23.69, 273.21),
    ("Mem0",        53.02, 19.17, 27.81, 2768.14),
    ("Memobase",    35.33, 29.97, 34.71, 293.30),
]

score = json.load(open(sys.argv[1]))["overall_score"]
results = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None

c = score["correct_qa_ratio"]["all"] * 100
h = score["hallucination_qa_ratio"]["all"] * 100
o = score["omission_qa_ratio"]["all"] * 100

ingest_min = None
if results:
    ms = 0.0
    with open(sys.argv[2], encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            for s in json.loads(line).get("sessions", []):
                ms += s.get("add_dialogue_duration_ms", 0.0)
    ingest_min = ms / 1000 / 60

rows = PUBLISHED + [("memrust", c, h, o, ingest_min)]
rows.sort(key=lambda r: -r[1])

print(f"\nHaluMem-Medium — memory question answering  (n={score['qa_num']})\n")
print(f"{'System':<14}{'Correct':>9}{'Halluc.':>9}{'Omission':>10}{'Ingest (min)':>14}")
print("-" * 56)
for name, cc, hh, oo, t in rows:
    star = "*" if name == "memrust" else " "
    tt = f"{t:,.1f}" if t else "-"
    print(f"{star}{name:<13}{cc:>8.2f}%{hh:>8.2f}%{oo:>9.2f}%{tt:>14}")
print("\n* measured here; all others transcribed from the HaluMem README.")
print("  memrust runs the QA task only — it has no LLM write path, so the")
print("  extraction and updating tasks do not apply to it.")
