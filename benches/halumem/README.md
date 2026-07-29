# memrust on HaluMem

[HaluMem](https://github.com/MemTensor/HaluMem) is an operation-level
hallucination benchmark for agent memory systems. Instead of scoring a memory
system end-to-end, it scores three stages separately — **memory extraction**,
**memory updating**, and **memory question answering** — so a failure can be
located rather than merely observed.

This directory runs the **question answering** task against memrust.

## Scope, stated up front

memrust has no LLM in its write path. It does not extract facts from a
conversation and it does not decide when one memory supersedes another — those
are the caller's decisions, by design. So the extraction and updating tasks do
not apply to it, and reporting a number for them would describe scope rather
than quality.

What *is* comparable is the QA task: given the same conversations and the same
question, does the memory system surface what the model needs to answer
correctly? That is the job memrust exists to do.

## Parity with the reference adapter

The point of this harness is that a skeptical reader can re-run it. Everything
that could tilt the result is held to whatever `eval_memzero.py` (the
benchmark's own Mem0 adapter) does:

| | Setting |
|---|---|
| Retrieved memories | `top_k = 20` |
| Answering prompt | `PROMPT_MEMZERO`, unmodified |
| Context format | `"{timestamp}: {memory}"`, same template shape |
| Judge | `eval_tools.evaluation_for_question`, unmodified |
| Model | whatever `OPENAI_MODEL` is set to, `temperature = 0.0` |

`evaluate_qa.py` exists only because the stock `evaluation.py` aggregates all
three tasks in one pass and divides by the extraction-record count, so a
QA-only artifact makes it raise `ZeroDivisionError` before it reaches the QA
section. The judging itself is imported from the benchmark and untouched; the
three ratios use the same formulas as `aggregate_eval_results`.

## Ingestion is sequential, and that matters

Questions attached to session *N* are answered after ingesting sessions
*1..N* and no further. Ingesting a user's whole history first and then asking
every question would let a session-3 question retrieve a session-40 memory,
which inflates the score for free. The reference adapter walks sessions in
order; so does this one.

## One asymmetry, disclosed

Mem0 accepts a caller-supplied timestamp, so its stored memories carry true
conversation times. memrust stamps `created_at` at write time and has no way to
backdate a memory, so under this harness every memory looks equally recent and
the recency signal is inert — a constant across candidates, which neither helps
nor hurts ranking.

To compensate, the conversation timestamp is written into the indexed text,
where both the answering model and BM25 can see it. That is a fair trade rather
than a free one. The real fix is an optional `created_at` on `remember`, which
memrust should have regardless of this benchmark.

## Running it

```bash
# 1. the dataset (~32 MB)
curl -sL -o HaluMem-Medium.jsonl \
  https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl

# 2. the benchmark, for its prompts, judge and llm client
git clone --depth 1 https://github.com/MemTensor/HaluMem
cp eval_memrust.py evaluate_qa.py report.py HaluMem/eval/
mv HaluMem-Medium.jsonl HaluMem/data/

# 3. an engine with a real embedder — the offline hash embedder will not do
memrust serve --addr 127.0.0.1:7801 --data-dir ./hm-data \
  --embedder openai --embedding-model text-embedding-3-small \
  --lifecycle-interval-secs 0

# 4. run, then score
cd HaluMem/eval
export OPENAI_MODEL=gpt-4o OPENAI_TEMPERATURE=0.0 OPENAI_MAX_TOKENS=16384 \
       OPENAI_TIMEOUT=300 RETRY_TIMES=3 WAIT_TIME_LOWER=10 WAIT_TIME_UPPER=30 \
       OPENAI_BASE_URL=https://api.openai.com/v1
python eval_memrust.py --version medium --max-workers 5
python evaluate_qa.py --results results/memrust-medium/memrust_eval_results.jsonl \
                      --out     results/memrust-medium/qa_score.json
python report.py results/memrust-medium/qa_score.json \
                 results/memrust-medium/memrust_eval_results.jsonl
```

`--limit N` runs the first N users only. Start there.

Lifecycle is disabled (`--lifecycle-interval-secs 0`) so consolidation cannot
rewrite memories mid-run and make the result depend on wall-clock timing.

## Cost

Measured from the pilot, for the full 3,467 questions: ~11.6M input and ~0.4M
output chat tokens, plus ~5.4M embedding tokens for the 60,146 dialogue turns.

| Model | Chat | Embeddings | Total |
|---|---|---|---|
| `gpt-4o` | $32.77 | $0.11 | **~$33** |
| `gpt-4o-mini` | $1.97 | $0.11 | **~$2** |

`gpt-4o` is what the benchmark's own `.env-example` specifies. Running the
judge on a different model produces a number that is not comparable to the
published leaderboard, so the cheap option is for iterating on the harness, not
for a citable result.

## Results

**Not yet complete.** The pilot below is one user out of twenty. Per-user
variance on a 20-persona benchmark is not something to wave away — treat this
as evidence the harness works, not as a score.

### Pilot — user 1 of 20, 164 questions, `gpt-4o` judge

| | memrust (pilot) |
|---|---|
| Correct | **76.22%** |
| Hallucination | 13.41% |
| Omission | 10.37% |

By question type: Memory Conflict 87.18%, Memory Boundary 84.62%,
Multi-hop Inference 55.56%.

### Published HaluMem-Medium leaderboard, QA task

Transcribed from the HaluMem README for reference. These are full 20-user runs;
the pilot above is not, and the two are not yet comparable.

| System | Correct | Hallucination | Omission | Ingest (min) |
|---|---|---|---|---|
| MemOS | 67.23% | 15.17% | 17.59% | 1,028.84 |
| Zep | 55.47% | 21.92% | 22.62% | — |
| Mem0-Graph | 54.66% | 19.28% | 26.06% | 2,840.07 |
| Supermemory | 54.07% | 22.24% | 23.69% | 273.21 |
| Mem0 | 53.02% | 19.17% | 27.81% | 2,768.14 |
| Memobase | 35.33% | 29.97% | 34.71% | 293.30 |

Two notes on reading that table. HaluMem is published by MemTensor, who also
build MemOS; a benchmark whose authors' own system leads it deserves the same
scrutiny any vendor benchmark does. And the ingest column is the benchmark
authors' own measurement of what an LLM write path costs: Mem0 spends **46
hours** ingesting the dialogues that memrust ingests in minutes.
