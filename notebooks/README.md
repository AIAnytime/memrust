# memrust notebooks

Three runnable notebooks. Everything installs with pip; the engine downloads
as a single static binary (no Rust toolchain needed).

| # | Notebook | What it covers | Open |
|---|---|---|---|
| 1 | `01_quickstart.ipynb` | install → remember → explained recall → strategies → dashboard, in ~2 minutes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIAnytime/memrust/blob/main/notebooks/01_quickstart.ipynb) |
| 2 | `02_rag_with_embeddings.ipynb` | sentence-transformers embeddings, vector storage, hybrid retrieval, a full RAG loop, memory lifecycle | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIAnytime/memrust/blob/main/notebooks/02_rag_with_embeddings.ipynb) |
| 3 | `03_pdf_rag_agents.ipynb` | pypdf ingestion, a LangGraph RAG agent that writes memories back, multi-agent visibility, the whole feature set | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AIAnytime/memrust/blob/main/notebooks/03_pdf_rag_agents.ipynb) |

## Opening the dashboard

`memrust serve` embeds a management UI at `http://127.0.0.1:7700/`. **In Colab
that address in your own browser will refuse the connection** — the server
lives inside the Colab VM, not on your machine. Every notebook has a cell that
proxies the port:

```python
from google.colab import output
output.serve_kernel_port_as_window(7700)
```

Running locally instead, `http://127.0.0.1:7700/` works directly.

## Notes

- **API keys are optional.** Notebook 1 needs none. Notebook 2 prints the
  grounded context instead of generating if you skip the key. Notebook 3's
  LangGraph agent needs an OpenAI key for generation; every memory cell runs
  without one.
- **Each notebook uses its own data dir** (`memory-quickstart`, `memory-rag`,
  `memory-pdf-rag`), so they never collide — a collection's vector dimension
  is fixed by the first vector stored in it.
- **Embeddings**: notebooks 2 and 3 use **BGE-large-en-v1.5** (1024-dim,
  ~1.3 GB — the first run downloads it) and embed locally, passing vectors
  explicitly (bring-your-own). Swap `BAAI/bge-base-en-v1.5` (768-dim) or
  `all-MiniLM-L6-v2` (384-dim) for a smaller/faster download. To let the
  engine embed server-side instead, start it with `--embedder openai
  --embedding-model text-embedding-3-small` and drop the `embedding=` /
  `query_embedding=` arguments.
- **BGE is asymmetric**: queries take an instruction prefix, stored passages
  don't. The notebooks model this with separate `emb()` / `emb_query()`
  helpers — getting it wrong quietly costs recall.
- **Running locally?** Replace the download in the bootstrap cell with
  `cargo build --release` and point `BIN` at `target/release/memrust`.
