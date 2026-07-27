use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Result};
use memrust::embed::{Embedder, HashEmbedder, RemoteEmbedder};
use memrust::engine::MemoryEngine;
use memrust::index::vector::{HnswConfig, Quantization, AUTO_QUANTIZE_DIM};
use memrust::rerank::LlmReranker;
use memrust::summarize::{ExtractiveSummarizer, RemoteSummarizer, Summarizer};
use memrust::types::{LifecycleConfig, MemoryKind, RecallRequest, RecallStrategy, RememberRequest};

const USAGE: &str = "\
memrust — agent-native memory engine

USAGE:
    memrust serve [--addr 127.0.0.1:7700] [--data-dir ./memrust-data] [options]
    memrust mcp   [--data-dir ./memrust-data] [--agent-id <name>] [options]
    memrust demo
    memrust bench [--n 20000] [--dim 256] [--engine]

SCALE OPTIONS:
    --quantize      force SQ8 codes in the vector index (1 byte/dim instead of 4)
    --no-quantize   force f32 vectors
                    Default: SQ8 automatically for vectors >= 1024 dims, where it
                    is as fast as f32 and uses 4x less memory; f32 below that.

MULTI-AGENT (mcp):
    --agent-id <name>   stamp remembers with this identity (private by default)
                        and scope recalls to what this agent may see

EMBEDDING OPTIONS:
    --embedder hash|openai|gemini   default: hash (offline, no API needed)
    --embedding-model <name>        default: text-embedding-3-small / gemini-embedding-001
    --embedding-url <base>          OpenAI-compatible base URL, default https://api.openai.com/v1
                                    (Ollama: http://localhost:11434/v1; HF TEI, LM Studio,
                                     Infinity and vLLM also serve this protocol)

    --embed-query-prefix <s>        prefix for query embeddings (e.g. \"query: \" for E5)
    --embed-passage-prefix <s>      prefix for stored-memory embeddings (e.g. \"passage: \")

    API key env vars: MEMRUST_EMBED_API_KEY, or OPENAI_API_KEY / GEMINI_API_KEY.
    Local OpenAI-compatible servers need no key.

RERANK OPTIONS (serve/mcp):
    --reranker openai               enable LLM reranking of recall results
    --reranker-model <name>         default: gpt-4o-mini
    --reranker-url <base>           OpenAI-compatible base URL, default https://api.openai.com/v1

LIFECYCLE OPTIONS (serve/mcp):
    --summarizer extract|openai       default: extract (offline, no LLM)
    --summarizer-model <name>         default: gpt-4o-mini
    --summarizer-url <base>           OpenAI-compatible base URL, default https://api.openai.com/v1
    --lifecycle-interval-secs <n>     default: 300; 0 disables the background pass
    --working-ttl-secs <n>            default: 86400 (1 day)
    --consolidate-after-secs <n>      default: 604800 (7 days)
";

fn build_summarizer(args: &[String]) -> Result<Box<dyn Summarizer>> {
    match flag(args, "--summarizer", "extract").as_str() {
        "extract" => Ok(Box::new(ExtractiveSummarizer::default())),
        "openai" => {
            let url = flag(args, "--summarizer-url", "https://api.openai.com/v1");
            let model = flag(args, "--summarizer-model", "gpt-4o-mini");
            let key = std::env::var("MEMRUST_SUMMARIZER_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .unwrap_or_default();
            Ok(Box::new(RemoteSummarizer::openai_compatible(
                &url, &model, &key,
            )))
        }
        other => bail!("unknown summarizer '{other}' (expected extract or openai)"),
    }
}

fn numeric_flag(args: &[String], name: &str, default: u64) -> Result<u64> {
    flag(args, name, &default.to_string())
        .parse()
        .map_err(|_| anyhow::anyhow!("{name} must be a non-negative integer"))
}

fn build_lifecycle_config(args: &[String]) -> Result<LifecycleConfig> {
    Ok(LifecycleConfig {
        working_ttl_secs: numeric_flag(args, "--working-ttl-secs", 86400)?,
        consolidate_after_secs: numeric_flag(args, "--consolidate-after-secs", 604800)?,
        ..LifecycleConfig::default()
    })
}

type Shared = Arc<RwLock<MemoryEngine>>;

fn open_engine(args: &[String]) -> Result<Shared> {
    let dir = PathBuf::from(flag(args, "--data-dir", "./memrust-data"));
    let quantize = match (
        args.iter().any(|a| a == "--quantize"),
        args.iter().any(|a| a == "--no-quantize"),
    ) {
        (true, true) => bail!("pass either --quantize or --no-quantize, not both"),
        (true, false) => Quantization::Always,
        (false, true) => Quantization::Never,
        // Auto: decided by the first vector's width (>= AUTO_QUANTIZE_DIM).
        (false, false) => Quantization::Auto,
    };
    let index_cfg = HnswConfig {
        quantize,
        ..HnswConfig::default()
    };
    let mut engine = MemoryEngine::open_with_options(
        &dir,
        build_embedder(args)?,
        build_summarizer(args)?,
        build_lifecycle_config(args)?,
        index_cfg,
    )?;
    match flag(args, "--reranker", "none").as_str() {
        "none" => {}
        "openai" => {
            let url = flag(args, "--reranker-url", "https://api.openai.com/v1");
            let model = flag(args, "--reranker-model", "gpt-4o-mini");
            let key = std::env::var("MEMRUST_RERANK_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .unwrap_or_default();
            engine.set_reranker(Box::new(LlmReranker::openai_compatible(&url, &model, &key)));
        }
        other => bail!("unknown reranker '{other}' (expected openai)"),
    }
    Ok(Arc::new(RwLock::new(engine)))
}

/// Index microbenchmark: insert rate, search throughput and recall@10 for
/// f32 vs SQ8 storage, on two data shapes:
/// - clustered: mixture of centers + noise, the shape real embedding models
///   produce (semantically similar texts form neighborhoods)
/// - uniform: i.i.d. random directions — in high dimensions every point is
///   nearly equidistant from every other, the known worst case for any ANN
///   method and a lower bound rather than an expectation
///
/// Run with --release for meaningful numbers.
/// End-to-end recall latency through the whole engine (fusion, filters,
/// record materialization) rather than the raw vector index, so the gap
/// against the HTTP number is attributable to transport alone.
fn bench_engine(args: &[String]) -> Result<()> {
    use memrust::index::vector::normalize;
    use memrust::types::{RecallRequest, RecallStrategy, RememberRequest};
    use std::time::Instant;

    let n = numeric_flag(args, "--n", 20_000)? as usize;
    let dim = numeric_flag(args, "--dim", 384)? as usize;
    let queries = 200usize;

    let mut seed = 42u64;
    let mut rand_vec = |dim: usize| -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        normalize(&mut v);
        v
    };

    let dir = std::env::temp_dir().join(format!("memrust-engine-bench-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    let mut engine = MemoryEngine::open(&dir)?;

    let items: Vec<RememberRequest> = (0..n)
        .map(|i| RememberRequest {
            text: format!("m{i}"),
            embedding: Some(rand_vec(dim)),
            ..Default::default()
        })
        .collect();
    let t = Instant::now();
    for chunk in items.chunks(1000) {
        engine.remember_batch(chunk.to_vec())?;
    }
    let ingest = t.elapsed();

    let qs: Vec<Vec<f32>> = (0..queries).map(|_| rand_vec(dim)).collect();
    let mut lat: Vec<f64> = Vec::with_capacity(queries);
    for q in &qs {
        let req = RecallRequest {
            query: "m".into(),
            query_embedding: Some(q.clone()),
            top_k: Some(10),
            strategy: RecallStrategy::Semantic,
            ..Default::default()
        };
        let t = Instant::now();
        std::hint::black_box(engine.recall(&req));
        lat.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    lat.sort_by(f64::total_cmp);
    println!(
        "engine bench: n={n} dim={dim}\n  ingest {:.0} rec/s (in-process)\n  recall p50 {:.3} ms  p95 {:.3} ms  (full engine path, no HTTP)",
        n as f64 / ingest.as_secs_f64(),
        lat[lat.len() / 2],
        lat[(lat.len() as f64 * 0.95) as usize],
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}

fn bench(args: &[String]) -> Result<()> {
    if args.iter().any(|a| a == "--engine") {
        return bench_engine(args);
    }
    use memrust::index::vector::{normalize, FlatIndex, Hnsw};
    use std::time::Instant;

    let n = numeric_flag(args, "--n", 20_000)? as usize;
    let dim = numeric_flag(args, "--dim", 256)? as usize;
    let ef = numeric_flag(args, "--ef", 100)? as usize;
    let queries = 200;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        }
        fn vec(&mut self, dim: usize) -> Vec<f32> {
            let mut v: Vec<f32> = (0..dim).map(|_| self.next()).collect();
            normalize(&mut v);
            v
        }
        fn near(&mut self, center: &[f32]) -> Vec<f32> {
            let mut v: Vec<f32> = center.iter().map(|x| x + 0.15 * self.next()).collect();
            normalize(&mut v);
            v
        }
    }
    let mut rng = Rng(42);

    println!("memrust bench: n={n} dim={dim} (HNSW m=16 ef_search={ef}; auto-quantize at >={AUTO_QUANTIZE_DIM} dims)");
    for clustered in [true, false] {
        let (vectors, query_set): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if clustered {
            let n_centers = (n / 100).max(10);
            let centers: Vec<Vec<f32>> = (0..n_centers).map(|_| rng.vec(dim)).collect();
            let pick = |r: &mut Rng| {
                ((r.next().abs() * 2.0 * n_centers as f32) as usize).min(n_centers - 1)
            };
            (
                (0..n)
                    .map(|_| {
                        let c = pick(&mut rng);
                        rng.near(&centers[c])
                    })
                    .collect(),
                (0..queries)
                    .map(|_| {
                        let c = pick(&mut rng);
                        rng.near(&centers[c])
                    })
                    .collect(),
            )
        } else {
            (
                (0..n).map(|_| rng.vec(dim)).collect(),
                (0..queries).map(|_| rng.vec(dim)).collect(),
            )
        };

        let mut flat = FlatIndex::new();
        for v in &vectors {
            flat.add(v.clone());
        }

        let shape = if clustered { "clustered" } else { "uniform  " };
        for quantize in [Quantization::Never, Quantization::Always] {
            let cfg = HnswConfig {
                quantize,
                ef_search: ef,
                ..HnswConfig::default()
            };
            let label = if quantize == Quantization::Always {
                "sq8"
            } else {
                "f32"
            };

            let t = Instant::now();
            let mut hnsw = Hnsw::new(cfg);
            for v in &vectors {
                hnsw.add(v.clone());
            }
            let build = t.elapsed();

            let t = Instant::now();
            for q in &query_set {
                std::hint::black_box(hnsw.search(q, 10));
            }
            let search = t.elapsed();

            let mut correct = 0usize;
            let mut total = 0usize;
            for q in query_set.iter().take(50) {
                let truth: std::collections::HashSet<usize> =
                    flat.search(q, 10).into_iter().map(|(i, _)| i).collect();
                correct += hnsw
                    .search(q, 10)
                    .iter()
                    .filter(|(i, _)| truth.contains(i))
                    .count();
                total += truth.len();
            }

            let vec_bytes = if quantize == Quantization::Always {
                n * (dim + 8)
            } else {
                n * dim * 4
            };
            println!(
                "  {shape} {label}: build {:>6.2}s ({:>6.0} inserts/s) | search {:>6.0} qps | recall@10 {:.3} | vectors {:.1} MB",
                build.as_secs_f64(),
                n as f64 / build.as_secs_f64(),
                queries as f64 / search.as_secs_f64(),
                correct as f64 / total as f64,
                vec_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }
    Ok(())
}

/// Periodic background lifecycle: sweep expired working memory, consolidate
/// old episodic memory into semantic summaries.
fn spawn_lifecycle(engine: Shared, interval_secs: u64) {
    if interval_secs == 0 {
        return;
    }
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        tick.tick().await; // skip the immediate first tick
        loop {
            tick.tick().await;
            let engine = engine.clone();
            match tokio::task::spawn_blocking(move || engine.write().unwrap().run_lifecycle()).await
            {
                Ok(Ok(report)) => {
                    if report.expired_swept > 0 || report.batches_consolidated > 0 {
                        eprintln!(
                            "memrust lifecycle: swept {} expired, consolidated {} batch(es) into {} summaries",
                            report.expired_swept,
                            report.batches_consolidated,
                            report.summaries.len()
                        );
                    }
                }
                Ok(Err(e)) => eprintln!("memrust lifecycle error: {e}"),
                Err(e) => eprintln!("memrust lifecycle task panicked: {e}"),
            }
        }
    });
}

fn build_embedder(args: &[String]) -> Result<Box<dyn Embedder>> {
    let key_for = |vendor: &str| {
        std::env::var("MEMRUST_EMBED_API_KEY")
            .or_else(|_| std::env::var(vendor))
            .unwrap_or_default()
    };
    match flag(args, "--embedder", "hash").as_str() {
        "hash" => Ok(Box::new(HashEmbedder::new(256))),
        "openai" => {
            let url = flag(args, "--embedding-url", "https://api.openai.com/v1");
            let model = flag(args, "--embedding-model", "text-embedding-3-small");
            let e = RemoteEmbedder::openai_compatible(&url, &model, &key_for("OPENAI_API_KEY"))?
                .with_prefixes(
                    &flag(args, "--embed-query-prefix", ""),
                    &flag(args, "--embed-passage-prefix", ""),
                );
            Ok(Box::new(e))
        }
        "gemini" => {
            let model = flag(args, "--embedding-model", "gemini-embedding-001");
            let e = RemoteEmbedder::gemini(&model, &key_for("GEMINI_API_KEY"))?.with_prefixes(
                &flag(args, "--embed-query-prefix", ""),
                &flag(args, "--embed-passage-prefix", ""),
            );
            Ok(Box::new(e))
        }
        other => bail!("unknown embedder '{other}' (expected hash, openai or gemini)"),
    }
}

fn flag(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let addr = flag(&args, "--addr", "127.0.0.1:7700");
            let engine = open_engine(&args)?;
            spawn_lifecycle(
                engine.clone(),
                numeric_flag(&args, "--lifecycle-interval-secs", 300)?,
            );
            memrust::server::http::serve(engine, &addr).await
        }
        Some("mcp") => {
            let engine = open_engine(&args)?;
            spawn_lifecycle(
                engine.clone(),
                numeric_flag(&args, "--lifecycle-interval-secs", 300)?,
            );
            let agent_id = match flag(&args, "--agent-id", "") {
                s if s.is_empty() => None,
                s => Some(s),
            };
            // MCP is stdio line-based; run it synchronously.
            tokio::task::spawn_blocking(move || memrust::server::mcp::run(engine, agent_id)).await?
        }
        Some("demo") => demo(),
        Some("bench") => bench(&args),
        _ => {
            print!("{USAGE}");
            Ok(())
        }
    }
}

fn demo() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("memrust-demo-{}", std::process::id()));
    let mut engine = MemoryEngine::open(&dir)?;

    let seed: [(&str, MemoryKind, f32); 8] = [
        ("User asked about OpenAI GPT-6 pricing during the enterprise deal review; they quoted $40/M input tokens.", MemoryKind::Episodic, 0.9),
        ("Acme Corp's procurement contact is Dana Whitfield; renewals happen every March.", MemoryKind::Semantic, 0.7),
        ("Deployment failed with error E1234 on cluster prod-west; root cause was an expired TLS cert.", MemoryKind::Episodic, 0.8),
        ("The user prefers concise answers with code examples in Rust.", MemoryKind::Semantic, 0.6),
        ("Currently drafting the Q3 infra cost comparison; waiting on GPU quota numbers.", MemoryKind::Working, 0.5),
        ("Reflection: my web searches for pricing keep returning stale blog posts; prefer official pricing pages.", MemoryKind::Reflection, 0.7),
        ("Tool call: fetch_url('openai.com/pricing') returned updated GPT-6 tiers on 2026-07-20.", MemoryKind::ToolCall, 0.6),
        ("Team decided to standardize on Postgres 17 for all new services.", MemoryKind::Semantic, 0.8),
    ];
    for (text, kind, importance) in seed {
        engine.remember(RememberRequest {
            text: text.to_string(),
            kind,
            importance: Some(importance),
            ..Default::default()
        })?;
    }

    for (query, strategy) in [
        (
            "previous discussions about GPT-6 pricing",
            RecallStrategy::Balanced,
        ),
        ("error E1234", RecallStrategy::Lexical),
        ("what does the user like", RecallStrategy::Semantic),
    ] {
        println!("\nrecall(\"{query}\", strategy={strategy:?})");
        let hits = engine.recall(&RecallRequest {
            query: query.to_string(),
            top_k: Some(3),
            strategy,
            ..Default::default()
        });
        for hit in hits {
            println!(
                "  [{:>6.4}] vec={:.4} lex={:.4} rec={:.2} | {:?} | {}",
                hit.score,
                hit.signals.vector,
                hit.signals.lexical,
                hit.signals.recency,
                hit.record.kind,
                hit.record.text
            );
        }
    }

    let stats = engine.stats();
    println!(
        "\n{} memories, {} vector-indexed, {} lexical-indexed, dim {}",
        stats.total_memories, stats.vector_indexed, stats.lexical_indexed, stats.embedding_dim
    );
    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
