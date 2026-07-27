//! Pluggable summarization for memory consolidation, mirroring the embedder
//! design: an offline default that always works, and an OpenAI-compatible
//! remote client for LLM-quality output (OpenAI, or local models via Ollama
//! / LM Studio / vLLM — same `/chat/completions` protocol).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::index::text::tokenize;

pub trait Summarizer: Send + Sync {
    /// Distill a batch of memory texts into one compact summary.
    fn summarize(&self, memories: &[&str]) -> Result<String>;
}

/// Extractive summarizer: no model, no network. Scores each memory by how
/// densely it shares vocabulary with the rest of the batch (lexical
/// centrality) and keeps the most representative ones, in chronological
/// order, within a character budget. Crude next to an LLM, but deterministic,
/// free, and it never hallucinates.
pub struct ExtractiveSummarizer {
    pub max_chars: usize,
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self { max_chars: 500 }
    }
}

impl Summarizer for ExtractiveSummarizer {
    fn summarize(&self, memories: &[&str]) -> Result<String> {
        if memories.is_empty() {
            bail!("nothing to summarize");
        }
        let token_sets: Vec<(Vec<String>, HashSet<String>)> = memories
            .iter()
            .map(|m| {
                let tokens = tokenize(m);
                let set = tokens.iter().cloned().collect();
                (tokens, set)
            })
            .collect();
        let mut df: HashMap<&str, usize> = HashMap::new();
        for (_, set) in &token_sets {
            for t in set {
                *df.entry(t.as_str()).or_insert(0) += 1;
            }
        }
        // Centrality: shared-vocabulary mass, length-dampened so short
        // fragments don't win by default.
        let mut scored: Vec<(f32, usize)> = token_sets
            .iter()
            .enumerate()
            .map(|(i, (tokens, set))| {
                let mass: f32 = set.iter().map(|t| df[t.as_str()] as f32).sum();
                (mass / (tokens.len() as f32 + 3.0), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut chosen: Vec<usize> = Vec::new();
        let mut used = 0usize;
        for (_, i) in scored {
            let len = memories[i].len();
            if used + len > self.max_chars && !chosen.is_empty() {
                continue;
            }
            chosen.push(i);
            used += len + 2;
            if used >= self.max_chars {
                break;
            }
        }
        chosen.sort_unstable();
        Ok(chosen
            .into_iter()
            .map(|i| memories[i].trim_end_matches(['.', ';']))
            .collect::<Vec<_>>()
            .join("; "))
    }
}

const SYSTEM_PROMPT: &str = "You compress an AI agent's episodic memories into durable semantic \
facts. Given a list of memories, produce a terse third-person summary (2-4 sentences) that \
preserves concrete names, numbers, identifiers, dates and decisions. Output only the summary.";

/// LLM summarization over the OpenAI-compatible `/chat/completions` protocol.
pub struct RemoteSummarizer {
    url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
}

impl RemoteSummarizer {
    pub fn openai_compatible(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(120))
                .build(),
        }
    }
}

impl Summarizer for RemoteSummarizer {
    fn summarize(&self, memories: &[&str]) -> Result<String> {
        if memories.is_empty() {
            bail!("nothing to summarize");
        }
        let bullets = memories
            .iter()
            .map(|m| format!("- {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        let mut req = self.agent.post(&self.url);
        if !self.api_key.is_empty() {
            req = req.set("authorization", &format!("Bearer {}", self.api_key));
        }
        let response: Value = req
            .send_json(json!({
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    { "role": "system", "content": SYSTEM_PROMPT },
                    { "role": "user", "content": bullets }
                ]
            }))?
            .into_json()?;
        response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .with_context(|| format!("unexpected chat completion response shape: {response}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn extractive_keeps_the_central_theme() {
        let s = ExtractiveSummarizer { max_chars: 80 };
        let memories = [
            "checkout latency spiked to 900ms on tuesday",
            "checkout latency alarms fired again",
            "latency in checkout traced to db pool",
            "the team ate birthday cake",
        ];
        let out = s.summarize(&memories).unwrap();
        assert!(out.contains("latency"), "got: {out}");
        assert!(!out.contains("birthday"), "got: {out}");
        assert!(out.len() <= 120);
    }

    #[test]
    fn remote_summarizer_speaks_chat_completions() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut data = Vec::new();
            let mut buf = [0u8; 4096];
            while let Ok(n) = stream.read(&mut buf) {
                if n == 0 {
                    break;
                }
                data.extend_from_slice(&buf[..n]);
                if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&data[..pos]).to_lowercase();
                    let content_length = headers
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:"))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if data.len() >= pos + 4 + content_length {
                        break;
                    }
                }
            }
            let body = r#"{"choices":[{"message":{"content":"Checkout latency regressed; root cause was the db pool."}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).ok();
        });

        let s = RemoteSummarizer::openai_compatible(&format!("http://{addr}"), "test-model", "");
        let out = s.summarize(&["a", "b"]).unwrap();
        assert!(out.contains("db pool"));
    }
}
