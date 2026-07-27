//! Pluggable reranking — the quality tier above rank fusion. RRF merges
//! rankings; a reranker actually *reads* the query against each candidate.
//! Same pattern as embedder/summarizer: optional, and the OpenAI-compatible
//! implementation works with hosted APIs or local models (Ollama, LM Studio,
//! vLLM). Reranking failures degrade to fused order, never to an error.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

pub trait Reranker: Send + Sync {
    /// Relevance of each document to the query, in input order, each in
    /// 0..=1. Must return exactly `docs.len()` scores.
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>>;
}

const SYSTEM_PROMPT: &str = "You score how relevant each document is to a query. Reply with ONLY \
a JSON array of numbers, one per document in order, each from 0 (irrelevant) to 10 (perfectly \
relevant). No other text.";

/// LLM reranking over the OpenAI-compatible `/chat/completions` protocol.
pub struct LlmReranker {
    url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
}

impl LlmReranker {
    pub fn openai_compatible(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build(),
        }
    }
}

impl Reranker for LlmReranker {
    fn rerank(&self, query: &str, docs: &[&str]) -> Result<Vec<f32>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let listing = docs
            .iter()
            .enumerate()
            .map(|(i, d)| format!("{}. {d}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let mut req = self.agent.post(&self.url);
        if !self.api_key.is_empty() {
            req = req.set("authorization", &format!("Bearer {}", self.api_key));
        }
        let response: Value = req
            .send_json(json!({
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    { "role": "system", "content": SYSTEM_PROMPT },
                    { "role": "user", "content": format!("Query: {query}\n\nDocuments:\n{listing}") }
                ]
            }))?
            .into_json()?;
        let content = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .with_context(|| format!("unexpected chat completion response shape: {response}"))?;
        let cleaned = content
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();
        let raw: Vec<f32> = serde_json::from_str(cleaned)
            .with_context(|| format!("reranker did not return a JSON number array: {content}"))?;
        if raw.len() != docs.len() {
            bail!(
                "reranker returned {} scores for {} documents",
                raw.len(),
                docs.len()
            );
        }
        Ok(raw
            .into_iter()
            .map(|s| (s / 10.0).clamp(0.0, 1.0))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn parses_scores_from_chat_response() {
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
            let body = r#"{"choices":[{"message":{"content":"```json\n[2, 9.5, 0]\n```"}}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).ok();
        });

        let r = LlmReranker::openai_compatible(&format!("http://{addr}"), "test-model", "");
        let scores = r.rerank("q", &["a", "b", "c"]).unwrap();
        assert_eq!(scores.len(), 3);
        assert!(scores[1] > scores[0] && scores[0] > scores[2]);
        assert!((scores[1] - 0.95).abs() < 1e-6);
    }
}
