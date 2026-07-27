//! Pluggable embeddings. The engine never assumes a specific model. Four ways
//! to get vectors in:
//!
//! 1. `HashEmbedder` (default): deterministic feature hashing, zero network,
//!    zero model weights. Not a learned model — but makes the engine fully
//!    functional offline and keeps tests hermetic.
//! 2. `RemoteEmbedder::openai_compatible`: speaks the de-facto standard
//!    `/embeddings` protocol. Covers OpenAI, Mistral, Voyage — and local
//!    sentence-transformers models served by Hugging Face TEI, Ollama,
//!    LM Studio, Infinity or vLLM, which all expose the same endpoint.
//! 3. `RemoteEmbedder::gemini`: Google's `:embedContent` API.
//! 4. Bring your own vectors: `RememberRequest.embedding` +
//!    `RecallRequest.query_embedding`, embedder never called.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

use crate::index::text::tokenize;
use crate::index::vector::normalize;

pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed many texts at once. Remote implementations override this with a
    /// single API round-trip; the default just loops.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Embed a *query* (as opposed to a stored passage). Asymmetric models
    /// (E5 family and friends) retrieve better with distinct query/passage
    /// prefixes; symmetric embedders just use `embed`.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }
}

// ---------------------------------------------------------------------------
// Offline feature-hashing embedder (default)
// ---------------------------------------------------------------------------

pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn bump(&self, v: &mut [f32], feature: &str, weight: f32) {
        let mut h = DefaultHasher::new();
        feature.hash(&mut h);
        let hash = h.finish();
        let idx = (hash % self.dim as u64) as usize;
        let sign = if hash & (1 << 63) == 0 { 1.0 } else { -1.0 };
        v[idx] += sign * weight;
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0f32; self.dim];
        let tokens = tokenize(text);
        for token in &tokens {
            self.bump(&mut v, token, 1.0);
            // Character trigrams give partial-word overlap (plural forms,
            // typos, compound identifiers).
            let chars: Vec<char> = token.chars().collect();
            if chars.len() > 3 {
                for w in chars.windows(3) {
                    let gram: String = w.iter().collect();
                    self.bump(&mut v, &format!("#{gram}"), 0.4);
                }
            }
        }
        // Word bigrams capture local phrase structure.
        for pair in tokens.windows(2) {
            self.bump(&mut v, &format!("{}_{}", pair[0], pair[1]), 0.7);
        }
        normalize(&mut v);
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Remote embedding APIs
// ---------------------------------------------------------------------------

enum Protocol {
    /// POST {url} with {"model", "input"} -> {"data":[{"embedding":[..]}]}
    OpenAiCompatible,
    /// POST {url} with {"content":{"parts":[{"text"}]}} -> {"embedding":{"values":[..]}}
    Gemini,
}

pub struct RemoteEmbedder {
    protocol: Protocol,
    url: String,
    model: String,
    api_key: String,
    dim: usize,
    agent: ureq::Agent,
    /// Prefixes for asymmetric retrieval models (e.g. E5's "query: " /
    /// "passage: "). Empty for symmetric models.
    query_prefix: String,
    passage_prefix: String,
}

impl RemoteEmbedder {
    /// `base_url` examples:
    ///   OpenAI            https://api.openai.com/v1
    ///   Ollama            http://localhost:11434/v1
    ///   HF TEI / LM Studio / Infinity / vLLM: their /v1 root
    pub fn openai_compatible(base_url: &str, model: &str, api_key: &str) -> Result<Self> {
        let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
        Self::probe(Protocol::OpenAiCompatible, url, model, api_key)
    }

    /// `model` example: gemini-embedding-001
    pub fn gemini(model: &str, api_key: &str) -> Result<Self> {
        let url =
            format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent");
        Self::probe(Protocol::Gemini, url, model, api_key)
    }

    /// The embedding dimension isn't knowable without asking the model, so
    /// construction issues one probe request. This also fails fast on bad
    /// keys/URLs instead of at first ingest.
    fn probe(protocol: Protocol, url: String, model: &str, api_key: &str) -> Result<Self> {
        let mut embedder = Self {
            protocol,
            url,
            model: model.to_string(),
            api_key: api_key.to_string(),
            dim: 0,
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(30))
                .build(),
            query_prefix: String::new(),
            passage_prefix: String::new(),
        };
        let probe = embedder
            .request("memrust dimension probe")
            .context("embedding provider probe failed (check URL, model name and API key)")?;
        if probe.is_empty() {
            return Err(anyhow!("embedding provider returned an empty vector"));
        }
        embedder.dim = probe.len();
        Ok(embedder)
    }

    /// Set asymmetric prefixes, e.g. `.with_prefixes("query: ", "passage: ")`
    /// for E5-family models.
    pub fn with_prefixes(mut self, query_prefix: &str, passage_prefix: &str) -> Self {
        self.query_prefix = query_prefix.to_string();
        self.passage_prefix = passage_prefix.to_string();
        self
    }

    fn request(&self, text: &str) -> Result<Vec<f32>> {
        let response: Value = match self.protocol {
            Protocol::OpenAiCompatible => {
                let mut req = self.agent.post(&self.url);
                if !self.api_key.is_empty() {
                    req = req.set("authorization", &format!("Bearer {}", self.api_key));
                }
                req.send_json(json!({ "model": self.model, "input": text }))?
                    .into_json()?
            }
            Protocol::Gemini => self
                .agent
                .post(&self.url)
                .set("x-goog-api-key", &self.api_key)
                .send_json(json!({ "content": { "parts": [{ "text": text }] } }))?
                .into_json()?,
        };

        let values = match self.protocol {
            Protocol::OpenAiCompatible => response
                .pointer("/data/0/embedding")
                .and_then(Value::as_array),
            Protocol::Gemini => response
                .pointer("/embedding/values")
                .and_then(Value::as_array),
        }
        .with_context(|| format!("unexpected embedding response shape: {response}"))?;

        let mut v: Vec<f32> = values
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as f32)
            .collect();
        // Not all providers return unit vectors (e.g. Gemini at non-default
        // output dims); the engine's cosine math assumes normalized input.
        normalize(&mut v);
        Ok(v)
    }
}

impl Embedder for RemoteEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.request(&format!("{}{}", self.passage_prefix, text))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.request(&format!("{}{}", self.query_prefix, text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.passage_prefix, t))
            .collect();
        let texts: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        let texts = &texts[..];
        match self.protocol {
            // One round-trip: the OpenAI protocol accepts an input array and
            // returns indexed embeddings (order is not guaranteed).
            Protocol::OpenAiCompatible => {
                if texts.is_empty() {
                    return Ok(Vec::new());
                }
                let mut req = self.agent.post(&self.url);
                if !self.api_key.is_empty() {
                    req = req.set("authorization", &format!("Bearer {}", self.api_key));
                }
                let response: Value = req
                    .send_json(json!({ "model": self.model, "input": texts }))?
                    .into_json()?;
                let data = response
                    .pointer("/data")
                    .and_then(Value::as_array)
                    .with_context(|| format!("unexpected batch response shape: {response}"))?;
                let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
                for (pos, item) in data.iter().enumerate() {
                    let idx = item
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|i| i as usize)
                        .unwrap_or(pos);
                    let mut v: Vec<f32> = item
                        .pointer("/embedding")
                        .and_then(Value::as_array)
                        .context("batch item missing embedding")?
                        .iter()
                        .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                        .collect();
                    normalize(&mut v);
                    if let Some(slot) = out.get_mut(idx) {
                        *slot = Some(v);
                    }
                }
                out.into_iter()
                    .enumerate()
                    .map(|(i, o)| o.with_context(|| format!("no embedding returned for input {i}")))
                    .collect()
            }
            // Gemini's batch endpoint has a different shape; the loop is
            // correct if not optimal.
            Protocol::Gemini => texts.iter().map(|t| self.request(t)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn similar_texts_are_closer() {
        let e = HashEmbedder::new(256);
        let a = e
            .embed("the pricing discussion for enterprise customers")
            .unwrap();
        let b = e.embed("enterprise customer pricing talks").unwrap();
        let c = e.embed("kubernetes cluster failure on friday").unwrap();
        assert!(cos(&a, &b) > cos(&a, &c));
    }

    #[test]
    fn deterministic() {
        let e = HashEmbedder::new(128);
        assert_eq!(
            e.embed("hello agent").unwrap(),
            e.embed("hello agent").unwrap()
        );
    }

    /// Serve one canned OpenAI-style response per connection.
    fn mock_openai_server(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let mut data = Vec::new();
                let mut buf = [0u8; 4096];
                // Read headers, then the content-length'd body.
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
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).ok();
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn batch_embeddings_are_reordered_by_index() {
        // The server answers every request with the same out-of-order batch;
        // the probe reads item 0, the batch call must reorder by `index`.
        let base = mock_openai_server(
            r#"{"data":[{"index":1,"embedding":[0.0,1.0]},{"index":0,"embedding":[1.0,0.0]}]}"#,
        );
        let e = RemoteEmbedder::openai_compatible(&base, "test-model", "").unwrap();
        let out = e.embed_batch(&["first", "second"]).unwrap();
        assert_eq!(out[0], vec![1.0, 0.0]);
        assert_eq!(out[1], vec![0.0, 1.0]);
    }

    #[test]
    fn openai_compatible_protocol_roundtrip() {
        let base = mock_openai_server(r#"{"data":[{"embedding":[1.0,2.0,2.0,4.0]}]}"#);
        let e = RemoteEmbedder::openai_compatible(&base, "test-model", "sk-test").unwrap();
        assert_eq!(e.dim(), 4);
        let v = e.embed("hello").unwrap();
        // [1,2,2,4] has norm 5, so the normalized last component is 0.8.
        assert!((v[3] - 0.8).abs() < 1e-6, "got {v:?}");
        assert!((cos(&v, &v) - 1.0).abs() < 1e-5);
    }
}
