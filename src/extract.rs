//! Optional memory extraction: distilling durable facts out of a conversation.
//!
//! This is the one place in memrust where an LLM can sit near a write, and it
//! is deliberately a *layer* rather than a step in `remember`. The engine's
//! write path stays deterministic — same input, same memory, no model
//! involved — and callers who want a model to decide what is worth remembering
//! opt in through `POST /v1/ingest`.
//!
//! Two design choices distinguish this from the extract-everything approach:
//!
//! **Raw turns are kept.** Systems that discard the conversation after
//! extraction do so because their write path costs two LLM calls per memory,
//! which makes storing everything prohibitive. memrust's does not, so it
//! stores the verbatim turns *and* the distilled facts, with the facts
//! carrying `sources` back to the turns they came from. Nothing is lost, and
//! an extraction mistake is recoverable because the original is still there.
//!
//! **Deduplication is deterministic.** Deciding whether a new fact is already
//! known is a cosine comparison against existing memories, not a second LLM
//! call. It costs nothing, it cannot hallucinate, and it gives the same answer
//! every time. Only superseding — deciding that a new fact makes an old one
//! *wrong* — needs judgement, and that stays off by default.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::types::MemoryKind;

/// Cap on memories per exchange. A long transcript without one produces a
/// wall of near-duplicate trivia that then competes for retrieval slots
/// forever.
pub const DEFAULT_MAX_FACTS: usize = 12;

/// One turn of the conversation being ingested.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// A durable fact an extractor proposes storing.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub text: String,
    pub kind: MemoryKind,
    pub importance: f32,
    pub tags: Vec<String>,
    /// Filled in during planning so the commit phase does not re-embed.
    pub embedding: Option<Vec<f32>>,
}

pub trait Extractor: Send + Sync {
    /// Propose durable memories from `turns`. `known` carries facts already in
    /// the store, so the model can avoid restating them — advisory only, since
    /// deduplication is enforced afterwards by cosine distance regardless of
    /// what the model does with this.
    fn extract(&self, turns: &[Turn], known: &[String]) -> Result<Vec<Candidate>>;

    /// Which of `existing` does `fact` make obsolete? Returns indices.
    ///
    /// The default is "none". Superseding deletes a memory the caller once
    /// believed, so an extractor that cannot make that judgement well should
    /// not make it at all — silently keeping both is recoverable, deleting the
    /// correct one is not.
    fn superseded_by(&self, _fact: &str, _existing: &[String]) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
}

const EXTRACT_SYSTEM: &str = "\
You extract durable memories from a conversation, for an agent that must recall them weeks later.

Keep: decisions, preferences, commitments, identifiers, names, numbers, dates, stable facts about \
people and systems, and how something was done.
Drop: pleasantries, acknowledgements, restatements of the question, anything true only during this \
exchange, and anything you are inferring rather than reading.

Write each memory as one self-contained third-person sentence that will still make sense with no \
surrounding context. Name the subject explicitly instead of writing \"the user\". Preserve \
identifiers, numbers and dates exactly as written.

Classify each as one of:
  semantic   - a stable fact or preference
  episodic   - something that happened, tied to a time
  procedural - how to do something: a workflow, runbook or convention
  reflection - a conclusion the assistant drew about its own behaviour

Score importance 0.0-1.0: how much it would hurt to have forgotten this in a month.

Reply with ONLY a JSON array, no prose and no code fence:
[{\"text\": \"...\", \"kind\": \"semantic\", \"importance\": 0.7}]
Return [] when the exchange contains nothing worth keeping. That is a normal answer, not a failure.";

const SUPERSEDE_SYSTEM: &str = "\
You decide which stored memories a new fact makes OBSOLETE — not merely related to.

A memory is obsolete only when the new fact makes it factually WRONG: the value changed, the \
decision was reversed, the preference was replaced.

A memory is NOT obsolete when it is merely older, similar, more detailed, less detailed, or about \
the same topic. Two facts that can both be true must both be kept.

Reply with ONLY a JSON array of the 1-based numbers of the obsolete memories. Reply [] if none are \
— that is the common answer, and guessing costs the user a memory they were relying on.";

/// Extraction over the OpenAI-compatible `/chat/completions` protocol —
/// hosted (OpenAI, Gemini's compatible endpoint) or local (Ollama, LM Studio,
/// vLLM, llama.cpp).
pub struct LlmExtractor {
    url: String,
    model: String,
    api_key: String,
    agent: ureq::Agent,
    /// Cap on memories per exchange. Without one, a long transcript produces a
    /// wall of near-duplicate trivia that then competes for retrieval slots.
    pub max_facts: usize,
}

impl LlmExtractor {
    pub fn openai_compatible(base_url: &str, model: &str, api_key: &str) -> Self {
        Self {
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(120))
                .build(),
            max_facts: DEFAULT_MAX_FACTS,
        }
    }

    fn chat(&self, system: &str, user: String) -> Result<String> {
        let mut req = self.agent.post(&self.url);
        if !self.api_key.is_empty() {
            req = req.set("authorization", &format!("Bearer {}", self.api_key));
        }
        let response: Value = req
            .send_json(json!({
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    { "role": "system", "content": system },
                    { "role": "user", "content": user }
                ]
            }))?
            .into_json()?;
        response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_string)
            .with_context(|| format!("unexpected chat completion shape: {response}"))
    }
}

/// Models wrap JSON in prose or a code fence often enough that refusing to
/// cope means a working extractor looks broken. Take the outermost array.
fn json_array(raw: &str) -> Result<Vec<Value>> {
    let trimmed = raw.trim();
    let start = trimmed.find('[');
    let end = trimmed.rfind(']');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e > s => &trimmed[s..=e],
        _ => bail!("extractor returned no JSON array: {trimmed}"),
    };
    serde_json::from_str::<Vec<Value>>(slice)
        .with_context(|| format!("extractor returned invalid JSON: {slice}"))
}

fn parse_kind(s: Option<&str>) -> MemoryKind {
    match s.unwrap_or("semantic") {
        "episodic" => MemoryKind::Episodic,
        "procedural" => MemoryKind::Procedural,
        "reflection" => MemoryKind::Reflection,
        "working" => MemoryKind::Working,
        "tool_call" => MemoryKind::ToolCall,
        _ => MemoryKind::Semantic,
    }
}

fn render(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| format!("{}: {}", t.role, t.content))
        .collect::<Vec<_>>()
        .join("\n")
}

impl Extractor for LlmExtractor {
    fn extract(&self, turns: &[Turn], known: &[String]) -> Result<Vec<Candidate>> {
        if turns.is_empty() {
            return Ok(Vec::new());
        }
        let mut user = String::new();
        if !known.is_empty() {
            user.push_str("Already stored, do not repeat:\n");
            for k in known {
                user.push_str("- ");
                user.push_str(k);
                user.push('\n');
            }
            user.push('\n');
        }
        user.push_str("Conversation:\n");
        user.push_str(&render(turns));

        let items = json_array(&self.chat(EXTRACT_SYSTEM, user)?)?;
        let mut out = Vec::new();
        for item in items {
            // A malformed entry is dropped rather than failing the exchange:
            // losing one candidate beats rejecting a conversation whose turns
            // were already stored verbatim.
            let Some(text) = item.get("text").and_then(Value::as_str) else {
                continue;
            };
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            out.push(Candidate {
                text: text.to_string(),
                kind: parse_kind(item.get("kind").and_then(Value::as_str)),
                importance: item
                    .get("importance")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.5)
                    .clamp(0.0, 1.0) as f32,
                tags: Vec::new(),
                embedding: None,
            });
            if out.len() >= self.max_facts {
                break;
            }
        }
        Ok(out)
    }

    fn superseded_by(&self, fact: &str, existing: &[String]) -> Result<Vec<usize>> {
        if existing.is_empty() {
            return Ok(Vec::new());
        }
        let listing = existing
            .iter()
            .enumerate()
            .map(|(i, e)| format!("{}. {e}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let user = format!("New fact:\n{fact}\n\nStored memories:\n{listing}");
        let items = json_array(&self.chat(SUPERSEDE_SYSTEM, user)?)?;
        Ok(items
            .iter()
            .filter_map(Value::as_u64)
            .filter(|&n| n >= 1 && n as usize <= existing.len())
            .map(|n| n as usize - 1)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A one-shot chat-completions server returning `body` as the content.
    fn mock_llm(content: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let payload = json!({
            "choices": [{ "message": { "content": content } }]
        })
        .to_string();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                // Read the whole request before replying. A single read can
                // return a partial one, and answering early makes the client
                // see a truncated response — which shows up as a bewildering
                // header error rather than anything about the test.
                let mut data = Vec::new();
                let mut buf = [0u8; 4096];
                while let Ok(n) = stream.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&data[..pos]).to_lowercase();
                        let len = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if data.len() >= pos + 4 + len {
                            break;
                        }
                    }
                }
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                );
                stream.write_all(resp.as_bytes()).ok();
                stream.flush().ok();
            }
        });
        format!("http://{addr}")
    }

    fn turns() -> Vec<Turn> {
        vec![Turn {
            role: "user".into(),
            content: "cap the Redis pool at 64".into(),
        }]
    }

    #[test]
    fn parses_a_clean_array() {
        let url = mock_llm(
            r#"[{"text":"The team capped the Redis pool at 64.","kind":"procedural","importance":0.8}]"#,
        );
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        let out = ex.extract(&turns(), &[]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, MemoryKind::Procedural);
        assert!((out[0].importance - 0.8).abs() < 1e-6);
    }

    #[test]
    fn tolerates_prose_and_code_fences_around_the_json() {
        let url = mock_llm(
            "Sure! Here are the memories:\n```json\n[{\"text\":\"Redis pool capped at 64.\"}]\n```\nHope that helps.",
        );
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        let out = ex.extract(&turns(), &[]).unwrap();
        assert_eq!(out.len(), 1);
        // Unspecified kind and importance take the documented defaults.
        assert_eq!(out[0].kind, MemoryKind::Semantic);
        assert!((out[0].importance - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empty_array_is_a_valid_answer_not_an_error() {
        let url = mock_llm("[]");
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        assert!(ex.extract(&turns(), &[]).unwrap().is_empty());
    }

    #[test]
    fn malformed_entries_are_dropped_not_fatal() {
        let url = mock_llm(r#"[{"nope":1},{"text":"   "},{"text":"A real fact."}]"#);
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        let out = ex.extract(&turns(), &[]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "A real fact.");
    }

    #[test]
    fn max_facts_caps_a_runaway_extraction() {
        let many: Vec<Value> = (0..50).map(|i| json!({ "text": format!("fact {i}") })).collect();
        let url = mock_llm(&serde_json::to_string(&many).unwrap());
        let mut ex = LlmExtractor::openai_compatible(&url, "m", "");
        ex.max_facts = 5;
        assert_eq!(ex.extract(&turns(), &[]).unwrap().len(), 5);
    }

    #[test]
    fn non_json_output_is_an_error_rather_than_silent_data_loss() {
        let url = mock_llm("I'm sorry, I can't help with that.");
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        assert!(ex.extract(&turns(), &[]).is_err());
    }

    #[test]
    fn supersede_indices_are_parsed_and_bounds_checked() {
        let url = mock_llm("[1, 3, 99, 0]");
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        let existing = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        // 99 and 0 are out of range and must not become indices.
        assert_eq!(ex.superseded_by("new", &existing).unwrap(), vec![0, 2]);
    }

    #[test]
    fn supersede_defaults_to_nothing_without_candidates() {
        let url = mock_llm("[1]");
        let ex = LlmExtractor::openai_compatible(&url, "m", "");
        assert!(ex.superseded_by("new", &[]).unwrap().is_empty());
    }
}
