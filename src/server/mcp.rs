//! Minimal MCP (Model Context Protocol) server over stdio, no SDK required.
//! Exposes memory_remember / memory_recall / memory_forget as tools, so any
//! MCP client (Claude Code, Claude Desktop, agent frameworks) can mount the
//! engine as native memory:
//!
//!   claude mcp add memory -- memrust mcp --data-dir ~/.memrust
//!
//! With `--agent-id <name>`, every remember is stamped with that identity
//! (private by default) and every recall runs as that agent — the multi-agent
//! visibility model without the agent having to manage it.

use std::io::{BufRead, Write};
use std::sync::{Arc, RwLock};

use serde_json::{json, Value};
use uuid::Uuid;

use crate::engine::MemoryEngine;
use crate::types::{RecallRequest, RememberRequest};

type Shared = Arc<RwLock<MemoryEngine>>;

pub fn run(engine: Shared, agent_id: Option<String>) -> anyhow::Result<()> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(id) = msg.get("id").cloned() else {
            continue; // notification, nothing to answer
        };
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2024-11-05"),
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "memrust", "version": env!("CARGO_PKG_VERSION") }
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => tools_call(&engine, &params, agent_id.as_deref()),
            _ => Err((-32601, format!("method not found: {method}"))),
        };

        let response = match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, message)) => json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": code, "message": message }
            }),
        };
        let mut out = stdout.lock();
        writeln!(out, "{response}")?;
        out.flush()?;
    }
    Ok(())
}

fn tools_list() -> Value {
    json!({
        "tools": [
            {
                "name": "memory_remember",
                "description": "Store a memory. Use kind=episodic for events/conversation, semantic for distilled facts, working for short-lived task state, reflection for your own conclusions, tool_call for tool results worth keeping, procedural for how-to knowledge and workflows.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "The memory content" },
                        "kind": { "type": "string", "enum": ["episodic", "semantic", "working", "reflection", "tool_call", "procedural"] },
                        "tags": { "type": "array", "items": { "type": "string" } },
                        "session_id": { "type": "string" },
                        "importance": { "type": "number", "minimum": 0, "maximum": 1 }
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "memory_recall",
                "description": "Retrieve relevant memories via hybrid search (semantic + keyword + entity graph + recency); strategy=relational follows entity links to find connected, not just similar, memories. Returns scored hits with per-signal explanations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "top_k": { "type": "integer", "minimum": 1, "maximum": 50 },
                        "strategy": { "type": "string", "enum": ["balanced", "semantic", "lexical", "recent", "relational"] }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "memory_forget",
                "description": "Permanently delete a memory by its id.",
                "inputSchema": {
                    "type": "object",
                    "properties": { "id": { "type": "string" } },
                    "required": ["id"]
                }
            }
        ]
    })
}

fn tools_call(
    engine: &Shared,
    params: &Value,
    agent_id: Option<&str>,
) -> Result<Value, (i64, String)> {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let payload = match name {
        "memory_remember" => {
            let mut req: RememberRequest = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("invalid arguments: {e}")))?;
            if req.agent_id.is_none() {
                req.agent_id = agent_id.map(String::from);
            }
            let record = engine
                .write()
                .unwrap()
                .remember(req)
                .map_err(|e| (-32603, e.to_string()))?;
            json!({ "stored": { "id": record.id, "kind": record.kind } })
        }
        "memory_recall" => {
            let mut req: RecallRequest = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("invalid arguments: {e}")))?;
            if req.as_agent.is_none() {
                req.as_agent = agent_id.map(String::from);
            }
            let hits = engine.read().unwrap().recall(&req);
            let memories: Vec<Value> = hits
                .iter()
                .map(|h| {
                    json!({
                        "id": h.record.id,
                        "text": h.record.text,
                        "kind": h.record.kind,
                        "score": h.score,
                        "signals": h.signals,
                        "created_at": h.record.created_at,
                        "tags": h.record.tags,
                    })
                })
                .collect();
            json!({ "memories": memories })
        }
        "memory_forget" => {
            let id = args
                .get("id")
                .and_then(Value::as_str)
                .and_then(|s| Uuid::parse_str(s).ok())
                .ok_or((-32602, "id must be a UUID".to_string()))?;
            let forgotten = engine
                .write()
                .unwrap()
                .forget(id)
                .map_err(|e| (-32603, e.to_string()))?;
            json!({ "forgotten": forgotten })
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };

    Ok(json!({
        "content": [{ "type": "text", "text": payload.to_string() }]
    }))
}
