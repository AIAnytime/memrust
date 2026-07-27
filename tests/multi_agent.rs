use memrust::engine::MemoryEngine;
use memrust::types::{RecallRequest, RememberRequest, Visibility};

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-ma-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn remember_as(
    engine: &mut MemoryEngine,
    text: &str,
    agent: Option<&str>,
    visibility: Option<Visibility>,
) {
    engine
        .remember(RememberRequest {
            text: text.into(),
            agent_id: agent.map(String::from),
            visibility,
            ..Default::default()
        })
        .unwrap();
}

fn recall_as(engine: &MemoryEngine, query: &str, agent: Option<&str>) -> Vec<String> {
    engine
        .recall(&RecallRequest {
            query: query.into(),
            as_agent: agent.map(String::from),
            top_k: Some(20),
            ..Default::default()
        })
        .into_iter()
        .map(|h| h.record.text)
        .collect()
}

#[test]
fn visibility_defaults_follow_ownership() {
    let dir = tmp_dir("defaults");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    let owned = engine
        .remember(RememberRequest {
            text: "planner's own note".into(),
            agent_id: Some("planner".into()),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        owned.visibility,
        Visibility::Private,
        "agent memories default private"
    );

    let global = engine
        .remember(RememberRequest {
            text: "unowned global note".into(),
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        global.visibility,
        Visibility::Shared,
        "unowned memories default shared"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn private_memories_are_invisible_to_other_agents() {
    let dir = tmp_dir("privacy");
    let mut engine = MemoryEngine::open(&dir).unwrap();
    remember_as(
        &mut engine,
        "planner secret roadmap decision",
        Some("planner"),
        None,
    );
    remember_as(
        &mut engine,
        "researcher shared finding about the market",
        Some("researcher"),
        Some(Visibility::Shared),
    );
    remember_as(&mut engine, "global team convention note", None, None);

    // The researcher cannot see the planner's private memory.
    let seen = recall_as(&engine, "roadmap decision finding note", Some("researcher"));
    assert!(!seen.iter().any(|t| t.contains("secret")), "{seen:?}");
    assert!(seen.iter().any(|t| t.contains("shared finding")));
    assert!(seen.iter().any(|t| t.contains("global team")));

    // The planner sees its own private memory plus everything shared.
    let seen = recall_as(&engine, "roadmap decision finding note", Some("planner"));
    assert!(seen.iter().any(|t| t.contains("secret")));
    assert!(seen.iter().any(|t| t.contains("shared finding")));

    // Unscoped recall (operator / single-agent mode) sees everything.
    let seen = recall_as(&engine, "roadmap decision finding note", None);
    assert_eq!(seen.len(), 3);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn visibility_survives_restart() {
    let dir = tmp_dir("durable");
    {
        let mut engine = MemoryEngine::open(&dir).unwrap();
        remember_as(&mut engine, "private planner fact", Some("planner"), None);
    }
    let engine = MemoryEngine::open(&dir).unwrap();
    assert!(recall_as(&engine, "planner fact", Some("other")).is_empty());
    assert!(!recall_as(&engine, "planner fact", Some("planner")).is_empty());
    std::fs::remove_dir_all(&dir).ok();
}
