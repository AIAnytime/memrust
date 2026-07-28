//! Namespace isolation and API-key enforcement, exercised through the real
//! HTTP surface rather than the internals — auth that only holds in a unit
//! test is not auth.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;

use memrust::server::tenancy::{EngineFactory, Registry};
use memrust::types::LifecycleConfig;

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("memrust-tenancy-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    dir
}

fn registry(dir: PathBuf) -> Arc<Registry> {
    Arc::new(Registry::new(
        dir,
        EngineFactory {
            embedder: Box::new(|| Ok(Box::new(memrust::embed::HashEmbedder::new(64)))),
            summarizer: Box::new(|| {
                Ok(Box::new(memrust::summarize::ExtractiveSummarizer::default()))
            }),
            reranker: None,
            lifecycle: LifecycleConfig::default(),
            index: Default::default(),
        },
    ))
}

/// Two namespaces must not see each other's memories, and each keeps its own
/// files on disk.
#[test]
fn namespaces_are_isolated_engines() {
    let dir = tmp_dir("isolation");
    let reg = registry(dir.clone());

    for (ns, text) in [
        ("acme", "acme roadmap secret"),
        ("globex", "globex pricing"),
    ] {
        let handle = reg.get_or_create(ns).unwrap();
        let record = handle
            .engine
            .read()
            .unwrap()
            .stage(memrust::types::RememberRequest {
                text: text.into(),
                ..Default::default()
            })
            .unwrap();
        handle
            .engine
            .write()
            .unwrap()
            .apply_staged(std::iter::once(record));
    }

    for (ns, expect, forbid) in [
        ("acme", "acme roadmap secret", "globex"),
        ("globex", "globex pricing", "acme"),
    ] {
        let handle = reg.get_or_create(ns).unwrap();
        let engine = handle.engine.read().unwrap();
        assert_eq!(engine.stats().total_memories, 1, "{ns} holds only its own");
        let hits = engine.recall(&memrust::types::RecallRequest {
            query: "roadmap secret pricing".into(),
            ..Default::default()
        });
        assert!(hits.iter().any(|h| h.record.text == expect));
        assert!(
            hits.iter().all(|h| !h.record.text.contains(forbid)),
            "{ns} must not see {forbid}"
        );
    }

    assert_eq!(reg.list(), vec!["acme".to_string(), "globex".to_string()]);
    assert!(dir.join("acme").join("memory.wal").exists());
    assert!(dir.join("globex").join("memory.wal").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dropping_a_namespace_removes_its_data() {
    let dir = tmp_dir("drop");
    let reg = registry(dir.clone());
    let handle = reg.get_or_create("temp").unwrap();
    let rec = handle
        .engine
        .read()
        .unwrap()
        .stage(memrust::types::RememberRequest {
            text: "throwaway".into(),
            ..Default::default()
        })
        .unwrap();
    handle
        .engine
        .write()
        .unwrap()
        .apply_staged(std::iter::once(rec));
    assert!(dir.join("temp").exists());

    assert!(reg.drop_namespace("temp").unwrap());
    assert!(!dir.join("temp").exists());
    assert!(
        !reg.drop_namespace("temp").unwrap(),
        "second drop is a no-op"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A data directory written before namespaces existed keeps working: the root
/// *is* the default namespace, and new namespaces live alongside it.
#[test]
fn legacy_data_directories_still_open() {
    let dir = tmp_dir("legacy");
    std::fs::create_dir_all(&dir).unwrap();
    {
        let mut engine = memrust::engine::MemoryEngine::open(&dir).unwrap();
        engine
            .remember(memrust::types::RememberRequest {
                text: "written before namespaces existed".into(),
                ..Default::default()
            })
            .unwrap();
    }
    assert!(dir.join("memory.wal").exists(), "legacy layout at root");

    let reg = registry(dir.clone());
    let handle = reg.get_or_create("default").unwrap();
    assert_eq!(handle.engine.read().unwrap().stats().total_memories, 1);

    // A new namespace does not disturb it.
    reg.get_or_create("fresh").unwrap();
    assert!(dir.join("fresh").exists());
    assert!(dir.join("memory.wal").exists());
    assert!(
        reg.drop_namespace("default").is_err(),
        "must refuse to delete the data-dir root"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// ---------------------------------------------------------------------------
// auth, over real HTTP
// ---------------------------------------------------------------------------

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn request(port: u16, method: &str, path: &str, key: Option<&str>, ns: Option<&str>) -> u16 {
    let mut cmd = Command::new("curl");
    cmd.args(["-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", method]);
    cmd.arg(format!("http://127.0.0.1:{port}{path}"));
    cmd.args(["-H", "content-type: application/json"]);
    if method == "POST" {
        cmd.args(["-d", r#"{"text":"probe","namespace":"probe"}"#]);
    }
    if let Some(k) = key {
        cmd.args(["-H", &format!("Authorization: Bearer {k}")]);
    }
    if let Some(n) = ns {
        cmd.args(["-H", &format!("X-Memrust-Namespace: {n}")]);
    }
    let out = cmd.output().expect("curl");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

#[test]
fn api_keys_gate_namespaces_over_http() {
    let bin = env!("CARGO_BIN_EXE_memrust");
    let dir = tmp_dir("auth");
    let port = free_port();
    let child = Command::new(bin)
        .args([
            "serve",
            "--addr",
            &format!("127.0.0.1:{port}"),
            "--data-dir",
            dir.to_str().unwrap(),
            "--lifecycle-interval-secs",
            "0",
            "--api-key",
            "admin-key-abcdef",
            "--api-key",
            "acme-key-abcdef:acme",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("start memrust");
    let _guard = Server(child);

    // Wait for it to bind.
    for _ in 0..60 {
        if request(port, "GET", "/health", Some("admin-key-abcdef"), None) == 200 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    assert_eq!(
        request(port, "POST", "/v1/remember", None, None),
        401,
        "no key"
    );
    assert_eq!(
        request(port, "POST", "/v1/remember", Some("wrong-key-abcdef"), None),
        401,
        "wrong key"
    );
    assert_eq!(
        request(
            port,
            "POST",
            "/v1/remember",
            Some("acme-key-abcdef"),
            Some("acme")
        ),
        200,
        "scoped key on its own namespace"
    );
    assert_eq!(
        request(
            port,
            "POST",
            "/v1/remember",
            Some("acme-key-abcdef"),
            Some("globex")
        ),
        403,
        "scoped key on someone else's namespace"
    );
    assert_eq!(
        request(
            port,
            "POST",
            "/v1/remember",
            Some("admin-key-abcdef"),
            Some("globex")
        ),
        200,
        "unrestricted key anywhere"
    );
    assert_eq!(
        request(port, "GET", "/v1/namespaces", Some("acme-key-abcdef"), None),
        403,
        "admin route rejects a scoped key"
    );
    assert_eq!(
        request(
            port,
            "GET",
            "/v1/namespaces",
            Some("admin-key-abcdef"),
            None
        ),
        200
    );
    std::fs::remove_dir_all(&dir).ok();
}
