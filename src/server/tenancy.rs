//! Namespaces and API keys — the two things that stand between "interesting
//! project" and "something I can deploy".
//!
//! A namespace is a *separate engine*, not a filter on a shared one: its own
//! indexes, its own WAL and checkpoint, its own embedding dimension, its own
//! directory. That is what makes isolation real — one tenant's ingest can't
//! slow another's recall, and dropping a namespace is deleting a directory
//! rather than a scan-and-delete over a shared index.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{bail, Result};

use crate::embed::Embedder;
use crate::engine::MemoryEngine;
use crate::index::vector::HnswConfig;
use crate::extract::Extractor;
use crate::rerank::Reranker;
use crate::summarize::Summarizer;
use crate::types::LifecycleConfig;

pub const DEFAULT_NAMESPACE: &str = "default";

/// Everything needed to build an engine for a namespace that doesn't exist
/// yet. Boxed closures because embedders and summarizers are trait objects
/// and each namespace needs its own.
pub type MakeEmbedder = Box<dyn Fn() -> Result<Box<dyn Embedder>> + Send + Sync>;
pub type MakeSummarizer = Box<dyn Fn() -> Result<Box<dyn Summarizer>> + Send + Sync>;
pub type MakeReranker = Box<dyn Fn() -> Box<dyn Reranker> + Send + Sync>;
/// Extractors are shared rather than cloned per namespace: they hold only a
/// URL, a model name and an HTTP agent, and the agent's connection pool is
/// worth reusing.
pub type MakeExtractor = Box<dyn Fn() -> Arc<dyn Extractor> + Send + Sync>;

pub struct EngineFactory {
    pub embedder: MakeEmbedder,
    pub summarizer: MakeSummarizer,
    pub reranker: Option<MakeReranker>,
    pub extractor: Option<MakeExtractor>,
    pub lifecycle: LifecycleConfig,
    pub index: HnswConfig,
}

/// One namespace: its engine, plus the commit lock that brackets the
/// two-phase write path (see `server::http`).
#[derive(Clone)]
pub struct Namespace {
    pub engine: Arc<RwLock<MemoryEngine>>,
    pub commit: Arc<Mutex<()>>,
}

pub struct Registry {
    root: PathBuf,
    factory: EngineFactory,
    /// Pre-v0.6 layouts kept the WAL at the data-dir root. When that is what
    /// we find, the root *is* the default namespace and stays where it is —
    /// upgrading must not orphan somebody's memories.
    legacy_root: bool,
    open: RwLock<HashMap<String, Namespace>>,
}

impl Registry {
    pub fn new(root: PathBuf, factory: EngineFactory) -> Self {
        let legacy_root = root.join("memory.wal").exists() || root.join("checkpoint.json").exists();
        Self {
            root,
            factory,
            legacy_root,
            open: RwLock::new(HashMap::new()),
        }
    }

    /// Namespace names become directory names, so they get the treatment any
    /// path segment from an untrusted header deserves.
    pub fn validate(name: &str) -> Result<()> {
        if name.is_empty() || name.len() > 64 {
            bail!("namespace must be 1-64 characters");
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            bail!("namespace may only contain letters, digits, '-' and '_'");
        }
        Ok(())
    }

    fn dir_for(&self, name: &str) -> PathBuf {
        if self.legacy_root && name == DEFAULT_NAMESPACE {
            self.root.clone()
        } else {
            self.root.join(name)
        }
    }

    /// Get a namespace, creating it on first use.
    pub fn get_or_create(&self, name: &str) -> Result<Namespace> {
        Self::validate(name)?;
        if let Some(ns) = self.open.read().unwrap().get(name) {
            return Ok(ns.clone());
        }
        let mut open = self.open.write().unwrap();
        // Another request may have created it while we waited for the lock.
        if let Some(ns) = open.get(name) {
            return Ok(ns.clone());
        }
        let mut engine = MemoryEngine::open_with_options(
            &self.dir_for(name),
            (self.factory.embedder)()?,
            (self.factory.summarizer)()?,
            self.factory.lifecycle.clone(),
            self.factory.index.clone(),
        )?;
        if let Some(make) = &self.factory.reranker {
            engine.set_reranker(make());
        }
        if let Some(make) = &self.factory.extractor {
            engine.set_extractor(make());
        }
        let ns = Namespace {
            engine: Arc::new(RwLock::new(engine)),
            commit: Arc::new(Mutex::new(())),
        };
        open.insert(name.to_string(), ns.clone());
        Ok(ns)
    }

    /// Namespaces with data on disk, plus any currently open. Sorted.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.open.read().unwrap().keys().cloned().collect();
        if self.legacy_root && !names.iter().any(|n| n == DEFAULT_NAMESPACE) {
            names.push(DEFAULT_NAMESPACE.to_string());
        }
        if let Ok(entries) = std::fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if entry.path().is_dir() {
                    if let Some(n) = entry.file_name().to_str() {
                        if Registry::validate(n).is_ok() && !names.iter().any(|x| x == n) {
                            names.push(n.to_string());
                        }
                    }
                }
            }
        }
        names.sort();
        names
    }

    /// Close a namespace and delete its data. Irreversible.
    pub fn drop_namespace(&self, name: &str) -> Result<bool> {
        Self::validate(name)?;
        if self.legacy_root && name == DEFAULT_NAMESPACE {
            bail!("refusing to delete the data directory root; move to a namespaced layout first");
        }
        self.open.write().unwrap().remove(name);
        let dir = self.dir_for(name);
        if !dir.exists() {
            return Ok(false);
        }
        std::fs::remove_dir_all(&dir)?;
        Ok(true)
    }

    /// Namespaces currently held in memory — the ones a background lifecycle
    /// pass should sweep.
    pub fn open_namespaces(&self) -> Vec<(String, Namespace)> {
        self.open
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Open every namespace already on disk, so a restart restores what was
    /// there rather than waiting for the first request to each.
    pub fn open_existing(&self) -> Result<usize> {
        let mut n = 0;
        for name in self.list() {
            let dir = self.dir_for(&name);
            if dir.join("memory.wal").exists() || dir.join("checkpoint.json").exists() {
                self.get_or_create(&name)?;
                n += 1;
            }
        }
        Ok(n)
    }
}

// ---------------------------------------------------------------------------
// API keys
// ---------------------------------------------------------------------------

/// What a key is allowed to touch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// Every namespace, including administrative operations.
    All,
    /// Only these namespaces.
    Only(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct ApiKey {
    pub secret: String,
    pub scope: Scope,
}

impl ApiKey {
    /// `KEY` for full access, `KEY:ns1,ns2` to scope it.
    pub fn parse(spec: &str) -> Result<Self> {
        let (secret, scope) = match spec.split_once(':') {
            None => (spec.trim(), Scope::All),
            Some((secret, list)) => {
                let names: Vec<String> = list
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if names.is_empty() {
                    bail!("api key '{spec}' lists no namespaces after ':'");
                }
                for n in &names {
                    Registry::validate(n)?;
                }
                (secret.trim(), Scope::Only(names))
            }
        };
        if secret.len() < 8 {
            bail!("api keys must be at least 8 characters");
        }
        Ok(Self {
            secret: secret.to_string(),
            scope,
        })
    }

    pub fn may_access(&self, namespace: &str) -> bool {
        match &self.scope {
            Scope::All => true,
            Scope::Only(list) => list.iter().any(|n| n == namespace),
        }
    }

    pub fn is_admin(&self) -> bool {
        self.scope == Scope::All
    }
}

/// The configured keys. Empty means authentication is disabled, which is the
/// right default for a laptop and the wrong one for a public address — the
/// server warns loudly about the latter at startup.
#[derive(Clone, Default)]
pub struct Auth {
    keys: Arc<Vec<ApiKey>>,
}

impl Auth {
    pub fn new(keys: Vec<ApiKey>) -> Self {
        Self {
            keys: Arc::new(keys),
        }
    }

    pub fn enabled(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Constant-time-ish comparison: compare every byte of every candidate so
    /// a wrong key can't be recovered by timing how fast it is rejected.
    pub fn authenticate(&self, presented: &str) -> Option<&ApiKey> {
        let mut found = None;
        for key in self.keys.iter() {
            let matches = key.secret.len() == presented.len()
                && key
                    .secret
                    .bytes()
                    .zip(presented.bytes())
                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                    == 0;
            if matches && found.is_none() {
                found = Some(key);
            }
        }
        found
    }
}

/// A bearer token, from `Authorization: Bearer <key>` or `X-API-Key`.
pub fn extract_key(headers: &axum::http::HeaderMap) -> Option<String> {
    if let Some(v) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = v.to_str() {
            if let Some(rest) = s
                .strip_prefix("Bearer ")
                .or_else(|| s.strip_prefix("bearer "))
            {
                return Some(rest.trim().to_string());
            }
        }
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
}

pub fn extract_namespace(headers: &axum::http::HeaderMap) -> String {
    headers
        .get("x-memrust-namespace")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_NAMESPACE)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_scopes_parse() {
        let full = ApiKey::parse("supersecretkey").unwrap();
        assert!(full.is_admin());
        assert!(full.may_access("anything"));

        let scoped = ApiKey::parse("supersecretkey:acme, globex").unwrap();
        assert!(!scoped.is_admin());
        assert!(scoped.may_access("acme"));
        assert!(scoped.may_access("globex"));
        assert!(!scoped.may_access("other"));

        assert!(ApiKey::parse("short").is_err());
        assert!(ApiKey::parse("supersecretkey:../etc").is_err());
    }

    #[test]
    fn namespace_names_cannot_escape_the_data_dir() {
        assert!(Registry::validate("acme").is_ok());
        assert!(Registry::validate("acme-prod_1").is_ok());
        for bad in ["..", "../etc", "a/b", "", "with space", "sym*link"] {
            assert!(Registry::validate(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn authenticate_matches_only_the_right_key() {
        let auth = Auth::new(vec![
            ApiKey::parse("alpha-key-1234").unwrap(),
            ApiKey::parse("beta-key-1234:acme").unwrap(),
        ]);
        assert!(auth.enabled());
        assert!(auth.authenticate("alpha-key-1234").unwrap().is_admin());
        assert!(!auth.authenticate("beta-key-1234").unwrap().is_admin());
        assert!(auth.authenticate("nope").is_none());
        assert!(!Auth::default().enabled());
    }
}
