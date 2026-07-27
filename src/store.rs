use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::MemoryRecord;

/// Write-ahead log. Every mutation is one JSON line, fsynced on append.
///
/// Recovery is checkpoint + tail: a checkpoint persists the full derived
/// state (records and both built indexes) as one binary file and truncates
/// the WAL, so open() deserializes the checkpoint and replays only the ops
/// that landed after it — no full-history replay, no index rebuild.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    Remember { record: Box<MemoryRecord> },
    Forget { id: Uuid },
}

pub struct Wal {
    path: PathBuf,
    file: File,
    appends: usize,
}

impl Wal {
    pub fn open(dir: &Path) -> Result<(Self, Vec<WalOp>)> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating data dir {}", dir.display()))?;
        let path = dir.join("memory.wal");

        let mut ops = Vec::new();
        if path.exists() {
            let reader = BufReader::new(File::open(&path)?);
            for (i, line) in reader.lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<WalOp>(&line) {
                    Ok(op) => ops.push(op),
                    // A torn final line from a crash is expected; anything
                    // mid-file is corruption worth surfacing.
                    Err(e) => {
                        eprintln!("memrust: skipping unreadable WAL line {}: {e}", i + 1);
                    }
                }
            }
        }

        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok((
            Self {
                path,
                file,
                appends: 0,
            },
            ops,
        ))
    }

    pub fn append(&mut self, op: &WalOp) -> Result<()> {
        let mut line = serde_json::to_string(op)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        self.file.sync_data()?;
        self.appends += 1;
        Ok(())
    }

    /// Ops appended since open or the last truncation — i.e. the size of the
    /// tail a restart would have to replay.
    pub fn appends_since_checkpoint(&self) -> usize {
        self.appends
    }

    /// Empty the log. Called after a checkpoint has durably captured
    /// everything the log contained.
    pub fn truncate(&mut self) -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        file.sync_all()?;
        self.file = OpenOptions::new().append(true).open(&self.path)?;
        self.appends = 0;
        Ok(())
    }
}

// JSON rather than a packed binary format: MemoryRecord's serde shape uses
// skipped optional fields and free-form JSON metadata, which only round-trip
// through a self-describing format. The costly thing a checkpoint avoids is
// HNSW graph *construction*; parsing is cheap next to that. A binary vector
// section is a later optimization.
const CHECKPOINT_FILE: &str = "checkpoint.json";

/// Atomically persist a checkpoint (write temp, fsync, rename).
pub fn save_checkpoint<T: Serialize>(dir: &Path, state: &T) -> Result<()> {
    let path = dir.join(CHECKPOINT_FILE);
    let tmp = dir.join("checkpoint.json.tmp");
    {
        let mut out = BufWriter::new(File::create(&tmp)?);
        serde_json::to_writer(&mut out, state).context("serializing checkpoint")?;
        out.flush()?;
        out.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

pub fn load_checkpoint<T: DeserializeOwned>(dir: &Path) -> Result<Option<T>> {
    let path = dir.join(CHECKPOINT_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let reader = BufReader::new(File::open(&path)?);
    let state = serde_json::from_reader(reader)
        .with_context(|| format!("reading checkpoint {}", path.display()))?;
    Ok(Some(state))
}
