//! Prometheus metrics and structured request logs.
//!
//! Hand-rolled rather than pulling in a metrics crate: the exposition format
//! is a few lines of text, and memrust's dependency list is a feature.
//!
//! Counters and latency histograms are recorded by middleware as requests
//! finish. Engine state (memory counts, index sizes, WAL depth) is *not*
//! mirrored into counters — it is read from the live registry at scrape time,
//! so it can never drift from the truth.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Instant;

use crate::server::tenancy::Registry;

/// Latency buckets in seconds, spanning "in-memory hit" to "something is
/// wrong". Cumulative, as Prometheus histograms require.
const BUCKETS: [f64; 12] = [
    0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0,
];

#[derive(Default)]
struct Histogram {
    counts: [u64; BUCKETS.len()],
    sum: f64,
    total: u64,
}

impl Histogram {
    fn record(&mut self, seconds: f64) {
        self.sum += seconds;
        self.total += 1;
        for (i, edge) in BUCKETS.iter().enumerate() {
            if seconds <= *edge {
                self.counts[i] += 1;
            }
        }
    }
}

pub struct Metrics {
    started: Instant,
    /// (route, status) -> count
    requests: Mutex<HashMap<(String, u16), u64>>,
    /// route -> latency histogram
    latency: Mutex<HashMap<String, Histogram>>,
    unauthorized: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            requests: Mutex::new(HashMap::new()),
            latency: Mutex::new(HashMap::new()),
            unauthorized: AtomicU64::new(0),
        }
    }

    pub fn record(&self, route: &str, status: u16, seconds: f64) {
        *self
            .requests
            .lock()
            .expect("metrics lock")
            .entry((route.to_string(), status))
            .or_insert(0) += 1;
        self.latency
            .lock()
            .expect("metrics lock")
            .entry(route.to_string())
            .or_default()
            .record(seconds);
        if status == 401 || status == 403 {
            self.unauthorized.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Prometheus text exposition. Engine gauges are read live from the
    /// registry so a scrape always reflects current state.
    pub fn render(&self, registry: &Registry) -> String {
        let mut out = String::with_capacity(4096);

        out.push_str("# HELP memrust_uptime_seconds Seconds since the server started.\n");
        out.push_str("# TYPE memrust_uptime_seconds gauge\n");
        out.push_str(&format!(
            "memrust_uptime_seconds {:.3}\n",
            self.started.elapsed().as_secs_f64()
        ));

        out.push_str("# HELP memrust_build_info Build metadata; the value is always 1.\n");
        out.push_str("# TYPE memrust_build_info gauge\n");
        out.push_str(&format!(
            "memrust_build_info{{version=\"{}\"}} 1\n",
            env!("CARGO_PKG_VERSION")
        ));

        // ---- per-namespace engine state, read live ----
        let namespaces = registry.open_namespaces();
        out.push_str("# HELP memrust_namespaces Namespaces currently open.\n");
        out.push_str("# TYPE memrust_namespaces gauge\n");
        out.push_str(&format!("memrust_namespaces {}\n", namespaces.len()));

        type Gauge = (&'static str, &'static str, Vec<(String, u64)>);
        let mut gauges: Vec<Gauge> = Vec::new();
        let mut memories = Vec::new();
        let mut vectors = Vec::new();
        let mut lexical = Vec::new();
        let mut entities = Vec::new();
        let mut wal = Vec::new();
        for (name, ns) in &namespaces {
            let stats = ns.engine.read().expect("engine lock").stats();
            memories.push((name.clone(), stats.total_memories as u64));
            vectors.push((name.clone(), stats.vector_indexed as u64));
            lexical.push((name.clone(), stats.lexical_indexed as u64));
            entities.push((name.clone(), stats.entities as u64));
            wal.push((name.clone(), stats.wal_tail_ops as u64));
        }
        gauges.push(("memrust_memories", "Live memories.", memories));
        gauges.push((
            "memrust_vector_indexed",
            "Memories in the vector index.",
            vectors,
        ));
        gauges.push((
            "memrust_lexical_indexed",
            "Memories in the lexical index.",
            lexical,
        ));
        gauges.push((
            "memrust_graph_entities",
            "Distinct entities in the graph index.",
            entities,
        ));
        gauges.push((
            "memrust_wal_tail_ops",
            "WAL operations a restart would replay.",
            wal,
        ));
        for (metric, help, series) in gauges {
            out.push_str(&format!("# HELP {metric} {help}\n# TYPE {metric} gauge\n"));
            for (ns, value) in series {
                out.push_str(&format!(
                    "{metric}{{namespace=\"{}\"}} {value}\n",
                    escape(&ns)
                ));
            }
        }

        // ---- request counters ----
        out.push_str("# HELP memrust_requests_total HTTP requests by route and status.\n");
        out.push_str("# TYPE memrust_requests_total counter\n");
        let requests = self.requests.lock().expect("metrics lock");
        let mut rows: Vec<_> = requests.iter().collect();
        rows.sort();
        for ((route, status), count) in rows {
            out.push_str(&format!(
                "memrust_requests_total{{route=\"{}\",status=\"{status}\"}} {count}\n",
                escape(route)
            ));
        }
        drop(requests);

        out.push_str(
            "# HELP memrust_requests_rejected_total Requests refused by authentication or scope.\n",
        );
        out.push_str("# TYPE memrust_requests_rejected_total counter\n");
        out.push_str(&format!(
            "memrust_requests_rejected_total {}\n",
            self.unauthorized.load(Ordering::Relaxed)
        ));

        // ---- latency ----
        out.push_str("# HELP memrust_request_duration_seconds Request latency by route.\n");
        out.push_str("# TYPE memrust_request_duration_seconds histogram\n");
        let latency = self.latency.lock().expect("metrics lock");
        let mut routes: Vec<_> = latency.iter().collect();
        routes.sort_by_key(|(r, _)| r.as_str());
        for (route, hist) in routes {
            let r = escape(route);
            for (i, edge) in BUCKETS.iter().enumerate() {
                out.push_str(&format!(
                    "memrust_request_duration_seconds_bucket{{route=\"{r}\",le=\"{edge}\"}} {}\n",
                    hist.counts[i]
                ));
            }
            out.push_str(&format!(
                "memrust_request_duration_seconds_bucket{{route=\"{r}\",le=\"+Inf\"}} {}\n",
                hist.total
            ));
            out.push_str(&format!(
                "memrust_request_duration_seconds_sum{{route=\"{r}\"}} {:.6}\n",
                hist.sum
            ));
            out.push_str(&format!(
                "memrust_request_duration_seconds_count{{route=\"{r}\"}} {}\n",
                hist.total
            ));
        }
        out
    }
}

/// Label values are quoted strings; backslash, quote and newline must escape.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "")
}

// ---------------------------------------------------------------------------
// structured logs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// Human-readable, one line per request.
    #[default]
    Text,
    /// One JSON object per line, for log shippers.
    Json,
    /// Request logging off.
    Off,
}

impl LogFormat {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s {
            "text" => Ok(LogFormat::Text),
            "json" => Ok(LogFormat::Json),
            "off" | "none" => Ok(LogFormat::Off),
            other => anyhow::bail!("unknown log format '{other}' (expected text, json or off)"),
        }
    }
}

static LOG_FORMAT: RwLock<LogFormat> = RwLock::new(LogFormat::Text);

pub fn set_log_format(format: LogFormat) {
    *LOG_FORMAT.write().expect("log format") = format;
}

pub fn log_request(method: &str, route: &str, namespace: &str, status: u16, ms: f64) {
    let format = *LOG_FORMAT.read().expect("log format");
    match format {
        LogFormat::Off => {}
        LogFormat::Text => {
            eprintln!("{method} {route} ns={namespace} {status} {ms:.1}ms");
        }
        LogFormat::Json => {
            let line = serde_json::json!({
                "ts": crate::types::now_ms(),
                "level": if status >= 500 { "error" } else if status >= 400 { "warn" } else { "info" },
                "method": method,
                "route": route,
                "namespace": namespace,
                "status": status,
                "duration_ms": (ms * 1000.0).round() / 1000.0,
            });
            eprintln!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative() {
        let mut h = Histogram::default();
        h.record(0.0004); // below every edge
        h.record(0.03); // above 0.025
        assert_eq!(h.total, 2);
        assert_eq!(h.counts[0], 1, "smallest bucket holds only the fast one");
        assert_eq!(
            h.counts[BUCKETS.len() - 1],
            2,
            "largest bucket holds both, cumulative"
        );
        assert!((h.sum - 0.0304).abs() < 1e-9);
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape(r#"a"b\c"#), r#"a\"b\\c"#);
        assert_eq!(escape("a\nb"), "ab");
    }

    #[test]
    fn log_format_parses() {
        assert_eq!(LogFormat::parse("json").unwrap(), LogFormat::Json);
        assert_eq!(LogFormat::parse("off").unwrap(), LogFormat::Off);
        assert!(LogFormat::parse("yaml").is_err());
    }
}
