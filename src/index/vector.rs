//! Vector indexes over L2-normalized embeddings (cosine similarity via dot
//! product). `Hnsw` is the production index; `FlatIndex` is the exact
//! brute-force baseline used for tests and benchmarks.
//!
//! Scale features:
//! - multi-accumulator dot kernels that LLVM auto-vectorizes (NEON/AVX)
//! - optional SQ8 scalar quantization: 1 byte/dim instead of 4, with
//!   distances computed directly on the codes
//! - the whole index (graph + vectors) is serde-serializable, so engine
//!   checkpoints persist the built graph instead of re-inserting on startup

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use serde::{Deserialize, Serialize};

/// f32 wrapper with a total order so it can live in heaps.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ord32(f32);

impl Eq for Ord32 {}
impl PartialOrd for Ord32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ord32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

const LANES: usize = 8;

/// Multi-accumulator dot product. The independent accumulator lanes break
/// the serial dependency chain so LLVM turns this into SIMD fma on
/// aarch64/x86-64.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let chunks = n / LANES;
    let mut acc = [0.0f32; LANES];
    for c in 0..chunks {
        let i = c * LANES;
        for l in 0..LANES {
            acc[l] = a[i + l].mul_add(b[i + l], acc[l]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in chunks * LANES..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot of an f32 query against raw u8 codes, same lane structure.
#[inline]
fn dot_codes(q: &[f32], codes: &[u8]) -> f32 {
    let n = q.len().min(codes.len());
    let chunks = n / LANES;
    let mut acc = [0.0f32; LANES];
    for c in 0..chunks {
        let i = c * LANES;
        for l in 0..LANES {
            acc[l] = q[i + l].mul_add(codes[i + l] as f32, acc[l]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in chunks * LANES..n {
        sum += q[i] * codes[i] as f32;
    }
    sum
}

/// Cosine distance for normalized f32 vectors.
#[inline]
fn dist_f32(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot(a, b)
}

/// Scalar-quantized vector: per-vector affine map to u8 codes.
/// value[i] ≈ min + delta * codes[i]. 1 byte/dim plus 8 bytes overhead —
/// a 4x smaller working set than f32.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantVec {
    min: f32,
    delta: f32,
    codes: Vec<u8>,
}

impl QuantVec {
    pub fn encode(v: &[f32]) -> Self {
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let delta = ((max - min) / 255.0).max(f32::MIN_POSITIVE);
        let codes = v
            .iter()
            .map(|x| (((x - min) / delta).round() as i32).clamp(0, 255) as u8)
            .collect();
        Self { min, delta, codes }
    }

    pub fn decode(&self) -> Vec<f32> {
        self.codes
            .iter()
            .map(|&c| self.min + self.delta * c as f32)
            .collect()
    }
}

/// A query vector with its component sum precomputed once, needed by the
/// quantized dot identity above.
struct QueryVec<'a> {
    v: &'a [f32],
    sum: f32,
}

impl<'a> QueryVec<'a> {
    fn new(v: &'a [f32]) -> Self {
        Self {
            v,
            sum: v.iter().sum(),
        }
    }
}

/// Contiguous vector storage: one flat allocation with a fixed stride, not
/// a Vec of Vecs. Sequential candidates land in cache lines instead of
/// pointer-chasing, and the layout is mmap-ready for the segment work ahead.
#[derive(Debug, Serialize, Deserialize)]
enum VecStore {
    F32 {
        dim: usize,
        data: Vec<f32>,
    },
    Sq8 {
        dim: usize,
        mins: Vec<f32>,
        deltas: Vec<f32>,
        codes: Vec<u8>,
    },
}

impl VecStore {
    fn push(&mut self, v: Vec<f32>) {
        match self {
            VecStore::F32 { dim, data } => {
                if data.is_empty() {
                    *dim = v.len();
                }
                data.extend_from_slice(&v);
            }
            VecStore::Sq8 {
                dim,
                mins,
                deltas,
                codes,
            } => {
                if codes.is_empty() {
                    *dim = v.len();
                }
                let q = QuantVec::encode(&v);
                mins.push(q.min);
                deltas.push(q.delta);
                codes.extend_from_slice(&q.codes);
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            VecStore::F32 { dim, data } => {
                if *dim == 0 {
                    0
                } else {
                    data.len() / dim
                }
            }
            VecStore::Sq8 { mins, .. } => mins.len(),
        }
    }

    #[inline]
    fn dist(&self, q: &QueryVec, i: usize) -> f32 {
        1.0 - match self {
            VecStore::F32 { dim, data } => dot(q.v, &data[i * dim..(i + 1) * dim]),
            VecStore::Sq8 {
                dim,
                mins,
                deltas,
                codes,
            } => mins[i] * q.sum + deltas[i] * dot_codes(q.v, &codes[i * dim..(i + 1) * dim]),
        }
    }

    /// Distance between two stored vectors without decoding to f32.
    /// For SQ8: dot(a, b) = Σ (mᵃ+dᵃcᵃᵢ)(mᵇ+dᵇcᵇᵢ)
    ///        = mᵃmᵇ·dim + mᵃdᵇΣcᵇ + dᵃmᵇΣcᵃ + dᵃdᵇΣcᵃᵢcᵇᵢ,
    /// with the code-product sum accumulated in integers.
    #[inline]
    fn dist_between(&self, a: usize, b: usize) -> f32 {
        1.0 - match self {
            VecStore::F32 { dim, data } => {
                dot(&data[a * dim..(a + 1) * dim], &data[b * dim..(b + 1) * dim])
            }
            VecStore::Sq8 {
                dim,
                mins,
                deltas,
                codes,
            } => {
                let ca = &codes[a * dim..(a + 1) * dim];
                let cb = &codes[b * dim..(b + 1) * dim];
                let mut sum_a = 0u32;
                let mut sum_b = 0u32;
                let mut prod = 0u64;
                for (&x, &y) in ca.iter().zip(cb) {
                    sum_a += x as u32;
                    sum_b += y as u32;
                    prod += x as u64 * y as u64;
                }
                mins[a] * mins[b] * *dim as f32
                    + mins[a] * deltas[b] * sum_b as f32
                    + deltas[a] * mins[b] * sum_a as f32
                    + deltas[a] * deltas[b] * prod as f32
            }
        }
    }
}

pub struct FlatIndex {
    vectors: Vec<Vec<f32>>,
    deleted: Vec<bool>,
}

impl Default for FlatIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            deleted: Vec::new(),
        }
    }

    pub fn add(&mut self, v: Vec<f32>) -> usize {
        self.vectors.push(v);
        self.deleted.push(false);
        self.vectors.len() - 1
    }

    pub fn remove(&mut self, id: usize) {
        if let Some(d) = self.deleted.get_mut(id) {
            *d = true;
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.deleted[*i])
            .map(|(i, v)| (i, dist_f32(query, v)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Max links per node on upper layers.
    pub m: usize,
    /// Max links per node on layer 0 (conventionally 2*M).
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    /// Store SQ8-quantized codes instead of f32 vectors.
    pub quantize: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 100,
            quantize: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hnsw {
    cfg: HnswConfig,
    level_mult: f64,
    entry: Option<usize>,
    max_level: usize,
    store: VecStore,
    /// node -> level -> neighbor ids
    links: Vec<Vec<Vec<usize>>>,
    deleted: Vec<bool>,
    live: usize,
    rng: u64,
}

impl Hnsw {
    pub fn new(cfg: HnswConfig) -> Self {
        let level_mult = 1.0 / (cfg.m as f64).ln();
        let store = if cfg.quantize {
            VecStore::Sq8 {
                dim: 0,
                mins: Vec::new(),
                deltas: Vec::new(),
                codes: Vec::new(),
            }
        } else {
            VecStore::F32 {
                dim: 0,
                data: Vec::new(),
            }
        };
        Self {
            cfg,
            level_mult,
            entry: None,
            max_level: 0,
            store,
            links: Vec::new(),
            deleted: Vec::new(),
            live: 0,
            rng: 0x9E3779B97F4A7C15,
        }
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.store, VecStore::Sq8 { .. })
    }

    /// Neighbor selection heuristic from the HNSW paper (Algorithm 4): keep
    /// a candidate only if it is closer to the base point than to any
    /// already-kept neighbor. This preserves edges in *different directions*
    /// instead of a tight clump of mutual neighbors — naive closest-M
    /// selection is exactly what makes recall collapse as the graph grows.
    /// Skipped candidates backfill remaining slots.
    fn select_neighbors(&self, candidates_by_dist: &[(f32, usize)], m: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let mut skipped: Vec<usize> = Vec::new();
        for &(d, e) in candidates_by_dist {
            if selected.len() >= m {
                break;
            }
            let diverse = selected.iter().all(|&s| d < self.store.dist_between(e, s));
            if diverse {
                selected.push(e);
            } else {
                skipped.push(e);
            }
        }
        for e in skipped {
            if selected.len() >= m {
                break;
            }
            selected.push(e);
        }
        selected
    }

    fn next_rand(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        let bits = x.wrapping_mul(0x2545F4914F6CDD1D) >> 11;
        bits as f64 / (1u64 << 53) as f64
    }

    fn random_level(&mut self) -> usize {
        let u = self.next_rand().max(f64::MIN_POSITIVE);
        (-u.ln() * self.level_mult) as usize
    }

    /// Beam search on one layer, returning up to `ef` (dist, id) pairs.
    ///
    /// With a predicate, this is *pre-filtering*: the beam still traverses
    /// every node (excluded nodes keep the graph navigable) but only
    /// predicate-passing nodes enter the result set, so a selective filter
    /// gets a full result set instead of scraps left over after post-hoc
    /// filtering. A visit cap bounds the walk when almost nothing matches.
    fn search_layer(
        &self,
        query: &QueryVec,
        entries: &[usize],
        ef: usize,
        level: usize,
        pred: Option<&dyn Fn(usize) -> bool>,
        max_visits: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = HashSet::new();
        // min-heap of candidates to expand
        let mut candidates: BinaryHeap<Reverse<(Ord32, usize)>> = BinaryHeap::new();
        // max-heap of current best results (worst on top)
        let mut results: BinaryHeap<(Ord32, usize)> = BinaryHeap::new();
        let passes = |i: usize| pred.map(|p| p(i)).unwrap_or(true);

        for &e in entries {
            if visited.insert(e) {
                let d = self.store.dist(query, e);
                candidates.push(Reverse((Ord32(d), e)));
                if passes(e) {
                    results.push((Ord32(d), e));
                }
            }
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse((Ord32(d), node))) = candidates.pop() {
            let worst = results.peek().map(|(Ord32(w), _)| *w).unwrap_or(f32::MAX);
            if d > worst && results.len() >= ef {
                break;
            }
            if visited.len() > max_visits {
                break;
            }
            for &nb in &self.links[node][level] {
                if !visited.insert(nb) {
                    continue;
                }
                let dn = self.store.dist(query, nb);
                let worst = results.peek().map(|(Ord32(w), _)| *w).unwrap_or(f32::MAX);
                if results.len() < ef || dn < worst {
                    candidates.push(Reverse((Ord32(dn), nb)));
                    if passes(nb) {
                        results.push((Ord32(dn), nb));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, usize)> = results.into_iter().map(|(Ord32(d), i)| (d, i)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    /// Greedy descent to the closest node on a layer (ef = 1).
    fn greedy(&self, query: &QueryVec, start: usize, level: usize) -> usize {
        let mut cur = start;
        let mut cur_d = self.store.dist(query, cur);
        loop {
            let mut improved = false;
            for &nb in &self.links[cur][level] {
                let d = self.store.dist(query, nb);
                if d < cur_d {
                    cur = nb;
                    cur_d = d;
                    improved = true;
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    pub fn add(&mut self, v: Vec<f32>) -> usize {
        let id = self.store.len();
        let level = self.random_level();
        let query_copy = v.clone();
        self.store.push(v);
        self.links.push(vec![Vec::new(); level + 1]);
        self.deleted.push(false);
        self.live += 1;

        let Some(entry) = self.entry else {
            self.entry = Some(id);
            self.max_level = level;
            return id;
        };

        let query = QueryVec::new(&query_copy);
        let mut ep = entry;

        // Descend through layers above the new node's level.
        let mut l = self.max_level;
        while l > level {
            ep = self.greedy(&query, ep, l);
            l -= 1;
        }

        // Connect on each layer from min(level, max_level) down to 0.
        let top = level.min(self.max_level);
        for lvl in (0..=top).rev() {
            let found = self.search_layer(
                &query,
                &[ep],
                self.cfg.ef_construction,
                lvl,
                None,
                usize::MAX,
            );
            let max_links = if lvl == 0 { self.cfg.m0 } else { self.cfg.m };
            let candidates: Vec<(f32, usize)> =
                found.iter().filter(|(_, n)| *n != id).copied().collect();
            let selected = self.select_neighbors(&candidates, max_links);

            self.links[id][lvl] = selected.clone();
            for nb in selected {
                self.links[nb][lvl].push(id);
                if self.links[nb][lvl].len() > max_links {
                    // Re-select nb's neighbors with the same diversity
                    // heuristic, from nb's point of view.
                    let mut cand: Vec<(f32, usize)> = self.links[nb][lvl]
                        .iter()
                        .map(|&x| (self.store.dist_between(nb, x), x))
                        .collect();
                    cand.sort_by(|a, b| a.0.total_cmp(&b.0));
                    self.links[nb][lvl] = self.select_neighbors(&cand, max_links);
                }
            }
            if let Some((_, best)) = found.first() {
                ep = *best;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }
        id
    }

    /// Tombstone a node. It stays in the graph for connectivity but is
    /// filtered from results; compaction rebuilds the index without it.
    pub fn remove(&mut self, id: usize) {
        if let Some(d) = self.deleted.get_mut(id) {
            if !*d {
                *d = true;
                self.live -= 1;
            }
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_filtered(query, k, None)
    }

    /// Search restricted to nodes passing `pred` (pre-filtering; see
    /// `search_layer`). Tombstones are always excluded.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        pred: Option<&dyn Fn(usize) -> bool>,
    ) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let q = QueryVec::new(query);
        let mut ep = entry;
        for l in (1..=self.max_level).rev() {
            ep = self.greedy(&q, ep, l);
        }
        // Over-fetch so tombstones don't starve the result set.
        let ef = self.cfg.ef_search.max(k * 2);
        // A selective filter can otherwise walk the whole graph; only an
        // externally filtered search gets the visit cap.
        let max_visits = if pred.is_some() { ef * 32 } else { usize::MAX };
        let live_pred = |i: usize| !self.deleted[i] && pred.map(|p| p(i)).unwrap_or(true);
        let found = self.search_layer(&q, &[ep], ef, 0, Some(&live_pred), max_visits);
        found.into_iter().take(k).map(|(d, i)| (i, d)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(seed: &mut u64, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push(((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5);
        }
        normalize(&mut v);
        v
    }

    fn recall_vs_flat(cfg: HnswConfig, n: usize, dim: usize, queries: usize) -> f64 {
        let mut seed = 42u64;
        let mut hnsw = Hnsw::new(cfg);
        let mut flat = FlatIndex::new();
        for _ in 0..n {
            let v = rand_vec(&mut seed, dim);
            hnsw.add(v.clone());
            flat.add(v);
        }
        let mut hits = 0usize;
        let mut total = 0usize;
        for _ in 0..queries {
            let q = rand_vec(&mut seed, dim);
            let truth: HashSet<usize> = flat.search(&q, 10).into_iter().map(|(i, _)| i).collect();
            let approx = hnsw.search(&q, 10);
            hits += approx.iter().filter(|(i, _)| truth.contains(i)).count();
            total += truth.len();
        }
        hits as f64 / total as f64
    }

    #[test]
    fn hnsw_recall_matches_flat() {
        let recall = recall_vs_flat(HnswConfig::default(), 2000, 64, 50);
        assert!(recall > 0.9, "HNSW recall@10 too low: {recall}");
    }

    #[test]
    fn quantized_hnsw_keeps_most_recall() {
        let cfg = HnswConfig {
            quantize: true,
            ..HnswConfig::default()
        };
        let recall = recall_vs_flat(cfg, 2000, 64, 50);
        assert!(recall > 0.85, "SQ8 HNSW recall@10 too low: {recall}");
    }

    #[test]
    fn quantization_roundtrip_error_is_small() {
        let mut seed = 3u64;
        let v = rand_vec(&mut seed, 128);
        let q = QuantVec::encode(&v);
        let back = q.decode();
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() < 2.0 / 255.0, "{a} vs {b}");
        }
        // The algebraic quantized dot (used by VecStore::dist) matches dot
        // against the decode.
        let mut seed2 = 9u64;
        let query = rand_vec(&mut seed2, 128);
        let mut store = VecStore::Sq8 {
            dim: 0,
            mins: Vec::new(),
            deltas: Vec::new(),
            codes: Vec::new(),
        };
        store.push(v.clone());
        let qv = QueryVec::new(&query);
        assert!((store.dist(&qv, 0) - (1.0 - dot(&query, &back))).abs() < 1e-3);
    }

    #[test]
    fn serialized_index_searches_identically() {
        let mut seed = 11u64;
        let mut hnsw = Hnsw::new(HnswConfig::default());
        for _ in 0..300 {
            hnsw.add(rand_vec(&mut seed, 32));
        }
        let json = serde_json::to_string(&hnsw).unwrap();
        let restored: Hnsw = serde_json::from_str(&json).unwrap();
        let q = rand_vec(&mut seed, 32);
        assert_eq!(hnsw.search(&q, 10), restored.search(&q, 10));
    }

    #[test]
    fn tombstones_are_filtered() {
        let mut seed = 7u64;
        let mut hnsw = Hnsw::new(HnswConfig::default());
        let mut ids = Vec::new();
        for _ in 0..100 {
            ids.push(hnsw.add(rand_vec(&mut seed, 16)));
        }
        let q = rand_vec(&mut seed, 16);
        let top = hnsw.search(&q, 5);
        let victim = top[0].0;
        hnsw.remove(victim);
        assert!(hnsw.search(&q, 5).iter().all(|(i, _)| *i != victim));
        assert_eq!(hnsw.len(), 99);
    }
}
