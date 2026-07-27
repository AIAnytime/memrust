//! In-memory BM25 inverted index. Exact-term recall is what vector search
//! reliably misses (IDs, error codes, product names, version strings), which
//! is why hybrid retrieval beats either signal alone for agent memory.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

const K1: f32 = 1.2;
const B: f32 = 0.75;

pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    /// term -> (doc id, term frequency)
    postings: HashMap<String, Vec<(usize, u32)>>,
    doc_len: Vec<u32>,
    deleted: Vec<bool>,
    live_docs: usize,
    live_len: u64,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new()
    }
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_len: Vec::new(),
            deleted: Vec::new(),
            live_docs: 0,
            live_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.live_docs
    }

    pub fn is_empty(&self) -> bool {
        self.live_docs == 0
    }

    pub fn add(&mut self, text: &str) -> usize {
        let id = self.doc_len.len();
        let tokens = tokenize(text);
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }
        for (term, freq) in tf {
            self.postings.entry(term).or_default().push((id, freq));
        }
        self.doc_len.push(tokens.len() as u32);
        self.deleted.push(false);
        self.live_docs += 1;
        self.live_len += tokens.len() as u64;
        id
    }

    pub fn remove(&mut self, id: usize) {
        if let Some(d) = self.deleted.get_mut(id) {
            if !*d {
                *d = true;
                self.live_docs -= 1;
                self.live_len -= self.doc_len[id] as u64;
            }
        }
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        self.search_filtered(query, k, None)
    }

    /// Score only documents passing `pred` (pre-filtering at accumulation
    /// time, so a selective filter still yields a full top-k).
    pub fn search_filtered(
        &self,
        query: &str,
        k: usize,
        pred: Option<&dyn Fn(usize) -> bool>,
    ) -> Vec<(usize, f32)> {
        if self.live_docs == 0 {
            return Vec::new();
        }
        let avg_len = self.live_len as f32 / self.live_docs as f32;
        let n = self.live_docs as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in tokenize(query) {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let df = posting.iter().filter(|(d, _)| !self.deleted[*d]).count() as f32;
            if df == 0.0 {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in posting {
                if self.deleted[doc] || !pred.map(|p| p(doc)).unwrap_or(true) {
                    continue;
                }
                let tf = tf as f32;
                let len_norm = 1.0 - B + B * (self.doc_len[doc] as f32 / avg_len);
                let score = idf * (tf * (K1 + 1.0)) / (tf + K1 * len_norm);
                *scores.entry(doc).or_insert(0.0) += score;
            }
        }

        let mut out: Vec<(usize, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_exact_terms_first() {
        let mut idx = Bm25Index::new();
        let a = idx.add("the deployment failed with error E1234 on cluster prod-west");
        let _b = idx.add("we discussed pricing for the enterprise tier");
        let _c = idx.add("general notes about deployments and clusters");

        let hits = idx.search("error E1234", 3);
        assert_eq!(hits[0].0, a);
    }

    #[test]
    fn removed_docs_disappear() {
        let mut idx = Bm25Index::new();
        let a = idx.add("rust memory engine");
        idx.remove(a);
        assert!(idx.search("rust", 5).is_empty());
        assert_eq!(idx.len(), 0);
    }
}
