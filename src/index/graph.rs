//! Entity graph: the third retrieval leg. Vectors find *similar* memories,
//! BM25 finds *matching* memories, the graph finds *related* ones — records
//! that share entities with the query, or whose entities co-occur with the
//! query's entities elsewhere in memory (1-hop expansion). No external graph
//! database: agent memory needs a lightweight in-engine store, not Neo4j.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Sentence-position-independent words that look like names but aren't.
const STOPWORDS: &[&str] = &[
    "the",
    "a",
    "an",
    "i",
    "we",
    "you",
    "it",
    "this",
    "that",
    "these",
    "those",
    "my",
    "our",
    "your",
    "their",
    "his",
    "her",
    "its",
    "they",
    "he",
    "she",
    "if",
    "when",
    "then",
    "also",
    "but",
    "and",
    "or",
    "not",
    "no",
    "yes",
    "today",
    "yesterday",
    "tomorrow",
];

fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w.to_lowercase().as_str())
}

/// Heuristic entity extraction: capitalized name runs ("Project Phoenix",
/// "Dana Whitfield"), identifiers with digits ("INC-90312", "E1234"),
/// short all-caps terms ("API", "GPU"), plus the record's tags. Offline and
/// deterministic; an LLM extractor can slot in later for higher precision.
pub fn extract_entities(text: &str, tags: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<String> = Vec::new();
    let mut push = |e: String| {
        let key = e.to_lowercase();
        if key.len() >= 2 && seen.insert(key.clone()) && out.len() < 16 {
            out.push(key);
        }
    };

    for tag in tags {
        push(tag.clone());
    }

    let raw: Vec<&str> = text.split_whitespace().collect();
    let words: Vec<&str> = raw
        .iter()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '-'))
        .collect();

    let is_name_word = |w: &str| -> bool {
        let mut chars = w.chars();
        match chars.next() {
            Some(c) if c.is_uppercase() => chars.all(|c| c.is_lowercase() || c.is_numeric()),
            _ => false,
        }
    };

    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if w.is_empty() {
            i += 1;
            continue;
        }
        // Identifiers: mixed letters+digits, optionally dashed (INC-90312, E1234).
        let has_digit = w.chars().any(|c| c.is_numeric());
        let has_alpha = w.chars().any(|c| c.is_alphabetic());
        if has_digit && has_alpha && w.len() >= 3 {
            push(w.to_string());
            i += 1;
            continue;
        }
        // Short all-caps terms (API, GPU, TLS).
        if w.len() >= 2 && w.len() <= 6 && w.chars().all(|c| c.is_uppercase()) && !has_digit {
            push(w.to_string());
            i += 1;
            continue;
        }
        // Capitalized runs. Multi-word runs are always names; a single
        // capitalized word only counts mid-sentence (not sentence-initial).
        if is_name_word(w) && !is_stopword(w) {
            let start = i;
            let mut end = i + 1;
            while end < words.len() && is_name_word(words[end]) && !is_stopword(words[end]) {
                end += 1;
            }
            let sentence_initial =
                start == 0 || raw[start - 1].ends_with(['.', '!', '?', ':', ';']);
            let run = words[start..end].join(" ");
            if end - start >= 2 || !sentence_initial {
                push(run);
            }
            i = end;
            continue;
        }
        i += 1;
    }
    out
}

#[derive(Default, Serialize, Deserialize)]
pub struct GraphIndex {
    /// entity -> ids of records mentioning it
    entity_records: HashMap<String, Vec<Uuid>>,
    /// entity -> co-occurring entity -> co-occurrence count
    edges: HashMap<String, HashMap<String, u32>>,
}

impl GraphIndex {
    pub fn is_empty(&self) -> bool {
        self.entity_records.is_empty()
    }

    pub fn entity_count(&self) -> usize {
        self.entity_records.len()
    }

    pub fn add(&mut self, id: Uuid, entities: &[String]) {
        for e in entities {
            self.entity_records.entry(e.clone()).or_default().push(id);
        }
        for (i, a) in entities.iter().enumerate() {
            for b in &entities[i + 1..] {
                if a != b {
                    *self
                        .edges
                        .entry(a.clone())
                        .or_default()
                        .entry(b.clone())
                        .or_insert(0) += 1;
                    *self
                        .edges
                        .entry(b.clone())
                        .or_default()
                        .entry(a.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    pub fn remove(&mut self, id: Uuid, entities: &[String]) {
        for e in entities {
            if let Some(recs) = self.entity_records.get_mut(e) {
                recs.retain(|r| *r != id);
                if recs.is_empty() {
                    self.entity_records.remove(e);
                }
            }
        }
        for (i, a) in entities.iter().enumerate() {
            for b in &entities[i + 1..] {
                if a == b {
                    continue;
                }
                for (x, y) in [(a, b), (b, a)] {
                    if let Some(nbrs) = self.edges.get_mut(x) {
                        if let Some(cnt) = nbrs.get_mut(y) {
                            *cnt = cnt.saturating_sub(1);
                            if *cnt == 0 {
                                nbrs.remove(y);
                            }
                        }
                        if nbrs.is_empty() {
                            self.edges.remove(x);
                        }
                    }
                }
            }
        }
    }

    /// Records related to the query entities: direct mentions weighted by
    /// entity rarity (common entities carry less signal), plus 1-hop
    /// expansion through the strongest co-occurring entities at reduced
    /// weight. Returns (id, relatedness) sorted descending.
    pub fn related(&self, query_entities: &[String], k: usize) -> Vec<(Uuid, f32)> {
        let mut scores: HashMap<Uuid, f32> = HashMap::new();
        let rarity = |e: &str| -> f32 {
            let df = self.entity_records.get(e).map(|r| r.len()).unwrap_or(0) as f32;
            1.0 / (1.0 + df.ln_1p())
        };

        for e in query_entities {
            let e = e.to_lowercase();
            if let Some(recs) = self.entity_records.get(&e) {
                let w = rarity(&e);
                for id in recs {
                    *scores.entry(*id).or_insert(0.0) += w;
                }
            }
            // 1-hop: entities that co-occur with a query entity somewhere in
            // memory pull in their records at reduced weight.
            if let Some(nbrs) = self.edges.get(&e) {
                let mut top: Vec<(&String, &u32)> = nbrs.iter().collect();
                top.sort_by(|a, b| b.1.cmp(a.1));
                for (nb, count) in top.into_iter().take(5) {
                    let w = 0.3 * rarity(nb) * (*count as f32).ln_1p();
                    if let Some(recs) = self.entity_records.get(nb) {
                        for id in recs {
                            *scores.entry(*id).or_insert(0.0) += w;
                        }
                    }
                }
            }
        }

        let mut out: Vec<(Uuid, f32)> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_names_identifiers_and_tags() {
        let ents = extract_entities(
            "Dana Whitfield filed INC-90312 against the billing API for Project Phoenix",
            &["oncall".into()],
        );
        assert!(ents.contains(&"dana whitfield".to_string()), "{ents:?}");
        assert!(ents.contains(&"inc-90312".to_string()), "{ents:?}");
        assert!(ents.contains(&"api".to_string()), "{ents:?}");
        assert!(ents.contains(&"project phoenix".to_string()), "{ents:?}");
        assert!(ents.contains(&"oncall".to_string()), "{ents:?}");
    }

    #[test]
    fn sentence_initial_single_words_are_skipped() {
        let ents = extract_entities("Deploys are failing again", &[]);
        assert!(!ents.contains(&"deploys".to_string()), "{ents:?}");
    }

    #[test]
    fn one_hop_expansion_finds_related_records() {
        let mut g = GraphIndex::default();
        let a = Uuid::new_v4(); // mentions phoenix + billing
        let b = Uuid::new_v4(); // mentions billing only
        let c = Uuid::new_v4(); // unrelated
        g.add(a, &["project phoenix".into(), "billing".into()]);
        g.add(b, &["billing".into()]);
        g.add(c, &["kubernetes".into()]);

        let related = g.related(&["project phoenix".into()], 10);
        let ids: Vec<Uuid> = related.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids[0], a, "direct mention ranks first");
        assert!(ids.contains(&b), "1-hop through billing should reach b");
        assert!(!ids.contains(&c));
    }

    #[test]
    fn remove_undoes_add() {
        let mut g = GraphIndex::default();
        let a = Uuid::new_v4();
        let ents: Vec<String> = vec!["alpha".into(), "beta".into()];
        g.add(a, &ents);
        g.remove(a, &ents);
        assert!(g.is_empty());
        assert!(g.related(&["alpha".into()], 5).is_empty());
    }
}
