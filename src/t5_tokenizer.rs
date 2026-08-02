//! SentencePiece Unigram tokenizer for the UMT5-XXL text encoder used by Wan2.1.
//!
//! This is not the BPE tokenizer used by the LLM path or the CLIP tokenizer used by SD. Unigram
//! picks the segmentation that maximises the summed log-probability of its pieces, which is a
//! Viterbi pass over the character lattice rather than a greedy merge sequence.
//!
//! Worth seeing why that distinction matters concretely. For `"a red fox"` the vocabulary has no
//! `▁fox` piece at all, so the model segments it as `▁`(-3.197) + `fox`(-13.09) — a greedy
//! longest-match tokenizer would instead produce `▁f` + `ox` and hand the encoder a different
//! sequence entirely.

use crate::gguf::{GgufFile, GgufValue};
use std::collections::HashMap;

/// SentencePiece's space marker (U+2581 LOWER ONE EIGHTH BLOCK).
const SPACE: char = '\u{2581}';

pub struct T5Tokenizer {
    pieces: HashMap<String, (u32, f32)>,
    /// Longest piece in bytes, so the Viterbi inner loop can stop early.
    max_piece_len: usize,
    pub eos_id: u32,
    pub pad_id: u32,
    pub unk_id: u32,
    pub add_eos: bool,
    pub add_space_prefix: bool,
}

impl T5Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let tokens: Vec<String> = match gguf.metadata.get("tokenizer.ggml.tokens")? {
            GgufValue::Array(a) => a.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect(),
            _ => return None,
        };
        let scores: Vec<f32> = match gguf.metadata.get("tokenizer.ggml.scores")? {
            GgufValue::Array(a) => a
                .iter()
                .map(|v| if let GgufValue::Float32(f) = v { *f } else { 0.0 })
                .collect(),
            _ => return None,
        };
        if tokens.len() != scores.len() {
            return None;
        }

        let mut pieces = HashMap::with_capacity(tokens.len());
        let mut max_piece_len = 1;
        for (i, tok) in tokens.iter().enumerate() {
            // Later duplicates must not displace earlier ids; SentencePiece treats the first
            // occurrence as canonical.
            pieces.entry(tok.clone()).or_insert((i as u32, scores[i]));
            max_piece_len = max_piece_len.max(tok.len());
        }

        let meta_u32 = |k: &str, d: u32| gguf.metadata.get(k).and_then(|v| v.as_u32()).unwrap_or(d);
        let meta_bool = |k: &str, d: bool| match gguf.metadata.get(k) {
            Some(GgufValue::Bool(b)) => *b,
            _ => d,
        };

        Some(Self {
            pieces,
            max_piece_len,
            eos_id: meta_u32("tokenizer.ggml.eos_token_id", 1),
            pad_id: meta_u32("tokenizer.ggml.padding_token_id", 0),
            unk_id: meta_u32("tokenizer.ggml.unknown_token_id", 2),
            add_eos: meta_bool("tokenizer.ggml.add_eos_token", true),
            add_space_prefix: meta_bool("tokenizer.ggml.add_space_prefix", true),
        })
    }

    /// Apply SentencePiece normalisation: spaces become the visible marker, and a leading marker
    /// is added so a word at the start of the string tokenises the same as one mid-sentence.
    fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + 3);
        if self.add_space_prefix && !text.starts_with(' ') {
            out.push(SPACE);
        }
        for ch in text.chars() {
            out.push(if ch == ' ' { SPACE } else { ch });
        }
        out
    }

    /// Viterbi over the lattice: `best[i]` is the score of the best segmentation of the first `i`
    /// bytes. Only char boundaries are considered, so multi-byte UTF-8 is never split mid-scalar.
    fn viterbi(&self, text: &str) -> Vec<u32> {
        let n = text.len();
        if n == 0 {
            return Vec::new();
        }
        let mut boundary = vec![false; n + 1];
        for (i, _) in text.char_indices() {
            boundary[i] = true;
        }
        boundary[n] = true;

        const NEG_INF: f32 = f32::NEG_INFINITY;
        let mut best = vec![NEG_INF; n + 1];
        let mut back: Vec<(usize, u32)> = vec![(0, 0); n + 1];
        best[0] = 0.0;

        for end in 1..=n {
            if !boundary[end] {
                continue;
            }
            let lo = end.saturating_sub(self.max_piece_len);
            for start in lo..end {
                if !boundary[start] || best[start] == NEG_INF {
                    continue;
                }
                if let Some(&(id, score)) = self.pieces.get(&text[start..end]) {
                    let cand = best[start] + score;
                    if cand > best[end] {
                        best[end] = cand;
                        back[end] = (start, id);
                    }
                }
            }
            // Nothing in the vocabulary covers this position: emit the single character as unk so
            // one unmappable scalar cannot fail the whole prompt.
            if best[end] == NEG_INF {
                let prev = text[..end].char_indices().next_back().map(|(i, _)| i).unwrap_or(0);
                if best[prev] != NEG_INF {
                    best[end] = best[prev] - 100.0;
                    back[end] = (prev, self.unk_id);
                }
            }
        }

        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (prev, id) = back[pos];
            ids.push(id);
            if prev == pos {
                break; // defensive: never loop on a malformed lattice
            }
            pos = prev;
        }
        ids.reverse();
        ids
    }

    /// Tokenize without padding. Appends EOS when the model asks for it.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = self.viterbi(&self.normalize(text));
        if self.add_eos {
            ids.push(self.eos_id);
        }
        ids
    }

    /// Tokenize and pad (or truncate) to exactly `len`, which is what the encoder expects — Wan
    /// runs the encoder over the full 512-token context regardless of the real prompt length.
    pub fn encode_padded(&self, text: &str, len: usize) -> Vec<u32> {
        let mut ids = self.encode(text);
        if ids.len() > len {
            ids.truncate(len);
            if self.add_eos {
                // Truncation must not silently drop the end-of-sequence marker.
                *ids.last_mut().unwrap() = self.eos_id;
            }
        }
        ids.resize(len, self.pad_id);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const PACK: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack";

    fn load() -> Option<T5Tokenizer> {
        let p = Path::new(PACK).join("umt5-xxl-encoder-Q4_K_M.gguf");
        if !p.exists() {
            eprintln!("skipping: {p:?} not present");
            return None;
        }
        T5Tokenizer::from_gguf(&GgufFile::open(&p).ok()?)
    }

    #[test]
    fn matches_reference_tokens_for_a_red_fox() {
        let Some(tk) = load() else { return };
        // Dumped from the shipped stable-diffusion.cpp engine via SAIENT_DUMP=1.
        // '▁fox' is absent from the vocabulary, so the correct segmentation is '▁' + 'fox'.
        assert_eq!(tk.encode("a red fox"), vec![289, 4062, 273, 56209, 1]);
    }

    #[test]
    fn pads_to_the_full_encoder_context() {
        let Some(tk) = load() else { return };
        let ids = tk.encode_padded("a red fox", 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(&ids[..5], &[289, 4062, 273, 56209, 1]);
        assert!(ids[5..].iter().all(|&i| i == tk.pad_id), "tail must be pad");
    }

    #[test]
    fn empty_prompt_matches_reference() {
        let Some(tk) = load() else { return };
        // This is what the unconditional branch encodes, so it has to be exactly right.
        // Reference dump is `273, 1, 0, 0, ...`: the space marker is prepended even when there is
        // no text, so an empty prompt is '▁' + '</s>' rather than a bare '</s>'. Asserted against
        // reference/t5_ids_empty.txt, not against intuition — intuition had this wrong.
        let ids = tk.encode_padded("", 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(&ids[..2], &[273, 1], "empty prompt must be marker + EOS");
        assert!(ids[2..].iter().all(|&i| i == tk.pad_id), "tail must be pad");
    }

    #[test]
    fn metadata_matches_the_model_card() {
        let Some(tk) = load() else { return };
        assert_eq!(tk.eos_id, 1);
        assert_eq!(tk.pad_id, 0);
        assert!(tk.add_eos, "UMT5 appends </s>");
        assert!(tk.add_space_prefix, "UMT5 prepends the space marker");
    }

    #[test]
    fn truncation_keeps_the_eos_marker() {
        let Some(tk) = load() else { return };
        let long = "a red fox ".repeat(200);
        let ids = tk.encode_padded(&long, 512);
        assert_eq!(ids.len(), 512);
        assert_eq!(*ids.last().unwrap(), tk.eos_id, "EOS must survive truncation");
    }

    #[test]
    fn viterbi_beats_greedy_longest_match() {
        let Some(tk) = load() else { return };
        // Greedy would take '▁f' (960) then 'ox'; unigram scoring prefers '▁' + 'fox'.
        let ids = tk.encode("fox");
        assert!(ids.contains(&56209), "expected the 'fox' piece, got {ids:?}");
        assert!(!ids.contains(&960), "greedy '▁f' split should not win, got {ids:?}");
    }
}
