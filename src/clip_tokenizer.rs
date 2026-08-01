//! CLIP byte-pair tokenizer used by Stable Diffusion 1.x.
//!
//! This is independent of Python/Transformers at runtime. The model pack carries
//! only the vocabulary and merge tables, and Quartz performs normalization, BPE,
//! truncation, and padding itself.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use anyhow::{Context, Result, bail};

pub struct ClipTokenizer {
    token_to_id: BTreeMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    byte_to_unicode: [char; 256],
    bos_id: u32,
    eos_id: u32,
}

impl ClipTokenizer {
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        let vocab_path = vocab_path.as_ref();
        let merges_path = merges_path.as_ref();
        let vocab_text = fs::read_to_string(vocab_path)
            .with_context(|| format!("cannot read CLIP vocabulary {}", vocab_path.display()))?;
        let token_to_id: BTreeMap<String, u32> = serde_json::from_str(&vocab_text)
            .with_context(|| format!("invalid CLIP vocabulary {}", vocab_path.display()))?;
        if token_to_id.is_empty() {
            bail!("CLIP vocabulary is empty");
        }
        let bos_id = token_to_id
            .get("<|startoftext|>")
            .copied()
            .context("CLIP vocabulary is missing <|startoftext|>")?;
        let eos_id = token_to_id
            .get("<|endoftext|>")
            .copied()
            .context("CLIP vocabulary is missing <|endoftext|>")?;

        let merges_text = fs::read_to_string(merges_path)
            .with_context(|| format!("cannot read CLIP merges {}", merges_path.display()))?;
        let mut merges = HashMap::new();
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut pieces = line.split_whitespace();
            let Some(first) = pieces.next() else { continue };
            let Some(second) = pieces.next() else {
                continue;
            };
            if pieces.next().is_some() {
                bail!("invalid CLIP merge line: {line}");
            }
            let rank = u32::try_from(merges.len()).context("too many CLIP merges")?;
            if merges
                .insert((first.to_string(), second.to_string()), rank)
                .is_some()
            {
                bail!("duplicate CLIP merge pair: {first} {second}");
            }
        }
        if merges.is_empty() {
            bail!("CLIP merge table is empty");
        }

        let (byte_to_unicode, _) = crate::tokenizer::build_byte_unicode_tables();
        Ok(Self {
            token_to_id,
            merges,
            byte_to_unicode,
            bos_id,
            eos_id,
        })
    }

    /// Encode a prompt to the fixed CLIP context used by SD1.x.
    pub fn encode(&self, text: &str) -> [u32; 77] {
        let mut ids = Vec::with_capacity(77);
        ids.push(self.bos_id);
        for token in pre_tokenize(text) {
            let encoded: String = token
                .as_bytes()
                .iter()
                .map(|byte| self.byte_to_unicode[*byte as usize])
                .collect();
            for piece in self.bpe(&encoded) {
                ids.push(self.token_to_id.get(&piece).copied().unwrap_or(self.eos_id));
                if ids.len() == 76 {
                    break;
                }
            }
            if ids.len() == 76 {
                break;
            }
        }
        ids.push(self.eos_id);
        ids.resize(77, self.eos_id);
        ids.try_into().expect("CLIP sequence is resized to 77")
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token
            .chars()
            .map(|character| character.to_string())
            .collect();
        let Some(last) = word.last_mut() else {
            return Vec::new();
        };
        last.push_str("</w>");
        if word.len() == 1 {
            return word;
        }

        loop {
            let mut best: Option<(u32, String, String)> = None;
            for pair in word.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                let Some(&rank) = self.merges.get(&key) else {
                    continue;
                };
                if best
                    .as_ref()
                    .is_none_or(|(best_rank, _, _)| rank < *best_rank)
                {
                    best = Some((rank, key.0, key.1));
                }
            }
            let Some((_, first, second)) = best else {
                break;
            };

            let mut merged = Vec::with_capacity(word.len());
            let mut index = 0;
            while index < word.len() {
                if index + 1 < word.len() && word[index] == first && word[index + 1] == second {
                    merged.push(format!("{first}{second}"));
                    index += 2;
                } else {
                    merged.push(word[index].clone());
                    index += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

/// Matches the token classes used by OpenAI CLIP without pulling a regex engine
/// into the mobile binary: special tokens, contractions, letter runs, individual
/// digits, and punctuation runs. Whitespace is normalized and discarded.
fn pre_tokenize(text: &str) -> Vec<String> {
    let normalized = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < normalized.len() {
        let rest = &normalized[index..];
        if rest.starts_with("<|startoftext|>") {
            tokens.push("<|startoftext|>".to_string());
            index += "<|startoftext|>".len();
            continue;
        }
        if rest.starts_with("<|endoftext|>") {
            tokens.push("<|endoftext|>".to_string());
            index += "<|endoftext|>".len();
            continue;
        }

        let character = rest
            .chars()
            .next()
            .expect("index remains on a character boundary");
        if character.is_whitespace() {
            index += character.len_utf8();
            continue;
        }
        if character == '\'' {
            if let Some(contraction) = ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"]
                .into_iter()
                .find(|candidate| rest.starts_with(candidate))
            {
                tokens.push(contraction.to_string());
                index += contraction.len();
                continue;
            }
        }
        if character.is_alphabetic() {
            let length = rest
                .char_indices()
                .take_while(|(_, next)| next.is_alphabetic())
                .map(|(offset, next)| offset + next.len_utf8())
                .last()
                .unwrap_or(character.len_utf8());
            tokens.push(rest[..length].to_string());
            index += length;
            continue;
        }
        if character.is_numeric() {
            tokens.push(character.to_string());
            index += character.len_utf8();
            continue;
        }

        let length = rest
            .char_indices()
            .take_while(|(_, next)| {
                !next.is_whitespace() && !next.is_alphabetic() && !next.is_numeric()
            })
            .map(|(offset, next)| offset + next.len_utf8())
            .last()
            .unwrap_or(character.len_utf8());
        tokens.push(rest[..length].to_string());
        index += length;
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture_tokenizer() -> (ClipTokenizer, std::path::PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("quartz-clip-tokenizer-{nonce}"));
        fs::create_dir(&dir).unwrap();
        fs::write(
            dir.join("vocab.json"),
            r#"{"<|startoftext|>":0,"<|endoftext|>":1,"h":2,"e":3,"l":4,"o</w>":5,"he":6,"hel":7,"hell":8,"hello</w>":9,"!</w>":10}"#,
        ).unwrap();
        fs::write(
            dir.join("merges.txt"),
            "#version: 0.2\nh e\nhe l\nhel l\nhell o</w>\n",
        )
        .unwrap();
        let tokenizer =
            ClipTokenizer::from_files(dir.join("vocab.json"), dir.join("merges.txt")).unwrap();
        (tokenizer, dir)
    }

    #[test]
    fn appends_end_of_word_and_pads_with_eos() {
        let (tokenizer, dir) = fixture_tokenizer();
        let ids = tokenizer.encode("  HELLO!  ");
        assert_eq!(&ids[..4], &[0, 9, 10, 1]);
        assert!(ids[4..].iter().all(|id| *id == 1));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pretokenizer_matches_clip_token_classes() {
        assert_eq!(
            pre_tokenize("Robot's 42 red... balloons!"),
            vec!["robot", "'s", "4", "2", "red", "...", "balloons", "!"]
        );
    }

    #[test]
    #[ignore = "requires QUARTZ_SD15_TOKENIZER_DIR pointing to official SD1.5 tokenizer files"]
    fn matches_official_sd15_golden_tokens() {
        let dir = std::env::var("QUARTZ_SD15_TOKENIZER_DIR").unwrap();
        let tokenizer = ClipTokenizer::from_files(
            Path::new(&dir).join("vocab.json"),
            Path::new(&dir).join("merges.txt"),
        )
        .unwrap();
        let cases: [(&str, &[u32]); 4] = [
            (
                "A photo of a lion in the wild, ultra realistic",
                &[
                    49406, 320, 1125, 539, 320, 5567, 530, 518, 3220, 267, 8118, 16157, 49407,
                ],
            ),
            ("", &[49406, 49407]),
            (
                "Robot's 42 red balloons!",
                &[49406, 8797, 568, 275, 273, 736, 18130, 256, 49407],
            ),
            (
                "café déjà vu",
                &[49406, 15304, 25466, 73, 21259, 13230, 49407],
            ),
        ];
        for (prompt, expected) in cases {
            assert_eq!(
                &tokenizer.encode(prompt)[..expected.len()],
                expected,
                "prompt: {prompt}"
            );
        }
    }
}
