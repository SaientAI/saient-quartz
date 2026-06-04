// BPE tokenizer loaded from GGUF metadata.
// Supports Qwen2, LLaMA-family, and any model using tokenizer.ggml.* keys.

use std::collections::HashMap;
use crate::gguf::{GgufFile, GgufValue};

pub struct Tokenizer {
    vocab:       Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges:      HashMap<(String, String), u32>,
    byte_to_u:   [char; 256],     // GPT-2 byte→unicode encoding table
    u_to_byte:   HashMap<char, u8>, // inverse
    spm:         bool,
    max_token_chars: usize,
    pub bos_id:       u32,
    pub eos_id:       u32,
    pub unk_id:       u32,
    pub im_start_id:  u32,
    pub im_end_id:    u32,
    pub msg_start_id: u32,  // <|start|> — GPT-oss role delimiter
    pub end_id:       u32,
    pub return_id:    u32,
    pub channel_id:   u32,  // <|channel|> — GPT-oss channel switch
    pub message_id:   u32,  // <|message|> — GPT-oss message start
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        // Vocab
        let vocab: Vec<String> = match gguf.metadata.get("tokenizer.ggml.tokens")? {
            GgufValue::Array(arr) => arr.iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            _ => return None,
        };

        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (i, tok) in vocab.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Merges
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        if let Some(GgufValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.merges") {
            for (rank, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() {
                    if let Some(pos) = s.find(' ') {
                        merges.insert((s[..pos].to_string(), s[pos+1..].to_string()), rank as u32);
                    }
                }
            }
        }

        let (byte_to_u, u_to_byte) = build_byte_unicode_tables();
        let model_name = gguf.metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let spm = model_name.contains("llama")
            || model_name.contains("sentencepiece")
            || vocab.iter().any(|tok| tok.starts_with('▁'));
        let max_token_chars = vocab.iter().map(|tok| tok.chars().count()).max().unwrap_or(1);

        // Special token IDs
        let find = |s: &str| token_to_id.get(s).copied().unwrap_or(u32::MAX);
        let bos_id      = find("<|endoftext|>")
            .min(find("<s>"))
            .min(gguf.metadata.get("tokenizer.ggml.bos_token_id")
                .and_then(|v| v.as_u32())
                .unwrap_or(u32::MAX));
        let unk_id      = find("<unk>")
            .min(gguf.metadata.get("tokenizer.ggml.unknown_token_id")
                .and_then(|v| v.as_u32())
                .unwrap_or(u32::MAX));
        let im_end_id   = find("<|im_end|>");
        let im_start_id = find("<|im_start|>");
        let end_id      = find("<|end|>");      // GPT-oss native EOS
        let return_id   = find("<|return|>");   // GPT-oss final turn terminator
        let msg_start_id = find("<|start|>");    // GPT-oss role delimiter
        let channel_id   = find("<|channel|>");  // GPT-oss channel switch
        let message_id   = find("<|message|>");  // GPT-oss message start
        let eos_id = if return_id != u32::MAX { return_id }
                     else if end_id != u32::MAX   { end_id }
                     else if im_end_id != u32::MAX { im_end_id }
                     else { token_to_id.get("</s>").copied().unwrap_or(bos_id) };

        Some(Self { vocab, token_to_id, merges, byte_to_u, u_to_byte,
                    spm, max_token_chars,
                    bos_id, eos_id, unk_id, im_start_id, im_end_id, msg_start_id,
                    end_id, return_id, channel_id, message_id })
    }

    // ── Encode ────────────────────────────────────────────────────────────────

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.spm {
            return self.encode_spm(text);
        }

        let mut ids = Vec::new();
        for word in pre_tokenize(text) {
            // Convert each byte to its GPT-2 unicode char
            let gpt2: String = word.bytes().map(|b| self.byte_to_u[b as usize]).collect();
            ids.extend(self.bpe(&gpt2));
        }
        ids
    }

    fn encode_spm(&self, text: &str) -> Vec<u32> {
        let mut norm = String::new();
        for ch in text.chars() {
            if ch == ' ' {
                norm.push('▁');
            } else {
                norm.push(ch);
            }
        }

        let chars: Vec<char> = norm.chars().collect();
        let mut ids = Vec::new();
        let mut i = 0usize;

        while i < chars.len() {
            let remaining = chars.len() - i;
            let max_len = self.max_token_chars.min(remaining);
            let mut matched = None;

            for len in (1..=max_len).rev() {
                let s: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&s) {
                    matched = Some((id, len));
                    break;
                }
            }

            if let Some((id, len)) = matched {
                ids.push(id);
                i += len;
                continue;
            }

            let ch = chars[i];
            for byte in ch.to_string().as_bytes() {
                let s = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.token_to_id.get(&s) {
                    ids.push(id);
                } else if self.unk_id != u32::MAX {
                    ids.push(self.unk_id);
                }
            }
            i += 1;
        }

        ids
    }

    // Encode a template string, treating <|...|> as atomic special tokens.
    pub fn encode_template(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            if rest.starts_with("<|") {
                if let Some(end) = rest.find("|>") {
                    let tok = &rest[..end + 2];
                    if let Some(&id) = self.token_to_id.get(tok) {
                        ids.push(id);
                        rest = &rest[end + 2..];
                        continue;
                    }
                }
            }
            let next = rest[1..].find("<|").map(|p| p + 1).unwrap_or(rest.len());
            ids.extend(self.encode(&rest[..next]));
            rest = &rest[next..];
        }
        ids
    }

    // Build Qwen2 instruct chat prompt and encode it.
    // Uses the default system message from the model's chat template.
    pub fn chat_encode(&self, user_text: &str) -> Vec<u32> {
        let system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        self.encode_messages(&[("system", system), ("user", user_text)])
    }

    // Encode an arbitrary multi-turn conversation (role, content) pairs.
    // Uses GPT-oss native format: <|start|>role<|message|>content<|end|>
    // Falls back to ChatML if <|start|> is not in the vocabulary.
    pub fn encode_messages(&self, messages: &[(&str, &str)]) -> Vec<u32> {
        let has_native = self.token_to_id.contains_key("<|start|>");
        let mut prompt = String::new();

        if has_native {
            prompt.push_str("<|start|>system<|message|>");
            prompt.push_str("Knowledge cutoff: 2024-06\n\n");
            prompt.push_str("Reasoning: low\n\n");   // keep gpt-oss snappy — minimal analysis ramble
            prompt.push_str("# Valid channels: analysis, final. Channel must be included for every message.");
            prompt.push_str("<|end|>");

            let mut first_message = 0usize;
            if let Some((role, content)) = messages.first() {
                if *role == "system" || *role == "developer" {
                    prompt.push_str("<|start|>developer<|message|>\n# Instructions\n\n");
                    prompt.push_str(content);
                    prompt.push_str("<|end|>");
                    first_message = 1;
                }
            }

            for (role, content) in &messages[first_message..] {
                match *role {
                    "system" | "developer" => {
                        prompt.push_str("<|start|>developer<|message|>\n# Instructions\n\n");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                    "assistant" => {
                        prompt.push_str("<|start|>assistant<|message|>");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                    _ => {
                        prompt.push_str("<|start|>user<|message|>");
                        prompt.push_str(content);
                        prompt.push_str("<|end|>");
                    }
                }
            }
            // gpt-oss harmony: end at `<|start|>assistant` and let the MODEL emit its own
            // channel (`analysis` for thinking, then `final` for the answer). Forcing
            // `<|message|>` with no channel garbles it; forcing `<|channel|>final` skips the
            // reasoning step and breaks anything that needs to think (e.g. a story → empty/junk).
            prompt.push_str("<|start|>assistant");
        } else if self.token_to_id.contains_key("<|im_start|>") {
            for (role, content) in messages {
                prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
            }
            prompt.push_str("<|im_start|>assistant\n");
        } else {
            if self.token_to_id.contains_key("<|user|>") && self.token_to_id.contains_key("<|assistant|>") {
                for (role, content) in messages {
                    match *role {
                        "system" | "developer" => {
                            prompt.push_str("<|system|>\n");
                            prompt.push_str(content);
                            prompt.push_str("</s>\n");
                        }
                        "assistant" => {
                            prompt.push_str("<|assistant|>\n");
                            prompt.push_str(content);
                            prompt.push_str("</s>\n");
                        }
                        _ => {
                            prompt.push_str("<|user|>\n");
                            prompt.push_str(content);
                            prompt.push_str("</s>\n");
                        }
                    }
                }
                prompt.push_str("<|assistant|>\n");
            } else {
                let mut saw_user = false;
                for (role, content) in messages {
                    match *role {
                        "system" | "developer" => {
                            prompt.push_str("SYSTEM: ");
                            prompt.push_str(content);
                            prompt.push('\n');
                        }
                        "assistant" => {
                            prompt.push_str(content);
                            prompt.push('\n');
                        }
                        _ => {
                            prompt.push_str("USER: ");
                            prompt.push_str(content);
                            prompt.push_str("\nASSISTANT:");
                            saw_user = true;
                        }
                    }
                }
                if !saw_user {
                    prompt.push_str("USER: hello\nASSISTANT:");
                }
            }
        }

        self.encode_template(&prompt)
    }

    // No system prompt version for testing
    pub fn chat_encode_nosys(&self, user_text: &str) -> Vec<u32> {
        self.encode_messages(&[("user", user_text)])
    }

    // ── Decode ────────────────────────────────────────────────────────────────

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            bytes.extend(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // Decode a single token (for streaming output).
    pub fn decode_token(&self, id: u32) -> String {
        self.decode(&[id])
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        if id as usize >= self.vocab.len() { return Vec::new(); }
        let tok = &self.vocab[id as usize];
        if self.is_special(id) { return Vec::new(); }
        if let Some(hex) = tok.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                return vec![byte];
            }
        }
        if self.spm {
            return tok.replace('▁', " ").into_bytes();
        }
        let mut bytes = Vec::new();
        for ch in tok.chars() {
            if let Some(&b) = self.u_to_byte.get(&ch) {
                bytes.push(b);
            }
        }
        bytes
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.vocab
            .get(id as usize)
            .map(|tok| {
                (tok.starts_with("<|") && tok.ends_with("|>"))
                    || tok == "<s>"
                    || tok == "</s>"
                    || tok == "<unk>"
            })
            .unwrap_or(false)
    }

    pub fn is_eos(&self, id: u32) -> bool {
        if self.return_id != u32::MAX {
            // GPT-oss native format: <|end|> closes a channel message block (not the turn).
            // Only <|return|> ends the full assistant turn.
            return id == self.return_id || id == self.bos_id;
        }
        id == self.eos_id || id == self.im_end_id || id == self.bos_id
    }

    // True for <|end|> in GPT-oss format — closes current channel message, not the turn.
    pub fn is_channel_end(&self, id: u32) -> bool {
        self.return_id != u32::MAX && id == self.end_id
    }

    // ── BPE internals ─────────────────────────────────────────────────────────

    fn bpe(&self, word: &str) -> Vec<u32> {
        if word.is_empty() { return vec![]; }

        // Direct vocab hit (common for short tokens)
        if let Some(&id) = self.token_to_id.get(word) {
            return vec![id];
        }

        // Initialise: one symbol per Unicode char
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Merge loop: repeatedly apply the lowest-rank merge
        loop {
            if syms.len() <= 1 { break; }

            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;

            for i in 0..syms.len() - 1 {
                let key = (syms[i].as_str(), syms[i + 1].as_str());
                if let Some(&rank) = self.merges.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if rank < best_rank { best_rank = rank; best_i = i; }
                    let _ = key;
                }
            }

            if best_i == usize::MAX { break; }

            let merged = format!("{}{}", syms[best_i], syms[best_i + 1]);
            syms[best_i] = merged;
            syms.remove(best_i + 1);
        }

        // Map symbols to IDs; byte-fallback for unknown symbols
        let mut ids = Vec::new();
        for sym in &syms {
            if let Some(&id) = self.token_to_id.get(sym) {
                ids.push(id);
            } else {
                for byte in sym.as_bytes() {
                    let s = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&s) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }
}

// ── Pre-tokenizer ─────────────────────────────────────────────────────────────
//
// Splits text so that spaces are attached to the FOLLOWING word
// (GPT-2 convention: " world" → one word starting with space).
// "hello world" → ["hello", " world"]
// "hi, there"   → ["hi", ",", " there"]

// Qwen2/tiktoken-style pre-tokenizer:
//   - space attaches to the following alphabetic run ("Ġworld")
//   - space before a digit or punctuation is emitted as a standalone "Ġ"
//   - digits are collected into runs (letters never merge across)
//   - punctuation chars are each emitted individually
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut chars = text.chars().peekable();

    while let Some(&ch) = chars.peek() {
        match ch {
            '\n' | '\r' => {
                chars.next();
                words.push(ch.to_string());
            }
            ' ' => {
                chars.next();
                // attach space to the next alphabetic run only
                let next_is_alpha = chars.peek().map(|c| c.is_alphabetic()).unwrap_or(false);
                let mut w = String::from(' ');
                if next_is_alpha {
                    while let Some(&c) = chars.peek() {
                        if c.is_alphabetic() { w.push(c); chars.next(); } else { break; }
                    }
                }
                words.push(w);
            }
            c if c.is_alphabetic() => {
                let mut w = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphabetic() { w.push(c); chars.next(); } else { break; }
                }
                words.push(w);
            }
            c if c.is_ascii_digit() => {
                let mut w = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() { w.push(c); chars.next(); } else { break; }
                }
                words.push(w);
            }
            _ => {
                chars.next();
                words.push(ch.to_string());
            }
        }
    }
    words
}

// ── GPT-2 byte ↔ unicode tables ───────────────────────────────────────────────
//
// Bytes in printable ASCII (33-126) and Latin supplement (161-172, 174-255)
// map to themselves. The remaining 68 bytes map to U+0100 ... U+0143.
// Space (0x20) → Ġ (U+0120), newline (0x0A) → Ċ (U+010A), etc.

fn build_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut b2u = ['\0'; 256];
    let mut u2b: HashMap<char, u8> = HashMap::new();

    // Keep-same ranges
    for b in 33u8..=126u8 {
        b2u[b as usize] = b as char;
        u2b.insert(b as char, b);
    }
    for b in 161u8..=172u8 {
        b2u[b as usize] = b as char;
        u2b.insert(b as char, b);
    }
    for b in 174u8..=255u8 {
        b2u[b as usize] = b as char;
        u2b.insert(b as char, b);
    }

    // Remaining bytes → U+0100 onwards (in byte order 0, 1, ..., 255)
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if b2u[b as usize] == '\0' {
            let ch = char::from_u32(0x100 + n).expect("valid unicode");
            b2u[b as usize] = ch;
            u2b.insert(ch, b);
            n += 1;
        }
    }

    (b2u, u2b)
}
