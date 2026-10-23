//! `llama.cpp` `Qwen2.5-VL-7B-Instruct` (`Q4_K_M`) Harvester & Content-Addressed Cache.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::schema::{IMPLICATE_LEXICON_GBNF, ShotLexicon};

/// Default local path for `Qwen2.5-VL-7B-Instruct` (`Q4_K_M`).
pub const DEFAULT_QWEN_VL_MODEL: &str =
    "/mnt/storage/models/llama.cpp/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";

/// Default local path for the `Qwen2.5-VL-7B-Instruct` multimodal projector (`Q8_0`).
pub const DEFAULT_QWEN_VL_MMPROJ: &str =
    "/mnt/storage/models/llama.cpp/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf";

/// Compute a deterministic 64-bit FNV-1a content hash for `.implicate_cache/` keys.
pub fn content_hash_hex(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// Configuration for invoking `llama-cli` with `Qwen2.5-VL-7B-Instruct` (`Q4_K_M`).
#[derive(Clone, Debug)]
pub struct QwenVlHarvesterConfig {
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub cache_dir: PathBuf,
    pub video_fps: f32,
    pub max_tokens: u32,
}

impl Default for QwenVlHarvesterConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from(DEFAULT_QWEN_VL_MODEL),
            mmproj_path: PathBuf::from(DEFAULT_QWEN_VL_MMPROJ),
            cache_dir: PathBuf::from(".implicate_cache"),
            video_fps: 1.0,
            max_tokens: 400,
        }
    }
}

/// Run `llama-cli` with `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` and `IMPLICATE_LEXICON_GBNF` on a video clip,
/// returning a parsed `ShotLexicon` (using content-addressed caching under `config.cache_dir`).
pub fn harvest_clip_with_qwen_vl(
    video_path: &Path,
    clip_duration_sec: u32,
    config: &QwenVlHarvesterConfig,
) -> Result<ShotLexicon, String> {
    let video_bytes = fs::read(video_path)
        .map_err(|e| format!("failed to read video {}: {e}", video_path.display()))?;
    let mut key_buf = video_bytes;
    key_buf.extend_from_slice(IMPLICATE_LEXICON_GBNF.as_bytes());
    let key = content_hash_hex(&key_buf);

    fs::create_dir_all(&config.cache_dir)
        .map_err(|e| format!("failed to create cache dir: {e}"))?;
    let cache_file = config.cache_dir.join(format!("shot_{key}.json"));
    if cache_file.exists() {
        let cached = fs::read_to_string(&cache_file)
            .map_err(|e| format!("failed to read cached json: {e}"))?;
        if let Ok(parsed) = serde_json::from_str::<ShotLexicon>(&cached) {
            return Ok(parsed);
        }
    }

    let gbnf_path = config.cache_dir.join("implicate_lexicon.gbnf");
    fs::write(&gbnf_path, IMPLICATE_LEXICON_GBNF)
        .map_err(|e| format!("failed to write gbnf file: {e}"))?;

    let prompt = format!(
        "Analyze this {clip_duration_sec}-second video clip (from 00:00:00 to 00:00:{clip_duration_sec:02}). \
         Output a JSON object describing the scene milieu, all active character/object holons \
         (id, canonical noun, appearance, location), and all action verbs with start and end \
         timestamps [HH:MM:SS - HH:MM:SS]."
    );

    let output = Command::new("llama-cli")
        .arg("-m")
        .arg(&config.model_path)
        .arg("--mmproj")
        .arg(&config.mmproj_path)
        .args(["-ngl", "99", "--video"])
        .arg(video_path)
        .args([
            "--video-fps",
            &format!("{:.1}", config.video_fps),
            "--video-timestamp-interval",
            "1000",
            "--grammar-file",
        ])
        .arg(&gbnf_path)
        .args([
            "--repeat-penalty",
            "1.15",
            "--temp",
            "0.0",
            "--seed",
            "42",
            "--single-turn",
            "-n",
            &config.max_tokens.to_string(),
            "-p",
            &prompt,
        ])
        .output()
        .map_err(|e| format!("failed to spawn llama-cli: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_slice = extract_json_object(&stdout)
        .ok_or_else(|| format!("no JSON object found in llama-cli output: {stdout}"))?;

    let parsed: ShotLexicon = serde_json::from_str(json_slice)
        .map_err(|e| format!("invalid ShotLexicon JSON ({e}): {json_slice}"))?;

    let _ = fs::write(
        &cache_file,
        serde_json::to_string_pretty(&parsed).unwrap_or_default(),
    );
    Ok(parsed)
}

/// Extract the outermost `{ ... }` JSON object following the prompt echo in `llama-cli` stdout.
pub fn extract_json_object(stdout: &str) -> Option<&str> {
    let search_in = stdout
        .rfind("timestamps [HH:MM:SS - HH:MM:SS].")
        .map(|idx| &stdout[idx + "timestamps [HH:MM:SS - HH:MM:SS].".len()..])
        .unwrap_or(stdout);

    let start = search_in.find('{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escape = false;
    for (offset, ch) in search_in[start..].char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_str {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(&search_in[start..=start + offset]);
                }
            }
        }
    }
    None
}
