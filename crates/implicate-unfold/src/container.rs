//! `.implicate` v1 Binary Container Format (Whitepaper §8).
//!
//! Layout:
//! - Magic: `"IMPL"` (`4 bytes`)
//! - Version: `1u32` (`4 bytes`, little-endian)
//! - Flags: `u32` (`4 bytes`, little-endian)
//! - Manifest JSON Length: `u64` (`8 bytes`, little-endian)
//! - UTF-8 JSON Manifest (`ImplicateContainerManifest`)
//! - Padding to 64-byte alignment
//! - Binary tensor payload

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use implicate_lexicon::{
    CORE_VERBS, ConceptLexicon, GatrHolonOutput, GatrHolonToken, GatrVerbBinder,
    VerbPotentialLexicon, synthesize_corpus_verb_clip,
};
use serde::{Deserialize, Serialize};

use crate::payload::EnfoldedClipSummary;

pub const IMPLICATE_MAGIC: &[u8; 4] = b"IMPL";
pub const IMPLICATE_VERSION: u32 = 1;
pub const IMPLICATE_FLAG_LEXICON: u32 = 0x01;
pub const IMPLICATE_FLAG_ENFOLDED_ASSETS: u32 = 0x02;

/// Manifest stored inside a `.implicate` v1 binary container.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImplicateContainerManifest {
    pub title: String,
    pub corpus_sources: Vec<String>,
    pub concept_lexicon: ConceptLexicon,
    pub verb_potentials: VerbPotentialLexicon,
    pub gatr_binder: GatrVerbBinder,
    #[serde(default)]
    pub enfolded_clips: Vec<EnfoldedClipSummary>,
}

impl ImplicateContainerManifest {
    /// Build the canonical Implicate Lexicon compiled from the 4 reference corpus sources
    /// (`pacman`, `smb3`, `raptor`, and *The Secret of NIMH*).
    pub fn canonical_corpus_manifest() -> Self {
        let concept_lexicon = ConceptLexicon::default();
        let num_train_clips = 12usize;
        let num_steps = 48usize;

        let mut ensembles = Vec::with_capacity(CORE_VERBS.len());
        for (v_idx, &verb) in CORE_VERBS.iter().enumerate() {
            let mut clips = Vec::with_capacity(num_train_clips);
            for c in 0..num_train_clips {
                let scale = 35.0 + 5.0 * (c as f64);
                let rot = 0.25 * (c as f64);
                let offset = [100.0 + 12.0 * (c as f64), 80.0 - 7.0 * (c as f64)];
                let seed = ((v_idx as u64) + 1) * 1000 + (c as u64);
                clips.push(synthesize_corpus_verb_clip(
                    verb, num_steps, offset, scale, rot, 0.012, seed,
                ));
            }
            ensembles.push((verb, clips));
        }

        let verb_potentials =
            VerbPotentialLexicon::from_labeled_ensembles(&concept_lexicon, &ensembles);

        let verb_spinors = concept_lexicon.embed_clauses(&CORE_VERBS);
        let noun_clauses = [
            "Mrs_Brisby hero mouse",
            "Jeremy_Crow trickster bird",
            "Mario hero jumper",
            "Red_Amulet self talisman",
        ];
        let noun_spinors = concept_lexicon.embed_clauses(&noun_clauses);

        let mut gatr_binder = GatrVerbBinder::new();
        for (v_idx, &verb_spinor) in verb_spinors.iter().enumerate() {
            let mut train_set = Vec::with_capacity(5);
            for clip_idx in 0..5 {
                let tau = (clip_idx as f32) * 0.2;
                let dist = 4.0 + 0.3 * (clip_idx as f32);
                let p_a = [0.0f32, 0.0, 0.0];
                let p_b = [dist, 0.4 * ((clip_idx % 3) as f32), 0.0];
                let n_up = [0.0f32, 1.0, 0.0];

                let rx = p_b[0] - p_a[0];
                let ry = p_b[1] - p_a[1];
                let rz = p_b[2] - p_a[2];
                let rn = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-3);
                let u_rel = [rx / rn, ry / rn, rz / rn];
                let dot_nr = n_up[0] * u_rel[0] + n_up[1] * u_rel[1] + n_up[2] * u_rel[2];
                let nx = n_up[0] - dot_nr * u_rel[0];
                let ny = n_up[1] - dot_nr * u_rel[1];
                let nz = n_up[2] - dot_nr * u_rel[2];
                let nn = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-3);
                let n_orth = [nx / nn, ny / nn, nz / nn];
                let u_tang = [
                    n_orth[1] * u_rel[2] - n_orth[2] * u_rel[1],
                    n_orth[2] * u_rel[0] - n_orth[0] * u_rel[2],
                    n_orth[0] * u_rel[1] - n_orth[1] * u_rel[0],
                ];

                let (w_rel, w_up, w_tang, beacon) = match v_idx {
                    0 => (0.8f32, 0.0f32, 0.4f32, [2.0f32, 1.0, 1.2, 0.0]),
                    1 => (1.2f32, 1.5f32, 0.0f32, [1.5f32, 0.0, 2.0, 0.5]),
                    2 => (2.0f32, 0.1f32, 0.0f32, [3.0f32, 0.0, 2.5, 0.0]),
                    3 => (-2.2f32, 0.3f32, 0.2f32, [3.0f32, 0.0, -2.5, 3.14]),
                    _ => (0.1f32, 0.0f32, 1.8f32, [2.5f32, 2.0, 1.8, 1.57]),
                };
                let dv = [
                    w_rel * u_rel[0] + w_up * n_orth[0] + w_tang * u_tang[0],
                    w_rel * u_rel[1] + w_up * n_orth[1] + w_tang * u_tang[1],
                    w_rel * u_rel[2] + w_up * n_orth[2] + w_tang * u_tang[2],
                ];
                let dw = [0.0f32, 0.5 * w_tang, 0.2 * w_up];

                let tok_a = GatrHolonToken {
                    position: p_a,
                    velocity: [0.2 * u_rel[0], 0.2 * u_rel[1], 0.2 * u_rel[2]],
                    angular_velocity: [0.0, 0.1, 0.0],
                    support_normal: n_up,
                    support_dist: 0.0,
                    noun_spinor: noun_spinors[v_idx % noun_spinors.len()],
                    verb_spinor,
                    target_idx: 1,
                    phase_tau: tau,
                };
                let tok_b = GatrHolonToken {
                    position: p_b,
                    velocity: [0.0, 0.0, 0.0],
                    angular_velocity: [0.0, 0.0, 0.0],
                    support_normal: n_up,
                    support_dist: 0.0,
                    noun_spinor: noun_spinors[(v_idx + 1) % noun_spinors.len()],
                    verb_spinor,
                    target_idx: 0,
                    phase_tau: tau,
                };
                let out_a = GatrHolonOutput {
                    delta_v: dv,
                    delta_omega: dw,
                    q_beacon_params: beacon,
                };
                let out_b = GatrHolonOutput {
                    delta_v: [-dv[0], -dv[1], -dv[2]],
                    delta_omega: [-dw[0], -dw[1], -dw[2]],
                    q_beacon_params: beacon,
                };
                train_set.push((vec![tok_a, tok_b], vec![out_a, out_b]));
            }
            gatr_binder.fit_verb_from_examples(verb_spinor, &train_set);
        }

        Self {
            title: "ImplicateModel Canonical Corpus Lexicon v1".to_string(),
            corpus_sources: vec![
                "pacman_240x308.mp4".to_string(),
                "smb3_w1_1_256x240.mp4".to_string(),
                "raptor_320x240.mp4".to_string(),
                "The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4".to_string(),
            ],
            concept_lexicon,
            verb_potentials,
            gatr_binder,
            enfolded_clips: Vec::new(),
        }
    }
}

/// Encode an `ImplicateContainerManifest` and optional binary tensor payload into `.implicate` v1 bytes.
pub fn encode_implicate_container(
    manifest: &ImplicateContainerManifest,
    flags: u32,
    binary_payload: &[u8],
) -> io::Result<Vec<u8>> {
    let json_bytes = serde_json::to_vec(manifest)
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("JSON encode error: {e}")))?;
    let header_and_json = 4 + 4 + 4 + 8 + json_bytes.len();
    let aligned_offset = (header_and_json + 63) & !63;
    let pad_len = aligned_offset - header_and_json;

    let mut out = Vec::with_capacity(aligned_offset + binary_payload.len());
    out.extend_from_slice(IMPLICATE_MAGIC);
    out.extend_from_slice(&IMPLICATE_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.resize(out.len() + pad_len, 0u8);
    out.extend_from_slice(binary_payload);
    Ok(out)
}

/// Decode a `.implicate` v1 binary container from raw bytes.
pub fn decode_implicate_container(
    bytes: &[u8],
) -> io::Result<(ImplicateContainerManifest, u32, &[u8])> {
    if bytes.len() < 20 {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            "Container shorter than 20-byte header",
        ));
    }
    if &bytes[0..4] != IMPLICATE_MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "Invalid .implicate magic header (expected 'IMPL')",
        ));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != IMPLICATE_VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("Unsupported .implicate version {version}"),
        ));
    }
    let flags = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let json_len = u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize;
    let json_end = 20usize.saturating_add(json_len);
    if bytes.len() < json_end {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            "Truncated JSON manifest in .implicate container",
        ));
    }
    let manifest: ImplicateContainerManifest = serde_json::from_slice(&bytes[20..json_end])
        .map_err(|e| io::Error::new(ErrorKind::InvalidData, format!("JSON decode error: {e}")))?;
    let aligned_offset = (json_end + 63) & !63;
    let payload = if bytes.len() >= aligned_offset {
        &bytes[aligned_offset..]
    } else {
        &[]
    };
    Ok((manifest, flags, payload))
}

pub fn save_implicate_container(
    path: &Path,
    manifest: &ImplicateContainerManifest,
    flags: u32,
    binary_payload: &[u8],
) -> io::Result<()> {
    let bytes = encode_implicate_container(manifest, flags, binary_payload)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, bytes)
}

pub fn load_implicate_container(
    path: &Path,
) -> io::Result<(ImplicateContainerManifest, u32, Vec<u8>)> {
    let bytes = fs::read(path)?;
    let (manifest, flags, payload) = decode_implicate_container(&bytes)?;
    Ok((manifest, flags, payload.to_vec()))
}
