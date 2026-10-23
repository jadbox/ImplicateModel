//! `implicate-unfold`: Prompt / `DzogLang` Scene Compiler, `.implicate` v1 Binary Container,
//! and Single-Pass Causal Unfoldment Engine (Subsystem III, Track R4.3).

pub mod container;
pub mod engine;
pub mod payload;
pub mod prompt;

pub use container::{
    IMPLICATE_FLAG_ENFOLDED_ASSETS, IMPLICATE_FLAG_LEXICON, IMPLICATE_MAGIC, IMPLICATE_VERSION,
    ImplicateContainerManifest, decode_implicate_container, encode_implicate_container,
    load_implicate_container, save_implicate_container,
};
pub use engine::{ActiveVerbBinding, CausalUnfoldEngine, UnfoldRunReport, VerbAdherenceMetric};
pub use payload::{
    ENFOLDED_PAYLOAD_MAGIC, ENFOLDED_PAYLOAD_VERSION, EnfoldedClipAssets, EnfoldedClipSummary,
    assimilate_clip_into_manifest, decode_enfolded_payload, encode_enfolded_payload,
    enfold_video_clip, load_implicate_container_with_assets, save_implicate_container_with_assets,
    upsert_enfolded_clip_to_container,
};
pub use prompt::{
    PromptBoundaryBlade, PromptHolonSpec, PromptVerbBinding, UnfoldSceneSpec,
};
