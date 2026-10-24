//! `implicate-lifter`: Subsystem I — Multi-Shot Video Enfoldment & The Clifford Lifter.
//!
//! Organized into modular subsystems matching Whitepaper §4:
//! - [`video_io`]: Streaming `ffmpeg` raw RGB24 frame reader (native and scaled) and H.264 writer.
//! - [`shot_segmenter`]: Stage 0 — Streaming 2D CFT cross-power + histogram KL-divergence shot cut & dissolve detector and `CameraRegime` classifier.
//! - [`multiplane_atlas`]: Stage 1 — Multi-plane depth-stratified ($1/z_l$) quantized RGB mode panorama reconstruction and additive luminescence cel separation.
//! - [`articulated_wavepacket`]: Stage 2 — `CliffordNet` Laplacian wedge-vorticity part segmentation and oriented Clifford Gaussian-Gabor wavepackets bound to $\text{Spin}(3,0,1)$ joint motors.
//! - [`enfold2d`]: 2D orthographic & scrolling CFT primitives (`ScrollEstimator`, `WorldAtlas`, `TileExtraction`, `ActorTrack`).

pub mod articulated_wavepacket;
pub mod enfold2d;
pub mod multiplane_atlas;
pub mod shot_segmenter;
pub mod video_io;

pub use articulated_wavepacket::{
    ArticulatedComparisonReport, ArticulatedHolon, CharacterObservation, CliffordDualDecomposition,
    ConstitutiveTuple, GpuModelSheetMonad, HolonModelSheet, LineOfAction, ModelSheetPartKernel,
    SpectralIdempotentDecomposition, SubHolonPartSpec, SubHolonPose, TurnaroundViewAspect,
    ViewRegime, WavepacketPartKernel, choose_part_and_view_counts, classify_view_regime,
    compute_cliffordnet_laplacian_features, compute_perceptual_patch_lpips,
    decompose_clifford_dual_product, decompose_masked_clifford_dual_product,
    decompose_observation_dual_product, dominant_silhouette, evaluate_vs_single_box_baseline,
    evaluate_vs_single_box_baseline_with_parts,
    extract_character_observation_tracks, extract_dominant_character_observations,
    extract_line_of_action, extract_line_of_action_weighted,
    factor_temporal_covariance_idempotents, fit_articulated_holon, induce_holon_model_sheet,
    mask_iou, render_articulated_holon_observation,
};
pub use enfold2d::{
    ActorKind, ActorTrack, AtlasAccumulator, Fft2, HudPartition, Rect, RgbImage8, ScrollEstimator,
    Sprite2D, TileExtraction, TileKind, TileOptions, WorldAtlas,
};
pub use multiplane_atlas::{
    LuminescenceFrame, MultiPlaneAtlas, MultiPlaneOptions, ParallaxLayer, build_multiplane_atlas,
};
pub use shot_segmenter::{
    CameraRegime, FrameTransitionMetrics, ShotSegment, ShotSegmentationReport,
    ShotSegmenterConfig, StreamingShotSegmenter, TransitionKind,
    evaluate_boundary_precision_recall,
};
pub use video_io::{
    FrameReader, RawFrame, VideoInfo, VideoWriter, mux_rgb_frames_and_wav_to_mp4, probe,
    probe_audio_stream,
};
