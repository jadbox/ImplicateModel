//! Stage 0: Streaming Shot Segmentation & Camera Regime Discovery (`shot_segmenter`).
//!
//! Split into:
//! - `mod`: `CameraRegime`, `TransitionKind`, `ShotSegmenterConfig`, `FrameTransitionMetrics`, `ShotSegment`, `ShotSegmentationReport`.
//! - `detector`: `StreamingShotSegmenter` state extraction, CFT cross-power & dissolve detection, and regime classifier.
//! - `metrics`: 2D CFT peak finders, KL-divergence, dissolve linear fit, HUD/bezel isolation, and P/R evaluation.

pub mod detector;
pub mod metrics;

use implicate_math::Motor;
use serde::{Deserialize, Serialize};

use crate::enfold2d::Rect;

pub use detector::StreamingShotSegmenter;
pub use metrics::{
    downsample_to_analysis, evaluate_boundary_precision_recall,
    find_primary_and_secondary_cft_peaks, grid_l1_distance, isolate_hud_and_bezels,
    linear_dissolve_fit, symmetric_kl,
};

/// Analysis lattice width (power of two for zero-padding-free 2D FFT).
pub const ANALYSIS_W: usize = 128;
/// Analysis lattice height (power of two for zero-padding-free 2D FFT).
pub const ANALYSIS_H: usize = 64;
/// Number of horizontal depth-parallax strips across `ANALYSIS_H`.
pub const PARALLAX_STRIPS: usize = 4;
/// Number of bins in the joint luminance-chrominance histogram ($32\text{ Y} + 16\text{ Cr} + 16\text{ Cb}$).
pub const HIST_BINS: usize = 64;

/// Camera ego-motion regime discovered in Stage 0 (Whitepaper §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraRegime {
    Ortho2DStatic,
    Ortho2DScroll,
    MultiPlane25D,
    Perspective3D,
}

impl CameraRegime {
    pub fn as_str(self) -> &'static str {
        match self {
            CameraRegime::Ortho2DStatic => "Ortho2D-Static",
            CameraRegime::Ortho2DScroll => "Ortho2D-Scroll",
            CameraRegime::MultiPlane25D => "MultiPlane-2.5D",
            CameraRegime::Perspective3D => "Perspective-3D",
        }
    }
}

/// Shot boundary transition type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionKind {
    StreamStart,
    HardCut,
    CrossDissolve { duration_frames: usize },
}

/// Configuration for the Stage 0 Streaming Shot Segmenter.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShotSegmenterConfig {
    pub tau_cut: f32,
    pub tau_hist: f32,
    pub tau_dissolve: f32,
    pub min_shot_duration_sec: f64,
    pub static_max_disp_px: f32,
    pub multiplane_shear_threshold: f32,
}

impl Default for ShotSegmenterConfig {
    fn default() -> Self {
        Self {
            tau_cut: 0.22,
            tau_hist: 0.32,
            tau_dissolve: 0.45,
            min_shot_duration_sec: 0.25,
            static_max_disp_px: 4.0,
            multiplane_shear_threshold: 0.20,
        }
    }
}

/// Per-frame spectral & histogram telemetry computed on the `128x64` analysis lattice.
#[derive(Clone, Debug)]
pub struct FrameTransitionMetrics {
    pub frame_idx: usize,
    pub cross_corr_peak: f32,
    pub secondary_peak_ratio: f32,
    pub kl_divergence: f32,
    pub grid_l1_jump: f32,
    pub global_flow: [f32; 2],
    pub strip_flows: [[f32; 2]; PARALLAX_STRIPS],
    pub radial_divergence: f32,
}

/// Discovered contiguous shot segment $S_m = [t_{\text{start}}, t_{\text{end}}]$ with camera motor metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShotSegment {
    pub shot_id: usize,
    pub frame_start: usize,
    pub frame_end: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transition_in: TransitionKind,
    pub camera_regime: CameraRegime,
    pub mean_camera_velocity: [f32; 2],
    pub max_camera_displacement: f32,
    pub parallax_shear_ratio: f32,
    pub divergence_score: f32,
    pub mean_cross_corr: f32,
    pub entry_kl_divergence: f32,
}

impl ShotSegment {
    pub fn frame_count(&self) -> usize {
        self.frame_end.saturating_sub(self.frame_start) + 1
    }

    pub fn duration_sec(&self) -> f64 {
        (self.end_sec - self.start_sec).max(0.0)
    }

    pub fn camera_step_motor(&self, dt_frames: f64) -> Motor<f64> {
        let dx = self.mean_camera_velocity[0] as f64 * dt_frames;
        let dy = self.mean_camera_velocity[1] as f64 * dt_frames;
        let dz = match self.camera_regime {
            CameraRegime::Perspective3D => self.divergence_score as f64 * dt_frames,
            _ => 0.0,
        };
        Motor::from_translation(dx, dy, dz)
    }
}

/// Complete output of Stage 0 Streaming Shot Segmentation.
#[derive(Clone, Debug)]
pub struct ShotSegmentationReport {
    pub native_width: u32,
    pub native_height: u32,
    pub fps: f64,
    pub total_frames: usize,
    pub elapsed_sec: f64,
    pub throughput_fps: f64,
    pub bezel_and_hud_rects: Vec<Rect>,
    pub active_y_bounds: (usize, usize),
    pub shots: Vec<ShotSegment>,
    pub frame_metrics: Vec<FrameTransitionMetrics>,
}
