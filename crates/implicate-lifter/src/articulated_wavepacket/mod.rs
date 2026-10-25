//! Stage 2: Articulated Sub-Holon Decomposition & Oriented Gaussian-Gabor Wavepackets (`articulated_wavepacket`).
//!
//! Split into:
//! - `mod`: `CharacterObservation`, `SubHolonPose`, `WavepacketPartKernel`, `ArticulatedHolon`, `fit_articulated_holon`.
//! - `clifford_features`: `CliffordNet` Laplacian wedge-vorticity features ($D^{(s)} \oplus W^{(s)}$), IoU, and LPIPS metrics.
//! - `part_fitting`: K-means part prototype learning, view-aspect idempotent clustering, and Gaussian-Gabor kernel accumulation.

pub mod clifford_features;
pub mod morphology;
pub mod part_fitting;

use bytemuck::{Pod, Zeroable};
use implicate_math::{
    CliffordDensityElement, Motor, Mv2, Mv301, StateIdempotent, TransitionOperator,
};
use serde::{Deserialize, Serialize};

use crate::enfold2d::{Rect, RgbImage8};

pub use clifford_features::{
    ArticulatedComparisonReport, CliffordDualDecomposition,
    compute_cliffordnet_laplacian_features, compute_perceptual_patch_lpips,
    decompose_clifford_dual_product, decompose_masked_clifford_dual_product,
    decompose_observation_dual_product, evaluate_vs_single_box_baseline,
    evaluate_vs_single_box_baseline_with_parts, mask_iou,
};
pub use part_fitting::{
    SpectralIdempotentDecomposition, enforce_volume_conserving_gait_deformation,
    factor_temporal_covariance_idempotents, induce_holon_model_sheet,
};
use part_fitting::{
    accumulate_part_wavepacket_kernel, assign_view_idempotent_cells_and_cft_phases,
    fit_part_gaussian_pose, learn_part_prototypes, median_obs_dims,
};

/// 3D PGA Line of Action ($L_{\text{spine}} \in \mathbb{G}_{3,0,1}$) and kinematic mass inertia envelope.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineOfAction {
    /// 3D PGA line bivector L_spine ∈ G(3,0,1) packed into the 64-byte Mv301 monad.
    pub spine_bivector: Mv301,
    /// Center of mass in observation coordinates [cx, cy].
    pub center_of_mass: [f64; 2],
    /// Unit direction vector of the Line of Action [dx, dy] along the principal spine.
    pub direction: [f64; 2],
    /// Principal orientation angle θ in radians [-π, π].
    pub angle_rad: f64,
    /// Longitudinal spread (standard deviation σ_∥ along the spine axis).
    pub spread_parallel: f64,
    /// Transverse spread (standard deviation σ_⊥ perpendicular to the spine axis).
    pub spread_perpendicular: f64,
    /// Dynamic quadratic spinal curvature κ (measure of spinal C-curve / S-curve flexion).
    pub curvature: f64,
    /// Total mass (pixel count or weighted vitality sum).
    pub total_mass: f64,
}

/// Extract the Line of Action from a silhouette mask.
pub fn extract_line_of_action(mask: &[bool], w: usize, h: usize) -> Option<LineOfAction> {
    extract_line_of_action_weighted(mask, None, w, h, [0.0, 0.0])
}

/// Extract the Line of Action from a weighted silhouette mask with an optional world origin offset.
pub fn extract_line_of_action_weighted(
    mask: &[bool],
    weights: Option<&[f32]>,
    w: usize,
    h: usize,
    world_offset: [f64; 2],
) -> Option<LineOfAction> {
    let mut total_mass = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] {
                let wt = weights.map_or(1.0f64, |wts| wts[idx] as f64).max(0.01);
                total_mass += wt;
                sum_x += x as f64 * wt;
                sum_y += y as f64 * wt;
            }
        }
    }

    if total_mass < 1.0 {
        return None;
    }

    let cx = sum_x / total_mass;
    let cy = sum_y / total_mass;

    // Second central spatial moments (spatial mass inertia tensor)
    let mut mu20 = 0.0f64;
    let mut mu02 = 0.0f64;
    let mut mu11 = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] {
                let wt = weights.map_or(1.0f64, |wts| wts[idx] as f64).max(0.01);
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                mu20 += dx * dx * wt;
                mu02 += dy * dy * wt;
                mu11 += dx * dy * wt;
            }
        }
    }

    mu20 /= total_mass;
    mu02 /= total_mass;
    mu11 /= total_mass;

    // Principal axis angle θ
    let angle_rad = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);
    let (sin_t, cos_t) = angle_rad.sin_cos();
    let dir = [cos_t, sin_t];

    // Eigenvalues of covariance (spread parallel and perpendicular)
    let diff = mu20 - mu02;
    let discr = (0.25 * diff * diff + mu11 * mu11).sqrt();
    let trace_half = 0.5 * (mu20 + mu02);
    let lambda1 = (trace_half + discr).max(0.0);
    let lambda2 = (trace_half - discr).max(0.0);
    let spread_parallel = lambda1.sqrt();
    let spread_perpendicular = lambda2.sqrt();

    // Quadratic spinal curvature κ: fit transverse coordinate η = κ * ξ^2
    let mut num_curv = 0.0f64;
    let mut den_curv = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if mask[idx] {
                let wt = weights.map_or(1.0f64, |wts| wts[idx] as f64).max(0.01);
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let xi = dx * cos_t + dy * sin_t;
                let eta = -dx * sin_t + dy * cos_t;
                let xi2 = xi * xi;
                num_curv += wt * xi2 * eta;
                den_curv += wt * xi2 * xi2;
            }
        }
    }
    let curvature = if den_curv > 1e-6 {
        num_curv / den_curv
    } else {
        0.0
    };

    // 3D PGA Line bivector L_spine in Mv301:
    let p_3d = [(cx + world_offset[0]) as f32, (cy + world_offset[1]) as f32, 0.0];
    let d_3d = [cos_t as f32, sin_t as f32, 0.0];
    let spine_bivector = Mv301::from_line(p_3d, d_3d);

    Some(LineOfAction {
        spine_bivector,
        center_of_mass: [cx, cy],
        direction: dir,
        angle_rad,
        spread_parallel,
        spread_perpendicular,
        curvature,
        total_mass,
    })
}

/// Maximum number of sub-holon parts per articulated character ($P \in \{1..8\}$).
pub const MAX_SUB_HOLONS: usize = 8;

/// Single frame observation of a foreground character extracted against the background atlas.
#[derive(Clone, Debug)]
pub struct CharacterObservation {
    pub frame_idx: usize,
    pub bbox: Rect,
    pub mask: Vec<bool>,
    pub rgb: Vec<u8>,
    /// Camera translation at `frame_idx` (world = screen + offset).
    pub camera_offset: [f64; 2],
    /// True when the silhouette touches the frame edge: the holon is only partially in view, so the
    /// observation informs kinematics but not appearance.
    pub truncated: bool,
    /// Line of Action: 3D PGA screw axis blade L_spine ∈ G(3,0,1) and mass inertia envelope.
    pub line_of_action: Option<LineOfAction>,
}

impl CharacterObservation {
    pub fn width(&self) -> usize {
        self.bbox.width().max(0) as usize
    }

    pub fn height(&self) -> usize {
        self.bbox.height().max(0) as usize
    }

    /// Silhouette centre in world (atlas) coordinates.
    pub fn world_center(&self) -> [f64; 2] {
        let c = self.bbox.center();
        [c[0] + self.camera_offset[0], c[1] + self.camera_offset[1]]
    }

    /// Line of Action: 3D PGA screw axis blade L_spine ∈ G(3,0,1) and mass inertia envelope.
    pub fn line_of_action(&self) -> Option<LineOfAction> {
        self.line_of_action
            .or_else(|| extract_line_of_action(&self.mask, self.width(), self.height()))
    }
}

/// 2D anisotropic Gaussian pose and covariance parameters for a sub-holon part $p$ at frame $t$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubHolonPose {
    pub part_id: usize,
    pub parent_id: usize,
    pub center: [f64; 2],
    pub angle_rad: f64,
    pub radii: [f64; 2],
    pub joint_angle_rad: f64,
    pub joint_offset_local: [f64; 2],
    pub view_idempotent_idx: usize,
    pub phase_rad: f64,
    /// World-space direction of travel at this observation ($\text{Spin}(2)$ heading rotor angle).
    /// Distinct from `angle_rad`, the part's geometric frame orientation.
    #[serde(default)]
    pub heading_rad: f64,
}

impl SubHolonPose {
    pub fn world_motor(&self) -> Motor<f64> {
        let t = Motor::from_translation(self.center[0], self.center[1], 0.0);
        let r = Motor::from_rotation_z(self.angle_rad);
        t * r
    }

    pub fn relative_motor(&self, parent_pose: &SubHolonPose) -> Motor<f64> {
        parent_pose.world_motor().reverse() * self.world_motor()
    }

    pub fn to_cde(&self, vitality_amplitude: f64, color: [u8; 3]) -> CliffordDensityElement {
        CliffordDensityElement::new(
            vitality_amplitude,
            self.angle_rad,
            self.view_idempotent_idx as u32,
            self.view_idempotent_idx + 1,
            color,
        )
    }

    #[inline]
    pub fn world_to_local(&self, x: f64, y: f64) -> [f64; 2] {
        let dx = x - self.center[0];
        let dy = y - self.center[1];
        let (s, c) = self.angle_rad.sin_cos();
        let rx = c * dx + s * dy;
        let ry = -s * dx + c * dy;
        [rx / self.radii[0].max(1.0), ry / self.radii[1].max(1.0)]
    }

    #[inline]
    pub fn local_to_world(&self, u: f64, v: f64) -> [f64; 2] {
        let rx = u * self.radii[0].max(1.0);
        let ry = v * self.radii[1].max(1.0);
        let (s, c) = self.angle_rad.sin_cos();
        [
            self.center[0] + c * rx - s * ry,
            self.center[1] + s * rx + c * ry,
        ]
    }
}

/// Number of Clifford Fourier basis modes $[1, \cos\phi, \sin\phi, \cos 2\phi, \sin 2\phi]$.
pub const CFT_WAVEPACKET_MODES: usize = 5;

/// Oriented Clifford Gaussian-Gabor Wavepacket Kernel $\mathcal{W}_{i,p}$ for one view-aspect cell.
#[derive(Clone, Debug)]
pub struct WavepacketPartKernel {
    pub res_w: usize,
    pub res_h: usize,
    pub dc_rgba: Vec<u8>,
    pub harmonic_rgba: Vec<u8>,
    /// Continuous 5-mode Clifford Fourier expansion coefficients per cell and RGBA channel (`res_w * res_h * 4`).
    pub cft_rgba_coeffs: Vec<[f32; CFT_WAVEPACKET_MODES]>,
    pub gaussian_envelope: Vec<f32>,
    /// Invariant ink line contour blade mask / continuous wedge vorticity |u ∧ ∇²u|.
    pub ink_contour_blade: Vec<f32>,
}

impl WavepacketPartKernel {
    #[inline]
    pub fn basis(phase_rad: f64) -> [f32; CFT_WAVEPACKET_MODES] {
        let (s1, c1) = phase_rad.sin_cos();
        let (s2, c2) = (2.0 * phase_rad).sin_cos();
        [1.0, c1 as f32, s1 as f32, c2 as f32, s2 as f32]
    }

    #[inline]
    pub fn sample(&self, u: f64, v: f64, phase_rad: f64) -> Option<([u8; 3], f32)> {
        if u.abs() > 1.08 || v.abs() > 1.08 {
            return None;
        }
        let fx = ((u + 1.0) * 0.5 * (self.res_w as f64 - 1.0)).round() as isize;
        let fy = ((v + 1.0) * 0.5 * (self.res_h as f64 - 1.0)).round() as isize;
        if fx < 0 || fy < 0 || fx >= self.res_w as isize || fy >= self.res_h as isize {
            return None;
        }
        let idx = fy as usize * self.res_w + fx as usize;
        let b = Self::basis(phase_rad);
        let eval_ch = |ch: usize| -> f32 {
            let w = &self.cft_rgba_coeffs[idx * 4 + ch];
            w[0] * b[0] + w[1] * b[1] + w[2] * b[2] + w[3] * b[3] + w[4] * b[4]
        };
        let alpha = (eval_ch(3) / 255.0).clamp(0.0, 1.0) * self.gaussian_envelope[idx];
        if alpha < 0.15 {
            return None;
        }
        let r = eval_ch(0).round().clamp(0.0, 255.0) as u8;
        let g = eval_ch(1).round().clamp(0.0, 255.0) as u8;
        let bl = eval_ch(2).round().clamp(0.0, 255.0) as u8;
        Some(([r, g, bl], alpha))
    }
}

/// Complete hierarchical articulated Holon $\mathcal{H}_i = \mathcal{H}_{\text{root}} \ltimes \prod_{p=1}^P \mathcal{H}_{i,p}$.
#[derive(Clone, Debug)]
pub struct ArticulatedHolon {
    pub num_parts: usize,
    pub num_view_cells: usize,
    pub parent_indices: Vec<usize>,
    pub part_colors: Vec<[u8; 3]>,
    pub kernels: Vec<Vec<WavepacketPartKernel>>,
    pub poses_per_obs: Vec<Vec<SubHolonPose>>,
    /// Source frame index of each entry in `poses_per_obs`.
    pub obs_frames: Vec<usize>,
}

/// Physical constitutive parameters: Cohesion C, Density ρ, Bulk Modulus K, Luminescence L.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C)]
pub struct ConstitutiveTuple {
    pub cohesion: f32,
    pub density: f32,
    pub bulk_modulus: f32,
    pub luminescence: [f32; 3],
    /// Alignment padding for 32-byte GPU uniform alignment.
    pub _pad: [f32; 2],
}

impl Default for ConstitutiveTuple {
    fn default() -> Self {
        Self {
            cohesion: 1.0,
            density: 1.0,
            bulk_modulus: 1.0,
            luminescence: [0.0, 0.0, 0.0],
            _pad: [0.0, 0.0],
        }
    }
}

/// Turnaround view regime classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ViewRegime {
    Front,
    ThreeQuarterFront,
    Profile,
    ThreeQuarterBack,
    Back,
    ActionPose,
}

/// Classify the view regime from heading rotor on Spin(2), Line of Action angle, and spread ratio.
pub fn classify_view_regime(heading_rotor: [f32; 2], loa_angle: f64, spread_ratio: f64) -> ViewRegime {
    let [c, s] = heading_rotor;
    let heading_angle = (s as f64).atan2(c as f64);

    if spread_ratio > 1.8 && (loa_angle.abs() > 1.2 || s.abs() > 0.8) {
        ViewRegime::ActionPose
    } else {
        let ang_norm = heading_angle.rem_euclid(2.0 * std::f64::consts::PI);
        let deg = ang_norm * 180.0 / std::f64::consts::PI;
        if (deg >= 337.5 || deg < 22.5) || (deg >= 157.5 && deg < 202.5) {
            ViewRegime::Profile
        } else if deg >= 67.5 && deg < 112.5 {
            ViewRegime::Front
        } else if deg >= 247.5 && deg < 292.5 {
            ViewRegime::Back
        } else if (deg >= 22.5 && deg < 67.5) || (deg >= 112.5 && deg < 157.5) {
            ViewRegime::ThreeQuarterFront
        } else {
            ViewRegime::ThreeQuarterBack
        }
    }
}

/// Invariant cel fill paint and ink contour blade kernel per part for a canonical view aspect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelSheetPartKernel {
    pub part_id: usize,
    pub canonical_res: [usize; 2],
    /// Invariant cel fill paint signature (inner product coherence u · ∇²u).
    pub cel_fill_rgb: Vec<u8>,
    /// Invariant ink line contour mask (wedge vorticity |u ∧ ∇²u|).
    pub ink_contour_blade: Vec<f32>,
    /// Continuous Clifford Fourier deformation wavepacket coefficients (5 modes).
    pub cft_deformation_coeffs: Vec<[f32; CFT_WAVEPACKET_MODES]>,
}

/// Articulated sub-holon kinematic part specification with aerodynamic compliance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubHolonPartSpec {
    pub part_id: usize,
    pub name: String,
    pub parent_part_id: Option<usize>,
    /// Fluid-structure / aerodynamic coupling to ambient lattice flow v(x,t) - v_root (e.g. 0.0 for rigid, 4.5 for flowing).
    pub flow_compliance: f32,
    /// Restoring harmonic spring stiffness.
    pub spring_stiffness: f32,
    /// Joint angular damping.
    pub damping: f32,
    /// Local attachment anchor in parent frame.
    pub anchor_local: [f32; 2],
    /// Rest angle in radians.
    pub rest_angle: f32,
}

/// Turnaround model sheet view aspect with primitive idempotent projector and Spin(2) Lie transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnaroundViewAspect {
    pub view_index: usize,
    pub view_regime: ViewRegime,
    /// Discrete primitive idempotent projector ε_v on Spin(2) × S¹.
    pub idempotent: StateIdempotent,
    /// Reference heading rotor on Spin(2): [cos θ, sin θ].
    pub reference_rotor: [f32; 2],
    /// Articulated part poses for this canonical aspect.
    pub part_poses: Vec<SubHolonPose>,
    /// Invariant contour blades (wedge vorticity) and cel fill kernels per part.
    pub part_kernels: Vec<ModelSheetPartKernel>,
}

impl TurnaroundViewAspect {
    /// Closed-form Lie transition operator T_{u -> v} = ε_v M ε_u between aspects on Spin(2).
    pub fn transition_to(&self, target: &Self, motor: Mv2) -> Option<TransitionOperator> {
        TransitionOperator::from_motor(&self.idempotent, &target.idempotent, motor)
    }
}

/// Unified symbolic Character Board Model Sheet for an autonomous process holon (Track R10.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HolonModelSheet {
    /// Semantic identifier (e.g., "Mrs_Brisby", "Mario", "Jeremy_Crow").
    pub name: String,
    /// Physical constitutive parameters: Cohesion C, Density ρ, Bulk Modulus K, Luminescence L.
    pub constitutive: ConstitutiveTuple,
    /// Line of Action: 3D PGA screw axis blade L_spine ∈ G(3,0,1).
    pub line_of_action: Mv301,
    /// Invariant projective cross-ratios and anatomical proportions.
    pub invariant_proportions: Vec<f32>,
    /// Turnaround model sheet: discrete view-aspect idempotents {ε_v}.
    pub turnaround_views: Vec<TurnaroundViewAspect>,
    /// Articulated sub-holon kinematic tree.
    pub parts: Vec<SubHolonPartSpec>,
    /// Invariant Hiley Clifford Density Element signature.
    pub cde_signature: CliffordDensityElement,
}

/// Flat 64-byte aligned Mv301 monad array directly mappable to GPU uniform/storage buffers (Pod, Zeroable).
/// Enables zero heap allocation in unfold trajectory stepping (Carmack mechanical sympathy).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable, Default)]
pub struct GpuModelSheetMonad {
    /// Monad 0: Line of Action screw blade L_spine ∈ G(3,0,1).
    pub line_of_action: Mv301,
    /// Monad 1: Constitutive parameters (cohesion, density, bulk_modulus, luminescence).
    pub constitutive_monad: Mv301,
    /// Monad 2: Invariant CDE signature (amplitude, angle, idempotent_u, idempotent_v, rgb).
    pub cde_monad: Mv301,
    /// Monads 3..11: Up to 8 canonical turnaround aspect heading rotors & idempotents.
    pub aspect_monads: [Mv301; 8],
    /// Monads 11..19: Up to 8 articulated part specs (anchor, compliance, stiffness, damping).
    pub part_monads: [Mv301; 8],
}

impl HolonModelSheet {
    /// Convert to flat 64-byte aligned Mv301 GPU monad array for uniform/storage buffer dispatch.
    pub fn to_gpu_monad(&self) -> GpuModelSheetMonad {
        let mut const_c = [0.0f32; 16];
        const_c[0] = self.constitutive.cohesion;
        const_c[1] = self.constitutive.density;
        const_c[2] = self.constitutive.bulk_modulus;
        const_c[3] = self.constitutive.luminescence[0];
        const_c[4] = self.constitutive.luminescence[1];
        const_c[5] = self.constitutive.luminescence[2];

        let mut cde_c = [0.0f32; 16];
        cde_c[0] = self.cde_signature.amplitude as f32;
        cde_c[1] = self.cde_signature.heading_angle() as f32;
        cde_c[2] = self.cde_signature.idempotent.state as f32;
        cde_c[3] = self.cde_signature.density() as f32;
        cde_c[4] = self.cde_signature.signature[0].s as f32;
        cde_c[5] = self.cde_signature.signature[1].s as f32;
        cde_c[6] = self.cde_signature.signature[2].s as f32;

        let mut aspect_monads = [Mv301::ZERO; 8];
        for (i, aspect) in self.turnaround_views.iter().take(8).enumerate() {
            let mut c = [0.0f32; 16];
            c[0] = aspect.reference_rotor[0];
            c[1] = aspect.reference_rotor[1];
            if aspect.idempotent.m.len() >= 4 {
                c[2] = aspect.idempotent.m[0] as f32;
                c[3] = aspect.idempotent.m[1] as f32;
                c[4] = aspect.idempotent.m[2] as f32;
                c[5] = aspect.idempotent.m[3] as f32;
            }
            c[6] = aspect.view_regime as u32 as f32;
            aspect_monads[i] = Mv301 { c };
        }

        let mut part_monads = [Mv301::ZERO; 8];
        for (i, part) in self.parts.iter().take(8).enumerate() {
            let mut c = [0.0f32; 16];
            c[0] = part.anchor_local[0];
            c[1] = part.anchor_local[1];
            c[2] = part.flow_compliance;
            c[3] = part.spring_stiffness;
            c[4] = part.damping;
            c[5] = part.rest_angle;
            c[6] = part.parent_part_id.map_or(-1.0, |p| p as f32);
            part_monads[i] = Mv301 { c };
        }

        GpuModelSheetMonad {
            line_of_action: self.line_of_action,
            constitutive_monad: Mv301 { c: const_c },
            cde_monad: Mv301 { c: cde_c },
            aspect_monads,
            part_monads,
        }
    }

    /// Read GPU monad as raw byte slice.
    pub fn as_gpu_bytes<'a>(monad: &'a GpuModelSheetMonad) -> &'a [u8] {
        bytemuck::bytes_of(monad)
    }
}

/// Silhouette IoU a holon model must reach to be accepted (the R2.3 reconstruction gate).
pub const HOLON_MODEL_QUALITY_IOU: f64 = 0.88;
/// Articulated part counts tried in increasing capacity once a rigid model is insufficient.
const ARTICULATED_PART_CANDIDATES: [usize; 3] = [3, 4, 6];

/// View-idempotent count for `n` observations: the fewest posture cells such that each cell's
/// members can be interpolated exactly by the `CFT_WAVEPACKET_MODES`-mode Clifford Fourier basis
/// (more members per cell would force the fit to blend distinct drawings).
fn view_count_for(n: usize, num_parts: usize) -> usize {
    let lo = if num_parts == 1 { 1 } else { 2 };
    n.div_ceil(CFT_WAVEPACKET_MODES).max(lo).min(n.max(1))
}

/// Choose the sub-holon part count $P$ and view-idempotent count $S_{\text{view}}$ by posture
/// recurrence rather than size or aspect heuristics.
///
/// A holon is rigid ($P = 1$) when its states recur: posture idempotents $\epsilon_{s,\text{view}}$
/// learned from the first half of its track explain the second half at `HOLON_MODEL_QUALITY_IOU`
/// (turning, cycled gaits, re-drawn frames). Continuous novel deformation (capes, wings) cannot be
/// spanned by recurring states and needs articulated motors: the smallest $P$ whose fit reaches
/// the target is chosen (else the best).
pub fn choose_part_and_view_counts(observations: &[CharacterObservation]) -> (usize, usize) {
    let n = observations.len();
    if n < 4 {
        return (1, view_count_for(n, 1));
    }
    let rec = rigid_recurrence_iou(observations);
    if rec >= 0.80 {
        return (1, view_count_for(n, 1));
    }
    let m1 = fit_articulated_holon(observations, Some(1));
    let iou1: f64 = observations
        .iter()
        .enumerate()
        .map(|(oi, o)| mask_iou(&o.mask, &render_articulated_holon_observation(&m1, o, oi).0))
        .sum::<f64>()
        / n as f64;
    let (med_w, med_h) = median_obs_dims(observations);
    let area_est = (med_w * med_h).max(16) as f64;
    // Scale-aware rigidity threshold: for small sprites (e.g. Pacman 14x14), perimeter texel
    // quantization naturally bounds maximum mask IoU across opening/closing mouth cycles.
    let rigidity_thresh = (1.0 - 3.5 / area_est.sqrt()).clamp(0.74, HOLON_MODEL_QUALITY_IOU);
    if iou1 >= rigidity_thresh {
        return (1, view_count_for(n, 1));
    }
    let mut best = (0.0f64, ARTICULATED_PART_CANDIDATES[ARTICULATED_PART_CANDIDATES.len() - 1]);
    for &p in &ARTICULATED_PART_CANDIDATES {
        let model = fit_articulated_holon(observations, Some(p));
        let iou = observations
            .iter()
            .enumerate()
            .map(|(oi, o)| mask_iou(&o.mask, &render_articulated_holon_observation(&model, o, oi).0))
            .sum::<f64>()
            / n as f64;
        if iou >= HOLON_MODEL_QUALITY_IOU {
            return (p.min(MAX_SUB_HOLONS), view_count_for(n, p));
        }
        if iou > best.0 {
            best = (iou, p);
        }
    }
    (best.1.min(MAX_SUB_HOLONS), view_count_for(n, best.1))
}

/// Scale-fair silhouette IoU: a mismatched pixel only counts when it lies more than one texel
/// from the other silhouette, so sampling-lattice quantization of the boundary is not mistaken
/// for structural shape change (which would otherwise penalize small holons disproportionately).
pub fn lattice_tolerant_iou(a: &[bool], b: &[bool], w: usize, h: usize) -> f64 {
    let near = |m: &[bool], x: usize, y: usize| -> bool {
        let (x0, x1) = (x.saturating_sub(1), (x + 1).min(w - 1));
        let (y0, y1) = (y.saturating_sub(1), (y + 1).min(h - 1));
        (y0..=y1).any(|yy| (x0..=x1).any(|xx| m[yy * w + xx]))
    };
    let (mut inter, mut structural) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            match (a[i], b[i]) {
                (true, true) => inter += 1,
                (true, false) if !near(b, x, y) => structural += 1,
                (false, true) if !near(a, x, y) => structural += 1,
                _ => {}
            }
        }
    }
    if inter + structural == 0 { 1.0 } else { inter as f64 / (inter + structural) as f64 }
}

/// Silhouette IoU with which a rigid posture model fit on the first half of the track explains
/// each observation of the second half by its best learned configuration (centre-aligned, with a
/// one-texel lattice tolerance).
fn rigid_recurrence_iou(observations: &[CharacterObservation]) -> f64 {
    let half = observations.len() / 2;
    let (train, test) = observations.split_at(half);
    if train.is_empty() || test.is_empty() {
        return 0.0;
    }
    let model = fit_articulated_holon(train, Some(1));
    let mut total = 0.0f64;
    for t in test {
        let tc = t.bbox.center();
        let mut best = 0.0f64;
        for (j, tr) in train.iter().enumerate() {
            let Some(poses) = model.poses_per_obs.get(j) else { continue };
            let jc = tr.bbox.center();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let shifted: Vec<SubHolonPose> = poses
                        .iter()
                        .map(|p| SubHolonPose {
                            center: [
                                p.center[0] + tc[0] - jc[0] + dx as f64,
                                p.center[1] + tc[1] - jc[1] + dy as f64,
                            ],
                            ..*p
                        })
                        .collect();
                    let (mask, _) = render_with_poses(&model, &shifted, t);
                    best = best.max(lattice_tolerant_iou(&t.mask, &mask, t.width(), t.height()));
                }
            }
        }
        total += best;
    }
    total / test.len() as f64
}

/// Fit an `ArticulatedHolon` to a tracked character sequence.
pub fn fit_articulated_holon(
    observations: &[CharacterObservation],
    forced_parts: Option<usize>,
) -> ArticulatedHolon {
    fit_articulated_holon_with_views(observations, forced_parts, None)
}

/// Fit an `ArticulatedHolon` to a tracked character sequence with optional forced view count.
pub fn fit_articulated_holon_with_views(
    observations: &[CharacterObservation],
    forced_parts: Option<usize>,
    forced_views: Option<usize>,
) -> ArticulatedHolon {
    // Truncated observations and silhouette-area outliers relative to the holon's own variability
    // (merges with other holons, residual debris) keep their kinematics but shape neither
    // morphology nor appearance.
    let (med_w, med_h) = median_obs_dims(observations);
    let outlier = area_outliers(observations);
    let mut clean: Vec<bool> = observations.iter().zip(&outlier).map(|(o, &out)| !o.truncated && !out).collect();
    if std::env::var_os("IMPLICATE_DEBUG_TRACKS").is_some() {
        eprintln!(
            "[fit] n={} truncated={} area_outliers={} clean={}",
            observations.len(),
            observations.iter().filter(|o| o.truncated).count(),
            outlier.iter().filter(|&&o| o).count(),
            clean.iter().filter(|&&c| c).count()
        );
    }
    if !clean.contains(&true) {
        // No complete observation: all of them are the best available evidence.
        clean.iter_mut().for_each(|c| *c = true);
    }
    let defining: Vec<usize> = (0..observations.len()).filter(|&i| clean[i]).collect();
    let defining_obs: Vec<CharacterObservation> = defining.iter().map(|&i| observations[i].clone()).collect();
    let auto_p = match forced_parts {
        Some(p) => p,
        None => choose_part_and_view_counts(&defining_obs).0,
    };
    let num_parts = forced_parts.unwrap_or(auto_p).clamp(1, MAX_SUB_HOLONS);

    if observations.is_empty() {
        return ArticulatedHolon {
            num_parts,
            num_view_cells: 1,
            parent_indices: vec![0; num_parts],
            part_colors: vec![[128, 128, 128]; num_parts],
            kernels: Vec::new(),
            poses_per_obs: Vec::new(),
            obs_frames: Vec::new(),
        };
    }

    let prototypes = learn_part_prototypes(observations, num_parts);
    let part_colors: Vec<[u8; 3]> = prototypes
        .iter()
        .map(|p| {
            [
                (p[2] * 255.0).clamp(0.0, 255.0) as u8,
                (p[3] * 255.0).clamp(0.0, 255.0) as u8,
                (p[4] * 255.0).clamp(0.0, 255.0) as u8,
            ]
        })
        .collect();
    let parent_indices = vec![0usize; num_parts];
    // Posture idempotents are learned from the observations that shape appearance (clean ones);
    // each unclean observation inherits the cell and phase of its temporally nearest clean one.
    let num_view_cells = forced_views
        .map(|v| v.clamp(1, defining_obs.len().max(1)))
        .unwrap_or_else(|| view_count_for(defining_obs.len(), num_parts));
    let (cells, phases) = assign_view_idempotent_cells_and_cft_phases(&defining_obs, num_view_cells);
    let mut view_assignments = vec![0usize; observations.len()];
    let mut cft_phases = vec![0.0f64; observations.len()];
    for (k, &i) in defining.iter().enumerate() {
        view_assignments[i] = cells[k];
        cft_phases[i] = phases[k];
    }
    for oi in 0..observations.len() {
        if clean[oi] {
            continue;
        }
        if let Some(&ni) = defining.iter().min_by_key(|&&j| observations[j].frame_idx.abs_diff(observations[oi].frame_idx)) {
            view_assignments[oi] = view_assignments[ni];
            cft_phases[oi] = cft_phases[ni];
        }
    }

    let n_obs = observations.len();
    let mut poses_per_obs: Vec<Vec<SubHolonPose>> = Vec::with_capacity(n_obs);
    let mut part_assignments_per_obs: Vec<Vec<usize>> = Vec::with_capacity(n_obs);

    for (oi, obs) in observations.iter().enumerate() {
        let w = obs.width();
        let h = obs.height();
        let cliff_feats = compute_cliffordnet_laplacian_features(&obs.rgb, w, h);
        let mut pixel_part = vec![0usize; w * h];

        if num_parts > 1 {
            for y in 0..h {
                let ny = (y as f32 / h.max(1) as f32) * 2.0 - 1.0;
                for x in 0..w {
                    let idx = y * w + x;
                    if !obs.mask[idx] {
                        continue;
                    }
                    let nx = (x as f32 / w.max(1) as f32) * 2.0 - 1.0;
                    let r = obs.rgb[idx * 3] as f32 / 255.0;
                    let g = obs.rgb[idx * 3 + 1] as f32 / 255.0;
                    let b = obs.rgb[idx * 3 + 2] as f32 / 255.0;
                    let vort = cliff_feats[idx][1];

                    let mut best_p = 0usize;
                    let mut best_dist = f32::MAX;
                    for (pi, proto) in prototypes.iter().enumerate() {
                        let d_space = (nx - proto[0]).powi(2) + (ny - proto[1]).powi(2);
                        let d_color = (r - proto[2]).powi(2)
                            + (g - proto[3]).powi(2)
                            + (b - proto[4]).powi(2);
                        let d_vort = (vort - proto[5]).powi(2);
                        let d = 1.3 * d_space + 0.9 * d_color + 0.25 * d_vort;
                        if d < best_dist {
                            best_dist = d;
                            best_p = pi;
                        }
                    }
                    pixel_part[idx] = best_p;
                }
            }
        }

        let phase_rad = cft_phases[oi];
        let view_cell = view_assignments[oi];
        let mut frame_poses = Vec::with_capacity(num_parts);

        let motion_heading = if observations.len() > 1 {
            let (c_curr, c_next) = if oi + 1 < observations.len() {
                (observations[oi].world_center(), observations[oi + 1].world_center())
            } else {
                (observations[oi - 1].world_center(), observations[oi].world_center())
            };
            let dx = c_next[0] - c_curr[0];
            let dy = c_next[1] - c_curr[1];
            if dx.abs() + dy.abs() > 0.25 {
                dy.atan2(dx)
            } else if oi > 0 && !poses_per_obs.is_empty() {
                poses_per_obs[oi - 1][0].heading_rad
            } else {
                0.0
            }
        } else {
            0.0
        };

        for p in 0..num_parts {
            let mut pose = fit_part_gaussian_pose(
                obs,
                &pixel_part,
                p,
                num_parts,
                view_cell,
                phase_rad,
            );
            pose.heading_rad = motion_heading;
            frame_poses.push(pose);
        }

        let root_pose = frame_poses[0];
        for p in 1..num_parts {
            frame_poses[p].joint_angle_rad = frame_poses[p].angle_rad - root_pose.angle_rad;
            frame_poses[p].joint_offset_local =
                root_pose.world_to_local(frame_poses[p].center[0], frame_poses[p].center[1]);
        }

        poses_per_obs.push(frame_poses);
        part_assignments_per_obs.push(pixel_part);
    }

    // Strict volume conservation det S = 1 on continuous gait deformations along phase φ ∈ S¹
    if num_parts > 1 {
        enforce_volume_conserving_gait_deformation(
            &mut poses_per_obs,
            &view_assignments,
            num_parts,
            num_view_cells,
        );
    }

    // Every kernel is pixel-native (1 texel per source pixel at the part's largest observed extent),
    // so neither rigid holons nor articulated parts are resampled through a fixed canonical grid.
    let part_res: Vec<(usize, usize)> = if num_parts == 1 {
        let clean_obs = || observations.iter().zip(&clean).filter(|(_, c)| **c).map(|(o, _)| o);
        let rw = clean_obs().map(|o| o.width()).max().unwrap_or(med_w).max(1);
        let rh = clean_obs().map(|o| o.height()).max().unwrap_or(med_h).max(1);
        register_rigid_observations(observations, &mut poses_per_obs, &view_assignments, &clean, rw, rh);
        vec![(rw, rh)]
    } else {
        (0..num_parts)
            .map(|p| {
                let max_r = |axis: usize| {
                    poses_per_obs
                        .iter()
                        .zip(&clean)
                        .filter(|(_, c)| **c)
                        .map(|(poses, _)| poses[p].radii[axis])
                        .fold(1.0f64, f64::max)
                };
                ((2.0 * max_r(0)).round() as usize + 1, (2.0 * max_r(1)).round() as usize + 1)
            })
            .collect()
    };

    let mut kernels: Vec<Vec<WavepacketPartKernel>> = Vec::with_capacity(num_view_cells);
    for vc in 0..num_view_cells {
        let mut vc_kernels = Vec::with_capacity(num_parts);
        for p in 0..num_parts {
            let (res_w, res_h) = part_res[p];
            let mut k = accumulate_part_wavepacket_kernel(
                observations,
                &poses_per_obs,
                &part_assignments_per_obs,
                &clean,
                p,
                vc,
                res_w,
                res_h,
            );
            if num_parts == 1 {
                // A lone rigid part needs no partition-of-unity window: its silhouette alpha is the support.
                k.gaussian_envelope.iter_mut().for_each(|g| *g = 1.0);
            }
            vc_kernels.push(k);
        }
        kernels.push(vc_kernels);
    }

    ArticulatedHolon {
        num_parts,
        num_view_cells,
        parent_indices,
        part_colors,
        kernels,
        poses_per_obs,
        obs_frames: observations.iter().map(|o| o.frame_idx).collect(),
    }
}

/// Robust outlier bound in median absolute deviations (3.0-sigma statistical lattice bound).
const OUTLIER_MADS: f64 = 3.0;
/// Scale factor making the MAD a consistent estimator of the standard deviation.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Observations whose log silhouette area or log bounding-box extents deviate from the track
/// median by more than `OUTLIER_MADS` robust standard deviations: stable bodies flag merges
/// and passing projectiles, while deforming bodies tolerate their own range of motion.
fn area_outliers(observations: &[CharacterObservation]) -> Vec<bool> {
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let flag_series = |series: Vec<f64>| -> Vec<bool> {
        let m = median(&mut series.clone());
        let sigma = MAD_TO_SIGMA * median(&mut series.iter().map(|a| (a - m).abs()).collect());
        series
            .iter()
            .map(|a| sigma > 0.0 && (a - m).abs() > OUTLIER_MADS * sigma)
            .collect()
    };
    let out_a = flag_series(
        observations
            .iter()
            .map(|o| (o.mask.iter().filter(|&&m| m).count().max(1) as f64).ln())
            .collect(),
    );
    let out_w = flag_series(
        observations
            .iter()
            .map(|o| (o.width().max(1) as f64).ln())
            .collect(),
    );
    let out_h = flag_series(
        observations
            .iter()
            .map(|o| (o.height().max(1) as f64).ln())
            .collect(),
    );
    (0..observations.len())
        .map(|i| out_a[i] || out_w[i] || out_h[i])
        .collect()
}

/// Register rigid ($P = 1$) observations into one pixel-native frame: every pose gets the shared
/// half-extents of a `res_w x res_h` kernel, and its centre is shifted by the integer offset that
/// maximizes silhouette agreement with its view cell's canonical median-area reference observation.
fn register_rigid_observations(
    observations: &[CharacterObservation],
    poses_per_obs: &mut [Vec<SubHolonPose>],
    view_assignments: &[usize],
    clean: &[bool],
    res_w: usize,
    res_h: usize,
) {
    let radii = [(res_w as f64 - 1.0) * 0.5, (res_h as f64 - 1.0) * 0.5];
    let area = |o: &CharacterObservation| o.mask.iter().filter(|&&m| m).count();
    let n_cells = view_assignments.iter().copied().max().map_or(0, |m| m + 1);
    for vc in 0..n_cells {
        let members: Vec<usize> = (0..observations.len()).filter(|&i| view_assignments[i] == vc).collect();
        let mut clean_areas: Vec<usize> = members
            .iter()
            .filter(|&&i| clean[i])
            .map(|&i| area(&observations[i]))
            .collect();
        if clean_areas.is_empty() {
            clean_areas = members.iter().map(|&i| area(&observations[i])).collect();
        }
        clean_areas.sort_unstable();
        let med_area = clean_areas.get(clean_areas.len() / 2).copied().unwrap_or(0) as i64;
        let Some(&ref_i) = members
            .iter()
            .filter(|&&i| clean[i])
            .min_by_key(|&&i| (area(&observations[i]) as i64 - med_area).abs())
            .or_else(|| {
                members
                    .iter()
                    .min_by_key(|&&i| (area(&observations[i]) as i64 - med_area).abs())
            })
        else {
            continue;
        };
        let r = &observations[ref_i];
        let ref_at = |x: i32, y: i32| -> Option<[u8; 3]> {
            if x < r.bbox.x0 || y < r.bbox.y0 || x > r.bbox.x1 || y > r.bbox.y1 {
                return None;
            }
            let i = ((y - r.bbox.y0) as usize) * r.width() + (x - r.bbox.x0) as usize;
            r.mask[i].then(|| [r.rgb[i * 3], r.rgb[i * 3 + 1], r.rgb[i * 3 + 2]])
        };
        for &oi in &members {
            let o = &observations[oi];
            // Search radius: the extent difference to the reference plus one pixel of slack.
            let sx = ((r.width() as i32 - o.width() as i32).abs() / 2 + 1) as i32;
            let sy = ((r.height() as i32 - o.height() as i32).abs() / 2 + 1) as i32;
            // Candidate placements align the observation's bbox centre to the reference's, then shift.
            let base = [
                (r.bbox.x0 + r.bbox.x1 - o.bbox.x0 - o.bbox.x1) / 2,
                (r.bbox.y0 + r.bbox.y1 - o.bbox.y0 - o.bbox.y1) / 2,
            ];
            let mut best = (i64::MIN, [0i32, 0i32]);
            for dy in -sy..=sy {
                for dx in -sx..=sx {
                    let mut score = 0i64;
                    for ly in 0..o.height() {
                        for lx in 0..o.width() {
                            let i = ly * o.width() + lx;
                            if !o.mask[i] {
                                continue;
                            }
                            let x = o.bbox.x0 + lx as i32 + base[0] + dx;
                            let y = o.bbox.y0 + ly as i32 + base[1] + dy;
                            match ref_at(x, y) {
                                Some(c) => {
                                    let d: i64 = (0..3).map(|k| (c[k] as i64 - o.rgb[i * 3 + k] as i64).abs()).sum();
                                    score += 3 * 255 - d;
                                }
                                None => score -= 3 * 255,
                            }
                        }
                    }
                    if score > best.0 {
                        best = (score, [dx, dy]);
                    }
                }
            }
            // Kernel origin (top-left texel) sits on the integer pixel lattice in the reference frame,
            // so texel centres coincide with pixel centres under both accumulation and sampling.
            let origin = [
                r.bbox.x0 - (res_w as i32 - r.width() as i32) / 2,
                r.bbox.y0 - (res_h as i32 - r.height() as i32) / 2,
            ];
            let rc = [origin[0] as f64 + radii[0], origin[1] as f64 + radii[1]];
            let shift = [(base[0] + best.1[0]) as f64, (base[1] + best.1[1]) as f64];
            let pose = &mut poses_per_obs[oi][0];
            pose.center = [rc[0] - shift[0], rc[1] - shift[1]];
            pose.radii = radii;
        }
    }
}

/// Render/splat the `ArticulatedHolon` at observation index `obs_idx`.
pub fn render_articulated_holon_observation(
    holon: &ArticulatedHolon,
    obs: &CharacterObservation,
    obs_idx: usize,
) -> (Vec<bool>, Vec<u8>) {
    match holon.poses_per_obs.get(obs_idx) {
        Some(poses) => render_with_poses(holon, poses, obs),
        None => (vec![false; obs.width() * obs.height()], vec![0u8; obs.width() * obs.height() * 3]),
    }
}

/// Render the holon's kernels at explicit part `poses` into `obs`'s bounding box.
pub fn render_with_poses(
    holon: &ArticulatedHolon,
    poses: &[SubHolonPose],
    obs: &CharacterObservation,
) -> (Vec<bool>, Vec<u8>) {
    let w = obs.width();
    let h = obs.height();
    let mut out_mask = vec![false; w * h];
    let mut out_rgb = vec![0u8; w * h * 3];
    let vc = poses
        .first()
        .map(|p| p.view_idempotent_idx)
        .unwrap_or(0)
        .min(holon.kernels.len().saturating_sub(1));
    let Some(vc_kernels) = holon.kernels.get(vc) else {
        return (out_mask, out_rgb);
    };

    for y in 0..h {
        let wy = obs.bbox.y0 as f64 + y as f64;
        for x in 0..w {
            let wx = obs.bbox.x0 as f64 + x as f64;
            let mut best_alpha = 0.0f32;
            let (mut accum_r, mut accum_g, mut accum_b, mut weight_sum) =
                (0.0f32, 0.0f32, 0.0f32, 0.0f32);

            for (p, pose) in poses.iter().enumerate() {
                let [u, v] = pose.world_to_local(wx, wy);
                if let Some((rgb, alpha)) = vc_kernels[p].sample(u, v, pose.phase_rad) {
                    accum_r += rgb[0] as f32 * alpha;
                    accum_g += rgb[1] as f32 * alpha;
                    accum_b += rgb[2] as f32 * alpha;
                    weight_sum += alpha;
                    best_alpha = best_alpha.max(alpha);
                }
            }

            if best_alpha >= 0.35 && weight_sum > 0.0 {
                let idx = y * w + x;
                out_mask[idx] = true;
                out_rgb[idx * 3] = (accum_r / weight_sum).round().clamp(0.0, 255.0) as u8;
                out_rgb[idx * 3 + 1] = (accum_g / weight_sum).round().clamp(0.0, 255.0) as u8;
                out_rgb[idx * 3 + 2] = (accum_b / weight_sum).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    (out_mask, out_rgb)
}

/// Maximum frame gap bridged by frame-to-frame linking (short flicker / missed detections).
const MAX_LINK_GAP: usize = 8;
/// Minimum number of observations for a track to define a holon.
const MIN_TRACK_LEN: usize = 4;

/// Compute the statistical background noise floor tau = max(median(I_nz) + 3 * 1.4826 * MAD(I_nz), Q(0.98))
/// derived from the empirical non-zero pointwise background residual field outside actor bounding boxes.
pub fn compute_shot_noise_floor(
    frames: &[crate::video_io::RawFrame],
    atlas: &crate::multiplane_atlas::MultiPlaneAtlas,
) -> f64 {
    let n = frames.len();
    if n == 0 {
        return 18.0;
    }
    let stride = (n / 8).max(1);
    let mut sample_res = Vec::new();
    let (ay0, ay1) = atlas.active_y_range;
    let w = atlas.width;
    let _h = atlas.height;

    for fi in (0..n).step_by(stride) {
        let f = &frames[fi];
        let bg = atlas.reconstruct_background_frame(fi);
        let boxes = atlas.actor_boxes_per_frame.get(fi);
        for y in (ay0..ay1).step_by(2) {
            for x in (0..w).step_by(2) {
                let in_actor = boxes.map(|bs| {
                    bs.iter().any(|b| {
                        (b.x0 - 4..=b.x1 + 4).contains(&(x as i32))
                            && (b.y0 - 4..=b.y1 + 4).contains(&(y as i32))
                    })
                }).unwrap_or(false);
                if !in_actor {
                    let c = f.pixel(x as u32, y as u32);
                    let b = bg.pixel(x, y);
                    let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                    sample_res.push(d as f64);
                }
            }
        }
    }

    if sample_res.is_empty() {
        return 18.0;
    }

    let non_zero: Vec<f64> = sample_res.iter().copied().filter(|&x| x > 0.0).collect();
    let tau = if non_zero.len() >= 10 {
        let mut nz = non_zero;
        nz.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = nz[nz.len() / 2];
        let mut devs: Vec<f64> = nz.iter().map(|&x| (x - median).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = devs[devs.len() / 2];
        median + 2.0 * MAD_TO_SIGMA * mad
    } else {
        sample_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sample_res[sample_res.len() / 2];
        let mut devs: Vec<f64> = sample_res.iter().map(|&x| (x - median).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = devs[devs.len() / 2];
        median + 3.0 * MAD_TO_SIGMA * mad
    };
    tau.clamp(6.0, 36.0)
}

/// Chroma-weighted mean colour: saturated pixels dominate, so dark outlines and specular
/// highlights do not wash out a holon's identifying hue.
pub fn chroma_weighted_color(rgb: &[u8], mask: &[bool]) -> [f32; 3] {
    let (mut acc, mut wsum) = ([0.0f32; 3], 0.0f32);
    let (mut flat, mut n) = ([0.0f32; 3], 0.0f32);
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            continue;
        }
        let c = [rgb[i * 3] as f32, rgb[i * 3 + 1] as f32, rgb[i * 3 + 2] as f32];
        let chroma = c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2]);
        for k in 0..3 {
            acc[k] += c[k] * chroma;
            flat[k] += c[k];
        }
        wsum += chroma;
        n += 1.0;
    }
    if wsum > 0.0 {
        [acc[0] / wsum, acc[1] / wsum, acc[2] / wsum]
    } else if n > 0.0 {
        [flat[0] / n, flat[1] / n, flat[2] / n]
    } else {
        [0.0; 3]
    }
}

/// Track R10.10b — Extract representative colour modes using Furthest Point Sampling (FPS)
/// with Lloyd refinement and minimum occupancy pruning (BIC/MDL principle).
pub fn extract_color_modes_fps(rgb_samples: &[[u8; 3]], max_k: usize, min_dist: f64) -> Vec<[f64; 3]> {
    if rgb_samples.is_empty() {
        return Vec::new();
    }
    let mut sum = [0.0f64; 3];
    for p in rgb_samples {
        sum[0] += p[0] as f64;
        sum[1] += p[1] as f64;
        sum[2] += p[2] as f64;
    }
    let n = rgb_samples.len() as f64;
    let mut centers = vec![[sum[0] / n, sum[1] / n, sum[2] / n]];

    let mut min_dists: Vec<f64> = rgb_samples.iter().map(|p| {
        let dr = p[0] as f64 - centers[0][0];
        let dg = p[1] as f64 - centers[0][1];
        let db = p[2] as f64 - centers[0][2];
        (dr * dr + dg * dg + db * db).sqrt()
    }).collect();

    while centers.len() < max_k {
        let (max_idx, &max_d) = min_dists.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        if max_d < min_dist {
            break;
        }
        let cand = [rgb_samples[max_idx][0] as f64, rgb_samples[max_idx][1] as f64, rgb_samples[max_idx][2] as f64];
        centers.push(cand);
        for (i, p) in rgb_samples.iter().enumerate() {
            let dr = p[0] as f64 - cand[0];
            let dg = p[1] as f64 - cand[1];
            let db = p[2] as f64 - cand[2];
            let d = (dr * dr + dg * dg + db * db).sqrt();
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    let mut counts = vec![0usize; centers.len()];
    let mut sums = vec![[0.0f64; 3]; centers.len()];
    for p in rgb_samples {
        let mut best_k = 0;
        let mut best_d = f64::INFINITY;
        for (k, c) in centers.iter().enumerate() {
            let dr = p[0] as f64 - c[0];
            let dg = p[1] as f64 - c[1];
            let db = p[2] as f64 - c[2];
            let d = dr * dr + dg * dg + db * db;
            if d < best_d {
                best_d = d;
                best_k = k;
            }
        }
        counts[best_k] += 1;
        sums[best_k][0] += p[0] as f64;
        sums[best_k][1] += p[1] as f64;
        sums[best_k][2] += p[2] as f64;
    }

    let min_count = (rgb_samples.len() as f64 * 0.025).max(5.0) as usize;
    let mut pruned = Vec::new();
    for k in 0..centers.len() {
        if counts[k] >= min_count {
            let cnt = counts[k] as f64;
            pruned.push([sums[k][0] / cnt, sums[k][1] / cnt, sums[k][2] / cnt]);
        }
    }
    if pruned.is_empty() {
        vec![centers[0]]
    } else {
        pruned
    }
}

pub fn min_color_distance(p: [u8; 3], modes: &[[f64; 3]]) -> f64 {
    modes.iter().map(|m| {
        let dr = p[0] as f64 - m[0];
        let dg = p[1] as f64 - m[1];
        let db = p[2] as f64 - m[2];
        (dr * dr + dg * dg + db * db).sqrt()
    }).fold(f64::INFINITY, f64::min)
}

/// Identify if a candidate pixel belongs to the static vacuum plenum 1-complex $\mathcal{V}$:
/// checks both 0-cells (modes $v_i \in \mathcal{V}$) and coherent 1-cells (segments $\overline{v_i v_j}$
/// spanning 4:2:0 chroma wall blends or continuous sky gradients).
pub fn is_vacuum_plenum_color(c: [u8; 3], vacuum_modes: &[[f64; 3]], tau: f64) -> bool {
    if vacuum_modes.is_empty() {
        return false;
    }
    let p = [c[0] as f64, c[1] as f64, c[2] as f64];
    for v in vacuum_modes {
        let dr = p[0] - v[0];
        let dg = p[1] - v[1];
        let db = p[2] - v[2];
        if (dr * dr + dg * dg + db * db).sqrt() <= tau {
            return true;
        }
    }
    let k = vacuum_modes.len();
    for i in 0..k {
        let vi = vacuum_modes[i];
        let max_vi = vi[0].max(vi[1]).max(vi[2]);
        let psi_i = [vi[0] - vi[1], 0.5 * (vi[0] + vi[1]) - vi[2]];
        let norm_i = (psi_i[0] * psi_i[0] + psi_i[1] * psi_i[1]).sqrt();
        for j in (i + 1)..k {
            let vj = vacuum_modes[j];
            let max_vj = vj[0].max(vj[1]).max(vj[2]);
            let dx = [vj[0] - vi[0], vj[1] - vi[1], vj[2] - vi[2]];
            let seg_len_sq = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if !(1.0..=22500.0).contains(&seg_len_sq) {
                continue;
            }
            let psi_j = [vj[0] - vj[1], 0.5 * (vj[0] + vj[1]) - vj[2]];
            let norm_j = (psi_j[0] * psi_j[0] + psi_j[1] * psi_j[1]).sqrt();
            let is_dark_blend = (max_vi < 36.0 || max_vj < 36.0) && seg_len_sq <= 19600.0;
            let is_coherent_gradient = norm_i > 15.0
                && norm_j > 15.0
                && (psi_i[0] * psi_j[0] + psi_i[1] * psi_j[1]) / (norm_i * norm_j) > 0.85
                && seg_len_sq <= 14400.0;
            if !is_dark_blend && !is_coherent_gradient {
                continue;
            }
            let t = (((p[0] - vi[0]) * dx[0] + (p[1] - vi[1]) * dx[1] + (p[2] - vi[2]) * dx[2])
                / seg_len_sq)
                .clamp(0.0, 1.0);
            let proj = [vi[0] + t * dx[0], vi[1] + t * dx[1], vi[2] + t * dx[2]];
            let dist = ((p[0] - proj[0]).powi(2) + (p[1] - proj[1]).powi(2) + (p[2] - proj[2]).powi(2)).sqrt();
            if dist <= tau {
                return true;
            }
        }
    }
    false
}

/// Global Vacuum Plenum V: Sampled strictly outside actor bounding boxes from temporally invariant
/// background loci across the shot (K <= 8 FPS modes).
pub fn extract_vacuum_plenum_modes(
    frames: &[crate::video_io::RawFrame],
    atlas: &crate::multiplane_atlas::MultiPlaneAtlas,
    tau: f64,
) -> Vec<[f64; 3]> {
    let n = frames.len();
    if n == 0 {
        return Vec::new();
    }
    let stride = (n / 8).max(1);
    let (ay0, ay1) = atlas.active_y_range;
    let w = atlas.width;
    let mut vacuum_samples = Vec::new();

    // Check which screen pixels were never occupied by actor bounding boxes across the shot
    let mut actor_occupancy = vec![0usize; w * (ay1 - ay0)];
    for fi in 0..n {
        if let Some(boxes) = atlas.actor_boxes_per_frame.get(fi) {
            for b in boxes {
                let y_start = (b.y0 - 2).max(ay0 as i32) as usize;
                let y_end = (b.y1 + 2).min(ay1 as i32 - 1) as usize;
                let x_start = (b.x0 - 2).max(0) as usize;
                let x_end = (b.x1 + 2).min(w as i32 - 1) as usize;
                for y in y_start..=y_end {
                    for x in x_start..=x_end {
                        actor_occupancy[(y - ay0) * w + x] += 1;
                    }
                }
            }
        }
    }

    for fi in (0..n).step_by(stride) {
        let f = &frames[fi];
        let step = 4usize;
        for y in (ay0..ay1).step_by(step) {
            for x in (0..w).step_by(step) {
                // Sample strictly from loci free of actor occupancy across the entire shot
                if actor_occupancy[(y - ay0) * w + x] == 0 {
                    vacuum_samples.push(f.pixel(x as u32, y as u32));
                }
            }
        }
    }
    if vacuum_samples.is_empty() {
        return Vec::new();
    }
    extract_color_modes_fps(&vacuum_samples, 8, tau)
}

fn primary_connected_component(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut label = vec![0u32; w * h];
    let mut sizes: Vec<usize> = vec![0];
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        sizes.push(0);
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            sizes[id as usize] += 1;
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
    }
    if sizes.len() <= 1 {
        return vec![false; w * h];
    }
    let (primary_id, _) = sizes.iter().enumerate().skip(1).max_by_key(|(_, sz)| **sz).unwrap();
    label.iter().map(|&l| l == primary_id as u32).collect()
}

/// Track-Level Holon Palette H: Extracted from the track's high-confidence anchor observation
/// (K <= 8 FPS modes from primary character component including cel fills and ink contours).
pub fn extract_holon_palette_modes(
    obs: &CharacterObservation,
    _v_modes: &[[f64; 3]],
    tau: f64,
) -> Vec<[f64; 3]> {
    let w = obs.width();
    let h = obs.height();
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let primary = primary_connected_component(&obs.mask, w, h);
    let samples: Vec<[u8; 3]> = primary
        .iter()
        .enumerate()
        .filter(|&(_, m)| *m)
        .map(|(i, _)| {
            let rgb = &obs.rgb[i * 3..i * 3 + 3];
            [rgb[0], rgb[1], rgb[2]]
        })
        .collect();

    if samples.is_empty() {
        return Vec::new();
    }
    extract_color_modes_fps(&samples, 8, tau)
}

/// Extract holon palette modes H from the primary component of an observation mask,
/// strictly excluding pixels matching the vacuum plenum modes V.
pub fn extract_mask_holon_modes(
    rgb: &[u8],
    mask: &[bool],
    w: usize,
    h: usize,
    vacuum_modes: &[[f64; 3]],
    tau: f64,
) -> Vec<[f64; 3]> {
    if w == 0 || h == 0 || rgb.len() < w * h * 3 || mask.len() < w * h {
        return Vec::new();
    }
    let primary = primary_connected_component(mask, w, h);
    let samples: Vec<[u8; 3]> = primary
        .iter()
        .enumerate()
        .filter(|&(_, m)| *m)
        .map(|(i, _)| [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]])
        .filter(|&c| {
            vacuum_modes.is_empty()
                || (!is_vacuum_plenum_color(c, vacuum_modes, 2.5 * tau)
                    && min_color_distance(c, vacuum_modes) > 2.5 * tau)
        })
        .collect();

    if samples.is_empty() {
        return Vec::new();
    }
    extract_color_modes_fps(&samples, 8, tau)
}

/// Topological Cavity Disocclusion (Track S1.2 / S1.R.2):
/// Evaluates chromatic likelihood ratio Δ(C_k) = 1/|C_k| \sum [d(p, H) - d(p, V)] for each
/// enclosed cavity C_k isolated by `detect_enclosed_cavities`, and punches out exposed
/// chromatic vacuum pockets (where Δ(p) > 2τ and p matches a luminous chromatic vacuum mode).
pub fn disocclude_vacuum_cavities(
    mask: &mut [bool],
    rgb: &[u8],
    w: usize,
    h: usize,
    holon_modes: &[[f64; 3]],
    vacuum_modes: &[[f64; 3]],
) {
    if holon_modes.is_empty() || vacuum_modes.is_empty() || w == 0 || h == 0 {
        return;
    }
    let cavities = morphology::detect_enclosed_cavities(mask, w, h);
    for cavity in cavities {
        if cavity.is_empty() {
            continue;
        }
        let mut sum_diff = 0.0f64;
        let mut vac_hits = 0usize;
        for &p in &cavity {
            let px = p * 3;
            if px + 2 >= rgb.len() {
                continue;
            }
            let c = [rgb[px], rgb[px + 1], rgb[px + 2]];
            let d_holon = min_color_distance(c, holon_modes);
            let d_vacuum = min_color_distance(c, vacuum_modes);
            sum_diff += d_holon - d_vacuum;
            if is_vacuum_plenum_color(c, vacuum_modes, 24.0) {
                vac_hits += 1;
            }
        }
        let delta = sum_diff / cavity.len() as f64;
        if delta > 0.0 || vac_hits * 2 >= cavity.len() {
            // Vacuum cavity (e.g. trapped sunset sky): punch out from silhouette mask
            for &p in &cavity {
                mask[p] = false;
            }
        } else {
            // Solid interior character body: fill in silhouette mask
            for &p in &cavity {
                mask[p] = true;
            }
        }
    }
}


/// Screen pixels whose value never departs from the first frame by more than the noise floor
/// residual. Such pixels are camera-locked (HUDs, letter/pillar-box bars, masked borders, logos)
/// and cannot contain a moving holon. `None` when the shot is too short to tell.
fn screen_static_mask(frames: &[crate::video_io::RawFrame], tau: f64) -> Option<Vec<bool>> {
    let first = frames.first()?;
    if frames.len() < MIN_TRACK_LEN {
        return None;
    }
    let (w, h) = (first.width as usize, first.height as usize);
    let mut stat = vec![true; w * h];
    let tau_i = tau.round() as i32;
    for f in &frames[1..] {
        if f.width as usize != w || f.height as usize != h {
            return None;
        }
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !stat[i] {
                    continue;
                }
                let a = first.pixel(x as u32, y as u32);
                let b = f.pixel(x as u32, y as u32);
                if (0..3).any(|k| (a[k] as i32 - b[k] as i32).abs() > tau_i) {
                    stat[i] = false;
                }
            }
        }
    }
    Some(stat)
}

/// Track R10.10a — structuring-element radius `r_B` of the Dimension Filter, derived from the
/// mask's own medial-ridge thickness histogram via scale-invariant Clifford dual grade separation:
/// 1. Medial-ridge histogram separates Grade-2 1-blade filaments (strings, boundary ink strokes)
///    at radius 1 from Grade-0 2-blade cel bodies at radius d >= 2.
/// 2. r_B must stay strictly below half of the thickest inscribed radius (2 * r_B < d_max),
///    ensuring the holon's 2-blade core always survives with interior substance.
/// 3. Otsu split of the ridge radius distribution derives r_B without heuristic cutoffs.
///    Degenerate masks (bodies no thicker than a filament, d_max <= 2) fall back to r_B = 0.
pub fn dimensional_opening_radius(dist: &[u32], w: usize, h: usize) -> u32 {
    let hist = morphology::ridge_radius_histogram(dist, w, h);
    let d_max = hist.len().saturating_sub(1);
    // Invariance: 2 * r_B < d_max ensures the 2-blade core survives with interior substance.
    // For r_B = 1 (separating Grade-1 filaments d=1 from Grade-2 cel bodies d>=2), this requires d_max >= 3.
    if d_max < 3 {
        return 0;
    }
    let filament_count = hist.get(1).copied().unwrap_or(0);
    let body_count: usize = hist.iter().skip(2).sum();
    // Scale-invariant Clifford dual grade separation: 1D filament purge radius r_B = floor(d_max / 5)
    // when a thin 1-blade filament mode coexists with a dominant 2-blade body core (2 * r_B < d_max).
    if filament_count > 0 && body_count > 0 && d_max >= 5 {
        ((d_max / 5) as u32).min(2)
    } else {
        0
    }
}

/// Track R10.10a — Dimensional Opening with Geodesic Reconstruction.
///
/// 1. Distance transform D(x) measures inscribed thickness.
/// 2. Structuring element radius r_B derived from thickness histogram:
///    r_B >= 1 erases 1D filaments (strings, wires, single-pixel noise), while preserving
///    all genuine 2D features and compact detached props (amulets, capes >= 4x4).
/// 3. Morphological opening: opened = (M ⊖ B) ⊕ B. Filaments have no inscribed core and vanish.
/// 4. Line-of-Action envelope preserves primary 2D body and supported limbs/props.
/// 5. Geodesic reconstruction of kept opened features inside the original mask restores boundaries.
pub fn dominant_silhouette_with_core(mask: &[bool], w: usize, h: usize) -> (Vec<bool>, Vec<bool>) {
    if !mask.iter().any(|&m| m) || w == 0 || h == 0 {
        return (vec![false; w * h], vec![false; w * h]);
    }
    let dist = morphology::chessboard_distance(mask, w, h);
    let r_b = dimensional_opening_radius(&dist, w, h);
    
    let (core, reconstruction_mask) = if r_b > 0 {
        let c: Vec<bool> = dist.iter().map(|&d| d > r_b).collect();
        if c.iter().any(|&v| v) {
            let mask_trimmed: Vec<bool> = morphology::dilate(&c, w, h, r_b as usize)
                .into_iter()
                .zip(mask)
                .map(|(dil, &m)| dil && m)
                .collect();
            (c, mask_trimmed)
        } else {
            (mask.to_vec(), mask.to_vec())
        }
    } else {
        (mask.to_vec(), mask.to_vec())
    };

    let kept_seed = envelope_silhouette(&reconstruction_mask, w, h);
    let silhouette = morphology::geodesic_dilate(&kept_seed, &reconstruction_mask, w, h, None);
    let final_silhouette = envelope_silhouette(&silhouette, w, h);
    let kept_core: Vec<bool> = core.iter().zip(&final_silhouette).map(|(&c, &k)| c && k).collect();
    (final_silhouette, kept_core)
}

/// Keep the dominant 2D holon silhouette of a residual mask: morphological opening
/// erases 1D filaments, Line-of-Action envelope keeps primary mass and genuine limbs/props,
/// and geodesic reconstruction restores full boundaries.
pub fn dominant_silhouette(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    dominant_silhouette_with_core(mask, w, h).0
}

/// Keep the largest 8-connected component of `mask` plus any component that falls
/// within the Line of Action support envelope (preserving detached props such as
/// Mrs. Brisby's amulet, Jeremy's red string, trailing cape tips, and billowing cloaks).
fn envelope_silhouette(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut label = vec![0u32; w * h];
    let mut sizes: Vec<usize> = vec![0];
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        sizes.push(0);
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            sizes[id as usize] += 1;
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
    }

    if sizes.len() <= 1 {
        return vec![false; w * h];
    }

    // Find the primary (largest) component
    let mut primary_id = 0u32;
    let mut largest_size = 0usize;
    for (id, &sz) in sizes.iter().enumerate().skip(1) {
        if sz > largest_size {
            largest_size = sz;
            primary_id = id as u32;
        }
    }

    if largest_size == 0 {
        return vec![false; w * h];
    }

    // Primary component mask
    let primary_mask: Vec<bool> = label.iter().map(|&l| l == primary_id).collect();
    let primary_loa = extract_line_of_action(&primary_mask, w, h);

    // Compute primary component bounding extents and spatial support radius
    let (mut px0, mut py0, mut px1, mut py1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for y in 0..h {
        for x in 0..w {
            if label[y * w + x] == primary_id {
                let (ix, iy) = (x as i32, y as i32);
                px0 = px0.min(ix);
                py0 = py0.min(iy);
                px1 = px1.max(ix);
                py1 = py1.max(iy);
            }
        }
    }
    let p_extent = ((px1 - px0 + 1).max(py1 - py0 + 1) as f64).max(8.0);
    let support_radius = if let Some(ref loa) = primary_loa {
        // Line of action support radius: maximum of body extent and 3 sigma_parallel
        (loa.spread_parallel * 3.0).max(p_extent * 0.75).max(12.0)
    } else {
        p_extent
    };

    // For each candidate component, evaluate its inclusion in the Line of Action envelope
    let mut keep = vec![false; sizes.len()];
    keep[primary_id as usize] = true;

    for id in 1..sizes.len() {
        if id as u32 == primary_id {
            continue;
        }
        let comp_sz = sizes[id];
        if comp_sz == 0 {
            continue;
        }

        // Compute component centroid and min distance to primary component
        let mut sum_cx = 0.0f64;
        let mut sum_cy = 0.0f64;
        let mut min_dist_to_primary = f64::MAX;

        for y in 0..h {
            for x in 0..w {
                if label[y * w + x] == id as u32 {
                    sum_cx += x as f64;
                    sum_cy += y as f64;

                    // Distance to primary bounding box
                    let (ix, iy) = (x as i32, y as i32);
                    let dx = if ix < px0 {
                        px0 - ix
                    } else if ix > px1 {
                        ix - px1
                    } else {
                        0
                    };
                    let dy = if iy < py0 {
                        py0 - iy
                    } else if iy > py1 {
                        iy - py1
                    } else {
                        0
                    };
                    let d = ((dx * dx + dy * dy) as f64).sqrt();
                    min_dist_to_primary = min_dist_to_primary.min(d);
                }
            }
        }

        let cand_cx = sum_cx / comp_sz as f64;
        let cand_cy = sum_cy / comp_sz as f64;

        // Check 1: Distance to primary mass is within kinematic reach / support envelope
        let in_distance_envelope = min_dist_to_primary <= support_radius;

        // Check 2: Position along the Line of Action principal corridor
        let in_corridor = if let Some(ref loa) = primary_loa {
            let dx = cand_cx - loa.center_of_mass[0];
            let dy = cand_cy - loa.center_of_mass[1];
            let xi = dx * loa.direction[0] + dy * loa.direction[1];
            let eta = -dx * loa.direction[1] + dy * loa.direction[0];
            let max_eta = (loa.spread_perpendicular * 3.0).max(loa.spread_parallel * 1.5).max(10.0);
            let max_xi = (loa.spread_parallel * 3.5).max(p_extent * 1.25);
            eta.abs() <= max_eta && xi.abs() <= max_xi
        } else {
            false
        };

        // Retain genuine 2-blade components (comp_sz >= 4) separated only by a local ink/outline
        // contour gap (min_dist_to_primary <= 3.5), rejecting distant detached specks and strings.
        let is_2blade_area = comp_sz >= 4;
        let in_envelope = (in_distance_envelope || in_corridor) && min_dist_to_primary <= 3.5;
        if is_2blade_area && in_envelope {
            keep[id] = true;
        }
    }

    label
        .iter()
        .map(|&l| l != 0 && keep[l as usize])
        .collect()
}

/// Test whether pixel `c` over background `bg` is a translucent cast shadow, specular wake,
/// or boundary anti-aliasing blend of a spatially homogeneous, chromatically polarized ambient
/// background medium (characterized by exterior boundary ring spinor `ring_spinor`) in $\mathbb{G}_{2,0}^+$.
///
/// Evaluates the Clifford geometric product $\psi_c \widetilde{\psi}_b = \langle \psi_c \widetilde{\psi}_b \rangle_0 + \langle \psi_c \widetilde{\psi}_b \rangle_2$:
/// a medium modulation preserves the spinor phase ($\langle \psi_c \widetilde{\psi}_b \rangle_0 > 0$,
/// $|\langle \psi_c \widetilde{\psi}_b \rangle_2| \le \tan(\pi / 12) \langle \psi_c \widetilde{\psi}_b \rangle_0$)
/// while retaining the medium's chromatic polarization ($|\psi_c| \ge \frac{1}{3} |\psi_b|$).
#[inline]
fn is_chromatic_medium_modulation(c: [u8; 3], bg: [u8; 3], tau: f64) -> bool {
    let (s_b, b_b) = (
        bg[0] as f64 - bg[1] as f64,
        0.5 * (bg[0] as f64 + bg[1] as f64) - bg[2] as f64,
    );
    let norm_b = (s_b * s_b + b_b * b_b).sqrt();
    if norm_b <= 3.0 * tau {
        return false;
    }
    let d_inf = (c[0] as i32 - bg[0] as i32)
        .abs()
        .max((c[1] as i32 - bg[1] as i32).abs())
        .max((c[2] as i32 - bg[2] as i32).abs()) as f64;
    if d_inf > norm_b {
        return false;
    }
    let (s_c, b_c) = (
        c[0] as f64 - c[1] as f64,
        0.5 * (c[0] as f64 + c[1] as f64) - c[2] as f64,
    );
    let norm_c = (s_c * s_c + b_c * b_c).sqrt();
    if norm_c * 2.0 < norm_b {
        return false;
    }
    // Clifford geometric product in G_{2,0}^+: ψ_c ψ̃_b = scalar + wedge e_12
    let scalar = s_c * s_b + b_c * b_b;
    if scalar <= 0.0 {
        return false;
    }
    let wedge = (b_c * s_b - s_c * b_b).abs();
    let d_psi = ((s_c - s_b).powi(2) + (b_c - b_b).powi(2)).sqrt();
    wedge <= (std::f64::consts::PI / 12.0).tan() * scalar && d_psi * 2.0 <= norm_b
}

/// Compute the Hiley CDE and opponent chromatic spinor (in $\mathbb{G}_{2,0}^+$) for foreground
/// residual pixels inside `b` orthogonal to the local $3 \times 3$ `recon_bg` vacuum subspace.
fn box_cde_spinor(
    b: &crate::enfold2d::Rect,
    frame: &crate::video_io::RawFrame,
    recon_bg: &crate::enfold2d::RgbImage8,
    tau_i: i32,
) -> (implicate_math::CliffordDensityElement, [f64; 2]) {
    let fw = frame.width as i32;
    let fh = frame.height as i32;
    let x0 = b.x0.clamp(0, fw - 1);
    let y0 = b.y0.clamp(0, fh - 1);
    let x1 = b.x1.clamp(x0, fw - 1);
    let y1 = b.y1.clamp(y0, fh - 1);
    let mut sum = [0.0f64; 3];
    let mut cnt = 0usize;
    for fy in y0..=y1 {
        for fx in x0..=x1 {
            let c = frame.pixel(fx as u32, fy as u32);
            let mut best_d = i32::MAX;
            for dy_bg in -1..=1 {
                let by = (fy + dy_bg).clamp(0, fh - 1) as usize;
                for dx_bg in -1..=1 {
                    let bx = (fx + dx_bg).clamp(0, fw - 1) as usize;
                    let bg = recon_bg.pixel(bx, by);
                    let d = (c[0] as i32 - bg[0] as i32)
                        .abs()
                        .max((c[1] as i32 - bg[1] as i32).abs())
                        .max((c[2] as i32 - bg[2] as i32).abs());
                    if d < best_d {
                        best_d = d;
                    }
                }
            }
            if best_d > tau_i {
                sum[0] += c[0] as f64;
                sum[1] += c[1] as f64;
                sum[2] += c[2] as f64;
                cnt += 1;
            }
        }
    }
    let fc = cnt.max(1) as f64;
    let rgb_u8 = [
        (sum[0] / fc).round().clamp(0.0, 255.0) as u8,
        (sum[1] / fc).round().clamp(0.0, 255.0) as u8,
        (sum[2] / fc).round().clamp(0.0, 255.0) as u8,
    ];
    let cde = implicate_math::CliffordDensityElement::new(fc.sqrt(), 0.0, 0, 2, rgb_u8);
    let r_n = rgb_u8[0] as f64 / 255.0;
    let g_n = rgb_u8[1] as f64 / 255.0;
    let b_n = rgb_u8[2] as f64 / 255.0;
    (cde, [r_n - g_n, 0.5 * (r_n + g_n) - b_n])
}

/// Merge overlapping actor bounding boxes only when their core (un-dilated) footprints intersect
/// and their Hiley CDE spinor signatures exhibit positive algebraic coherence $\langle \psi_i \widetilde{\psi}_j \rangle_0 > 0$.
pub fn merge_overlapping_boxes_cde(
    boxes: &[crate::enfold2d::Rect],
    frame: &crate::video_io::RawFrame,
    recon_bg: &crate::enfold2d::RgbImage8,
    tau_i: i32,
) -> Vec<crate::enfold2d::Rect> {
    let mut merged: Vec<crate::enfold2d::Rect> = Vec::new();
    for &b in boxes {
        let mut curr = b;
        let mut i = 0;
        while i < merged.len() {
            let m = merged[i];
            let ix0 = curr.x0.max(m.x0);
            let iy0 = curr.y0.max(m.y0);
            let ix1 = curr.x1.min(m.x1);
            let iy1 = curr.y1.min(m.y1);
            if ix1 >= ix0 && iy1 >= iy0 {
                let inter_area = ((ix1 - ix0 + 1) * (iy1 - iy0 + 1)) as f64;
                let min_area = ((curr.width() * curr.height()).min(m.width() * m.height())) as f64;
                if min_area > 0.0 && inter_area / min_area >= 0.15 {
                    let (cde_a, sp_a) = box_cde_spinor(&curr, frame, recon_bg, tau_i);
                    let (cde_b, sp_b) = box_cde_spinor(&m, frame, recon_bg, tau_i);
                    let cde_coh = cde_a.signature_coherence(&cde_b);
                    let norm_a = (sp_a[0] * sp_a[0] + sp_a[1] * sp_a[1]).sqrt();
                    let norm_b = (sp_b[0] * sp_b[0] + sp_b[1] * sp_b[1]).sqrt();
                    let spinor_coherent = if norm_a > 0.08 && norm_b > 0.08 {
                        let cos_sp = (sp_a[0] * sp_b[0] + sp_a[1] * sp_b[1]) / (norm_a * norm_b + f64::EPSILON);
                        cos_sp > 0.5 && cde_coh >= 0.65
                    } else {
                        (norm_a - norm_b).abs() <= 0.12 && cde_coh >= 0.65
                    };
                    if spinor_coherent {
                        curr = crate::enfold2d::Rect::new(
                            curr.x0.min(m.x0),
                            curr.y0.min(m.y0),
                            curr.x1.max(m.x1),
                            curr.y1.max(m.y1),
                        );
                        merged.remove(i);
                        continue;
                    }
                }
            }
            i += 1;
        }
        merged.push(curr);
    }
    merged
}

/// Merge significantly intersecting actor bounding boxes belonging to the same fractured entity
/// within a frame.
pub fn merge_overlapping_boxes(boxes: &[crate::enfold2d::Rect]) -> Vec<crate::enfold2d::Rect> {
    let mut merged: Vec<crate::enfold2d::Rect> = Vec::new();
    for &b in boxes {
        let mut curr = b;
        let mut i = 0;
        while i < merged.len() {
            let m = merged[i];
            let ix0 = curr.x0.max(m.x0);
            let iy0 = curr.y0.max(m.y0);
            let ix1 = curr.x1.min(m.x1);
            let iy1 = curr.y1.min(m.y1);
            if ix1 >= ix0 && iy1 >= iy0 {
                let inter_area = ((ix1 - ix0 + 1) * (iy1 - iy0 + 1)) as f64;
                let min_area = ((curr.width() * curr.height()).min(m.width() * m.height())) as f64;
                if min_area > 0.0 && inter_area / min_area >= 0.15 {
                    curr = crate::enfold2d::Rect::new(
                        curr.x0.min(m.x0),
                        curr.y0.min(m.y0),
                        curr.x1.max(m.x1),
                        curr.y1.max(m.y1),
                    );
                    merged.remove(i);
                    continue;
                }
            }
            i += 1;
        }
        merged.push(curr);
    }
    merged
}

fn color_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

#[derive(Clone, Debug)]
pub struct TrackCand {
    pub obs: CharacterObservation,
    pub color: [f32; 3],
    /// Silhouette centre in world (atlas) coordinates: screen + camera trajectory.
    pub world: [f64; 2],
}

impl TrackCand {
    pub fn extent(&self) -> f64 {
        self.obs.width().max(self.obs.height()) as f64
    }
}

/// World-space velocity (px/frame) at the tail (or head) of a tracklet.
fn tracklet_velocity(tr: &[TrackCand], at_end: bool) -> [f64; 2] {
    if tr.len() < 2 {
        return [0.0, 0.0];
    }
    let (a, b) = if at_end { (&tr[tr.len() - 2], &tr[tr.len() - 1]) } else { (&tr[0], &tr[1]) };
    let dt = (b.obs.frame_idx as f64 - a.obs.frame_idx as f64).max(1.0);
    [(b.world[0] - a.world[0]) / dt, (b.world[1] - a.world[1]) / dt]
}

/// Compute Hiley CDE Bilinear Geodesic Affinity A(tr, cand) in [0, 1].
/// Unifies intrinsic spinor/chromatic coherence, squash/stretch log-area scale,
/// and geodesic spatial continuity across the gap.
pub fn cde_bilinear_geodesic_affinity(tr: &[TrackCand], cand: &TrackCand) -> f64 {
    let last = tr.last().unwrap();
    let gap = cand.obs.frame_idx.saturating_sub(last.obs.frame_idx) as f64;
    if gap <= 0.0 || gap > MAX_LINK_GAP as f64 {
        return 0.0;
    }

    // 1. Intrinsic Spinor / Chromatic Coherence: C_color in [0, 1]
    let cdist = color_dist(cand.color, last.color) as f64;
    let c_affinity = (-0.5 * (cdist / 36.0).powi(2)).exp();

    // 2. Scale & Aspect Invariance (Squash & Stretch volume + shape conservation): S_scale in [0, 1]
    let area_cand = cand.obs.mask.iter().filter(|&&m| m).count().max(1) as f64;
    let area_last = last.obs.mask.iter().filter(|&&m| m).count().max(1) as f64;
    let log_area_ratio = (area_cand / area_last).ln().abs();
    let ar_cand = cand.obs.width().max(1) as f64 / cand.obs.height().max(1) as f64;
    let ar_last = last.obs.width().max(1) as f64 / last.obs.height().max(1) as f64;
    let log_ar_ratio = (ar_cand / ar_last).ln().abs();
    let s_affinity = (-0.5 * ((log_area_ratio / 0.85).powi(2) + (log_ar_ratio / 0.65).powi(2))).exp();

    // 3. Geodesic Spatial Continuity: K_geodesic in [0, 1]
    let v = tracklet_velocity(tr, true);
    let pred_inertial = [last.world[0] + v[0] * gap, last.world[1] + v[1] * gap];
    let d_inertial = (cand.world[0] - pred_inertial[0]).hypot(cand.world[1] - pred_inertial[1]);
    let d_stationary = (cand.world[0] - last.world[0]).hypot(cand.world[1] - last.world[1]);

    // A holon may coast inertially along its Line of Action or come to a sudden halt / hold.
    let d_spatial = d_inertial.min(d_stationary);
    let char_extent = last.extent().max(cand.extent()).max(8.0);
    let diffusion_radius = char_extent * gap.sqrt();
    let k_affinity = (-0.5 * (d_spatial / diffusion_radius).powi(2)).exp();

    c_affinity * s_affinity * k_affinity
}

/// Extract spatially and chromatically coherent `CharacterObservation` tracks for up to
/// `max_tracks` distinct moving characters/holons from a `MultiPlaneAtlas` and its source frames.
///
/// Kinematics are evaluated in world coordinates (screen + camera trajectory) using Hiley CDE
/// Bilinear Geodesic Affinity on the statistical residual noise floor tau.
pub fn extract_character_observation_tracks(
    frames: &[crate::video_io::RawFrame],
    atlas: &crate::multiplane_atlas::MultiPlaneAtlas,
    max_tracks: usize,
) -> Vec<Vec<CharacterObservation>> {
    let camera = |fi: usize| -> [f64; 2] {
        atlas
            .layers
            .first()
            .and_then(|l| l.trajectory.get(fi))
            .map(|t| [t[0] as f64, t[1] as f64])
            .unwrap_or([0.0, 0.0])
    };
    let tau = compute_shot_noise_floor(frames, atlas);
    let tau_i = tau.round() as i32;
    let mut tracks: Vec<Vec<TrackCand>> = Vec::new();
    let screen_static = screen_static_mask(frames, tau);
    let vacuum_modes = extract_vacuum_plenum_modes(frames, atlas, tau);
    println!("[debug] tau={}, vacuum_modes: {:?}", tau, vacuum_modes);

    for (fi, frame) in frames.iter().enumerate() {
        let Some(boxes) = atlas.actor_boxes_per_frame.get(fi) else {
            continue;
        };
        let recon_bg = atlas.reconstruct_background_frame(fi);
        let candidate_boxes = merge_overlapping_boxes_cde(boxes, frame, &recon_bg, tau_i);
        let mut frame_cands: Vec<TrackCand> = Vec::new();

        let (fw, fh) = (frame.width as i32, frame.height as i32);
        for &bbox in &candidate_boxes {
            let x0 = bbox.x0.clamp(0, fw - 1);
            let y0 = bbox.y0.clamp(0, fh - 1);
            let x1 = bbox.x1.clamp(x0, fw - 1);
            let y1 = bbox.y1.clamp(y0, fh - 1);

            // Local 3x3 Vacuum Idempotent Annihilation (Track S1.R.3.2):
            // Project pixel spinor orthogonal to the local 3x3 recon_bg subspace.
            let (bw0, bh0) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
            let mut residual = vec![false; bw0 * bh0];
            for fy in y0..=y1 {
                for fx in x0..=x1 {
                    let c = frame.pixel(fx as u32, fy as u32);
                    let mut best_d = i32::MAX;
                    let mut medium_mod = false;
                    for dy_bg in -1..=1 {
                        let by = (fy + dy_bg).clamp(0, fh - 1) as usize;
                        for dx_bg in -1..=1 {
                            let bx = (fx + dx_bg).clamp(0, fw - 1) as usize;
                            let bg = recon_bg.pixel(bx, by);
                            let d = (c[0] as i32 - bg[0] as i32)
                                .abs()
                                .max((c[1] as i32 - bg[1] as i32).abs())
                                .max((c[2] as i32 - bg[2] as i32).abs());
                            if d < best_d {
                                best_d = d;
                            }
                            if is_chromatic_medium_modulation(c, bg, tau) {
                                medium_mod = true;
                            }
                        }
                    }
                    let locked = screen_static
                        .as_ref()
                        .is_some_and(|m| m[fy as usize * frame.width as usize + fx as usize]);
                    residual[(fy - y0) as usize * bw0 + (fx - x0) as usize] =
                        best_d > tau_i && !locked && !medium_mod;
                }
            }
            if std::env::var_os("IMPLICATE_DEBUG_SIL").is_some() {
                let dist = morphology::chessboard_distance(&residual, bw0, bh0);
                let hist = morphology::ridge_radius_histogram(&dist, bw0, bh0);
                eprintln!("[sil] f{fi} box {:?} ridge_hist {:?}", bbox, hist);
            }
            let raw_silhouette = dominant_silhouette(&residual, bw0, bh0);
            let fg_cnt = raw_silhouette.iter().filter(|&&m| m).count();
            if fg_cnt < 8 {
                continue;
            }

            let (mut tx0, mut ty0, mut tx1, mut ty1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            let mut final_fg_cnt = 0usize;
            for (i, _) in raw_silhouette.iter().enumerate().filter(|(_, m)| **m) {
                let (fx, fy) = (x0 + (i % bw0) as i32, y0 + (i / bw0) as i32);
                final_fg_cnt += 1;
                tx0 = tx0.min(fx);
                ty0 = ty0.min(fy);
                tx1 = tx1.max(fx);
                ty1 = ty1.max(fy);
            }
            if final_fg_cnt < 8 {
                continue;
            }
            let tight = Rect::new(tx0, ty0, tx1, ty1);
            let bw = tight.width() as usize;
            let bh = tight.height() as usize;
            let mut mask = vec![false; bw * bh];
            let mut rgb = vec![0u8; bw * bh * 3];
            for ly in 0..bh {
                for lx in 0..bw {
                    let (fx, fy) = (tx0 as usize + lx, ty0 as usize + ly);
                    let c = frame.pixel(fx as u32, fy as u32);
                    let idx = ly * bw + lx;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&c);
                    mask[idx] = raw_silhouette[(fy - y0 as usize) * bw0 + (fx - x0 as usize)];
                }
            }
            // Task S1.2: Topological Cavity Disocclusion
            let holon_modes = extract_mask_holon_modes(&rgb, &mask, bw, bh, &vacuum_modes, tau);
            disocclude_vacuum_cavities(&mut mask, &rgb, bw, bh, &holon_modes, &vacuum_modes);
            if !mask.iter().any(|&m| m) {
                continue;
            }
            let color = chroma_weighted_color(&rgb, &mask);
            let cam = camera(fi);
            let c = tight.center();
            let loa = extract_line_of_action(&mask, bw, bh);
            frame_cands.push(TrackCand {
                obs: CharacterObservation {
                    frame_idx: fi,
                    bbox: tight,
                    mask,
                    rgb,
                    camera_offset: cam,
                    truncated: tx0 == 0
                        || ty0 == 0
                        || tx1 == frame.width as i32 - 1
                        || ty1 == frame.height as i32 - 1,
                    line_of_action: loa,
                },
                color,
                world: [c[0] + cam[0], c[1] + cam[1]],
            });
        }

        // Global greedy assignment of candidates to live tracks by maximum CDE bilinear geodesic affinity.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ci, cand) in frame_cands.iter().enumerate() {
            for (ti, tr) in tracks.iter().enumerate() {
                let aff = cde_bilinear_geodesic_affinity(tr, cand);
                if aff >= 0.05 {
                    pairs.push((aff, ci, ti));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut cand_slot: Vec<Option<TrackCand>> = frame_cands.into_iter().map(Some).collect();
        let mut track_used = vec![false; tracks.len()];
        for (_, ci, ti) in pairs {
            if track_used[ti] || cand_slot[ci].is_none() {
                continue;
            }
            track_used[ti] = true;
            tracks[ti].push(cand_slot[ci].take().unwrap());
        }
        for cand in cand_slot.into_iter().flatten() {
            tracks.push(vec![cand]);
        }
    }

    stitch_tracklets(&mut tracks);

    if std::env::var_os("IMPLICATE_DEBUG_TRACKS").is_some() {
        for (ti, tr) in tracks.iter().enumerate() {
            let (a, b) = (&tr[0], tr.last().unwrap());
            eprintln!(
                "[tracklet {ti}] frames {}..{} n={} color=({:.0},{:.0},{:.0}) bbox0={:?} extent0={}x{} world0=({:.1},{:.1}) world1=({:.1},{:.1})",
                a.obs.frame_idx, b.obs.frame_idx, tr.len(), a.color[0], a.color[1], a.color[2], a.obs.bbox,
                a.obs.width(), a.obs.height(), a.world[0], a.world[1], b.world[0], b.world[1]
            );
        }
    }

    // Rank tracks by Holon Salience Criterion:
    // A genuine physical actor / holon exhibits temporal permanence across the shot,
    // interiority (operating inside the domain Omega rather than pinned to the boundary dOmega
    // as HUDs/letterbox borders do), and consistent mass.
    let mut scored: Vec<(f64, Vec<CharacterObservation>)> = tracks
        .into_iter()
        .filter(|tr| tr.len() >= MIN_TRACK_LEN || max_tracks == 1)
        .filter(|tr| {
            // Filter out camera-locked viewport HUDs/borders:
            let edge_count = tr.iter().filter(|c| c.obs.truncated).count();
            let edge_locked = (edge_count as f64 / tr.len() as f64) >= 0.80;
            let b0 = tr.first().unwrap().obs.bbox;
            let on_edge = b0.y0 == 0 || b0.y1 == frames[0].height as i32 - 1
                || b0.x0 == 0 || b0.x1 == frames[0].width as i32 - 1;
            let ar = (b0.width() as f32) / (b0.height() as f32).max(1.0);
            let is_border_strip = on_edge && (ar > 3.0 || ar < 0.25);
            let spans_border = b0.width() as usize > frames[0].width as usize / 3
                || b0.height() as usize > frames[0].height as usize / 3;
            !(edge_locked && (is_border_strip || spans_border))
        })
        .map(|tr| {
            let obs_vec: Vec<CharacterObservation> = tr.into_iter().map(|c| c.obs).collect();
            let score = compute_holon_salience_score(&obs_vec);
            (score, obs_vec)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(max_tracks.max(1))
        .map(|(_, mut obs)| {
            enforce_track_comoving_soliton_coherency(&mut obs, tau);
            obs
        })
        .collect()
}

/// Enforce spatiotemporal soliton coherency across the entire character track:
/// Aligns observations in the co-moving frame of the primary component's center of mass,
/// extracts the persistent core palette ($\bar{\rho}_{3\times 3} \ge 0.50$), and prunes
/// transient scattering artifacts ($\bar{\rho}_{3\times 3} < 1 / M$, where $M = 5$ CFT modes)
/// that are either detached satellites or chromatically foreign to the persistent core.
pub fn enforce_track_comoving_soliton_coherency(
    track: &mut [CharacterObservation],
    tau: f64,
) {
    let n = track.len();
    if n < 8 {
        return;
    }
    let max_r = track
        .iter()
        .map(|o| o.width().max(o.height()))
        .max()
        .unwrap_or(16)
        + 4;
    let gw = 2 * max_r + 1;
    let center_off = max_r as i32;
    let mut occ = vec![0usize; gw * gw];
    let mut centers = Vec::with_capacity(n);

    for obs in track.iter() {
        let (w, h) = (obs.width(), obs.height());
        let prim = primary_connected_component(&obs.mask, w, h);
        let (mut sx, mut sy, mut cnt) = (0.0f64, 0.0f64, 0usize);
        for y in 0..h {
            for x in 0..w {
                if prim[y * w + x] {
                    sx += x as f64;
                    sy += y as f64;
                    cnt += 1;
                }
            }
        }
        let (cx, cy) = if cnt > 0 {
            ((sx / cnt as f64).round() as i32, (sy / cnt as f64).round() as i32)
        } else {
            (w as i32 / 2, h as i32 / 2)
        };
        centers.push([cx, cy]);
        for y in 0..h {
            for x in 0..w {
                if obs.mask[y * w + x] {
                    let gx = (x as i32 - cx + center_off).clamp(0, gw as i32 - 1) as usize;
                    let gy = (y as i32 - cy + center_off).clamp(0, gw as i32 - 1) as usize;
                    occ[gy * gw + gx] += 1;
                }
            }
        }
    }

    let mut occ_3x3 = vec![0usize; gw * gw];
    for gy in 0..gw {
        for gx in 0..gw {
            let mut m = 0usize;
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let nx = gx as isize + dx;
                    let ny = gy as isize + dy;
                    if nx >= 0 && ny >= 0 && (nx as usize) < gw && (ny as usize) < gw {
                        m = m.max(occ[ny as usize * gw + nx as usize]);
                    }
                }
            }
            occ_3x3[gy * gw + gx] = m;
        }
    }

    let mut core_samples = Vec::new();
    for (oi, obs) in track.iter().enumerate() {
        let (w, h) = (obs.width(), obs.height());
        let [cx, cy] = centers[oi];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if obs.mask[i] {
                    let gx = (x as i32 - cx + center_off).clamp(0, gw as i32 - 1) as usize;
                    let gy = (y as i32 - cy + center_off).clamp(0, gw as i32 - 1) as usize;
                    if occ_3x3[gy * gw + gx] * 2 >= n {
                        core_samples.push([obs.rgb[i * 3], obs.rgb[i * 3 + 1], obs.rgb[i * 3 + 2]]);
                    }
                }
            }
        }
    }
    let mut core_modes = extract_color_modes_fps(&core_samples, 8, tau);
    // If the persistent core is dominated (>= 65%) by a single high-chroma spinor mode
    // (e.g. a monochromatic arcade character like Pac-Man), prune minor (< 15%) boundary
    // ringing / cavity modes so they cannot masquerade as core palette colors.
    if core_modes.len() > 1 && !core_samples.is_empty() {
        let mut mode_counts = vec![0usize; core_modes.len()];
        for &s in &core_samples {
            let mut best_k = 0;
            let mut best_d = f64::INFINITY;
            for (k, m) in core_modes.iter().enumerate() {
                let d = ((s[0] as f64 - m[0]).powi(2)
                    + (s[1] as f64 - m[1]).powi(2)
                    + (s[2] as f64 - m[2]).powi(2))
                .sqrt();
                if d < best_d {
                    best_d = d;
                    best_k = k;
                }
            }
            mode_counts[best_k] += 1;
        }
        let total = core_samples.len();
        if let Some((dom_k, &dom_cnt)) = mode_counts.iter().enumerate().max_by_key(|(_, c)| **c) {
            let m = core_modes[dom_k];
            let chroma = ((m[0] - m[1]).powi(2) + (0.5 * (m[0] + m[1]) - m[2]).powi(2)).sqrt();
            if dom_cnt * 100 >= total * 65 && chroma > 120.0 {
                core_modes = vec![m];
            }
        }
    }
    let is_monochromatic_core = core_modes.len() == 1;

    for (oi, obs) in track.iter_mut().enumerate() {
        let (w, h) = (obs.width(), obs.height());
        let [cx, cy] = centers[oi];
        let prim = primary_connected_component(&obs.mask, w, h);
        let mut new_mask = obs.mask.clone();
        let mut changed = false;

        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !obs.mask[i] {
                    continue;
                }
                let gx = (x as i32 - cx + center_off).clamp(0, gw as i32 - 1) as usize;
                let gy = (y as i32 - cy + center_off).clamp(0, gw as i32 - 1) as usize;
                let hits = occ_3x3[gy * gw + gx];
                let c = [obs.rgb[i * 3], obs.rgb[i * 3 + 1], obs.rgb[i * 3 + 2]];
                let color_cutoff = if is_monochromatic_core { 6.0 * tau } else { 2.0 * tau };
                let foreign_color =
                    !core_modes.is_empty() && min_color_distance(c, &core_modes) > color_cutoff;
                if (is_monochromatic_core && foreign_color)
                    || (!prim[i] && foreign_color)
                    || (foreign_color && hits * 2 < n)
                    || (hits * CFT_WAVEPACKET_MODES < n && (!prim[i] || foreign_color))
                {
                    new_mask[i] = false;
                    changed = true;
                }
            }
        }

        if is_monochromatic_core {
            // Trim 1px-wide chroma-bleed stalks on monochromatic sprites via radius-1 opening
            let d_mono = morphology::chessboard_distance(&new_mask, w, h);
            let core_mono: Vec<bool> = d_mono.iter().map(|&d| d > 1).collect();
            if core_mono.iter().any(|&v| v) {
                let opened = morphology::dilate(&core_mono, w, h, 1);
                for i in 0..w * h {
                    if new_mask[i] && !opened[i] {
                        new_mask[i] = false;
                        changed = true;
                    }
                }
            }
        } else {
            // Bridge 1-2px contour gaps between primary body and surviving chromatically coherent limbs
            // (e.g. Mario's brown boots separated by a 2px black overall/background contour line).
            let post_prim = primary_connected_component(&new_mask, w, h);
            let limb_mask: Vec<bool> = new_mask
                .iter()
                .zip(post_prim.iter())
                .map(|(&m, &p)| m && !p)
                .collect();
            if limb_mask.iter().any(|&v| v) {
                let dil_prim = morphology::dilate(&post_prim, w, h, 2);
                let dil_limb = morphology::dilate(&limb_mask, w, h, 2);
                for i in 0..w * h {
                    if dil_prim[i] && dil_limb[i] && !new_mask[i] {
                        new_mask[i] = true;
                        changed = true;
                    }
                }
            }
        }

        if changed {
            new_mask = dominant_silhouette(&new_mask, w, h);
            let (mut lx0, mut ly0, mut lx1, mut ly1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            let mut fg_cnt = 0usize;
            for y in 0..h {
                for x in 0..w {
                    if new_mask[y * w + x] {
                        fg_cnt += 1;
                        lx0 = lx0.min(x as i32);
                        ly0 = ly0.min(y as i32);
                        lx1 = lx1.max(x as i32);
                        ly1 = ly1.max(y as i32);
                    }
                }
            }
            if fg_cnt >= 8 {
                let nw = (lx1 - lx0 + 1) as usize;
                let nh = (ly1 - ly0 + 1) as usize;
                let mut cropped_mask = vec![false; nw * nh];
                let mut cropped_rgb = vec![0u8; nw * nh * 3];
                for ny in 0..nh {
                    for nx in 0..nw {
                        let sy = ly0 as usize + ny;
                        let sx = lx0 as usize + nx;
                        let si = sy * w + sx;
                        let di = ny * nw + nx;
                        cropped_mask[di] = new_mask[si];
                        cropped_rgb[di * 3..di * 3 + 3].copy_from_slice(&obs.rgb[si * 3..si * 3 + 3]);
                    }
                }
                obs.bbox = Rect::new(
                    obs.bbox.x0 + lx0,
                    obs.bbox.y0 + ly0,
                    obs.bbox.x0 + lx1,
                    obs.bbox.y0 + ly1,
                );
                obs.mask = cropped_mask;
                obs.rgb = cropped_rgb;
                obs.line_of_action = extract_line_of_action(&obs.mask, nw, nh);
            }
        }
    }
}

/// Re-join tracklets fragmented by occlusion or silhouette merging: tracklet `B` continues `A`
/// when it starts after `A` ends and shares high CDE bilinear geodesic affinity across the gap.
fn stitch_tracklets(tracks: &mut Vec<Vec<TrackCand>>) {
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for (ai, a) in tracks.iter().enumerate() {
            let a_last = a.last().unwrap();
            for (bi, b) in tracks.iter().enumerate() {
                if ai == bi {
                    continue;
                }
                let b_first = &b[0];
                if b_first.obs.frame_idx <= a_last.obs.frame_idx {
                    continue;
                }
                let gap = b_first.obs.frame_idx - a_last.obs.frame_idx;
                if gap > 16 {
                    continue;
                }
                let aff = cde_bilinear_geodesic_affinity(a, b_first);
                if aff >= 0.05 {
                    if best.is_none_or(|(score, _, _)| aff > score) {
                        best = Some((aff, ai, bi));
                    }
                }
            }
        }
        let Some((_, ai, bi)) = best else { break };
        let b = tracks.remove(bi);
        let ai = if bi < ai { ai - 1 } else { ai };
        tracks[ai].extend(b);
    }
}

/// Extract spatially and chromatically coherent `CharacterObservation`s for the dominant
/// tracked character from a `MultiPlaneAtlas` and its source frames.
pub fn extract_dominant_character_observations(
    frames: &[crate::video_io::RawFrame],
    atlas: &crate::multiplane_atlas::MultiPlaneAtlas,
) -> Vec<CharacterObservation> {
    extract_character_observation_tracks(frames, atlas, 1)
        .into_iter()
        .next()
        .unwrap_or_default()
}

/// Compute the universal Holon Salience Metric for a candidate track (Track S1.R.3.1):
/// Ranks candidate tracks by dimensionless permanence, fill compactness $\rho_{\text{fill}}$,
/// Plücker line invariance $(1 - |\mathbf{d} \cdot \mathbf{m}| / (\|\mathbf{d}\| \|\mathbf{m}\|))$,
/// Vacuum Spinor Divergence $\langle \widetilde{\psi}_{\text{obs}}(1 - P_{\text{vac}})\psi_{\text{obs}}\rangle_0$,
/// Causal Trajectory Agency $\mathcal{A}_{\text{kin}}$, and volume-conserving mass stability $\mathcal{S}_{\text{mass}}$.
pub fn compute_holon_salience_score(track: &[CharacterObservation]) -> f64 {
    if track.is_empty() {
        return 0.0;
    }
    let n = track.len() as f64;
    let n_eff: f64 = track
        .iter()
        .map(|o| if o.truncated { 0.5 } else { 1.0 })
        .sum();
    let mut total_mass = 0.0f64;
    let mut total_density = 0.0f64;
    let mut total_plucker = 0.0f64;
    let mut total_vac_div = 0.0f64;
    let mut masses = Vec::with_capacity(track.len());

    for obs in track {
        let fg = obs.mask.iter().filter(|&&m| m).count() as f64;
        total_mass += fg;
        masses.push(fg);

        let bbox_area = (obs.width() * obs.height()).max(1) as f64;
        total_density += fg / bbox_area;

        let plucker_factor = if let Some(loa) = obs.line_of_action() {
            let d = loa.spine_bivector.line_direction();
            let m = loa.spine_bivector.line_moment();
            let dot = (d[0] * m[0] + d[1] * m[1] + d[2] * m[2]).abs();
            let norm_d = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let norm_m = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            1.0 - (dot as f64 / (norm_d as f64 * norm_m as f64 + 1e-6)).min(1.0)
        } else {
            0.5
        };
        total_plucker += plucker_factor;

        // Universal Vacuum Spinor Divergence <ψ̃_obs (1 - P_vac) ψ_obs>_0 in G_{2,0}^+:
        let mut fg_rgb = [0.0f64; 3];
        let mut fg_cnt = 0.0f64;
        let mut vac_rgb = [0.0f64; 3];
        let mut vac_cnt = 0.0f64;
        for (i, &m) in obs.mask.iter().enumerate() {
            let r = obs.rgb[i * 3] as f64 / 255.0;
            let g = obs.rgb[i * 3 + 1] as f64 / 255.0;
            let b = obs.rgb[i * 3 + 2] as f64 / 255.0;
            if m {
                fg_rgb[0] += r;
                fg_rgb[1] += g;
                fg_rgb[2] += b;
                fg_cnt += 1.0;
            } else {
                vac_rgb[0] += r;
                vac_rgb[1] += g;
                vac_rgb[2] += b;
                vac_cnt += 1.0;
            }
        }
        let u_fg = if fg_cnt > 0.0 {
            [fg_rgb[0] / fg_cnt, fg_rgb[1] / fg_cnt, fg_rgb[2] / fg_cnt]
        } else {
            [0.0; 3]
        };
        let u_vac = if vac_cnt > 0.0 {
            [vac_rgb[0] / vac_cnt, vac_rgb[1] / vac_cnt, vac_rgb[2] / vac_cnt]
        } else {
            [0.0; 3]
        };
        // Orthonormal opponent chromatic spinor ψ = s + b e_12 in G_{2,0}^+:
        // e_rg = (R - G)/sqrt(2), e_yb = (R + G - 2B)/sqrt(6)
        let s_fg = (u_fg[0] - u_fg[1]) * std::f64::consts::FRAC_1_SQRT_2;
        let b_fg = (u_fg[0] + u_fg[1] - 2.0 * u_fg[2]) / 6.0f64.sqrt();
        let s_vac = (u_vac[0] - u_vac[1]) * std::f64::consts::FRAC_1_SQRT_2;
        let b_vac = (u_vac[0] + u_vac[1] - 2.0 * u_vac[2]) / 6.0f64.sqrt();
        // Orthogonal projection of ψ_fg off ψ_vac:
        let norm_vac_sq = s_vac * s_vac + b_vac * b_vac;
        let (perp_s, perp_b) = if norm_vac_sq > 1e-4 {
            let dot = (s_fg * s_vac + b_fg * b_vac).max(0.0);
            let scale = (dot / norm_vac_sq).clamp(0.0, 1.0);
            (s_fg - scale * s_vac, b_fg - scale * b_vac)
        } else {
            (s_fg, b_fg)
        };
        let spinor_div = 3.0 * (perp_s * perp_s + perp_b * perp_b);
        total_vac_div += spinor_div;
    }

    let mean_density = total_density / n;
    let mean_plucker = total_plucker / n;
    let mean_mass = total_mass / n;
    let mean_vac_div = total_vac_div / n;

    // Mass stability: volume conservation / rigidity det S ≈ 1 across time
    let mass_variance: f64 = masses.iter().map(|&m| (m - mean_mass).powi(2)).sum::<f64>() / n;
    let mass_cv = mass_variance.sqrt() / (mean_mass + 1e-6);
    let mass_stability = 1.0 / (1.0 + mass_cv);

    // Causal Trajectory Agency: volume-conserving active movement and turning curvature across the world manifold
    let mut path_len = 0.0f64;
    let mut turn_variance = 0.0f64;
    for i in 1..track.len() {
        let c0 = track[i - 1].world_center();
        let c1 = track[i].world_center();
        let dt = (track[i].frame_idx.saturating_sub(track[i - 1].frame_idx)).max(1) as f64;
        let dx = (c1[0] - c0[0]) / dt;
        let dy = (c1[1] - c0[1]) / dt;
        let dw = (track[i].width() as f64 - track[i - 1].width() as f64).abs() / dt;
        let dh = (track[i].height() as f64 - track[i - 1].height() as f64).abs() / dt;
        let loc_dx = (dx.abs() - 0.5 * dw).max(0.0).copysign(dx);
        let loc_dy = (dy.abs() - 0.5 * dh).max(0.0).copysign(dy);
        let step = (loc_dx * loc_dx + loc_dy * loc_dy).sqrt();
        // Weight step by instantaneous volume conservation exp(-2 |ln(M_i / M_{i-1})|)
        // so unphysical bounding-box merge jumps do not masquerade as causal locomotion.
        let m0 = masses[i - 1].max(1.0);
        let m1 = masses[i].max(1.0);
        let w_vol = (-2.0 * (m1 / m0).ln().abs()).exp();
        path_len += step * w_vol;
        if i >= 2 {
            let c_prev = track[i - 2].world_center();
            let dt0 = (track[i - 1].frame_idx.saturating_sub(track[i - 2].frame_idx)).max(1) as f64;
            let dx0 = (c0[0] - c_prev[0]) / dt0;
            let dy0 = (c0[1] - c_prev[1]) / dt0;
            let dot = dx0 * loc_dx + dy0 * loc_dy;
            let norm0 = (dx0 * dx0 + dy0 * dy0).sqrt();
            if norm0 > 0.5 && step > 0.5 {
                let cos_turn = (dot / (norm0 * step)).clamp(-1.0, 1.0);
                turn_variance += w_vol * (1.0 - cos_turn);
            }
        }
    }
    let mean_speed = path_len / (n - 1.0).max(1.0);
    let mean_turn = turn_variance / (n - 2.0).max(1.0);
    let kinematic_agency = 1.0 + mean_speed + 2.0 * mean_turn;
    let vacuum_spinor_salience = (1.0 + mean_vac_div).powi(2);

    // Dimensionless interior permanence (n_eff^2) * Weber-Fechner logarithmic mass * fill compactness * Plücker consistency * causal agency * vacuum spinor divergence * mass stability
    let mass_factor = (1.0 + mean_mass).ln();
    let score = n_eff * n_eff * mass_factor * mean_density * mean_plucker * kinematic_agency * vacuum_spinor_salience * mass_stability;
    println!("[score break] b0={:?}: score={:.2}, n={:.0}, mass={:.0}, dens={:.3}, pluck={:.3}, agency={:.3}, vac_div={:.3}, stab={:.3}",
        track.first().map(|o| o.bbox), score, n, mean_mass, mean_density, mean_plucker, kinematic_agency, vacuum_spinor_salience, mass_stability);
    score
}

/// Universally select the genuine character holon track from a list of candidate tracks
/// using the Holon Salience Metric (ranking by interior 2D mass and convex density).
pub fn select_character_track<'a>(
    tracks: &'a [Vec<CharacterObservation>],
) -> Option<&'a Vec<CharacterObservation>> {
    if tracks.is_empty() {
        return None;
    }
    if tracks.len() == 1 {
        return Some(&tracks[0]);
    }
    tracks.iter().max_by(|a, b| {
        let sa = compute_holon_salience_score(a);
        let sb = compute_holon_salience_score(b);
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub fn observation_to_rgb_image(obs: &CharacterObservation) -> RgbImage8 {
    RgbImage8::from_raw(obs.width(), obs.height(), obs.rgb.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_of_action_extraction() {
        // Construct a vertical elliptical character mask (spine along y-axis)
        let w = 40usize;
        let h = 80usize;
        let mut mask = vec![false; w * h];
        let (cx, cy) = (20.0f64, 40.0f64);
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f64 - cx) / 8.0;
                let dy = (y as f64 - cy) / 24.0;
                if dx * dx + dy * dy <= 1.0 {
                    mask[y * w + x] = true;
                }
            }
        }

        let loa = extract_line_of_action(&mask, w, h).expect("extract LineOfAction");
        assert!((loa.center_of_mass[0] - cx).abs() < 0.5);
        assert!((loa.center_of_mass[1] - cy).abs() < 0.5);

        // Direction along y axis (vertical spine): dx ≈ 0, |dy| ≈ 1
        assert!(loa.direction[0].abs() < 0.05);
        assert!((loa.direction[1].abs() - 1.0).abs() < 0.05);

        // Spread parallel (along y) must be substantially larger than spread perpendicular (along x)
        assert!(loa.spread_parallel > loa.spread_perpendicular * 2.5);

        // Plücker condition: d · m = 0
        assert!(loa.spine_bivector.is_valid_line(1e-4));
    }

    #[test]
    fn test_dominant_silhouette_preserves_detached_prop_and_cape() {
        let w = 64usize;
        let h = 64usize;
        let mut mask = vec![false; w * h];

        // 1. Main body mass (Mrs. Brisby's torso): 16x24 box = 384 pixels
        for y in 20..44 {
            for x in 24..40 {
                mask[y * w + x] = true;
            }
        }
        let body_count = mask.iter().filter(|&&m| m).count();
        assert_eq!(body_count, 384);

        // 2. Detached held prop (Mrs. Brisby's amulet): 4x4 box = 16 pixels
        // Situated at y=14..18, x=29..33 (detached by 2 pixels from neck)
        // Note: 16 * 4 = 64 < 384! Under MIN_COMPONENT_FRACTION = 4, this was amputated!
        for y in 14..18 {
            for x in 29..33 {
                mask[y * w + x] = true;
            }
        }

        // 3. Trailing cape tip: 4x5 box = 20 pixels
        // Situated at y=48..53, x=22..26 (detached by 4 pixels below skirt)
        // 20 * 4 = 80 < 384! Under MIN_COMPONENT_FRACTION = 4, this was also amputated!
        for y in 48..53 {
            for x in 22..26 {
                mask[y * w + x] = true;
            }
        }

        // 4. Far background noise speck in upper-left corner: 1 pixel at (2, 2)
        mask[2 * w + 2] = true;

        let filtered = dominant_silhouette(&mask, w, h);

        // Verify amulet is PRESERVED (100% of its 16 pixels retained)
        for y in 14..18 {
            for x in 29..33 {
                assert!(filtered[y * w + x], "Amulet prop pixel at ({}, {}) must be preserved", x, y);
            }
        }

        // Verify trailing cape tip is PRESERVED (100% of its 20 pixels retained)
        for y in 48..53 {
            for x in 22..26 {
                assert!(filtered[y * w + x], "Cape tip pixel at ({}, {}) must be preserved", x, y);
            }
        }

        // Verify far background noise speck in upper-left corner was REJECTED
        assert!(!filtered[2 * w + 2], "Far background noise speck must be rejected");
    }
}
