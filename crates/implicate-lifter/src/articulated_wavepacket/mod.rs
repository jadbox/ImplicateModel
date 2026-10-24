//! Stage 2: Articulated Sub-Holon Decomposition & Oriented Gaussian-Gabor Wavepackets (`articulated_wavepacket`).
//!
//! Split into:
//! - `mod`: `CharacterObservation`, `SubHolonPose`, `WavepacketPartKernel`, `ArticulatedHolon`, `fit_articulated_holon`.
//! - `clifford_features`: `CliffordNet` Laplacian wedge-vorticity features ($D^{(s)} \oplus W^{(s)}$), IoU, and LPIPS metrics.
//! - `part_fitting`: K-means part prototype learning, view-aspect idempotent clustering, and Gaussian-Gabor kernel accumulation.

pub mod clifford_features;
pub mod part_fitting;

use implicate_math::{CliffordDensityElement, Motor};
use serde::{Deserialize, Serialize};

use crate::enfold2d::{Rect, RgbImage8};

pub use clifford_features::{
    ArticulatedComparisonReport, compute_cliffordnet_laplacian_features,
    compute_perceptual_patch_lpips, evaluate_vs_single_box_baseline,
    evaluate_vs_single_box_baseline_with_parts, mask_iou,
};
use part_fitting::{
    accumulate_part_wavepacket_kernel, assign_view_idempotent_cells_and_cft_phases,
    fit_part_gaussian_pose, learn_part_prototypes, median_obs_dims,
};

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
}

/// 2D anisotropic Gaussian pose and covariance parameters for a sub-holon part $p$ at frame $t$.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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
    if rec >= HOLON_MODEL_QUALITY_IOU {
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
    let num_view_cells = view_count_for(defining_obs.len(), num_parts);
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

    // Rigid holons use a pixel-native kernel frame (1 texel = 1 source pixel) with half-extents
    // shared by all observations, registered per view cell; articulated parts use a canonical grid.
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

/// Robust outlier bound in median absolute deviations (the conventional 3-sigma-equivalent).
const OUTLIER_MADS: f64 = 3.0;
/// Scale factor making the MAD a consistent estimator of the standard deviation.
const MAD_TO_SIGMA: f64 = 1.4826;

/// Observations whose log silhouette area deviates from the track median by more than
/// `OUTLIER_MADS` robust standard deviations: stable bodies flag merges, deforming bodies tolerate
/// their own range of motion.
fn area_outliers(observations: &[CharacterObservation]) -> Vec<bool> {
    let la: Vec<f64> = observations
        .iter()
        .map(|o| (o.mask.iter().filter(|&&m| m).count().max(1) as f64).ln())
        .collect();
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let m = median(&mut la.clone());
    let sigma = MAD_TO_SIGMA * median(&mut la.iter().map(|a| (a - m).abs()).collect());
    la.iter().map(|a| sigma > 0.0 && (a - m).abs() > OUTLIER_MADS * sigma).collect()
}

/// Register rigid ($P = 1$) observations into one pixel-native frame: every pose gets the shared
/// half-extents of a `res_w x res_h` kernel, and its centre is shifted by the integer offset that
/// maximizes silhouette agreement with its view cell's reference (largest-area) observation.
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
        let Some(&ref_i) = members
            .iter()
            .filter(|&&i| clean[i])
            .max_by_key(|&&i| area(&observations[i]))
            .or_else(|| members.iter().max_by_key(|&&i| area(&observations[i])))
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

/// Per-channel residual (8-bit) above which a pixel differs from the reconstructed background.
const FOREGROUND_RESIDUAL: i32 = 26;
/// Minimum foreground pixel count for an observation to carry a recognizable silhouette.
const MIN_OBSERVATION_PIXELS: usize = 12;
/// Maximum frame gap bridged by frame-to-frame linking (short flicker / missed detections).
const MAX_LINK_GAP: usize = 4;
/// Maximum chroma-weighted colour distance (8-bit max-channel) for two observations of one holon.
const MAX_APPEARANCE_DIST: f32 = 55.0;
/// Maximum ratio between successive silhouette extents of one holon.
const MAX_EXTENT_RATIO: f64 = 1.6;
/// Minimum number of observations for a track to define a holon.
const MIN_TRACK_LEN: usize = 4;

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

/// Screen pixels whose value never departs from the first frame by more than the foreground
/// residual. Such pixels are camera-locked (HUDs, letter/pillar-box bars, masked borders, logos)
/// and cannot contain a moving holon. `None` when the shot is too short to tell.
fn screen_static_mask(frames: &[crate::video_io::RawFrame]) -> Option<Vec<bool>> {
    let first = frames.first()?;
    if frames.len() < MIN_TRACK_LEN {
        return None;
    }
    let (w, h) = (first.width as usize, first.height as usize);
    let mut stat = vec![true; w * h];
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
                if (0..3).any(|k| (a[k] as i32 - b[k] as i32).abs() > FOREGROUND_RESIDUAL) {
                    stat[i] = false;
                }
            }
        }
    }
    Some(stat)
}

/// Components smaller than this fraction of the largest are residual debris, not body parts.
const MIN_COMPONENT_FRACTION: usize = 4;

/// Keep the largest 8-connected component of `mask` plus any component at least
/// `1 / MIN_COMPONENT_FRACTION` of its area (detached limbs, props held by the body).
pub fn dominant_silhouette(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
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
    let largest = sizes.iter().copied().max().unwrap_or(0);
    label
        .iter()
        .map(|&l| l != 0 && sizes[l as usize] * MIN_COMPONENT_FRACTION >= largest)
        .collect()
}

fn color_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs())
}

struct TrackCand {
    obs: CharacterObservation,
    color: [f32; 3],
    /// Silhouette centre in world (atlas) coordinates: screen + camera trajectory.
    world: [f64; 2],
}

impl TrackCand {
    fn extent(&self) -> f64 {
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

/// Number of observations at a tracklet end used for robust (median) extent estimates.
const EXTENT_WINDOW: usize = 5;

/// Median silhouette extent `(w, h)` over the last (or first) `EXTENT_WINDOW` observations, so a
/// single observation contaminated by a merge does not redefine the holon's size.
fn robust_extent(tr: &[TrackCand], at_end: bool) -> (usize, usize) {
    let n = tr.len().min(EXTENT_WINDOW);
    let slice = if at_end { &tr[tr.len() - n..] } else { &tr[..n] };
    let mut ws: Vec<usize> = slice.iter().map(|c| c.obs.width()).collect();
    let mut hs: Vec<usize> = slice.iter().map(|c| c.obs.height()).collect();
    ws.sort_unstable();
    hs.sort_unstable();
    (ws[ws.len() / 2], hs[hs.len() / 2])
}

fn extents_compatible(a: (usize, usize), b: (usize, usize)) -> bool {
    let r = |x: usize, y: usize| x.max(y) as f64 / x.min(y).max(1) as f64;
    r(a.0, b.0) <= MAX_EXTENT_RATIO && r(a.1, b.1) <= MAX_EXTENT_RATIO
}

/// Extract spatially and chromatically coherent `CharacterObservation` tracks for up to
/// `max_tracks` distinct moving characters/holons from a `MultiPlaneAtlas` and its source frames.
///
/// Kinematics are evaluated in world coordinates (screen + camera trajectory) so scenery that merely
/// scrolls with the camera is recognised as stationary. Linking gates are relative to each holon's
/// own extent and velocity; fragments split by occlusion are re-joined by reachability.
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
    let mut tracks: Vec<Vec<TrackCand>> = Vec::new();
    let screen_static = screen_static_mask(frames);

    for (fi, frame) in frames.iter().enumerate() {
        let Some(boxes) = atlas.actor_boxes_per_frame.get(fi) else {
            continue;
        };
        let recon_bg = atlas.reconstruct_background_frame(fi);
        let mut frame_cands: Vec<TrackCand> = Vec::new();

        for &bbox in boxes {
            let x0 = bbox.x0.clamp(0, frame.width as i32 - 1);
            let y0 = bbox.y0.clamp(0, frame.height as i32 - 1);
            let x1 = bbox.x1.clamp(x0, frame.width as i32 - 1);
            let y1 = bbox.y1.clamp(y0, frame.height as i32 - 1);

            // Residual mask inside the (dilated) actor box, reduced to the dominant silhouette.
            let (bw0, bh0) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);
            let mut residual = vec![false; bw0 * bh0];
            for fy in y0..=y1 {
                for fx in x0..=x1 {
                    let c = frame.pixel(fx as u32, fy as u32);
                    let bg = recon_bg.pixel(fx as usize, fy as usize);
                    let d = (0..3).map(|k| (c[k] as i32 - bg[k] as i32).abs()).max().unwrap_or(0);
                    let locked = screen_static
                        .as_ref()
                        .is_some_and(|m| m[fy as usize * frame.width as usize + fx as usize]);
                    residual[(fy - y0) as usize * bw0 + (fx - x0) as usize] = d > FOREGROUND_RESIDUAL && !locked;
                }
            }
            let silhouette = dominant_silhouette(&residual, bw0, bh0);
            let (mut tx0, mut ty0, mut tx1, mut ty1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            let mut fg_cnt = 0usize;
            for (i, _) in silhouette.iter().enumerate().filter(|(_, m)| **m) {
                let (fx, fy) = (x0 + (i % bw0) as i32, y0 + (i / bw0) as i32);
                fg_cnt += 1;
                tx0 = tx0.min(fx);
                ty0 = ty0.min(fy);
                tx1 = tx1.max(fx);
                ty1 = ty1.max(fy);
            }
            if fg_cnt < MIN_OBSERVATION_PIXELS {
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
                    mask[idx] = silhouette[(fy - y0 as usize) * bw0 + (fx - x0 as usize)];
                }
            }
            let color = chroma_weighted_color(&rgb, &mask);
            let cam = camera(fi);
            let c = tight.center();
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
                },
                color,
                world: [c[0] + cam[0], c[1] + cam[1]],
            });
        }

        // Global greedy assignment of candidates to live tracks by normalized prediction cost.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ci, cand) in frame_cands.iter().enumerate() {
            for (ti, tr) in tracks.iter().enumerate() {
                let last = tr.last().unwrap();
                let gap = fi.saturating_sub(last.obs.frame_idx);
                if gap == 0 || gap > MAX_LINK_GAP {
                    continue;
                }
                if !extents_compatible((cand.obs.width(), cand.obs.height()), robust_extent(tr, true)) {
                    continue;
                }
                let cdist = color_dist(cand.color, last.color);
                if cdist > MAX_APPEARANCE_DIST {
                    continue;
                }
                let v = tracklet_velocity(tr, true);
                let pred = [last.world[0] + v[0] * gap as f64, last.world[1] + v[1] * gap as f64];
                let err = (cand.world[0] - pred[0]).hypot(cand.world[1] - pred[1]);
                // One body extent of unexplained displacement per elapsed frame.
                let scale = last.extent().max(cand.extent()) * gap as f64;
                if err > scale {
                    continue;
                }
                pairs.push((err / scale + (cdist / MAX_APPEARANCE_DIST) as f64, ci, ti));
            }
        }
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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

    let mut scored: Vec<(usize, Vec<CharacterObservation>)> = tracks
        .into_iter()
        .filter(|tr| tr.len() >= MIN_TRACK_LEN || max_tracks == 1)
        .filter(|tr| {
            if max_tracks == 1 || tr.len() <= 1 {
                return true;
            }
            // A holon must move relative to the world by more than half its own extent. The robust
            // (inter-decile) spread of world positions ignores centroid jitter from partial silhouettes.
            let spread = |axis: usize| {
                let mut v: Vec<f64> = tr.iter().map(|c| c.world[axis]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
                q(0.9) - q(0.1)
            };
            let mean_extent = tr.iter().map(TrackCand::extent).sum::<f64>() / tr.len() as f64;
            spread(0).hypot(spread(1)) > 0.5 * mean_extent
        })
        .map(|tr| {
            let total_fg: usize = tr.iter().map(|c| c.obs.mask.iter().filter(|&&m| m).count()).sum();
            let score = tr.len() * 1000 + total_fg;
            (score, tr.into_iter().map(|c| c.obs).collect())
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(max_tracks.max(1)).map(|(_, obs)| obs).collect()
}

/// Re-join tracklets fragmented by occlusion or silhouette merging: tracklet `B` continues `A`
/// when it starts after `A` ends, shares its appearance and extent, and lies within the region
/// reachable at the pair's observed speed plus one body extent.
fn stitch_tracklets(tracks: &mut Vec<Vec<TrackCand>>) {
    loop {
        let mut best: Option<(f64, usize, usize)> = None;
        for (ai, a) in tracks.iter().enumerate() {
            let a_last = a.last().unwrap();
            let va = tracklet_velocity(a, true);
            for (bi, b) in tracks.iter().enumerate() {
                if ai == bi {
                    continue;
                }
                let b_first = &b[0];
                if b_first.obs.frame_idx <= a_last.obs.frame_idx {
                    continue;
                }
                if !extents_compatible(robust_extent(a, true), robust_extent(b, false)) {
                    continue;
                }
                let cdist = color_dist(a_last.color, b_first.color);
                if cdist > MAX_APPEARANCE_DIST {
                    continue;
                }
                let gap = (b_first.obs.frame_idx - a_last.obs.frame_idx) as f64;
                let vb = tracklet_velocity(b, false);
                let speed = va[0].hypot(va[1]).max(vb[0].hypot(vb[1]));
                let reach = a_last.extent().max(b_first.extent()) + speed * gap;
                let d = (b_first.world[0] - a_last.world[0]).hypot(b_first.world[1] - a_last.world[1]);
                if d > reach {
                    continue;
                }
                let cost = d / reach + (cdist / MAX_APPEARANCE_DIST) as f64;
                if best.is_none_or(|(c, _, _)| cost < c) {
                    best = Some((cost, ai, bi));
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

pub fn observation_to_rgb_image(obs: &CharacterObservation) -> RgbImage8 {
    RgbImage8::from_raw(obs.width(), obs.height(), obs.rgb.clone())
}
