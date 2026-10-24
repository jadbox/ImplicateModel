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
    compute_perceptual_patch_lpips, evaluate_vs_single_box_baseline, mask_iou,
};
use part_fitting::{
    accumulate_part_wavepacket_kernel, assign_view_idempotent_cells_and_cft_phases,
    fit_part_gaussian_pose, learn_part_prototypes, median_obs_dims,
};

/// Maximum number of sub-holon parts per articulated character ($P \in \{1..8\}$).
pub const MAX_SUB_HOLONS: usize = 8;
/// Canonical local grid resolution ($K \times K$) for each sub-holon's oriented wavepacket kernel.
pub const PART_KERNEL_RES: usize = 24;
/// Maximum number of discrete view-aspect / posture idempotent cells ($\epsilon_{s,\text{view}}$).
pub const MAX_VIEW_IDEMPOTENTS: usize = 8;

/// Single frame observation of a foreground character extracted against the background atlas.
#[derive(Clone, Debug)]
pub struct CharacterObservation {
    pub frame_idx: usize,
    pub bbox: Rect,
    pub mask: Vec<bool>,
    pub rgb: Vec<u8>,
}

impl CharacterObservation {
    pub fn width(&self) -> usize {
        self.bbox.width().max(0) as usize
    }

    pub fn height(&self) -> usize {
        self.bbox.height().max(0) as usize
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
            MAX_VIEW_IDEMPOTENTS,
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
}

/// Automatically determine the optimal sub-holon part count $P \in \{1..8\}$ and view-aspect
/// idempotent count $S_{\text{view}} \in \{1..8\}$ from the sequence's deformation complexity.
pub fn choose_part_and_view_counts(observations: &[CharacterObservation]) -> (usize, usize) {
    if observations.is_empty() {
        return (1, 1);
    }
    let mut ws: Vec<i32> = observations.iter().map(|o| o.bbox.width()).collect();
    let mut hs: Vec<i32> = observations.iter().map(|o| o.bbox.height()).collect();
    ws.sort_unstable();
    hs.sort_unstable();
    let med_w = ws[ws.len() / 2].max(1) as f64;
    let med_h = hs[hs.len() / 2].max(1) as f64;
    let max_dim = med_w.max(med_h);

    let mean_aspect = observations
        .iter()
        .map(|o| o.bbox.width().max(1) as f64 / o.bbox.height().max(1) as f64)
        .sum::<f64>()
        / observations.len() as f64;
    let aspect_var = observations
        .iter()
        .map(|o| {
            let a = o.bbox.width().max(1) as f64 / o.bbox.height().max(1) as f64;
            (a - mean_aspect) * (a - mean_aspect)
        })
        .sum::<f64>()
        / observations.len() as f64;
    let aspect_cv = aspect_var.sqrt() / mean_aspect.max(1e-3);

    if max_dim <= 56.0 && aspect_cv < 0.28 {
        return (1, 1);
    }

    let num_parts = if max_dim >= 96.0 || aspect_cv >= 0.30 {
        6
    } else if max_dim >= 64.0 || aspect_cv >= 0.22 {
        4
    } else {
        3
    };
    let num_views = (observations.len().div_ceil(3)).clamp(2, MAX_VIEW_IDEMPOTENTS);
    (num_parts.min(MAX_SUB_HOLONS), num_views)
}

/// Fit an `ArticulatedHolon` to a tracked character sequence.
pub fn fit_articulated_holon(
    observations: &[CharacterObservation],
    forced_parts: Option<usize>,
) -> ArticulatedHolon {
    let (auto_p, auto_views) = choose_part_and_view_counts(observations);
    let num_parts = forced_parts.unwrap_or(auto_p).clamp(1, MAX_SUB_HOLONS);
    let num_view_cells = if num_parts == 1 {
        (observations.len().div_ceil(3)).clamp(1, MAX_VIEW_IDEMPOTENTS)
    } else {
        auto_views
    };

    if observations.is_empty() {
        return ArticulatedHolon {
            num_parts,
            num_view_cells: 1,
            parent_indices: vec![0; num_parts],
            part_colors: vec![[128, 128, 128]; num_parts],
            kernels: Vec::new(),
            poses_per_obs: Vec::new(),
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
    let (view_assignments, cft_phases) =
        assign_view_idempotent_cells_and_cft_phases(observations, num_view_cells);

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
                (observations[oi].bbox.center(), observations[oi + 1].bbox.center())
            } else {
                (observations[oi - 1].bbox.center(), observations[oi].bbox.center())
            };
            let dx = c_next[0] - c_curr[0];
            let dy = c_next[1] - c_curr[1];
            if dx.abs() + dy.abs() > 0.25 {
                dy.atan2(dx)
            } else if oi > 0 && !poses_per_obs.is_empty() {
                poses_per_obs[oi - 1][0].angle_rad
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
            if num_parts == 1 {
                pose.angle_rad = motion_heading;
            }
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

    let res = if num_parts == 1 {
        let (mw, mh) = median_obs_dims(observations);
        mw.max(mh).clamp(16, 48)
    } else {
        PART_KERNEL_RES
    };

    let mut kernels: Vec<Vec<WavepacketPartKernel>> = Vec::with_capacity(num_view_cells);
    for vc in 0..num_view_cells {
        let mut vc_kernels = Vec::with_capacity(num_parts);
        for p in 0..num_parts {
            vc_kernels.push(accumulate_part_wavepacket_kernel(
                observations,
                &poses_per_obs,
                &part_assignments_per_obs,
                p,
                vc,
                res,
                res,
            ));
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
    }
}

/// Render/splat the `ArticulatedHolon` at observation index `obs_idx`.
pub fn render_articulated_holon_observation(
    holon: &ArticulatedHolon,
    obs: &CharacterObservation,
    obs_idx: usize,
) -> (Vec<bool>, Vec<u8>) {
    let w = obs.width();
    let h = obs.height();
    let mut out_mask = vec![false; w * h];
    let mut out_rgb = vec![0u8; w * h * 3];
    let Some(poses) = holon.poses_per_obs.get(obs_idx) else {
        return (out_mask, out_rgb);
    };
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

/// Extract spatially and chromatically coherent `CharacterObservation` tracks for up to
/// `max_tracks` distinct moving characters/holons from a `MultiPlaneAtlas` and its source frames.
pub fn extract_character_observation_tracks(
    frames: &[crate::video_io::RawFrame],
    atlas: &crate::multiplane_atlas::MultiPlaneAtlas,
    max_tracks: usize,
) -> Vec<Vec<CharacterObservation>> {
    struct Cand {
        obs: CharacterObservation,
        color: [f32; 3],
    }
    let mut tracks: Vec<Vec<Cand>> = Vec::new();

    for (fi, frame) in frames.iter().enumerate() {
        let Some(boxes) = atlas.actor_boxes_per_frame.get(fi) else {
            continue;
        };
        let recon_bg = atlas.reconstruct_background_frame(fi);
        let mut frame_cands: Vec<Cand> = Vec::new();

        for &bbox in boxes {
            let x0 = bbox.x0.clamp(0, frame.width as i32 - 1);
            let y0 = bbox.y0.clamp(0, frame.height as i32 - 1);
            let x1 = bbox.x1.clamp(x0, frame.width as i32 - 1);
            let y1 = bbox.y1.clamp(y0, frame.height as i32 - 1);
            let clipped = Rect::new(x0, y0, x1, y1);
            let bw = clipped.width() as usize;
            let bh = clipped.height() as usize;
            if bw < 6 || bh < 6 {
                continue;
            }

            let mut mask = vec![false; bw * bh];
            let mut rgb = vec![0u8; bw * bh * 3];
            let mut fg_cnt = 0usize;
            let mut col_sum = [0.0f32; 3];

            for ly in 0..bh {
                let fy = y0 as usize + ly;
                for lx in 0..bw {
                    let fx = x0 as usize + lx;
                    let c = frame.pixel(fx as u32, fy as u32);
                    let bg = recon_bg.pixel(fx, fy);
                    let idx = ly * bw + lx;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&c);
                    let d = (c[0] as i32 - bg[0] as i32)
                        .abs()
                        .max((c[1] as i32 - bg[1] as i32).abs())
                        .max((c[2] as i32 - bg[2] as i32).abs());
                    if d > 26 {
                        mask[idx] = true;
                        fg_cnt += 1;
                        col_sum[0] += c[0] as f32;
                        col_sum[1] += c[1] as f32;
                        col_sum[2] += c[2] as f32;
                    }
                }
            }
            if fg_cnt >= 12 {
                let inv = 1.0 / fg_cnt as f32;
                frame_cands.push(Cand {
                    obs: CharacterObservation {
                        frame_idx: fi,
                        bbox: clipped,
                        mask,
                        rgb,
                    },
                    color: [col_sum[0] * inv, col_sum[1] * inv, col_sum[2] * inv],
                });
            }
        }

        let mut used_track = vec![false; tracks.len()];
        for cand in frame_cands {
            let cc = cand.obs.bbox.center();
            let mut best_ti = None;
            let mut best_cost = f64::MAX;
            for (ti, tr) in tracks.iter().enumerate() {
                if used_track[ti] {
                    continue;
                }
                let last = tr.last().unwrap();
                if fi.saturating_sub(last.obs.frame_idx) > 4 {
                    continue;
                }
                let lc = last.obs.bbox.center();
                let dist = (cc[0] - lc[0]).hypot(cc[1] - lc[1]);
                let cdist = (cand.color[0] - last.color[0])
                    .abs()
                    .max((cand.color[1] - last.color[1]).abs())
                    .max((cand.color[2] - last.color[2]).abs()) as f64;
                let dw = (cand.obs.bbox.width() - last.obs.bbox.width()).abs();
                let dh = (cand.obs.bbox.height() - last.obs.bbox.height()).abs();
                if dist <= 32.0 && cdist <= 55.0 && dw <= 10 && dh <= 10 {
                    let cost = dist + 0.4 * cdist;
                    if cost < best_cost {
                        best_cost = cost;
                        best_ti = Some(ti);
                    }
                }
            }
            if let Some(ti) = best_ti {
                used_track[ti] = true;
                tracks[ti].push(cand);
            } else {
                tracks.push(vec![cand]);
                used_track.push(true);
            }
        }
    }

    let mut scored: Vec<(usize, Vec<CharacterObservation>)> = tracks
        .into_iter()
        .filter(|tr| tr.len() >= 4 || max_tracks == 1)
        .filter(|tr| {
            if tr.len() <= 1 {
                return true;
            }
            let c0 = tr[0].obs.bbox.center();
            let mut max_disp = 0.0f64;
            for cand in tr {
                let c = cand.obs.bbox.center();
                let d = (c[0] - c0[0]).hypot(c[1] - c0[1]);
                if d > max_disp {
                    max_disp = d;
                }
            }
            max_disp > 2.0 || max_tracks == 1
        })
        .map(|tr| {
            let total_fg: usize = tr
                .iter()
                .map(|c| c.obs.mask.iter().filter(|&&m| m).count())
                .sum();
            let score = tr.len() * 1000 + total_fg;
            let obs_vec = tr.into_iter().map(|c| c.obs).collect();
            (score, obs_vec)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .take(max_tracks.max(1))
        .map(|(_, obs)| obs)
        .collect()
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
