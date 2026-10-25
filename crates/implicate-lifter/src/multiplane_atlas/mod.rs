//! Stage 1: Multi-Plane Depth-Stratified Panorama Reconstruction (`multiplane_atlas`).
//!
//! Split into:
//! - `mod`: `ParallaxLayer`, `LuminescenceFrame`, `MultiPlaneOptions`, `MultiPlaneAtlas`, `build_multiplane_atlas`.
//! - `layer_accum`: Band CFT trajectory estimation, $\pm 2\text{ px}$ L1 refinement, and actor-masked mode inpainting.

pub mod layer_accum;

use serde::{Deserialize, Serialize};

use crate::enfold2d::{
    Rect, RgbImage8, WorldAtlas, color_dist, connected_boxes, partition_hud_and_viewport,
};
use crate::video_io::RawFrame;
use layer_accum::{
    accumulate_band_atlas, accumulate_band_atlas_masked, band_registration_error,
    band_spatial_gradient, estimate_band_trajectory, trajectory_rms_diff,
};

/// Maximum number of discrete multi-plane depth layers ($L \in \{1..4\}$).
pub const MAX_PARALLAX_LAYERS: usize = 4;

/// A single reconstructed depth plane $A_l(\mathbf{X}_l)$ in a `MultiPlaneAtlas`.
#[derive(Clone, Debug)]
pub struct ParallaxLayer {
    pub layer_index: usize,
    pub inverse_depth: f32,
    pub atlas: WorldAtlas,
    pub alpha: Vec<u8>,
    pub trajectory: Vec<[i32; 2]>,
    pub y_band: (usize, usize),
}

impl ParallaxLayer {
    #[inline]
    pub fn sample_rgba(&self, wx: i32, wy: i32) -> Option<[u8; 4]> {
        let lx = wx - self.atlas.origin[0];
        let ly = wy - self.atlas.origin[1];
        if lx < 0
            || ly < 0
            || lx >= self.atlas.image.width as i32
            || ly >= self.atlas.image.height as i32
        {
            return None;
        }
        let ux = lx as usize;
        let uy = ly as usize;
        let a = self.alpha[uy * self.atlas.image.width + ux];
        if a == 0 {
            return None;
        }
        let rgb = self.atlas.image.pixel(ux, uy);
        Some([rgb[0], rgb[1], rgb[2], a])
    }
}

/// Extracted additive luminescence cel excitation for a frame.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LuminescenceFrame {
    pub frame_idx: usize,
    pub glow_boxes: Vec<Rect>,
    pub mean_glow_rgb: [f32; 3],
    pub active_fraction: f32,
}

/// Options for multi-plane depth-stratified atlas reconstruction.
#[derive(Clone, Debug)]
pub struct MultiPlaneOptions {
    pub max_layers: usize,
    pub active_y_range: Option<(usize, usize)>,
    pub actor_residual_threshold: i32,
    pub actor_min_area: usize,
    pub actor_mask_dilation: i32,
    pub held_out_frames: Vec<usize>,
}

impl Default for MultiPlaneOptions {
    fn default() -> Self {
        Self {
            max_layers: MAX_PARALLAX_LAYERS,
            active_y_range: None,
            actor_residual_threshold: 24,
            actor_min_area: 32,
            actor_mask_dilation: 5,
            held_out_frames: Vec::new(),
        }
    }
}

/// Complete multi-plane depth-stratified atlas and foreground residual tracks for a shot.
#[derive(Clone, Debug)]
pub struct MultiPlaneAtlas {
    pub width: usize,
    pub height: usize,
    pub active_y_range: (usize, usize),
    pub layers: Vec<ParallaxLayer>,
    /// Optional camera-locked ($\mathbf{v}_{\text{screen}} = \mathbf{0}$) HUD holon plane.
    pub camera_locked_hud: Option<ParallaxLayer>,
    pub actor_boxes_per_frame: Vec<Vec<Rect>>,
    pub actor_masks_per_frame: Vec<Vec<bool>>,
    pub luminescence_per_frame: Vec<LuminescenceFrame>,
}

impl MultiPlaneAtlas {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn reconstruct_background_frame(&self, frame_idx: usize) -> RgbImage8 {
        let mut out = RgbImage8::new(self.width, self.height);
        if self.layers.is_empty() && self.camera_locked_hud.is_none() {
            return out;
        }
        for layer in self.layers.iter().rev() {
            let off = layer.trajectory.get(frame_idx).copied().unwrap_or([0, 0]);
            for y in layer.y_band.0..layer.y_band.1.min(self.height) {
                let wy = y as i32 + off[1];
                for x in 0..self.width {
                    let wx = x as i32 + off[0];
                    if let Some([r, g, b, a]) = layer.sample_rgba(wx, wy) {
                        if a == 255 {
                            out.set_pixel(x, y, [r, g, b]);
                        } else if a > 0 {
                            let prev = out.pixel(x, y);
                            let af = a as u32;
                            let inv = 255 - af;
                            out.set_pixel(
                                x,
                                y,
                                [
                                    ((r as u32 * af + prev[0] as u32 * inv) / 255) as u8,
                                    ((g as u32 * af + prev[1] as u32 * inv) / 255) as u8,
                                    ((b as u32 * af + prev[2] as u32 * inv) / 255) as u8,
                                ],
                            );
                        }
                    }
                }
            }
        }
        if let Some(hud) = &self.camera_locked_hud {
            for y in hud.y_band.0..hud.y_band.1.min(self.height) {
                for x in 0..self.width {
                    if let Some([r, g, b, a]) = hud.sample_rgba(x as i32, y as i32) {
                        if a > 0 {
                            out.set_pixel(x, y, [r, g, b]);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn evaluate_background_psnr(&self, frame_idx: usize, frame: &RawFrame) -> f64 {
        let recon = self.reconstruct_background_frame(frame_idx);
        let (y0, y1) = self.active_y_range;
        let mask = self.actor_masks_per_frame.get(frame_idx);
        let boxes = self
            .actor_boxes_per_frame
            .get(frame_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let mut sse = 0.0f64;
        let mut count = 0usize;
        for y in y0..y1.min(self.height) {
            for x in 0..self.width {
                let in_actor = mask.map_or(false, |m| m[y * self.width + x])
                    || boxes.iter().any(|r| r.contains(x as i32, y as i32));
                if in_actor {
                    continue;
                }
                let p = frame.pixel(x as u32, y as u32);
                let q = recon.pixel(x, y);
                let dr = p[0] as f64 - q[0] as f64;
                let dg = p[1] as f64 - q[1] as f64;
                let db = p[2] as f64 - q[2] as f64;
                sse += (dr * dr + dg * dg + db * db) / 3.0;
                count += 1;
            }
        }
        if count == 0 {
            return 99.0;
        }
        let mse = sse / count as f64;
        if mse <= 1e-6 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        }
    }

    pub fn measure_ghost_contamination_ratio(&self, frames: &[RawFrame]) -> f64 {
        if frames.len() < 4 || self.layers.is_empty() {
            return 0.0;
        }
        let (y0, y1) = self.active_y_range;
        let mut checked_loci = 0usize;
        let mut ghost_loci = 0usize;

        for (t, boxes) in self.actor_boxes_per_frame.iter().enumerate() {
            for b in boxes {
                let cx = ((b.x0 + b.x1) / 2).clamp(0, self.width as i32 - 1) as usize;
                let cy = ((b.y0 + b.y1) / 2).clamp(y0 as i32, y1 as i32 - 1) as usize;
                let actor_rgb = frames[t].pixel(cx as u32, cy as u32);
                let recon_t = self.reconstruct_background_frame(t);
                let bg_rgb = recon_t.pixel(cx, cy);

                let Some(layer) = self
                    .layers
                    .iter()
                    .find(|l| cy >= l.y_band.0 && cy < l.y_band.1)
                    .or_else(|| self.layers.first())
                else {
                    continue;
                };
                let wx = cx as i32 + layer.trajectory[t][0];
                let wy = cy as i32 + layer.trajectory[t][1];

                for (t_other, f_other) in frames.iter().enumerate() {
                    if t_other.abs_diff(t) < 16 {
                        continue;
                    }
                    let ox = wx - layer.trajectory[t_other][0];
                    let oy = wy - layer.trajectory[t_other][1];
                    if ox < 0
                        || oy < y0 as i32
                        || ox >= self.width as i32
                        || oy >= y1 as i32
                        || b.expand(8).contains(ox, oy)
                    {
                        continue;
                    }
                    let occupied_other = self.actor_masks_per_frame[t_other]
                        [oy as usize * self.width + ox as usize]
                        || self.actor_boxes_per_frame[t_other]
                            .iter()
                            .any(|r| r.expand(6).contains(ox, oy));
                    if occupied_other {
                        continue;
                    }
                    let true_bg = f_other.pixel(ox as u32, oy as u32);
                    checked_loci += 1;
                    let d_true = color_dist(bg_rgb, true_bg);
                    let d_actor = color_dist(bg_rgb, actor_rgb);
                    if d_true > 44 && d_actor + 18 < d_true {
                        ghost_loci += 1;
                    }
                    break;
                }
            }
        }
        if checked_loci == 0 {
            0.0
        } else {
            ghost_loci as f64 / checked_loci as f64
        }
    }
}

/// Reconstruct a `MultiPlaneAtlas` from a contiguous shot's frames.
pub fn build_multiplane_atlas(frames: &[RawFrame], opts: &MultiPlaneOptions) -> MultiPlaneAtlas {
    if frames.is_empty() {
        return MultiPlaneAtlas {
            width: 0,
            height: 0,
            active_y_range: (0, 0),
            layers: Vec::new(),
            camera_locked_hud: None,
            actor_boxes_per_frame: Vec::new(),
            actor_masks_per_frame: Vec::new(),
            luminescence_per_frame: Vec::new(),
        };
    }

    let w = frames[0].width as usize;
    let h = frames[0].height as usize;
    let traj_span = |tr: &[[i32; 2]]| -> f32 {
        tr.iter()
            .map(|&[dx, dy]| ((dx * dx + dy * dy) as f32).sqrt())
            .fold(0.0f32, f32::max)
    };
    let full_probe = estimate_band_trajectory(frames, 0, h);
    let camera_is_static = traj_span(&full_probe) <= 3.5;

    let (mut y0, mut y1) = opts.active_y_range.unwrap_or_else(|| {
        if camera_is_static {
            (0, h)
        } else {
            let hud = partition_hud_and_viewport(frames);
            (0, hud.top.min(h))
        }
    });
    y0 = y0.min(h.saturating_sub(1));
    y1 = y1.clamp(y0 + 1, h);
    let active_h = y1 - y0;

    let n_bands = opts.max_layers.clamp(1, MAX_PARALLAX_LAYERS);
    let band_h = (active_h / n_bands).max(8);
    let mut raw_trajs: Vec<Vec<[i32; 2]>> = Vec::with_capacity(n_bands);
    let mut raw_grads: Vec<f64> = Vec::with_capacity(n_bands);
    let mut band_ranges: Vec<(usize, usize)> = Vec::with_capacity(n_bands);

    for b in 0..n_bands {
        let by0 = y0 + b * band_h;
        let by1 = if b + 1 == n_bands {
            y1
        } else {
            (by0 + band_h).min(y1)
        };
        if by1 <= by0 + 4 {
            continue;
        }
        band_ranges.push((by0, by1));
        raw_grads.push(band_spatial_gradient(frames, by0, by1));
        raw_trajs.push(estimate_band_trajectory(frames, by0, by1));
    }

    let max_world_span = raw_trajs.iter().map(|t| traj_span(t)).fold(0.0f32, f32::max);

    // Isolate a Camera-Locked HUD Holon band at the bottom or top border if partitioned by
    // `partition_hud_and_viewport` or if its CFT trajectory is stationary while the world translates.
    let zero_traj = vec![[0i32, 0]; frames.len()];
    let mut hud_band: Option<(usize, usize)> = if !camera_is_static && y1 < h {
        Some((y1, h))
    } else if !camera_is_static && y0 > 0 {
        Some((0, y0))
    } else {
        None
    };
    if hud_band.is_none() && !camera_is_static && max_world_span >= 8.0 && band_ranges.len() >= 2 {
        let last_idx = band_ranges.len() - 1;
        let (by0, by1) = band_ranges[last_idx];
        let static_err = band_registration_error(frames, by0, by1, &zero_traj);
        if traj_span(&raw_trajs[last_idx]) <= 2.5 && static_err < 4.5 {
            hud_band = Some((by0, h));
            y1 = by0;
            band_ranges.pop();
            raw_trajs.pop();
            raw_grads.pop();
        } else if traj_span(&raw_trajs[0]) <= 2.5
            && band_registration_error(frames, band_ranges[0].0, band_ranges[0].1, &zero_traj) < 4.5
        {
            hud_band = Some((0, band_ranges[0].1));
            y0 = band_ranges[0].1;
            band_ranges.remove(0);
            raw_trajs.remove(0);
            raw_grads.remove(0);
        }
    }

    let full_traj = estimate_band_trajectory(frames, y0, y1);
    let mut best_traj = full_traj;
    let mut best_err = band_registration_error(frames, y0, y1, &best_traj);
    for tr in &raw_trajs {
        let err = band_registration_error(frames, y0, y1, tr);
        if err < best_err {
            best_err = err;
            best_traj = tr.clone();
        }
    }
    let global_traj = best_traj;

    let mut band_trajs: Vec<Vec<[i32; 2]>> = Vec::with_capacity(band_ranges.len());
    for (i, &(by0, by1)) in band_ranges.iter().enumerate() {
        let grad = raw_grads[i];
        let tr = &raw_trajs[i];
        let cand_traj = if grad >= 8.0 {
            let err_cand = band_registration_error(frames, by0, by1, tr);
            let err_glob = band_registration_error(frames, by0, by1, &global_traj);
            if trajectory_rms_diff(tr, &global_traj) > 3.0 && err_cand + 2.0 < err_glob * 0.80 {
                tr.clone()
            } else {
                global_traj.clone()
            }
        } else {
            global_traj.clone()
        };
        band_trajs.push(cand_traj);
    }

    let mut merged_trajs: Vec<Vec<[i32; 2]>> = Vec::new();
    let mut merged_bands: Vec<(usize, usize)> = Vec::new();

    for (traj, range) in band_trajs.into_iter().zip(band_ranges.into_iter()) {
        if let Some(last_idx) = merged_trajs.len().checked_sub(1) {
            if trajectory_rms_diff(&merged_trajs[last_idx], &traj) <= 2.2 {
                merged_bands[last_idx].1 = range.1;
                continue;
            }
        }
        merged_trajs.push(traj);
        merged_bands.push(range);
    }

    let mut layer_spans: Vec<f32> = merged_trajs
        .iter()
        .map(|tr| {
            tr.iter()
                .map(|&[dx, dy]| ((dx * dx + dy * dy) as f32).sqrt())
                .fold(0.0f32, f32::max)
        })
        .collect();
    let max_span = layer_spans.iter().copied().fold(0.0f32, f32::max).max(1.0);
    for sp in &mut layer_spans {
        *sp = (*sp / max_span).clamp(0.1, 1.0);
    }

    let n_layers = merged_trajs.len();
    let mut pass1_layers: Vec<WorldAtlas> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (by0, by1) = merged_bands[l];
        let py0 = if l == 0 { 0 } else { by0.saturating_sub(4) };
        let py1 = if l + 1 == n_layers { h } else { (by1 + 4).min(h) };
        pass1_layers.push(accumulate_band_atlas(
            frames,
            &merged_trajs[l],
            py0,
            py1,
            &[],
            &opts.held_out_frames,
        ));
    }

    let n_frames = frames.len();
    let mut raw_actor_masks: Vec<Vec<bool>> = vec![vec![false; w * h]; n_frames];
    let mut luminescence_per_frame: Vec<LuminescenceFrame> = Vec::with_capacity(n_frames);

    for (fi, frame) in frames.iter().enumerate() {
        let mut glow_mask = vec![false; w * h];
        let mut glow_sum = [0.0f32; 3];
        let mut glow_cnt = 0usize;

        for y in y0..y1 {
            for x in 0..w {
                let c = frame.pixel(x as u32, y as u32);
                let mut best_dist = i32::MAX;
                let mut base_luma = 0.0f32;
                for l in 0..n_layers {
                    let (by0, by1) = merged_bands[l];
                    if !((y >= by0.saturating_sub(4) && y < by1 + 4) || n_layers == 1) {
                        continue;
                    }
                    let off = merged_trajs[l][fi];
                    let (wx, wy) = (x as i32 + off[0], y as i32 + off[1]);
                    for dy_a in -1..=1 {
                        for dx_a in -1..=1 {
                            if let Some(bg) = pass1_layers[l].pixel(wx + dx_a, wy + dy_a) {
                                let d = (c[0] as i32 - bg[0] as i32)
                                    .abs()
                                    .max((c[1] as i32 - bg[1] as i32).abs())
                                    .max((c[2] as i32 - bg[2] as i32).abs());
                                if d < best_dist {
                                    best_dist = d;
                                    base_luma = 0.299 * bg[0] as f32
                                        + 0.587 * bg[1] as f32
                                        + 0.114 * bg[2] as f32;
                                }
                            }
                        }
                    }
                }

                if best_dist != i32::MAX && best_dist > opts.actor_residual_threshold {
                    raw_actor_masks[fi][y * w + x] = true;
                    let cur_luma =
                        0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32;
                    if cur_luma > 195.0 && cur_luma - base_luma > 45.0 {
                        glow_mask[y * w + x] = true;
                        glow_sum[0] += (c[0] as f32 - base_luma).max(0.0) / 255.0;
                        glow_sum[1] += (c[1] as f32 - base_luma).max(0.0) / 255.0;
                        glow_sum[2] += (c[2] as f32 - base_luma).max(0.0) / 255.0;
                        glow_cnt += 1;
                    }
                }
            }
        }

        let glow_boxes = connected_boxes(&glow_mask, w, h, 16);
        let mean_glow_rgb = if glow_cnt > 0 {
            [
                glow_sum[0] / glow_cnt as f32,
                glow_sum[1] / glow_cnt as f32,
                glow_sum[2] / glow_cnt as f32,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        luminescence_per_frame.push(LuminescenceFrame {
            frame_idx: fi,
            glow_boxes,
            mean_glow_rgb,
            active_fraction: glow_cnt as f32 / (w * active_h).max(1) as f32,
        });
    }

    let actor_boxes_per_frame = kinematic_co_motion_binding(
        frames,
        &raw_actor_masks,
        w,
        h,
        opts.actor_min_area,
        opts.actor_mask_dilation,
    );

    /*
    // Track S1.1: Mask out active actor bounding tracks from temporal-mode accumulators.
    // Form coherent spatiotemporal tracks and enforce stationary hold envelopes.
    let active_tracks = extract_active_actor_bounding_tracks(&actor_boxes_per_frame, n_frames, w, h);
    for tr in &active_tracks {
        for fi in 0..n_frames {
            if let Some(b) = tr[fi] {
                if !actor_boxes_per_frame[fi].contains(&b) {
                    actor_boxes_per_frame[fi].push(b);
                }
            }
        }
    }
    */

    let mut dilated_masks: Vec<Vec<bool>> = vec![vec![false; w * h]; n_frames];
    for fi in 0..n_frames {
        let boxes = &actor_boxes_per_frame[fi];
        for y in 0..h {
            for x in 0..w {
                if raw_actor_masks[fi][y * w + x]
                    || boxes.iter().any(|r| r.contains(x as i32, y as i32))
                {
                    dilated_masks[fi][y * w + x] = true;
                }
            }
        }
    }

    let mut layers: Vec<ParallaxLayer> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (by0, by1) = merged_bands[l];
        let py0 = if l == 0 { y0 } else { by0 };
        let py1 = if l + 1 == n_layers { y1 } else { by1 };
        let clean_atlas = accumulate_band_atlas_masked(
            frames,
            &merged_trajs[l],
            py0,
            py1,
            &dilated_masks,
            &opts.held_out_frames,
            &pass1_layers[l],
        );
        let alpha = vec![255u8; clean_atlas.image.width * clean_atlas.image.height];
        layers.push(ParallaxLayer {
            layer_index: l,
            inverse_depth: layer_spans[l],
            atlas: clean_atlas,
            alpha,
            trajectory: merged_trajs[l].clone(),
            y_band: (py0, py1),
        });
    }

    layers.sort_by(|a, b| {
        b.inverse_depth
            .partial_cmp(&a.inverse_depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (idx, layer) in layers.iter_mut().enumerate() {
        layer.layer_index = idx;
    }

    let camera_locked_hud = hud_band.map(|(hy0, hy1)| {
        let hud_atlas = accumulate_band_atlas(
            frames,
            &zero_traj,
            hy0,
            hy1,
            &[],
            &opts.held_out_frames,
        );
        let alpha = vec![255u8; hud_atlas.image.width * hud_atlas.image.height];
        ParallaxLayer {
            layer_index: layers.len(),
            inverse_depth: f32::INFINITY,
            atlas: hud_atlas,
            alpha,
            trajectory: zero_traj,
            y_band: (hy0, hy1),
        }
    });

    MultiPlaneAtlas {
        width: w,
        height: h,
        active_y_range: (y0, y1),
        layers,
        camera_locked_hud,
        actor_boxes_per_frame,
        actor_masks_per_frame: dilated_masks,
        luminescence_per_frame,
    }
}

/// Estimate 2D translation displacement of a component between adjacent frames
fn estimate_component_displacement(
    curr_frame: &RawFrame,
    other_frame: &RawFrame,
    mask: &[bool],
    w: usize,
    h: usize,
    bbox: &Rect,
    forward: bool,
) -> [f32; 2] {
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_sad = u64::MAX;

    let y_start = bbox.y0.max(0).min(h.saturating_sub(1) as i32);
    let y_end = bbox.y1.max(0).min(h.saturating_sub(1) as i32);
    let x_start = bbox.x0.max(0).min(w.saturating_sub(1) as i32);
    let x_end = bbox.x1.max(0).min(w.saturating_sub(1) as i32);

    for dy in -4..=4 {
        for dx in -4..=4 {
            let mut sad = 0u64;
            let mut count = 0u64;

            for y in y_start..=y_end {
                let ny = y + dy;
                if ny < 0 || ny >= h as i32 {
                    continue;
                }
                for x in x_start..=x_end {
                    let nx = x + dx;
                    if nx < 0 || nx >= w as i32 {
                        continue;
                    }
                    let idx = y as usize * w + x as usize;
                    if !mask[idx] {
                        continue;
                    }
                    let p_cur = curr_frame.pixel(x as u32, y as u32);
                    let p_oth = other_frame.pixel(nx as u32, ny as u32);
                    let diff = (p_cur[0] as i32 - p_oth[0] as i32).abs()
                        + (p_cur[1] as i32 - p_oth[1] as i32).abs()
                        + (p_cur[2] as i32 - p_oth[2] as i32).abs();
                    sad += diff as u64;
                    count += 1;
                }
            }

            if count > 0 && sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    if forward {
        [best_dx as f32, best_dy as f32]
    } else {
        [-best_dx as f32, -best_dy as f32]
    }
}

/// Kinematic Co-Motion Binding (Track R10.7a):
/// Groups co-moving adjacent residual components sharing spatial proximity and velocity
/// displacement v before forming tracklets, restoring fragmented animated entities
/// (e.g. Mrs. Brisby's head, ears, snout, torso, cloak).
pub fn kinematic_co_motion_binding(
    frames: &[RawFrame],
    raw_masks: &[Vec<bool>],
    w: usize,
    h: usize,
    min_comp_area: usize,
    dilation: i32,
) -> Vec<Vec<Rect>> {
    let n_frames = frames.len();
    let mut out = Vec::with_capacity(n_frames);

    for fi in 0..n_frames {
        let mask = &raw_masks[fi];
        let boxes = crate::enfold2d::connected_boxes(mask, w, h, min_comp_area);
        if boxes.is_empty() {
            out.push(Vec::new());
            continue;
        }

        let other_fi = if fi + 1 < n_frames { fi + 1 } else { fi.saturating_sub(1) };
        let forward = fi + 1 < n_frames;

        let mut velocities = Vec::with_capacity(boxes.len());
        for b in &boxes {
            let v = if n_frames > 1 {
                estimate_component_displacement(&frames[fi], &frames[other_fi], mask, w, h, b, forward)
            } else {
                [0.0, 0.0]
            };
            velocities.push(v);
        }

        let mut parent: Vec<usize> = (0..boxes.len()).collect();
        fn find(p: &mut [usize], i: usize) -> usize {
            if p[i] == i { i } else { p[i] = find(p, p[i]); p[i] }
        }

        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let bi = &boxes[i];
                let bj = &boxes[j];

                let dx = (bi.x0 - bj.x1).max(bj.x0 - bi.x1).max(0);
                let dy = (bi.y0 - bj.y1).max(bj.y0 - bi.y1).max(0);
                let gap = dx.max(dy);

                // Kinematic Co-Motion Binding (Track R10.7a / R10.9):
                // Union co-moving adjacent residual components sharing spatial proximity (gap <= 16)
                // and velocity displacement v (v_diff <= 3.5 for articulated limb swing) within a localized actor scale.
                if gap <= 16 {
                    let union_w = (bi.x1.max(bj.x1) - bi.x0.min(bj.x0) + 1) as usize;
                    let union_h = (bi.y1.max(bj.y1) - bi.y0.min(bj.y0) + 1) as usize;
                    if union_w <= w / 2 && union_h <= h / 2 {
                        let vi = velocities[i];
                        let vj = velocities[j];
                        let v_diff = (vi[0] - vj[0]).hypot(vi[1] - vj[1]);
                        if v_diff <= 3.5 {
                            let ri = find(&mut parent, i);
                            let rj = find(&mut parent, j);
                            if ri != rj {
                                parent[ri] = rj;
                            }
                        }
                    }
                }
            }
        }

        let mut united_boxes: Vec<Option<Rect>> = vec![None; boxes.len()];

        for i in 0..boxes.len() {
            let root = find(&mut parent, i);
            let b = &boxes[i];
            united_boxes[root] = match united_boxes[root] {
                Some(prev) => Some(Rect::new(
                    prev.x0.min(b.x0),
                    prev.y0.min(b.y0),
                    prev.x1.max(b.x1),
                    prev.y1.max(b.y1),
                )),
                None => Some(*b),
            };
        }

        let mut merged: Vec<Rect> = Vec::new();
        for opt_b in united_boxes.into_iter().flatten() {
            merged.push(opt_b.expand(dilation));
        }

        for b in &mut merged {
            b.x0 = b.x0.clamp(0, w as i32 - 1);
            b.y0 = b.y0.clamp(0, h as i32 - 1);
            b.x1 = b.x1.clamp(b.x0, w as i32 - 1);
            b.y1 = b.y1.clamp(b.y0, h as i32 - 1);
        }

        out.push(merged);
    }

    out
}

/// Form coherent active actor bounding tracks across the shot from per-frame candidate boxes (Track S1.1).
/// Identifies stationary holds (e.g. speaking holds where motion v ≈ 0) and extends the complete
/// bounding envelope across the entire hold duration, eliminating bake-in leakage.
pub fn extract_active_actor_bounding_tracks(
    boxes_per_frame: &[Vec<Rect>],
    n_frames: usize,
    w: usize,
    h: usize,
) -> Vec<Vec<Option<Rect>>> {
    if n_frames == 0 {
        return Vec::new();
    }
    let mut raw_tracks: Vec<Vec<(usize, Rect)>> = Vec::new();

    for (fi, boxes) in boxes_per_frame.iter().enumerate() {
        for &b in boxes {
            let c = b.center();
            let mut best_ti = None;
            let mut best_dist = f64::MAX;

            for (ti, tr) in raw_tracks.iter().enumerate() {
                let (last_fi, last_b) = tr.last().unwrap();
                let f_gap = fi.saturating_sub(*last_fi);
                if fi > *last_fi && f_gap <= 3 {
                    let last_c = last_b.center();
                    let d = ((c[0] - last_c[0]).powi(2) + (c[1] - last_c[1]).powi(2)).sqrt();
                    let max_d = 16.0 * f_gap as f64;
                    // Check center proximity and aspect ratio consistency
                    if d <= max_d && d < best_dist {
                        let ar_cur = (b.width() as f32) / (b.height() as f32).max(1.0);
                        let ar_last = (last_b.width() as f32) / (last_b.height() as f32).max(1.0);
                        if (ar_cur - ar_last).abs() < 1.5 {
                            best_dist = d;
                            best_ti = Some(ti);
                        }
                    }
                }
            }

            if let Some(ti) = best_ti {
                raw_tracks[ti].push((fi, b));
            } else {
                raw_tracks.push(vec![(fi, b)]);
            }
        }
    }

    let min_track_span = 4usize.min(n_frames);
    let active_tracks: Vec<Vec<(usize, Rect)>> = raw_tracks
        .into_iter()
        .filter(|tr| {
            let span = tr.last().unwrap().0 - tr.first().unwrap().0 + 1;
            tr.len() >= min_track_span || span >= min_track_span
        })
        .collect();

    let mut out_tracks = Vec::with_capacity(active_tracks.len());

    for tr in active_tracks {
        let mut per_frame: Vec<Option<Rect>> = vec![None; n_frames];
        for &(fi, b) in &tr {
            per_frame[fi] = match per_frame[fi] {
                Some(prev) => Some(Rect::new(
                    prev.x0.min(b.x0),
                    prev.y0.min(b.y0),
                    prev.x1.max(b.x1),
                    prev.y1.max(b.y1),
                )),
                None => Some(b),
            };
        }

        let first_fi = tr.first().unwrap().0;
        let last_fi = tr.last().unwrap().0;
        for fi in first_fi..=last_fi {
            if per_frame[fi].is_none() {
                let prev_b = (first_fi..fi).rev().find_map(|f| per_frame[f]);
                let next_b = (fi + 1..=last_fi).find_map(|f| per_frame[f]);
                per_frame[fi] = match (prev_b, next_b) {
                    (Some(p), Some(n)) => Some(Rect::new(
                        p.x0.min(n.x0),
                        p.y0.min(n.y0),
                        p.x1.max(n.x1),
                        p.y1.max(n.y1),
                    )),
                    (Some(p), None) => Some(p),
                    (None, Some(n)) => Some(n),
                    (None, None) => None,
                };
            }
        }

        let mut hold_start = first_fi;
        while hold_start <= last_fi {
            let Some(b0) = per_frame[hold_start] else {
                hold_start += 1;
                continue;
            };
            let c0 = b0.center();
            let mut hold_end = hold_start;
            let mut union_box = b0;

            for fi in (hold_start + 1)..=last_fi {
                if let Some(b) = per_frame[fi] {
                    let c = b.center();
                    let d = ((c[0] - c0[0]).powi(2) + (c[1] - c0[1]).powi(2)).sqrt();
                    if d <= 6.0 {
                        hold_end = fi;
                        union_box = Rect::new(
                            union_box.x0.min(b.x0),
                            union_box.y0.min(b.y0),
                            union_box.x1.max(b.x1),
                            union_box.y1.max(b.y1),
                        );
                    } else {
                        break;
                    }
                }
            }

            if hold_end >= hold_start + 2 {
                for fi in hold_start..=hold_end {
                    per_frame[fi] = Some(union_box);
                }
                if hold_start <= 3 {
                    for fi in 0..hold_start {
                        per_frame[fi] = Some(union_box);
                    }
                }
                hold_start = hold_end + 1;
            } else {
                hold_start += 1;
            }
        }

        for opt_b in &mut per_frame {
            if let Some(b) = opt_b {
                b.x0 = b.x0.clamp(0, w as i32 - 1);
                b.y0 = b.y0.clamp(0, h as i32 - 1);
                b.x1 = b.x1.clamp(b.x0, w as i32 - 1);
                b.y1 = b.y1.clamp(b.y0, h as i32 - 1);
            }
        }

        out_tracks.push(per_frame);
    }

    out_tracks
}
