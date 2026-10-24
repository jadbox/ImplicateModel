//! `CliffordNet` Laplacian wedge-vorticity features ($D^{(s)} \oplus W^{(s)}$),
//! silhouette IoU, and multi-scale perceptual patch distance (`articulated_wavepacket::clifford_features`).

use super::{
    CharacterObservation, fit_articulated_holon,
    part_fitting::{fit_rigid_single_box_baseline, render_rigid_baseline_observation},
    render_articulated_holon_observation,
};

#[derive(Clone, Debug)]
pub struct ArticulatedComparisonReport {
    pub num_parts: usize,
    pub num_view_cells: usize,
    pub baseline_single_box_iou: f64,
    pub baseline_single_box_lpips: f64,
    pub articulated_wavepacket_iou: f64,
    pub articulated_wavepacket_lpips: f64,
}

/// Compute `CliffordNet` shifted geometric product features ($D^{(s)} \oplus W^{(s)}$)
/// over the factorized discrete Laplacian context $\mathcal{C}_{\text{loc}}(H) \approx \nabla^2 H$.
pub fn compute_cliffordnet_laplacian_features(rgb: &[u8], w: usize, h: usize) -> Vec<[f32; 2]> {
    let n = w * h;
    if n == 0 {
        return Vec::new();
    }
    let mut state = vec![[0.0f32; 4]; n];
    for i in 0..n {
        let r = rgb[i * 3] as f32 / 255.0;
        let g = rgb[i * 3 + 1] as f32 / 255.0;
        let b = rgb[i * 3 + 2] as f32 / 255.0;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        state[i] = [r, g, b, y];
    }

    let smooth1 = box_blur_3x3(&state, w, h);
    let smooth2 = box_blur_3x3(&smooth1, w, h);

    let mut out = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let h_vec = state[i];
        let mut c_vec = [0.0f32; 4];
        for c in 0..4 {
            c_vec[c] = smooth2[i][c] - h_vec[c];
        }
        let mut d_acc = 0.0f32;
        let mut w_acc = 0.0f32;
        for s in [1usize, 2usize] {
            for c in 0..4 {
                let cs = (c + s) & 3;
                let prod1 = h_vec[c] * c_vec[cs];
                let prod2 = c_vec[c] * h_vec[cs];
                let silu = prod1 / (1.0 + (-prod1 * 4.0).exp());
                d_acc += silu.abs();
                w_acc += (prod1 - prod2).abs();
            }
        }
        out[i] = [(d_acc * 2.0).clamp(0.0, 1.0), (w_acc * 3.0).clamp(0.0, 1.0)];
    }
    out
}

/// Dual Clifford Geometric Product Decomposition ($u \nabla^2 u = u \cdot \nabla^2 u + u \wedge \nabla^2 u$).
///
/// In the Implicate Video Engine (Track R10.2, Whitepaper §1, §2.3):
/// - Grade 0 scalar inner product $D(\mathbf{x}) = u \cdot \nabla^2 u$ measures cel paint fill coherence and vitality diffusion.
/// - Grade 2 bivector wedge product $W(\mathbf{x}) = u \wedge \nabla^2 u$ measures boundary vorticity blades (animated film ink contours).
#[derive(Clone, Debug, PartialEq)]
pub struct CliffordDualDecomposition {
    pub width: usize,
    pub height: usize,
    /// Cel paint fill scalar coherence field $D(\mathbf{x}) = \langle u \cdot \nabla^2 u \rangle_0 \in [0, 1]$ per pixel.
    pub cel_fill: Vec<f32>,
    /// Ink contour blade bivector vorticity magnitude field $W(\mathbf{x}) = \|\langle u \wedge \nabla^2 u \rangle_2\| \in [0, 1]$ per pixel.
    pub ink_contour: Vec<f32>,
    /// Statistical noise floor τ_W derived from MAD on non-zero wedge vorticity.
    pub ink_blade_threshold: f32,
    /// Discrete binary ink contour blade mask ($W(\mathbf{x}) \ge \tau_W$).
    pub ink_blade_mask: Vec<bool>,
    /// Discrete binary cel paint fill mask (silhouette interior where $W(\mathbf{x}) < \tau_W$).
    pub cel_fill_mask: Vec<bool>,
}

/// Decompose an RGB image into its Clifford dual product components:
/// $u \nabla^2 u = u \cdot \nabla^2 u + u \wedge \nabla^2 u$.
pub fn decompose_clifford_dual_product(rgb: &[u8], w: usize, h: usize) -> CliffordDualDecomposition {
    decompose_masked_clifford_dual_product(rgb, None, w, h)
}

/// Decompose an RGB image with an optional silhouette mask into its Clifford dual product components.
pub fn decompose_masked_clifford_dual_product(
    rgb: &[u8],
    mask: Option<&[bool]>,
    w: usize,
    h: usize,
) -> CliffordDualDecomposition {
    let n = w * h;
    if n == 0 {
        return CliffordDualDecomposition {
            width: 0,
            height: 0,
            cel_fill: Vec::new(),
            ink_contour: Vec::new(),
            ink_blade_threshold: 0.0,
            ink_blade_mask: Vec::new(),
            cel_fill_mask: Vec::new(),
        };
    }

    let mut state = vec![[0.0f32; 4]; n];
    for i in 0..n {
        let r = rgb[i * 3] as f32 / 255.0;
        let g = rgb[i * 3 + 1] as f32 / 255.0;
        let b = rgb[i * 3 + 2] as f32 / 255.0;
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        state[i] = [r, g, b, y];
    }

    let smooth1 = box_blur_3x3(&state, w, h);
    let smooth2 = box_blur_3x3(&smooth1, w, h);

    let mut cel_fill = vec![0.0f32; n];
    let mut ink_contour = vec![0.0f32; n];
    let mut non_zero_w = Vec::with_capacity(n / 4);

    for i in 0..n {
        if let Some(m) = mask {
            if !m[i] {
                continue;
            }
        }

        let u = state[i];
        let mut lap = [0.0f32; 4];
        for c in 0..4 {
            lap[c] = smooth2[i][c] - u[c];
        }

        // 1. Grade 0 Scalar Inner Product: u · ∇²u
        let dot = u[0] * lap[0] + u[1] * lap[1] + u[2] * lap[2] + u[3] * lap[3];

        // 2. Grade 2 Bivector Exterior Wedge Product: u ∧ ∇²u
        // 6 independent bivector components in 4D:
        let mut wedge_sq = 0.0f32;
        for j in 0..4 {
            for k in (j + 1)..4 {
                let w_jk = u[j] * lap[k] - u[k] * lap[j];
                wedge_sq += w_jk * w_jk;
            }
        }
        let w_mag = (wedge_sq.sqrt() * 4.0).clamp(0.0, 1.0);
        ink_contour[i] = w_mag;

        if w_mag > 1e-4 {
            non_zero_w.push(w_mag);
        }

        // Cel paint fill vitality: smooth inner product coherence modulated by absence of wedge vorticity
        let coherence = (-0.5 * (w_mag / 0.25).powi(2)).exp();
        let vitality = (u[3] + dot.abs()).clamp(0.0, 1.0);
        cel_fill[i] = (vitality * coherence).clamp(0.0, 1.0);
    }

    // Statistical noise floor for ink blade detection τ_W = median + 1.5 * 1.4826 * MAD
    let tau_w = if !non_zero_w.is_empty() {
        non_zero_w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = non_zero_w[non_zero_w.len() / 2];
        let mut devs: Vec<f32> = non_zero_w.iter().map(|&x| (x - med).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = devs[devs.len() / 2];
        (med + 1.5 * 1.4826 * mad).clamp(0.06, 0.40)
    } else {
        0.12
    };

    let mut ink_blade_mask = vec![false; n];
    let mut cel_fill_mask = vec![false; n];

    for i in 0..n {
        let inside = mask.map_or(true, |m| m[i]);
        if inside {
            if ink_contour[i] >= tau_w {
                ink_blade_mask[i] = true;
            } else if state[i][3] > 0.02 || state[i][0] > 0.02 || state[i][1] > 0.02 || state[i][2] > 0.02 {
                cel_fill_mask[i] = true;
            }
        }
    }

    CliffordDualDecomposition {
        width: w,
        height: h,
        cel_fill,
        ink_contour,
        ink_blade_threshold: tau_w,
        ink_blade_mask,
        cel_fill_mask,
    }
}

/// Decompose a character observation into its Clifford dual product components.
pub fn decompose_observation_dual_product(obs: &CharacterObservation) -> CliffordDualDecomposition {
    decompose_masked_clifford_dual_product(&obs.rgb, Some(&obs.mask), obs.width(), obs.height())
}

fn box_blur_3x3(src: &[[f32; 4]], w: usize, h: usize) -> Vec<[f32; 4]> {
    let mut dst = vec![[0.0f32; 4]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = [0.0f32; 4];
            let mut cnt = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let v = src[ny * w + nx];
                    for c in 0..4 {
                        sum[c] += v[c];
                    }
                    cnt += 1.0;
                }
            }
            for c in 0..4 {
                dst[y * w + x][c] = sum[c] / cnt;
            }
        }
    }
    dst
}

pub fn evaluate_vs_single_box_baseline(
    observations: &[CharacterObservation],
) -> ArticulatedComparisonReport {
    evaluate_vs_single_box_baseline_with_parts(observations, None)
}

/// As `evaluate_vs_single_box_baseline`, with an optional forced part count.
pub fn evaluate_vs_single_box_baseline_with_parts(
    observations: &[CharacterObservation],
    forced_parts: Option<usize>,
) -> ArticulatedComparisonReport {
    let articulated = fit_articulated_holon(observations, forced_parts);
    let baseline = fit_rigid_single_box_baseline(observations);

    let (mut base_iou_sum, mut base_lpips_sum) = (0.0f64, 0.0f64);
    let (mut art_iou_sum, mut art_lpips_sum) = (0.0f64, 0.0f64);
    let n = observations.len().max(1) as f64;

    for (oi, obs) in observations.iter().enumerate() {
        let (b_mask, b_rgb) = render_rigid_baseline_observation(&baseline, obs);
        let (a_mask, a_rgb) = render_articulated_holon_observation(&articulated, obs, oi);

        base_iou_sum += mask_iou(&obs.mask, &b_mask);
        base_lpips_sum += compute_perceptual_patch_lpips(
            &obs.rgb,
            &obs.mask,
            &b_rgb,
            &b_mask,
            obs.width(),
            obs.height(),
        );

        art_iou_sum += mask_iou(&obs.mask, &a_mask);
        art_lpips_sum += compute_perceptual_patch_lpips(
            &obs.rgb,
            &obs.mask,
            &a_rgb,
            &a_mask,
            obs.width(),
            obs.height(),
        );
    }

    ArticulatedComparisonReport {
        num_parts: articulated.num_parts,
        num_view_cells: articulated.num_view_cells,
        baseline_single_box_iou: base_iou_sum / n,
        baseline_single_box_lpips: base_lpips_sum / n,
        articulated_wavepacket_iou: art_iou_sum / n,
        articulated_wavepacket_lpips: art_lpips_sum / n,
    }
}

pub fn mask_iou(gt: &[bool], pred: &[bool]) -> f64 {
    let mut inter = 0usize;
    let mut union = 0usize;
    for (&a, &b) in gt.iter().zip(pred.iter()) {
        if a && b {
            inter += 1;
        }
        if a || b {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

pub fn compute_perceptual_patch_lpips(
    gt_rgb: &[u8],
    gt_mask: &[bool],
    pred_rgb: &[u8],
    pred_mask: &[bool],
    w: usize,
    h: usize,
) -> f64 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let clean_gt = mask_rgb_patch(gt_rgb, gt_mask);
    let clean_pr = mask_rgb_patch(pred_rgb, pred_mask);

    let mut total_dist = 0.0f64;
    let mut scale_weights = 0.0f64;
    let mut cur_gt = clean_gt;
    let mut cur_pr = clean_pr;
    let mut cw = w;
    let mut ch = h;

    for scale in 0..3 {
        if cw < 4 || ch < 4 {
            break;
        }
        let w_s = match scale {
            0 => 0.5,
            1 => 0.3,
            _ => 0.2,
        };
        total_dist += w_s * single_scale_clifford_perceptual_dist(&cur_gt, &cur_pr, cw, ch);
        scale_weights += w_s;

        if cw >= 8 && ch >= 8 {
            cur_gt = downsample_patch_2x(&cur_gt, cw, ch);
            cur_pr = downsample_patch_2x(&cur_pr, cw, ch);
            cw /= 2;
            ch /= 2;
        } else {
            break;
        }
    }

    if scale_weights <= 0.0 {
        0.0
    } else {
        total_dist / scale_weights
    }
}

fn mask_rgb_patch(rgb: &[u8], mask: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; rgb.len()];
    for (i, &fg) in mask.iter().enumerate() {
        if fg {
            out[i * 3..i * 3 + 3].copy_from_slice(&rgb[i * 3..i * 3 + 3]);
        }
    }
    out
}

fn single_scale_clifford_perceptual_dist(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let cliff_a = compute_cliffordnet_laplacian_features(a, w, h);
    let cliff_b = compute_cliffordnet_laplacian_features(b, w, h);

    let mut diff_sum = 0.0f64;
    let mut active_px = 0usize;
    for i in 0..w * h {
        let (ar, ag, ab) = (
            a[i * 3] as f32 / 255.0,
            a[i * 3 + 1] as f32 / 255.0,
            a[i * 3 + 2] as f32 / 255.0,
        );
        let (br, bg, bb) = (
            b[i * 3] as f32 / 255.0,
            b[i * 3 + 1] as f32 / 255.0,
            b[i * 3 + 2] as f32 / 255.0,
        );

        if ar + ag + ab < 1e-3 && br + bg + bb < 1e-3 {
            continue;
        }
        active_px += 1;

        let ay = 0.299 * ar + 0.587 * ag + 0.114 * ab;
        let by = 0.299 * br + 0.587 * bg + 0.114 * bb;
        let fa = [
            ay,
            (ar - ag) * 0.7,
            (ay - ab) * 0.7,
            cliff_a[i][0] * 0.5,
            cliff_a[i][1] * 0.5,
        ];
        let fb = [
            by,
            (br - bg) * 0.7,
            (by - bb) * 0.7,
            cliff_b[i][0] * 0.5,
            cliff_b[i][1] * 0.5,
        ];
        let norm_a = (fa.iter().map(|v| v * v).sum::<f32>() + 1e-4).sqrt();
        let norm_b = (fb.iter().map(|v| v * v).sum::<f32>() + 1e-4).sqrt();
        let mut sq = 0.0f32;
        for k in 0..5 {
            let d = fa[k] / norm_a - fb[k] / norm_b;
            sq += d * d;
        }
        diff_sum += (sq * 0.25) as f64;
    }
    if active_px == 0 {
        0.0
    } else {
        diff_sum / active_px as f64
    }
}

fn downsample_patch_2x(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let nw = w / 2;
    let nh = h / 2;
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        for x in 0..nw {
            let i00 = ((2 * y) * w + 2 * x) * 3;
            let i01 = ((2 * y) * w + 2 * x + 1) * 3;
            let i10 = ((2 * y + 1) * w + 2 * x) * 3;
            let i11 = ((2 * y + 1) * w + 2 * x + 1) * 3;
            let dst = (y * nw + x) * 3;
            for c in 0..3 {
                let s = rgb[i00 + c] as u32
                    + rgb[i01 + c] as u32
                    + rgb[i10 + c] as u32
                    + rgb[i11 + c] as u32;
                out[dst + c] = ((s + 2) / 4) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clifford_dual_product_separates_contour_from_cel_fill() {
        // Construct a 32x32 image with a flat red cel fill disk surrounded by a dark ink contour
        let w = 32usize;
        let h = 32usize;
        let mut rgb = vec![255u8; w * h * 3]; // white background
        let mut mask = vec![false; w * h];
        let center = [16.0f64, 16.0f64];
        let radius_inner = 9.0f64;
        let radius_outer = 11.0f64;

        for y in 0..h {
            for x in 0..w {
                let d = ((x as f64 - center[0]).powi(2) + (y as f64 - center[1]).powi(2)).sqrt();
                let idx = y * w + x;
                if d <= radius_inner {
                    // Cel paint fill (pure red)
                    rgb[idx * 3] = 230;
                    rgb[idx * 3 + 1] = 40;
                    rgb[idx * 3 + 2] = 40;
                    mask[idx] = true;
                } else if d <= radius_outer {
                    // animated film ink contour line (dark pigment)
                    rgb[idx * 3] = 20;
                    rgb[idx * 3 + 1] = 15;
                    rgb[idx * 3 + 2] = 25;
                    mask[idx] = true;
                }
            }
        }

        let decomp = decompose_masked_clifford_dual_product(&rgb, Some(&mask), w, h);

        // Center pixel (16, 16) is deep inside the cel paint fill:
        // Wedge vorticity u ∧ ∇²u must be very small, and it must be classified as cel fill.
        let center_idx = 16 * w + 16;
        assert!(decomp.ink_contour[center_idx] < decomp.ink_blade_threshold,
            "Center cel fill wedge vorticity ({:.4}) must be below threshold ({:.4})",
            decomp.ink_contour[center_idx], decomp.ink_blade_threshold);
        assert!(decomp.cel_fill_mask[center_idx], "Center must be in cel_fill_mask");
        assert!(!decomp.ink_blade_mask[center_idx], "Center must not be in ink_blade_mask");

        // Boundary pixels near r = 10 must have strong wedge vorticity and be marked as ink blade
        let mut ink_count = 0usize;
        let mut cel_count = 0usize;
        for i in 0..w * h {
            if decomp.ink_blade_mask[i] {
                ink_count += 1;
            }
            if decomp.cel_fill_mask[i] {
                cel_count += 1;
            }
        }

        assert!(ink_count > 20, "Must detect ink contour blade along the boundary (got {})", ink_count);
        assert!(cel_count > 150, "Must detect cel paint fill in the disk interior (got {})", cel_count);
    }
}
