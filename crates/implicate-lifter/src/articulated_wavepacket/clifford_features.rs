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
    let articulated = fit_articulated_holon(observations, None);
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
