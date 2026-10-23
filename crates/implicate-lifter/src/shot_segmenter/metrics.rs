//! Spectral, histogram, and evaluation metrics for Stage 0 Shot Segmentation (`shot_segmenter::metrics`).

use rustfft::num_complex::Complex;

use super::{ANALYSIS_H, ANALYSIS_W, HIST_BINS};
use crate::enfold2d::Rect;
use crate::video_io::RawFrame;

/// Fast box/bilinear downsample of an arbitrary `RawFrame` into `128x64` (`ANALYSIS_W x ANALYSIS_H`).
pub fn downsample_to_analysis(src: &RawFrame, dst: &mut RawFrame) {
    let sw = src.width as usize;
    let sh = src.height as usize;
    dst.width = ANALYSIS_W as u32;
    dst.height = ANALYSIS_H as u32;
    dst.rgb.resize(ANALYSIS_W * ANALYSIS_H * 3, 0);
    if sw == ANALYSIS_W && sh == ANALYSIS_H {
        dst.rgb.copy_from_slice(&src.rgb);
        return;
    }
    for y in 0..ANALYSIS_H {
        let sy = (y * sh) / ANALYSIS_H;
        for x in 0..ANALYSIS_W {
            let sx = (x * sw) / ANALYSIS_W;
            let si = (sy * sw + sx) * 3;
            let di = (y * ANALYSIS_W + x) * 3;
            dst.rgb[di..di + 3].copy_from_slice(&src.rgb[si..si + 3]);
        }
    }
}

pub fn symmetric_kl(p: &[f32; HIST_BINS], q: &[f32; HIST_BINS]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..HIST_BINS {
        let pi = p[i].max(1e-6);
        let qi = q[i].max(1e-6);
        sum += 0.5 * (pi * (pi / qi).ln() + qi * (qi / pi).ln());
    }
    sum
}

pub fn grid_l1_distance(a: &[f32; 16 * 8 * 3], b: &[f32; 16 * 8 * 3]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum / a.len() as f32
}

/// Fit `mid ≈ (1 - alpha) * start + alpha * end` in L2/L1 sense, returning `(alpha, l1_residual)`.
pub fn linear_dissolve_fit(
    start: &[f32; 16 * 8 * 3],
    mid: &[f32; 16 * 8 * 3],
    end: &[f32; 16 * 8 * 3],
) -> (f32, f32) {
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..start.len() {
        let d = end[i] - start[i];
        let m = mid[i] - start[i];
        num += m * d;
        den += d * d;
    }
    let alpha = if den > 1e-6 {
        (num / den).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let mut err = 0.0f32;
    for i in 0..start.len() {
        let pred = (1.0 - alpha) * start[i] + alpha * end[i];
        err += (mid[i] - pred).abs();
    }
    (alpha, err / start.len() as f32)
}

/// Locate the primary CFT correlation peak (with parabolic sub-pixel refinement)
/// and the strongest well-separated secondary peak ($\|\mathbf{d}_2 - \mathbf{d}_1\| \ge 3\text{ px}$).
pub fn find_primary_and_secondary_cft_peaks(
    plane: &[Complex<f32>],
    w: usize,
    h: usize,
    max_shift: i32,
) -> (f32, f32, [f32; 2]) {
    let max_sy = max_shift.min((h as i32 / 2) - 1);
    let max_sx = max_shift.min((w as i32 / 2) - 1);
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;
    let mut best_val = 0.0f32;

    let sample = |dx: i32, dy: i32| -> f32 {
        let xx = dx.rem_euclid(w as i32) as usize;
        let yy = dy.rem_euclid(h as i32) as usize;
        plane[yy * w + xx].re.max(0.0)
    };

    for dy in -max_sy..=max_sy {
        for dx in -max_sx..=max_sx {
            let v = sample(dx, dy);
            if v > best_val {
                best_val = v;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    let sub_x = if best_dx > -max_sx && best_dx < max_sx {
        let l = sample(best_dx - 1, best_dy);
        let r = sample(best_dx + 1, best_dy);
        let denom = 2.0 * (2.0 * best_val - l - r);
        if denom.abs() > 1e-5 {
            ((r - l) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };
    let sub_y = if best_dy > -max_sy && best_dy < max_sy {
        let u = sample(best_dx, best_dy - 1);
        let d = sample(best_dx, best_dy + 1);
        let denom = 2.0 * (2.0 * best_val - u - d);
        if denom.abs() > 1e-5 {
            ((d - u) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mut sec_val = 0.0f32;
    for dy in -max_sy..=max_sy {
        for dx in -max_sx..=max_sx {
            let dist_sq = (dx - best_dx) * (dx - best_dx) + (dy - best_dy) * (dy - best_dy);
            // Exclude neighborhood of primary peak and stationary (0,0) HUD/bezel origin when camera moves.
            if dist_sq < 9 || ((best_dx.abs() > 1 || best_dy.abs() > 1) && dx.abs() <= 1 && dy.abs() <= 1) {
                continue;
            }
            let v = sample(dx, dy);
            if v > sec_val
                && v >= sample(dx - 1, dy)
                && v >= sample(dx + 1, dy)
                && v >= sample(dx, dy - 1)
                && v >= sample(dx, dy + 1)
            {
                sec_val = v;
            }
        }
    }

    let sec_ratio = if best_val > 1e-4 {
        (sec_val / best_val).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let (fx, fy) = if best_dx == 0 && best_dy == 0 && sub_x.hypot(sub_y) < 0.08 {
        (0.0, 0.0)
    } else {
        (best_dx as f32 + sub_x, best_dy as f32 + sub_y)
    };

    (
        best_val.clamp(0.0, 1.0),
        sec_ratio,
        [fx, fy],
    )
}

/// Isolate zero-parallax HUD bars and letterbox bezels from per-pixel temporal variance $\sigma_t^2(x, y)$.
pub fn isolate_hud_and_bezels(
    var_sum: &[f32],
    var_sq_sum: &[f32],
    n_frames: usize,
    native_w: u32,
    native_h: u32,
) -> (Vec<Rect>, (usize, usize)) {
    if n_frames < 4 {
        return (Vec::new(), (0, native_h as usize));
    }
    let nf = n_frames as f32;
    let mut row_std = vec![0.0f32; ANALYSIS_H];
    for y in 0..ANALYSIS_H {
        let mut stds = Vec::with_capacity(ANALYSIS_W);
        for x in 0..ANALYSIS_W {
            let idx = y * ANALYSIS_W + x;
            let mean = var_sum[idx] / nf;
            let var = (var_sq_sum[idx] / nf - mean * mean).max(0.0);
            stds.push(var.sqrt());
        }
        stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        row_std[y] = stds[stds.len() / 2];
    }

    let mut top_row = 0usize;
    for y in 0..ANALYSIS_H / 3 {
        if row_std[y] < 3.5 {
            top_row = y + 1;
        } else {
            break;
        }
    }
    let mut bot_row = ANALYSIS_H;
    for y in (ANALYSIS_H * 2 / 3..ANALYSIS_H).rev() {
        if row_std[y] < 3.5 {
            bot_row = y;
        } else {
            break;
        }
    }

    let sy = |r: usize| -> usize { (r * native_h as usize) / ANALYSIS_H };
    let y_top_native = sy(top_row);
    let y_bot_native = sy(bot_row).max(y_top_native + 1);

    let mut rects = Vec::new();
    if y_top_native > 0 {
        rects.push(Rect::new(0, 0, native_w as i32 - 1, y_top_native as i32 - 1));
    }
    if y_bot_native < native_h as usize {
        rects.push(Rect::new(
            0,
            y_bot_native as i32,
            native_w as i32 - 1,
            native_h as i32 - 1,
        ));
    }
    (rects, (y_top_native, y_bot_native))
}

/// Evaluate Precision, Recall, and F1 of detected shot boundary frame indices against ground truth.
pub fn evaluate_boundary_precision_recall(
    detected_frames: &[usize],
    ground_truth_frames: &[usize],
    tol_frames: usize,
) -> (f64, f64, f64) {
    if detected_frames.is_empty() && ground_truth_frames.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    if detected_frames.is_empty() || ground_truth_frames.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut matched_gt = vec![false; ground_truth_frames.len()];
    let mut tp = 0usize;
    for &d in detected_frames {
        let mut best_g = None;
        let mut best_dist = usize::MAX;
        for (gi, &g) in ground_truth_frames.iter().enumerate() {
            if matched_gt[gi] {
                continue;
            }
            let dist = d.abs_diff(g);
            if dist <= tol_frames && dist < best_dist {
                best_dist = dist;
                best_g = Some(gi);
            }
        }
        if let Some(gi) = best_g {
            matched_gt[gi] = true;
            tp += 1;
        }
    }
    let precision = tp as f64 / detected_frames.len() as f64;
    let recall = tp as f64 / ground_truth_frames.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}
