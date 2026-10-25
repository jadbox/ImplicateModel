//! Band trajectory estimation and actor-masked temporal mode accumulation (`multiplane_atlas::layer_accum`).

use std::collections::HashMap;

use crate::enfold2d::{Bucket, RgbImage8, ScrollEstimator, WorldAtlas};
use crate::video_io::RawFrame;

pub fn estimate_band_trajectory(frames: &[RawFrame], y0: usize, y1: usize) -> Vec<[i32; 2]> {
    let w = frames[0].width as usize;
    let band_h = (y1 - y0).max(8);
    let mut est = ScrollEstimator::new(w, band_h);
    let mut cum = [0i32, 0];
    let mut out = Vec::with_capacity(frames.len());
    let mut sub_frame = RawFrame::blank(w as u32, band_h as u32);

    for (fi, f) in frames.iter().enumerate() {
        for y in 0..band_h {
            let sy = (y0 + y).min(f.height as usize - 1);
            let src_off = sy * w * 3;
            let dst_off = y * w * 3;
            sub_frame.rgb[dst_off..dst_off + w * 3]
                .copy_from_slice(&f.rgb[src_off..src_off + w * 3]);
        }
        let (mut delta, conf) = est.push(&sub_frame);
        if fi > 0 && conf < 2.2 {
            delta = refine_band_delta(&frames[fi - 1], f, y0, y1, delta);
        }
        cum[0] += delta[0];
        cum[1] += delta[1];
        out.push(cum);
    }
    out
}

pub fn refine_band_delta(
    prev: &RawFrame,
    curr: &RawFrame,
    y0: usize,
    y1: usize,
    init_delta: [i32; 2],
) -> [i32; 2] {
    let w = prev.width as i32;
    let y0i = y0 as i32;
    let y1i = y1 as i32;

    let eval_delta = |dx: i32, dy: i32| -> Option<u64> {
        let mut err = 0u64;
        let mut cnt = 0u64;
        let mut y = y0i;
        while y < y1i {
            let py = y + dy;
            if py >= y0i && py < y1i {
                let mut x = 8;
                while x < w - 8 {
                    let px = x + dx;
                    if px >= 0 && px < w {
                        let i_cur = (y as usize * w as usize + x as usize) * 3;
                        let i_prv = (py as usize * w as usize + px as usize) * 3;
                        err += (curr.rgb[i_cur] as i32 - prev.rgb[i_prv] as i32).unsigned_abs()
                            as u64
                            + (curr.rgb[i_cur + 1] as i32 - prev.rgb[i_prv + 1] as i32)
                                .unsigned_abs() as u64
                            + (curr.rgb[i_cur + 2] as i32 - prev.rgb[i_prv + 2] as i32)
                                .unsigned_abs() as u64;
                        cnt += 1;
                    }
                    x += 2;
                }
            }
            y += 2;
        }
        if cnt > 16 {
            let reg = ((dx - init_delta[0]).unsigned_abs() as u64
                + (dy - init_delta[1]).unsigned_abs() as u64)
                * 64;
            Some(err * 1024 / cnt + reg)
        } else {
            None
        }
    };

    let mut best = init_delta;
    let mut best_err = eval_delta(init_delta[0], init_delta[1]).unwrap_or(u64::MAX);

    for dy in (init_delta[1] - 2)..=(init_delta[1] + 2) {
        for dx in (init_delta[0] - 2)..=(init_delta[0] + 2) {
            if let Some(norm_err) = eval_delta(dx, dy) {
                if norm_err < best_err {
                    best_err = norm_err;
                    best = [dx, dy];
                }
            }
        }
    }
    best
}

/// Mean spatial gradient magnitude inside `[y0, y1)` across sampled frames.
pub fn band_spatial_gradient(frames: &[RawFrame], y0: usize, y1: usize) -> f64 {
    if frames.is_empty() || y1 <= y0 + 2 {
        return 0.0;
    }
    let w = frames[0].width as usize;
    let step_f = (frames.len() / 6).max(1);
    let mut sum = 0u64;
    let mut cnt = 0u64;
    for f in frames.iter().step_by(step_f) {
        for y in (y0..y1 - 1).step_by(2) {
            for x in (4..w.saturating_sub(5)).step_by(2) {
                let i = (y * w + x) * 3;
                let ir = (y * w + x + 1) * 3;
                let id = ((y + 1) * w + x) * 3;
                let gx = (f.rgb[i] as i32 - f.rgb[ir] as i32).unsigned_abs()
                    + (f.rgb[i + 1] as i32 - f.rgb[ir + 1] as i32).unsigned_abs()
                    + (f.rgb[i + 2] as i32 - f.rgb[ir + 2] as i32).unsigned_abs();
                let gy = (f.rgb[i] as i32 - f.rgb[id] as i32).unsigned_abs()
                    + (f.rgb[i + 1] as i32 - f.rgb[id + 1] as i32).unsigned_abs()
                    + (f.rgb[i + 2] as i32 - f.rgb[id + 2] as i32).unsigned_abs();
                sum += (gx + gy) as u64;
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum as f64 / (cnt as f64 * 3.0)
    }
}

/// Mean inter-frame RGB L1 alignment residual inside `[y0, y1)` along `traj`.
pub fn band_registration_error(
    frames: &[RawFrame],
    y0: usize,
    y1: usize,
    traj: &[[i32; 2]],
) -> f64 {
    if frames.len() < 2 || traj.len() < frames.len() {
        return 0.0;
    }
    let w = frames[0].width as i32;
    let y0i = y0 as i32;
    let y1i = y1 as i32;
    let mut sum = 0u64;
    let mut cnt = 0u64;
    for fi in 1..frames.len() {
        let dx = traj[fi][0] - traj[fi - 1][0];
        let dy = traj[fi][1] - traj[fi - 1][1];
        let prev = &frames[fi - 1];
        let curr = &frames[fi];
        let mut y = y0i;
        while y < y1i {
            let py = y + dy;
            if py >= y0i && py < y1i {
                let mut x = 8;
                while x < w - 8 {
                    let px = x + dx;
                    if px >= 0 && px < w {
                        let ic = (y as usize * w as usize + x as usize) * 3;
                        let ip = (py as usize * w as usize + px as usize) * 3;
                        sum += (curr.rgb[ic] as i32 - prev.rgb[ip] as i32).unsigned_abs() as u64
                            + (curr.rgb[ic + 1] as i32 - prev.rgb[ip + 1] as i32).unsigned_abs()
                                as u64
                            + (curr.rgb[ic + 2] as i32 - prev.rgb[ip + 2] as i32).unsigned_abs()
                                as u64;
                        cnt += 1;
                    }
                    x += 2;
                }
            }
            y += 2;
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum as f64 / (cnt as f64 * 3.0)
    }
}

pub fn trajectory_rms_diff(a: &[[i32; 2]], b: &[[i32; 2]]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let dx = (a[i][0] - b[i][0]) as f32;
        let dy = (a[i][1] - b[i][1]) as f32;
        sum_sq += dx * dx + dy * dy;
    }
    (sum_sq / n as f32).sqrt()
}

pub fn accumulate_band_atlas(
    frames: &[RawFrame],
    traj: &[[i32; 2]],
    y0: usize,
    y1: usize,
    masks: &[Vec<bool>],
    held_out: &[usize],
) -> WorldAtlas {
    let w = frames[0].width as usize;
    let h = frames[0].height as usize;
    let y1 = y1.min(h);
    let mut buckets: HashMap<(i32, i32), Bucket> = HashMap::new();
    let mut min = [i32::MAX, i32::MAX];
    let mut max = [i32::MIN, i32::MIN];

    for &[ox, oy] in traj {
        min[0] = min[0].min(ox);
        min[1] = min[1].min(y0 as i32 + oy);
        max[0] = max[0].max(w as i32 - 1 + ox);
        max[1] = max[1].max(y1 as i32 - 1 + oy);
    }

    for (fi, frame) in frames.iter().enumerate() {
        if held_out.contains(&fi) {
            continue;
        }
        let off = traj[fi];
        let mask = masks.get(fi);
        for y in y0..y1 {
            let wy = y as i32 + off[1];
            for x in 0..w {
                if mask.map_or(false, |m| m[y * w + x]) {
                    continue;
                }
                let wx = x as i32 + off[0];
                let c = frame.pixel(x as u32, y as u32);
                buckets.entry((wx, wy)).or_default().add(c);
            }
        }
    }

    let cw = (max[0] - min[0] + 1).max(1) as usize;
    let ch = (max[1] - min[1] + 1).max(1) as usize;
    let mut image = RgbImage8::new(cw, ch);
    for (&(wx, wy), b) in &buckets {
        if let Some((c, _)) = b.mode() {
            let ix = (wx - min[0]) as usize;
            let iy = (wy - min[1]) as usize;
            if ix < cw && iy < ch {
                image.set_pixel(ix, iy, c);
            }
        }
    }
    WorldAtlas { image, origin: min }
}

pub fn accumulate_band_atlas_masked(
    frames: &[RawFrame],
    traj: &[[i32; 2]],
    y0: usize,
    y1: usize,
    masks: &[Vec<bool>],
    held_out: &[usize],
    fallback: &WorldAtlas,
) -> WorldAtlas {
    let mut atlas = accumulate_band_atlas(frames, traj, y0, y1, masks, held_out);
    let cw = atlas.image.width;
    let ch = atlas.image.height;

    let mut valid_count = vec![0u16; cw * ch];
    let mut obs_count = vec![0u16; cw * ch];
    let mut occl_count = vec![0u16; cw * ch];
    let w = frames[0].width as usize;
    let y1 = y1.min(frames[0].height as usize);
    for (fi, _) in frames.iter().enumerate() {
        if held_out.contains(&fi) {
            continue;
        }
        let off = traj[fi];
        let mask = masks.get(fi);
        for y in y0..y1 {
            let wy = y as i32 + off[1];
            for x in 0..w {
                let wx = x as i32 + off[0];
                let ix = (wx - atlas.origin[0]) as usize;
                let iy = (wy - atlas.origin[1]) as usize;
                if ix < cw && iy < ch {
                    let idx = iy * cw + ix;
                    obs_count[idx] += 1;
                    if mask.map_or(false, |m| m[y * w + x]) {
                        occl_count[idx] += 1;
                    } else {
                        valid_count[idx] += 1;
                    }
                }
            }
        }
    }

    let has_any_valid = valid_count.iter().any(|&v| v > 0);
    if !has_any_valid {
        for iy in 0..ch {
            for ix in 0..cw {
                let wx = atlas.origin[0] + ix as i32;
                let wy = atlas.origin[1] + iy as i32;
                if let Some(c) = fallback.pixel(wx, wy) {
                    atlas.image.set_pixel(ix, iy, c);
                }
            }
        }
        return atlas;
    }

    // Unobserved texels (outside the camera's sweep) take the temporal-median fallback. Texels that
    // were observed but occluded in >= 50% of temporal frames (or never validly observed with >= 4 frames)
    // are unknown or actor-dominated background: fill them with the harmonic (Laplace, $\nabla^2 I = 0$)
    // continuation of the surrounding observed ambient background plenum (purging stationary character holds).
    let mut hole = vec![false; cw * ch];
    for iy in 0..ch {
        for ix in 0..cw {
            let idx = iy * cw + ix;
            // Dimensionless occlusion ratio: a locus is clean background if it has at least one
            // valid unoccluded observation and the occlusion fraction is strictly under half (majority unoccluded).
            if valid_count[idx] > 0 && occl_count[idx] * 2 < obs_count[idx] {
                continue;
            }
            if obs_count[idx] == 0 {
                let wx = atlas.origin[0] + ix as i32;
                let wy = atlas.origin[1] + iy as i32;
                if let Some(c) = fallback.pixel(wx, wy) {
                    atlas.image.set_pixel(ix, iy, c);
                    continue;
                }
            }
            hole[idx] = true;
        }
    }
    harmonic_inpaint(&mut atlas.image, &hole);
    atlas
}

/// Fill `hole` texels with the solution of Laplace's equation whose Dirichlet boundary is every
/// non-hole texel (minimal-gradient-energy continuation). Successive over-relaxation with the
/// optimal factor for the hole's extent; iterates until no texel changes by half a quantization step.
pub fn harmonic_inpaint(img: &mut crate::enfold2d::RgbImage8, hole: &[bool]) {
    let (w, h) = (img.width, img.height);
    let idx: Vec<usize> = (0..w * h).filter(|&i| hole[i]).collect();
    if idx.is_empty() || idx.len() == w * h {
        return;
    }
    let mut f: Vec<[f32; 3]> = (0..w * h)
        .map(|i| {
            let c = img.pixel(i % w, i / w);
            [c[0] as f32, c[1] as f32, c[2] as f32]
        })
        .collect();
    // Initial guess: mean of the known boundary ring, so iteration starts near the solution.
    let mut ring = [0.0f32; 3];
    let mut n_ring = 0.0f32;
    for &i in &idx {
        let (x, y) = (i % w, i / w);
        for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
            if nx < w && ny < h && !hole[ny * w + nx] {
                for k in 0..3 {
                    ring[k] += f[ny * w + nx][k];
                }
                n_ring += 1.0;
            }
        }
    }
    if n_ring > 0.0 {
        for &i in &idx {
            f[i] = ring.map(|v| v / n_ring);
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (w, 0, h, 0);
    for &i in &idx {
        x0 = x0.min(i % w);
        x1 = x1.max(i % w);
        y0 = y0.min(i / w);
        y1 = y1.max(i / w);
    }
    let extent = (x1 - x0).max(y1 - y0) as f32 + 2.0;
    let omega = 2.0 / (1.0 + (std::f32::consts::PI / extent).sin());
    let max_sweeps = (extent * extent) as usize;
    for _ in 0..max_sweeps {
        let mut max_delta = 0.0f32;
        for &i in &idx {
            let (x, y) = (i % w, i / w);
            let mut acc = [0.0f32; 3];
            let mut n = 0.0f32;
            for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                if nx < w && ny < h {
                    let v = f[ny * w + nx];
                    for k in 0..3 {
                        acc[k] += v[k];
                    }
                    n += 1.0;
                }
            }
            for k in 0..3 {
                let target = acc[k] / n;
                let d = omega * (target - f[i][k]);
                f[i][k] += d;
                max_delta = max_delta.max(d.abs());
            }
        }
        if max_delta < 0.5 {
            break;
        }
    }
    for &i in &idx {
        let c = f[i].map(|v| v.round().clamp(0.0, 255.0) as u8);
        img.set_pixel(i % w, i / w, c);
    }
}

