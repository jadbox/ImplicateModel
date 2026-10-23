//! K-means part prototype learning, posture idempotent clustering, and Gaussian-Gabor kernel accumulation.

use super::{
    CFT_WAVEPACKET_MODES, CharacterObservation, SubHolonPose, WavepacketPartKernel,
    clifford_features::compute_cliffordnet_laplacian_features,
};
use crate::enfold2d::Bucket;

pub fn median_obs_dims(observations: &[CharacterObservation]) -> (usize, usize) {
    if observations.is_empty() {
        return (16, 16);
    }
    let mut ws: Vec<usize> = observations.iter().map(|o| o.width()).collect();
    let mut hs: Vec<usize> = observations.iter().map(|o| o.height()).collect();
    ws.sort_unstable();
    hs.sort_unstable();
    (ws[ws.len() / 2].max(8), hs[hs.len() / 2].max(8))
}

pub fn learn_part_prototypes(
    observations: &[CharacterObservation],
    num_parts: usize,
) -> Vec<[f32; 6]> {
    if num_parts == 1 {
        let mut sum = [0.0f32; 6];
        let mut cnt = 0.0f32;
        for obs in observations {
            for i in 0..obs.width() * obs.height() {
                if obs.mask[i] {
                    sum[2] += obs.rgb[i * 3] as f32 / 255.0;
                    sum[3] += obs.rgb[i * 3 + 1] as f32 / 255.0;
                    sum[4] += obs.rgb[i * 3 + 2] as f32 / 255.0;
                    cnt += 1.0;
                }
            }
        }
        if cnt > 0.0 {
            sum[2] /= cnt;
            sum[3] /= cnt;
            sum[4] /= cnt;
        }
        return vec![sum];
    }

    let mut samples: Vec<[f32; 6]> = Vec::new();
    for obs in observations {
        let (w, h) = (obs.width(), obs.height());
        let cliff = compute_cliffordnet_laplacian_features(&obs.rgb, w, h);
        for y in (0..h).step_by(2) {
            let ny = (y as f32 / h.max(1) as f32) * 2.0 - 1.0;
            for x in (0..w).step_by(2) {
                let idx = y * w + x;
                if !obs.mask[idx] {
                    continue;
                }
                let nx = (x as f32 / w.max(1) as f32) * 2.0 - 1.0;
                samples.push([
                    nx,
                    ny,
                    obs.rgb[idx * 3] as f32 / 255.0,
                    obs.rgb[idx * 3 + 1] as f32 / 255.0,
                    obs.rgb[idx * 3 + 2] as f32 / 255.0,
                    cliff[idx][1],
                ]);
            }
        }
    }

    if samples.is_empty() {
        return vec![[0.0, 0.0, 0.5, 0.5, 0.5, 0.0]; num_parts];
    }

    let mut centroids = vec![[0.0f32; 6]; num_parts];
    centroids[0] = [0.0, 0.0, 0.5, 0.5, 0.5, 0.1];
    for p in 1..num_parts {
        let ang = 2.0 * std::f32::consts::PI * (p - 1) as f32 / (num_parts - 1) as f32
            - std::f32::consts::FRAC_PI_2;
        centroids[p] = [0.55 * ang.cos(), 0.55 * ang.sin(), 0.5, 0.5, 0.5, 0.2];
    }
    for c in &mut centroids {
        if let Some(&nearest) = samples.iter().min_by(|a, b| {
            let da = (a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2);
            let db = (b[0] - c[0]).powi(2) + (b[1] - c[1]).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            *c = nearest;
        }
    }

    for _iter in 0..8 {
        let mut sums = vec![[0.0f32; 6]; num_parts];
        let mut counts = vec![0usize; num_parts];
        for s in &samples {
            let mut best_p = 0usize;
            let mut best_d = f32::MAX;
            for (p, c) in centroids.iter().enumerate() {
                let d = 1.3 * ((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2))
                    + 0.9
                        * ((s[2] - c[2]).powi(2)
                            + (s[3] - c[3]).powi(2)
                            + (s[4] - c[4]).powi(2))
                    + 0.25 * (s[5] - c[5]).powi(2);
                if d < best_d {
                    best_d = d;
                    best_p = p;
                }
            }
            counts[best_p] += 1;
            for k in 0..6 {
                sums[best_p][k] += s[k];
            }
        }
        for p in 0..num_parts {
            if counts[p] > 0 {
                let inv = 1.0 / counts[p] as f32;
                for k in 0..6 {
                    centroids[p][k] = sums[p][k] * inv;
                }
            }
        }
    }

    centroids.sort_by(|a, b| {
        let da = a[0] * a[0] + a[1] * a[1];
        let db = b[0] * b[0] + b[1] * b[1];
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    centroids
}

/// Compute a 64D 2D Clifford Fourier Transform (CFT) posture & spinor density descriptor
/// for a character observation in normalized canonical coordinates $(u, v) \in [-1, 1]^2$.
pub fn compute_cft_posture_descriptor(obs: &CharacterObservation) -> [f32; 64] {
    let mut d = [0.0f32; 64];
    let (w, h) = (obs.width().max(1), obs.height().max(1));
    let mut grid = [0.0f32; 64];
    let mut chroma = [0.0f32; 64];
    for gy in 0..8 {
        for gx in 0..8 {
            let idx = ((gy * h) / 8) * w + ((gx * w) / 8);
            if obs.mask[idx] {
                let r = obs.rgb[idx * 3] as f32 / 255.0;
                let g = obs.rgb[idx * 3 + 1] as f32 / 255.0;
                let b = obs.rgb[idx * 3 + 2] as f32 / 255.0;
                let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                grid[gy * 8 + gx] = 0.5 + 0.5 * luma;
                chroma[gy * 8 + gx] = r - b;
            }
        }
    }
    // First 32 entries: low-frequency 2D Clifford Fourier modes (kx, ky in -1..=2)
    let mut k_out = 0usize;
    for ky in -1i32..=2 {
        for kx in -1i32..=2 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for gy in 0..8 {
                let v = (gy as f32 / 7.0) * 2.0 - 1.0;
                for gx in 0..8 {
                    let u = (gx as f32 / 7.0) * 2.0 - 1.0;
                    let ang = -std::f32::consts::PI * (kx as f32 * u + ky as f32 * v);
                    let (s, c) = ang.sin_cos();
                    let a = grid[gy * 8 + gx];
                    let b = chroma[gy * 8 + gx];
                    re += a * c - b * s;
                    im += a * s + b * c;
                }
            }
            d[k_out] = re / 16.0;
            d[k_out + 1] = im / 16.0;
            k_out += 2;
        }
    }
    // Remaining 32 entries: 4x8 spatial spinor density
    for i in 0..32 {
        d[32 + i] = 0.5 * (grid[2 * i] + grid[2 * i + 1]);
    }
    d
}

pub fn assign_view_idempotent_cells_and_cft_phases(
    observations: &[CharacterObservation],
    num_view_cells: usize,
) -> (Vec<usize>, Vec<f64>) {
    let n = observations.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let descs: Vec<[f32; 64]> = observations
        .iter()
        .map(compute_cft_posture_descriptor)
        .collect();

    let k = num_view_cells.clamp(1, n);
    let mut protos: Vec<[f32; 64]> = (0..k).map(|i| descs[(i * n) / k]).collect();
    let mut assign = vec![0usize; n];

    if k > 1 {
        for _ in 0..8 {
            for (oi, d) in descs.iter().enumerate() {
                let mut best_c = 0usize;
                let mut best_err = f32::MAX;
                for (ci, p) in protos.iter().enumerate() {
                    let err: f32 = d.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                    if err < best_err {
                        best_err = err;
                        best_c = ci;
                    }
                }
                assign[oi] = best_c;
            }
            for ci in 0..k {
                let mut sum = [0.0f32; 64];
                let mut cnt = 0usize;
                for (oi, &c) in assign.iter().enumerate() {
                    if c == ci {
                        for j in 0..64 {
                            sum[j] += descs[oi][j];
                        }
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    for j in 0..64 {
                        protos[ci][j] = sum[j] / cnt as f32;
                    }
                }
            }
        }
    }

    // Assign intrinsic CFT phase within each view cell by sorting observations along their principal CFT mode
    let mut phases = vec![0.0f64; n];
    for ci in 0..k {
        let mut members: Vec<(usize, f32)> = (0..n)
            .filter(|&oi| assign[oi] == ci)
            .map(|oi| {
                let score: f32 = (0..32)
                    .map(|j| (descs[oi][j] - protos[ci][j]) * if j % 2 == 0 { 1.0 } else { 0.7 })
                    .sum();
                (oi, score)
            })
            .collect();
        members.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let m_len = members.len().max(1);
        for (rank, &(oi, _)) in members.iter().enumerate() {
            phases[oi] = 2.0 * std::f64::consts::PI * (rank as f64) / (m_len as f64);
        }
    }
    (assign, phases)
}

pub fn fit_part_gaussian_pose(
    obs: &CharacterObservation,
    pixel_part: &[usize],
    part_id: usize,
    num_parts: usize,
    view_cell: usize,
    phase_rad: f64,
) -> SubHolonPose {
    let (w, h) = (obs.width(), obs.height());
    if num_parts == 1 {
        return SubHolonPose {
            part_id: 0,
            parent_id: 0,
            center: [
                obs.bbox.x0 as f64 + (w.saturating_sub(1) as f64) * 0.5,
                obs.bbox.y0 as f64 + (h.saturating_sub(1) as f64) * 0.5,
            ],
            angle_rad: 0.0,
            radii: [
                ((w.saturating_sub(1) as f64) * 0.5).max(1.0),
                ((h.saturating_sub(1) as f64) * 0.5).max(1.0),
            ],
            joint_angle_rad: 0.0,
            joint_offset_local: [0.0, 0.0],
            view_idempotent_idx: view_cell,
            phase_rad,
        };
    }

    let (mut sum_x, mut sum_y, mut cnt) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if obs.mask[idx] && pixel_part[idx] == part_id {
                sum_x += obs.bbox.x0 as f64 + x as f64;
                sum_y += obs.bbox.y0 as f64 + y as f64;
                cnt += 1.0;
            }
        }
    }

    if cnt < 4.0 {
        return SubHolonPose {
            part_id,
            parent_id: 0,
            center: [
                obs.bbox.x0 as f64 + w as f64 * 0.5,
                obs.bbox.y0 as f64 + h as f64 * 0.5,
            ],
            angle_rad: 0.0,
            radii: [(w as f64 * 0.25).max(2.0), (h as f64 * 0.25).max(2.0)],
            joint_angle_rad: 0.0,
            joint_offset_local: [0.0, 0.0],
            view_idempotent_idx: view_cell,
            phase_rad,
        };
    }

    let (cx, cy) = (sum_x / cnt, sum_y / cnt);
    let (mut s_xx, mut s_yy, mut s_xy) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if obs.mask[idx] && pixel_part[idx] == part_id {
                let dx = (obs.bbox.x0 as f64 + x as f64) - cx;
                let dy = (obs.bbox.y0 as f64 + y as f64) - cy;
                s_xx += dx * dx;
                s_yy += dy * dy;
                s_xy += dx * dy;
            }
        }
    }
    s_xx /= cnt;
    s_yy /= cnt;
    s_xy /= cnt;

    let angle_rad = 0.5 * (2.0 * s_xy).atan2(s_xx - s_yy);
    let (s, c) = angle_rad.sin_cos();

    let (mut max_u, mut max_v) = (2.0f64, 2.0f64);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if obs.mask[idx] && pixel_part[idx] == part_id {
                let dx = (obs.bbox.x0 as f64 + x as f64) - cx;
                let dy = (obs.bbox.y0 as f64 + y as f64) - cy;
                max_u = max_u.max((c * dx + s * dy).abs() + 1.0);
                max_v = max_v.max((-s * dx + c * dy).abs() + 1.0);
            }
        }
    }

    SubHolonPose {
        part_id,
        parent_id: 0,
        center: [cx, cy],
        angle_rad,
        radii: [max_u, max_v],
        joint_angle_rad: 0.0,
        joint_offset_local: [0.0, 0.0],
        view_idempotent_idx: view_cell,
        phase_rad,
    }
}

fn solve_5x5(mut a: [[f64; 5]; 5], mut b: [f64; 5]) -> [f32; 5] {
    for i in 0..5 {
        let mut piv = i;
        let mut max_v = a[i][i].abs();
        for r in (i + 1)..5 {
            if a[r][i].abs() > max_v {
                max_v = a[r][i].abs();
                piv = r;
            }
        }
        if max_v < 1e-9 {
            continue;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let diag = a[i][i];
        for c in i..5 {
            a[i][c] /= diag;
        }
        b[i] /= diag;
        for r in 0..5 {
            if r != i {
                let f = a[r][i];
                for c in i..5 {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }
    }
    [
        b[0] as f32,
        b[1] as f32,
        b[2] as f32,
        b[3] as f32,
        b[4] as f32,
    ]
}

pub fn accumulate_part_wavepacket_kernel(
    observations: &[CharacterObservation],
    poses_per_obs: &[Vec<SubHolonPose>],
    part_assignments: &[Vec<usize>],
    part_id: usize,
    view_cell: usize,
    res_w: usize,
    res_h: usize,
) -> WavepacketPartKernel {
    let n_cells = res_w * res_h;
    let mut buckets = vec![Bucket::default(); n_cells];
    let mut fg_hits = vec![0u32; n_cells];
    let mut cell_obs_indices: Vec<(usize, [f64; 5])> = Vec::new();
    let mut gram = [[0.0f64; 5]; 5];
    gram[0][0] = 1e-4;
    for m in 1..5 {
        gram[m][m] = 0.08;
    }

    for (oi, obs) in observations.iter().enumerate() {
        let pose = &poses_per_obs[oi][part_id];
        if pose.view_idempotent_idx != view_cell {
            continue;
        }
        let bf = WavepacketPartKernel::basis(pose.phase_rad);
        let b = [
            bf[0] as f64,
            bf[1] as f64,
            bf[2] as f64,
            bf[3] as f64,
            bf[4] as f64,
        ];
        for r in 0..5 {
            for c in 0..5 {
                gram[r][c] += b[r] * b[c];
            }
        }
        cell_obs_indices.push((oi, b));

        let (w, h) = (obs.width(), obs.height());
        let p_assign = &part_assignments[oi];
        for ky in 0..res_h {
            let v = (ky as f64 / (res_h as f64 - 1.0).max(1.0)) * 2.0 - 1.0;
            for kx in 0..res_w {
                let u = (kx as f64 / (res_w as f64 - 1.0).max(1.0)) * 2.0 - 1.0;
                let [wx, wy] = pose.local_to_world(u, v);
                let lx = (wx - obs.bbox.x0 as f64).round() as i32;
                let ly = (wy - obs.bbox.y0 as f64).round() as i32;
                if lx < 0 || ly < 0 || lx >= w as i32 || ly >= h as i32 {
                    continue;
                }
                let p_idx = ly as usize * w + lx as usize;
                let k_idx = ky * res_w + kx;
                if obs.mask[p_idx] && p_assign[p_idx] == part_id {
                    let c = [
                        obs.rgb[p_idx * 3],
                        obs.rgb[p_idx * 3 + 1],
                        obs.rgb[p_idx * 3 + 2],
                    ];
                    buckets[k_idx].add(c);
                    fg_hits[k_idx] += 1;
                }
            }
        }
    }

    let mut dc_rgba = vec![0u8; n_cells * 4];
    let harmonic_rgba = vec![128u8; n_cells * 4];
    let mut cft_rgba_coeffs = vec![[0.0f32; CFT_WAVEPACKET_MODES]; n_cells * 4];
    let mut gaussian_envelope = vec![1.0f32; n_cells];
    let mut mode_rgb = vec![[0u8; 3]; n_cells];

    for k_idx in 0..n_cells {
        if let Some((c, _)) = buckets[k_idx].mode() {
            mode_rgb[k_idx] = c;
            dc_rgba[k_idx * 4..k_idx * 4 + 3].copy_from_slice(&c);
        }
    }

    let mut rhs = vec![[0.0f64; 5]; n_cells * 4];
    for &(oi, b) in &cell_obs_indices {
        let obs = &observations[oi];
        let pose = &poses_per_obs[oi][part_id];
        let (w, h) = (obs.width(), obs.height());
        let p_assign = &part_assignments[oi];

        for ky in 0..res_h {
            let v = (ky as f64 / (res_h as f64 - 1.0).max(1.0)) * 2.0 - 1.0;
            for kx in 0..res_w {
                let u = (kx as f64 / (res_w as f64 - 1.0).max(1.0)) * 2.0 - 1.0;
                let [wx, wy] = pose.local_to_world(u, v);
                let lx = (wx - obs.bbox.x0 as f64).round() as i32;
                let ly = (wy - obs.bbox.y0 as f64).round() as i32;
                let k_idx = ky * res_w + kx;
                let mut is_fg = false;
                let mut col = mode_rgb[k_idx];
                if lx >= 0 && ly >= 0 && lx < w as i32 && ly < h as i32 {
                    let p_idx = ly as usize * w + lx as usize;
                    if obs.mask[p_idx] && p_assign[p_idx] == part_id {
                        is_fg = true;
                        col = [
                            obs.rgb[p_idx * 3],
                            obs.rgb[p_idx * 3 + 1],
                            obs.rgb[p_idx * 3 + 2],
                        ];
                    }
                }
                let a_val = if is_fg { 255.0 } else { 0.0 };
                let targets = [col[0] as f64, col[1] as f64, col[2] as f64, a_val];
                for ch in 0..4 {
                    for m in 0..5 {
                        rhs[k_idx * 4 + ch][m] += targets[ch] * b[m];
                    }
                }
            }
        }
    }

    for ky in 0..res_h {
        let v = (ky as f32 / (res_h as f32 - 1.0).max(1.0)) * 2.0 - 1.0;
        for kx in 0..res_w {
            let u = (kx as f32 / (res_w as f32 - 1.0).max(1.0)) * 2.0 - 1.0;
            let k_idx = ky * res_w + kx;
            let r2 = u * u + v * v;
            gaussian_envelope[k_idx] = (-0.08 * r2 * r2).exp();
            if fg_hits[k_idx] > 0 {
                for ch in 0..4 {
                    let coeffs = solve_5x5(gram, rhs[k_idx * 4 + ch]);
                    cft_rgba_coeffs[k_idx * 4 + ch] = coeffs;
                }
                dc_rgba[k_idx * 4 + 3] = cft_rgba_coeffs[k_idx * 4 + 3][0]
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }

    WavepacketPartKernel {
        res_w,
        res_h,
        dc_rgba,
        harmonic_rgba,
        cft_rgba_coeffs,
        gaussian_envelope,
    }
}

pub struct RigidSingleBoxBaseline {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
}

pub fn fit_rigid_single_box_baseline(
    observations: &[CharacterObservation],
) -> RigidSingleBoxBaseline {
    let (mw, mh) = median_obs_dims(observations);
    let n = mw * mh;
    let mut buckets = vec![Bucket::default(); n];
    let mut hits = vec![0u32; n];
    let total = observations.len().max(1) as u32;

    for obs in observations {
        let (ow, oh) = (obs.width(), obs.height());
        let ox0 = (ow as i32 - mw as i32) / 2;
        let oy0 = (oh as i32 - mh as i32) / 2;
        for sy in 0..mh {
            let py = oy0 + sy as i32;
            if py < 0 || py >= oh as i32 {
                continue;
            }
            for sx in 0..mw {
                let px = ox0 + sx as i32;
                if px < 0 || px >= ow as i32 {
                    continue;
                }
                let p_idx = py as usize * ow + px as usize;
                if obs.mask[p_idx] {
                    let s_idx = sy * mw + sx;
                    let c = [
                        obs.rgb[p_idx * 3],
                        obs.rgb[p_idx * 3 + 1],
                        obs.rgb[p_idx * 3 + 2],
                    ];
                    buckets[s_idx].add(c);
                    hits[s_idx] += 1;
                }
            }
        }
    }

    let mut rgba = vec![0u8; n * 4];
    let min_hits = (total * 2 / 5).max(1);
    for i in 0..n {
        if hits[i] >= min_hits {
            if let Some((c, _)) = buckets[i].mode() {
                rgba[i * 4..i * 4 + 3].copy_from_slice(&c);
                rgba[i * 4 + 3] = 255;
            }
        }
    }
    RigidSingleBoxBaseline { w: mw, h: mh, rgba }
}

pub fn render_rigid_baseline_observation(
    base: &RigidSingleBoxBaseline,
    obs: &CharacterObservation,
) -> (Vec<bool>, Vec<u8>) {
    let (ow, oh) = (obs.width(), obs.height());
    let mut mask = vec![false; ow * oh];
    let mut rgb = vec![0u8; ow * oh * 3];
    let ox0 = (ow as i32 - base.w as i32) / 2;
    let oy0 = (oh as i32 - base.h as i32) / 2;

    for sy in 0..base.h {
        let py = oy0 + sy as i32;
        if py < 0 || py >= oh as i32 {
            continue;
        }
        for sx in 0..base.w {
            let px = ox0 + sx as i32;
            if px < 0 || px >= ow as i32 {
                continue;
            }
            let s_idx = sy * base.w + sx;
            if base.rgba[s_idx * 4 + 3] > 127 {
                let p_idx = py as usize * ow + px as usize;
                mask[p_idx] = true;
                rgb[p_idx * 3..p_idx * 3 + 3].copy_from_slice(&base.rgba[s_idx * 4..s_idx * 4 + 3]);
            }
        }
    }
    (mask, rgb)
}
