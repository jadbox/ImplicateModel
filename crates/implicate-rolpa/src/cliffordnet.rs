//! 4-Block No-FFN `CliffordNet-Lite` `@rolpa` Visual Refiner (Track R3.3, Whitepaper §3.2 & §7.1).
//!
//! Implements Algorithm 1 of Ji (2026, `reference_cliffordnet.pdf`) over the 12-channel composite
//! $X_{\text{in}} = (I_{\text{base}}, \nabla p, \nabla \langle \mathcal{Q}_{\text{Cl}} \rangle_0, C, \mathbf{L})$,
//! computing the factorized discrete Laplacian context $\mathcal{C}_{\text{loc}}(H) \approx \nabla^2 H$
//! and shifted Clifford geometric products $D^{(s)} \oplus W^{(s)}$ ($\mathcal{S} = \{1, 2, 4, 8, 16\}$)
//! with Gated Geometric Residuals (GGR).

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

pub const CLIFFORDNET_BLOCKS: usize = 4;
pub const CLIFFORDNET_SHIFTS: [usize; 5] = [1, 2, 4, 8, 16];

/// Uniform parameters shared between CPU and GPU (`wgpu`) `CliffordNet-Lite` refiners.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct RolpaUniforms {
    pub width: u32,
    pub height: u32,
    pub bypass_mode: u32,
    pub _pad0: u32,
    /// Readout weights `[w_lap1, w_lap2, w_cliff_d, w_cliff_w]`.
    pub readout_weights: [f32; 4],
    /// Per-block Gated Geometric Residual gains `[gamma_0, gamma_1, gamma_2, gamma_3]`.
    pub block_gammas: [f32; 4],
}

impl Default for RolpaUniforms {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            bypass_mode: 0,
            _pad0: 0,
            readout_weights: [1.35, -0.38, 0.12, 0.18],
            block_gammas: [0.35, 0.30, 0.25, 0.20],
        }
    }
}

/// 12-channel `@rolpa` conditioning grid packed as three `vec4<f32>` buffers per pixel:
/// 1. `base_c`: `(R_base, G_base, B_base, cohesion_C)`
/// 2. `grad_pq`: `(dp_dx, dp_dy, dQ_dx, dQ_dy)`
/// 3. `lum`: `(L_r, L_g, L_b, edge_prior)`
#[derive(Clone, Debug)]
pub struct RolpaInputGrid {
    pub width: usize,
    pub height: usize,
    pub base_c: Vec<[f32; 4]>,
    pub grad_pq: Vec<[f32; 4]>,
    pub lum: Vec<[f32; 4]>,
}

impl RolpaInputGrid {
    /// Construct the 12-channel `@rolpa` input tensor from a base wavepacket composite `base_rgb`
    /// (`[f32; 3]` per pixel in `[0, 1]`), computing physical pressure gradients $\nabla p$,
    /// Bohmian potential gradients $\nabla \langle \mathcal{Q}_{\text{Cl}} \rangle_0$, cohesion $C$,
    /// and chromatic luminescence $\mathbf{L}$.
    pub fn from_base_rgb(width: usize, height: usize, base_rgb: &[[f32; 3]]) -> Self {
        let n = width * height;
        assert_eq!(base_rgb.len(), n);
        let mut base_c = vec![[0.0f32; 4]; n];
        let mut grad_pq = vec![[0.0f32; 4]; n];
        let mut lum = vec![[0.0f32; 4]; n];

        let mut r_env = vec![0.0f32; n];
        for i in 0..n {
            let [r, g, b] = base_rgb[i];
            let l = (0.299 * r + 0.587 * g + 0.114 * b).max(0.02);
            r_env[i] = l.sqrt();
        }

        let mut q_scalar = vec![0.0f32; n];
        for y in 0..height {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(height - 1);
            for x in 0..width {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(width - 1);
                let idx = y * width + x;
                let rc = r_env[idx];
                let lap_r = r_env[ym * width + x]
                    + r_env[yp * width + x]
                    + r_env[y * width + xm]
                    + r_env[y * width + xp]
                    - 4.0 * rc;
                q_scalar[idx] = (-lap_r / rc).clamp(-4.0, 4.0);
            }
        }

        for y in 0..height {
            let ym = y.saturating_sub(1);
            let yp = (y + 1).min(height - 1);
            for x in 0..width {
                let xm = x.saturating_sub(1);
                let xp = (x + 1).min(width - 1);
                let idx = y * width + x;
                let [r, g, b] = base_rgb[idx];

                let dp_dx = 0.5 * (r_env[y * width + xp] - r_env[y * width + xm]);
                let dp_dy = 0.5 * (r_env[yp * width + x] - r_env[ym * width + x]);
                let dq_dx = 0.5 * (q_scalar[y * width + xp] - q_scalar[y * width + xm]);
                let dq_dy = 0.5 * (q_scalar[yp * width + x] - q_scalar[ym * width + x]);
                let edge_mag = (dp_dx * dp_dx + dp_dy * dp_dy).sqrt();
                let cohesion = (1.0 - 2.0 * edge_mag).clamp(0.2, 1.0);

                base_c[idx] = [r, g, b, cohesion];
                grad_pq[idx] = [dp_dx, dp_dy, dq_dx, dq_dy];
                lum[idx] = [r * 0.25, g * 0.25, b * 0.25, edge_mag];
            }
        }

        Self {
            width,
            height,
            base_c,
            grad_pq,
            lum,
        }
    }
}

/// CPU reference implementation and closed-form calibrator for the 4-block `CliffordNet-Lite` refiner.
#[derive(Clone, Debug, Default)]
pub struct CliffordNetLiteRefiner {
    pub uniforms: RolpaUniforms,
}

impl CliffordNetLiteRefiner {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            uniforms: RolpaUniforms {
                width,
                height,
                ..RolpaUniforms::default()
            },
        }
    }

    /// Extract per-pixel 4-component `CliffordNet-Lite` basis features
    /// `[lap1_rgb, lap2_rgb, cliff_d_acc, cliff_w_acc]` across the 4 No-FFN blocks.
    pub fn extract_basis_features(&self, grid: &RolpaInputGrid) -> Vec<[[f32; 3]; 4]> {
        let w = grid.width;
        let h = grid.height;
        let n = w * h;
        let mut state = vec![[0.0f32; 4]; n];
        for i in 0..n {
            let bc = grid.base_c[i];
            let gpq = grid.grad_pq[i];
            let lm = grid.lum[i];
            state[i] = [
                bc[0] + 0.02 * gpq[0] + 0.02 * lm[0],
                bc[1] + 0.02 * gpq[1] + 0.02 * lm[1],
                bc[2] + 0.02 * gpq[2] + 0.02 * lm[2],
                0.333_333_34 * (bc[0] + bc[1] + bc[2]) * bc[3],
            ];
        }

        let mut basis = vec![[[0.0f32; 3]; 4]; n];
        let mut next_state = state.clone();

        for b in 0..CLIFFORDNET_BLOCKS {
            let gamma_b = self.uniforms.block_gammas[b];
            for y in 0..h {
                let ym = y.saturating_sub(1);
                let yp = (y + 1).min(h - 1);
                for x in 0..w {
                    let xm = x.saturating_sub(1);
                    let xp = (x + 1).min(w - 1);
                    let idx = y * w + x;
                    let hc = state[idx];

                    let n_up = state[ym * w + x];
                    let n_dn = state[yp * w + x];
                    let n_lt = state[y * w + xm];
                    let n_rt = state[y * w + xp];
                    let n_ul = state[ym * w + xm];
                    let n_ur = state[ym * w + xp];
                    let n_dl = state[yp * w + xm];
                    let n_dr = state[yp * w + xp];

                    let mut c_loc = [0.0f32; 4];
                    for c in 0..4 {
                        let card = 0.20 * (n_up[c] + n_dn[c] + n_lt[c] + n_rt[c]);
                        let diag = 0.05 * (n_ul[c] + n_ur[c] + n_dl[c] + n_dr[c]);
                        c_loc[c] = (card + diag) - hc[c];
                    }

                    if b == 0 {
                        basis[idx][0] = [-c_loc[0], -c_loc[1], -c_loc[2]];
                    } else if b == 1 {
                        basis[idx][1] = [-c_loc[0], -c_loc[1], -c_loc[2]];
                    }

                    let c_s1 = [c_loc[1], c_loc[2], c_loc[3], c_loc[0]];
                    let h_s1 = [hc[1], hc[2], hc[3], hc[0]];
                    let c_s2 = [c_loc[2], c_loc[3], c_loc[0], c_loc[1]];
                    let h_s2 = [hc[2], hc[3], hc[0], hc[1]];

                    let mut d_sum = [0.0f32; 4];
                    let mut w_sum = [0.0f32; 4];
                    for c in 0..4 {
                        let d1 = silu_f32(hc[c] * c_s1[c]);
                        let w1 = hc[c] * c_s1[c] - c_loc[c] * h_s1[c];
                        let d2 = silu_f32(hc[c] * c_s2[c]);
                        let w2 = hc[c] * c_s2[c] - c_loc[c] * h_s2[c];
                        let d16 = silu_f32(hc[c] * c_loc[c]);
                        d_sum[c] = 0.40 * d1 + 0.30 * d2 + 0.30 * d16;
                        w_sum[c] = 0.60 * w1 + 0.40 * w2;
                        let g_feat = 0.35 * d_sum[c] + 0.35 * w_sum[c] - 0.50 * c_loc[c];
                        next_state[idx][c] = hc[c] + gamma_b * g_feat;
                    }

                    for c in 0..3 {
                        basis[idx][2][c] += d_sum[c];
                        basis[idx][3][c] += w_sum[c];
                    }
                }
            }
            state.copy_from_slice(&next_state);
        }
        basis
    }

    /// Calibrate the 4 readout weights `[w_lap1, w_lap2, w_cliff_d, w_cliff_w]` from a pair of
    /// `(grid, gt_rgb)` frames via $4 \times 4$ regularized normal equations.
    pub fn calibrate_from_pair(&mut self, grid: &RolpaInputGrid, gt_rgb: &[[f32; 3]]) {
        let basis = self.extract_basis_features(grid);
        let mut ata = [[0.0f64; 4]; 4];
        let mut atb = [0.0f64; 4];

        for (i, feat) in basis.iter().enumerate() {
            let base = grid.base_c[i];
            let gt = gt_rgb[i];
            let res = [
                (gt[0] - base[0]) as f64,
                (gt[1] - base[1]) as f64,
                (gt[2] - base[2]) as f64,
            ];
            for m in 0..4 {
                let fm = [feat[m][0] as f64, feat[m][1] as f64, feat[m][2] as f64];
                atb[m] += fm[0] * res[0] + fm[1] * res[1] + fm[2] * res[2];
                for k in 0..4 {
                    let fk = [feat[k][0] as f64, feat[k][1] as f64, feat[k][2] as f64];
                    ata[m][k] += fm[0] * fk[0] + fm[1] * fk[1] + fm[2] * fk[2];
                }
            }
        }
        for m in 0..4 {
            ata[m][m] += 1e-2;
        }
        let w = solve_sym_4(ata, atb);
        self.uniforms.readout_weights = [w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32];
    }

    /// Execute the 1-pass `CliffordNet-Lite` forward refinement on CPU.
    pub fn refine_cpu(&self, grid: &RolpaInputGrid) -> Vec<[f32; 3]> {
        let n = grid.width * grid.height;
        if self.uniforms.bypass_mode != 0 {
            return grid
                .base_c
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect();
        }
        let basis = self.extract_basis_features(grid);
        let rw = self.uniforms.readout_weights;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bc = grid.base_c[i];
            let f = &basis[i];
            let mut rgb = [0.0f32; 3];
            for c in 0..3 {
                let delta = rw[0] * f[0][c] + rw[1] * f[1][c] + rw[2] * f[2][c] + rw[3] * f[3][c];
                rgb[c] = (bc[c] + delta).clamp(0.0, 1.0);
            }
            out.push(rgb);
        }
        out
    }
}

/// Simulate raw Gaussian-Gabor wavepacket splatting ($5 \times 5$ separable binomial Gaussian kernel)
/// prior to 1-pass `CliffordNet-Lite` boundary/contour refinement.
pub fn simulate_raw_wavepacket_splat(
    width: usize,
    height: usize,
    gt_rgb: &[[f32; 3]],
    passes: usize,
) -> Vec<[f32; 3]> {
    let k = [0.0625f32, 0.25, 0.375, 0.25, 0.0625];
    let mut curr = gt_rgb.to_vec();
    let mut temp = vec![[0.0f32; 3]; width * height];

    for _ in 0..passes.max(1) {
        // Horizontal pass
        for y in 0..height {
            for x in 0..width {
                let mut acc = [0.0f32; 3];
                for (oi, &kw) in k.iter().enumerate() {
                    let xx = (x as isize + (oi as isize) - 2).clamp(0, (width - 1) as isize) as usize;
                    let p = curr[y * width + xx];
                    acc[0] += kw * p[0];
                    acc[1] += kw * p[1];
                    acc[2] += kw * p[2];
                }
                temp[y * width + x] = acc;
            }
        }
        // Vertical pass
        for y in 0..height {
            for x in 0..width {
                let mut acc = [0.0f32; 3];
                for (oi, &kw) in k.iter().enumerate() {
                    let yy = (y as isize + (oi as isize) - 2).clamp(0, (height - 1) as isize) as usize;
                    let p = temp[yy * width + x];
                    acc[0] += kw * p[0];
                    acc[1] += kw * p[1];
                    acc[2] += kw * p[2];
                }
                curr[y * width + x] = acc;
            }
        }
    }
    curr
}

/// Compute Peak Signal-to-Noise Ratio (PSNR in dB) between two `[0, 1]` RGB images.
pub fn compute_psnr_rgb(pred: &[[f32; 3]], gt: &[[f32; 3]]) -> f64 {
    let n = pred.len().min(gt.len());
    if n == 0 {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for i in 0..n {
        for c in 0..3 {
            let d = (pred[i][c] - gt[i][c]) as f64;
            mse += d * d;
        }
    }
    mse /= (n * 3) as f64;
    if mse <= 1e-12 {
        99.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// Compute multi-scale Sobel-Gabor perceptual patch distance (LPIPS proxy) between two RGB images.
pub fn compute_lpips_rgb(
    pred: &[[f32; 3]],
    gt: &[[f32; 3]],
    width: usize,
    height: usize,
) -> f64 {
    if width < 3 || height < 3 {
        return 0.0;
    }
    let lum = |img: &[[f32; 3]], x: usize, y: usize| -> f64 {
        let p = img[y * width + x];
        (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) as f64
    };

    let mut diff_sum = 0.0f64;
    let mut count = 0usize;
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx_p = lum(pred, x + 1, y) - lum(pred, x - 1, y);
            let gy_p = lum(pred, x, y + 1) - lum(pred, x, y - 1);
            let lap_p = lum(pred, x + 1, y)
                + lum(pred, x - 1, y)
                + lum(pred, x, y + 1)
                + lum(pred, x, y - 1)
                - 4.0 * lum(pred, x, y);

            let gx_g = lum(gt, x + 1, y) - lum(gt, x - 1, y);
            let gy_g = lum(gt, x, y + 1) - lum(gt, x, y - 1);
            let lap_g = lum(gt, x + 1, y)
                + lum(gt, x - 1, y)
                + lum(gt, x, y + 1)
                + lum(gt, x, y - 1)
                - 4.0 * lum(gt, x, y);

            diff_sum += (gx_p - gx_g).abs() + (gy_p - gy_g).abs() + 0.5 * (lap_p - lap_g).abs();
            count += 1;
        }
    }
    diff_sum / (count.max(1) as f64)
}

/// Compute temporal warping error $E_{\text{warp}}$ across consecutive refined frames `(frame_t, frame_t1)`
/// on static/slowly moving non-occluded regions (where ground-truth inter-frame change is $< \tau_{\text{occ}}$).
pub fn compute_temporal_warping_error(
    refined_t: &[[f32; 3]],
    refined_t1: &[[f32; 3]],
    gt_t: &[[f32; 3]],
    gt_t1: &[[f32; 3]],
) -> f64 {
    let n = refined_t
        .len()
        .min(refined_t1.len())
        .min(gt_t.len())
        .min(gt_t1.len());
    if n == 0 {
        return 0.0;
    }
    let mut flicker_sum = 0.0f64;
    let mut valid = 0usize;

    for i in 0..n {
        let gt_diff = ((gt_t1[i][0] - gt_t[i][0]).abs()
            + (gt_t1[i][1] - gt_t[i][1]).abs()
            + (gt_t1[i][2] - gt_t[i][2]).abs())
            / 3.0;
        // Evaluate on static or slowly moving non-occluded pixels
        if gt_diff < 0.015 {
            for c in 0..3 {
                let d_ref = (refined_t1[i][c] - refined_t[i][c]) as f64;
                let d_gt = (gt_t1[i][c] - gt_t[i][c]) as f64;
                let excess = (d_ref - d_gt).abs();
                flicker_sum += excess * excess;
            }
            valid += 3;
        }
    }
    if valid == 0 {
        0.0
    } else {
        flicker_sum / (valid as f64)
    }
}

#[inline]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x.clamp(-12.0, 12.0)).exp())
}

fn solve_sym_4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for k in 0..4 {
        let diag = a[k][k];
        if diag.abs() < 1e-12 {
            continue;
        }
        for i in (k + 1)..4 {
            let f = a[i][k] / diag;
            b[i] -= f * b[k];
            for j in k..4 {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        let mut s = b[i];
        for j in (i + 1)..4 {
            s -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-12 {
            s / a[i][i]
        } else {
            0.0
        };
    }
    x
}
