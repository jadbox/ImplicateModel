//! K-means part prototype learning, posture idempotent clustering, and Gaussian-Gabor kernel accumulation.

use super::{
    CFT_WAVEPACKET_MODES, CharacterObservation, ConstitutiveTuple, HolonModelSheet,
    ModelSheetPartKernel, SubHolonPartSpec, SubHolonPose, TurnaroundViewAspect,
    WavepacketPartKernel, classify_view_regime, fit_articulated_holon,
    clifford_features::{
        compute_cliffordnet_laplacian_features, decompose_masked_clifford_dual_product,
        decompose_observation_dual_product,
    },
};
use crate::enfold2d::Bucket;
use implicate_math::{CliffordDensityElement, Mv301, StateIdempotent};

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

/// Spectral Idempotent Factorization on the Temporal Density Covariance Operator:
/// Γ = ∫ Ψ(t) Ψ̃(t) dt = ∑ σ_k ε_k (Hiley Clifford Pregeometry, Track R10.3).
#[derive(Clone, Debug)]
pub struct SpectralIdempotentDecomposition {
    /// Number of canonical view aspects discovered algebraically (without hardcoding).
    pub k_aspects: usize,
    /// Primitive orthogonal state idempotents {ε_v} satisfying ε_v² = ε_v and ε_u ε_v = 0.
    pub idempotents: Vec<StateIdempotent>,
    /// Canonical heading rotors on Spin(2): [cos θ, sin θ].
    pub aspect_rotors: Vec<[f32; 2]>,
    /// Singular values (energies) σ_k of the temporal density covariance operator.
    pub singular_values: Vec<f64>,
    /// Per-observation canonical aspect assignment.
    pub observation_assignments: Vec<usize>,
    /// Continuous CFT phase within each aspect cell.
    pub observation_phases: Vec<f64>,
}

/// Extract normalized Clifford process state vector Ψ(t) for an observation.
pub fn extract_observation_state_vector(obs: &CharacterObservation, heading: f64) -> Vec<f64> {
    let mut vec = Vec::with_capacity(80);
    // 1. Heading rotor on Spin(2): [cos φ, sin φ]
    let (s_h, c_h) = heading.sin_cos();
    vec.push(c_h * 4.0);
    vec.push(s_h * 4.0);

    // 2. Line of Action 3D PGA features
    if let Some(loa) = obs.line_of_action() {
        let (s_t, c_t) = loa.angle_rad.sin_cos();
        vec.push(c_t * 2.5);
        vec.push(s_t * 2.5);
        vec.push(loa.spread_parallel / (loa.spread_parallel + loa.spread_perpendicular).max(1.0) * 2.0);
        vec.push(loa.spread_perpendicular / (loa.spread_parallel + loa.spread_perpendicular).max(1.0) * 2.0);
        vec.push((loa.curvature * 100.0).clamp(-5.0, 5.0) * 2.0);
    } else {
        vec.extend_from_slice(&[2.5, 0.0, 1.0, 1.0, 0.0]);
    }

    // 3. Clifford dual decomposition features
    let decomp = decompose_observation_dual_product(obs);
    let total_px = (obs.width() * obs.height()).max(1) as f64;
    let cel_px = decomp.cel_fill_mask.iter().filter(|&&m| m).count() as f64;
    let ink_px = decomp.ink_blade_mask.iter().filter(|&&m| m).count() as f64;
    vec.push(cel_px / total_px * 2.0);
    vec.push(ink_px / total_px * 2.0);

    // 4. Mean RGB of cel paint fill
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut cnt = 0.0f64;
    for (i, &is_cel) in decomp.cel_fill_mask.iter().enumerate() {
        if is_cel {
            r_sum += obs.rgb[i * 3] as f64 / 255.0;
            g_sum += obs.rgb[i * 3 + 1] as f64 / 255.0;
            b_sum += obs.rgb[i * 3 + 2] as f64 / 255.0;
            cnt += 1.0;
        }
    }
    if cnt > 0.0 {
        vec.push(r_sum / cnt);
        vec.push(g_sum / cnt);
        vec.push(b_sum / cnt);
    } else {
        vec.extend_from_slice(&[0.5, 0.5, 0.5]);
    }

    // 5. Low-frequency 2D Clifford Fourier modes
    let cft = compute_cft_posture_descriptor(obs);
    for &val in &cft[..16] {
        vec.push(val as f64 * 0.5);
    }

    // L2 normalize state vector
    let norm = vec.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    for x in &mut vec {
        *x /= norm;
    }
    vec
}

/// Compute physical headings along the trajectory using world coordinates when moving,
/// or Line of Action orientation when stationary.
pub fn compute_trajectory_headings(obs: &[CharacterObservation]) -> Vec<f64> {
    let n = obs.len();
    let mut headings = Vec::with_capacity(n);
    let mut last_h = 0.0f64;
    for i in 0..n {
        let (wdx, wdy) = if i + 1 < n {
            let c0 = obs[i].world_center();
            let c1 = obs[i + 1].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else if i > 0 {
            let c0 = obs[i - 1].world_center();
            let c1 = obs[i].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else {
            (0.0, 0.0)
        };

        let spd = wdx.hypot(wdy);
        if spd > 0.3 {
            if (wdx.abs() - wdy.abs()).abs() <= 0.25 && spd < 1.5 {
                // Universal kinematic abstraction: trajectory heading hysteresis at corner intersections.
                // When sub-pixel velocity components are balanced, kinematic momentum preserves the heading.
            } else if wdy.abs() > 1.2 || (wdy.abs() > wdx.abs() * 1.2) {
                last_h = if wdy > 0.0 { std::f64::consts::FRAC_PI_2 } else { -std::f64::consts::FRAC_PI_2 };
            } else if wdx.abs() > 0.35 {
                last_h = if wdx > 0.0 { 0.0 } else { std::f64::consts::PI };
            } else {
                last_h = wdy.atan2(wdx);
            }
        } else if let Some(loa) = obs[i].line_of_action() {
            last_h = loa.angle_rad;
        }
        headings.push(last_h);
    }
    headings
}

/// Symmetric Jacobi eigendecomposition: A = V * diag(λ) * V^T.
pub fn symmetric_jacobi_eigensolver(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..80 {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        if max_off < 1e-10 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-12 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                for r in 0..n {
                    if r != p && r != q {
                        let arp = a[r * n + p];
                        let arq = a[r * n + q];
                        a[r * n + p] = c * arp - s * arq;
                        a[p * n + r] = a[r * n + p];
                        a[r * n + q] = s * arp + c * arq;
                        a[q * n + r] = a[r * n + q];
                    }
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut eigvals = vec![0.0; n];
    for i in 0..n {
        eigvals[i] = a[i * n + i];
    }
    (eigvals, v)
}

/// Factor the temporal appearance-orientation covariance operator into primitive orthogonal idempotents:
/// Γ = ∫ Ψ(t) Ψ̃(t) dt = ∑ σ_k ε_k (Hiley Clifford Pregeometry, Track R10.3).
pub fn factor_temporal_covariance_idempotents(
    observations: &[CharacterObservation],
    forced_k: Option<usize>,
) -> SpectralIdempotentDecomposition {
    let n = observations.len();
    if n == 0 {
        return SpectralIdempotentDecomposition {
            k_aspects: 1,
            idempotents: vec![StateIdempotent::diagonal(0, 2, 0)],
            aspect_rotors: vec![[1.0, 0.0]],
            singular_values: vec![1.0],
            observation_assignments: Vec::new(),
            observation_phases: Vec::new(),
        };
    }

    let headings = compute_trajectory_headings(observations);
    let states: Vec<Vec<f64>> = observations
        .iter()
        .zip(&headings)
        .map(|(o, &h)| extract_observation_state_vector(o, h))
        .collect();

    let mut gram = vec![0.0f64; n * n];
    for i in 0..n {
        gram[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let dot: f64 = states[i].iter().zip(&states[j]).map(|(a, b)| a * b).sum();
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }

    let (eigvals, v) = symmetric_jacobi_eigensolver(gram, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        eigvals[j]
            .partial_cmp(&eigvals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_aspects = match forced_k {
        Some(k) => k.clamp(1, n),
        None => {
            let total_var: f64 = eigvals.iter().map(|&x| x.max(0.0)).sum();
            let mut cum = 0.0f64;
            let mut k = 1;
            for (idx, &col) in order.iter().enumerate() {
                cum += eigvals[col].max(0.0);
                if eigvals[col] > 0.05 * eigvals[order[0]] {
                    k = idx + 1;
                }
                if cum / total_var >= 0.85 && k >= 2 {
                    break;
                }
            }
            k.clamp(1, 8.min(n))
        }
    };

    let mut proj = vec![vec![0.0f64; k_aspects]; n];
    for i in 0..n {
        for (k, &col) in order.iter().take(k_aspects).enumerate() {
            let sigma = eigvals[col].max(0.0).sqrt();
            proj[i][k] = v[i * n + col] * sigma;
        }
        let norm = proj[i].iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        for x in &mut proj[i] {
            *x /= norm;
        }
    }

    let mut centroids = vec![proj[0].clone()];
    while centroids.len() < k_aspects {
        let next = proj
            .iter()
            .max_by(|a, b| {
                let da: f64 = centroids
                    .iter()
                    .map(|c| a.iter().zip(c).map(|(x, y)| (x - y).powi(2)).sum::<f64>())
                    .fold(f64::MAX, f64::min);
                let db: f64 = centroids
                    .iter()
                    .map(|c| b.iter().zip(c).map(|(x, y)| (x - y).powi(2)).sum::<f64>())
                    .fold(f64::MAX, f64::min);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .clone();
        centroids.push(next);
    }

    let mut assignments = vec![0usize; n];
    for _iter in 0..25 {
        for i in 0..n {
            let mut best_k = 0;
            let mut best_sim = -f64::MAX;
            for (k, c) in centroids.iter().enumerate() {
                let sim: f64 = proj[i].iter().zip(c).map(|(x, y)| x * y).sum();
                if sim > best_sim {
                    best_sim = sim;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }
        for k in 0..k_aspects {
            let mut sum = vec![0.0f64; k_aspects];
            let mut count = 0.0f64;
            for i in 0..n {
                if assignments[i] == k {
                    for d in 0..k_aspects {
                        sum[d] += proj[i][d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                for x in &mut sum {
                    *x /= norm;
                }
                centroids[k] = sum;
            }
        }
    }

    let mut aspect_rotors = Vec::with_capacity(k_aspects);
    let mut idempotents = Vec::with_capacity(k_aspects);
    for k in 0..k_aspects {
        let mut sum_c = 0.0f64;
        let mut sum_s = 0.0f64;
        let mut cnt = 0.0f64;
        for i in 0..n {
            if assignments[i] == k {
                let (s, c) = headings[i].sin_cos();
                sum_c += c;
                sum_s += s;
                cnt += 1.0;
            }
        }
        let (mean_c, mean_s) = if cnt > 0.0 && (sum_c * sum_c + sum_s * sum_s) > 1e-6 {
            let norm = (sum_c * sum_c + sum_s * sum_s).sqrt();
            (sum_c / norm, sum_s / norm)
        } else {
            (1.0, 0.0)
        };
        aspect_rotors.push([mean_c as f32, mean_s as f32]);
        idempotents.push(StateIdempotent::from_coeffs(k as u32, [mean_c, mean_s]));
    }

    let mut phases = vec![0.0f64; n];
    for ci in 0..k_aspects {
        let mut members: Vec<(usize, f64)> = (0..n)
            .filter(|&oi| assignments[oi] == ci)
            .map(|oi| {
                let score = if k_aspects > 1 {
                    proj[oi][(ci + 1) % k_aspects]
                } else {
                    oi as f64
                };
                (oi, score)
            })
            .collect();
        members.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let m_len = members.len().max(1);
        for (rank, &(oi, _)) in members.iter().enumerate() {
            phases[oi] = 2.0 * std::f64::consts::PI * (rank as f64) / (m_len as f64);
        }
    }

    let singular_values = order.iter().map(|&col| eigvals[col]).collect();
    SpectralIdempotentDecomposition {
        k_aspects,
        idempotents,
        aspect_rotors,
        singular_values,
        observation_assignments: assignments,
        observation_phases: phases,
    }
}

pub fn assign_view_idempotent_cells_and_cft_phases(
    observations: &[CharacterObservation],
    num_view_cells: usize,
) -> (Vec<usize>, Vec<f64>) {
    let decomp = factor_temporal_covariance_idempotents(observations, Some(num_view_cells));
    (decomp.observation_assignments, decomp.observation_phases)
}

/// Compute continuous volume-conserving scale factor s(φ) along gait phase φ ∈ S¹
/// such that det S = s(φ) · (1 / s(φ)) = 1 identically.
#[inline]
pub fn continuous_gait_volume_scale(phase_rad: f64, coeffs: [f64; 4]) -> (f64, f64) {
    let (s1, c1) = phase_rad.sin_cos();
    let (s2, c2) = (2.0 * phase_rad).sin_cos();
    let log_s = coeffs[0] * c1 + coeffs[1] * s1 + coeffs[2] * c2 + coeffs[3] * s2;
    let s = log_s.clamp(-0.8, 0.8).exp();
    (s, 1.0 / s)
}

/// Enforce strict volume conservation det S = 1 on continuous gait deformations along phase φ ∈ S¹.
///
/// For each canonical view-aspect cell and sub-holon part, calculates the canonical reference area
/// A_ref = exp(1/N ∑ ln(r_u · r_v)). Scales radii by k = √(A_ref / (r_u · r_v)) such that
/// det S = (r_u · k)(r_v · k) / A_ref = 1 identically across all phases φ ∈ S¹,
/// eliminating outline blur, squashing artifacts, and Gibbs ringing.
pub fn enforce_volume_conserving_gait_deformation(
    poses_per_obs: &mut [Vec<SubHolonPose>],
    view_assignments: &[usize],
    num_parts: usize,
    num_view_cells: usize,
) {
    let n_obs = poses_per_obs.len();
    if n_obs == 0 {
        return;
    }

    for vc in 0..num_view_cells {
        let members: Vec<usize> = (0..n_obs)
            .filter(|&i| view_assignments.get(i).copied().unwrap_or(0) == vc)
            .collect();
        if members.is_empty() {
            continue;
        }

        for p in 0..num_parts {
            // 1. Calculate geometric mean reference area A_ref
            let mut sum_log_area = 0.0f64;
            let mut count = 0.0f64;
            for &oi in &members {
                if let Some(pose) = poses_per_obs[oi].get(p) {
                    let a = (pose.radii[0] * pose.radii[1]).max(1.0);
                    sum_log_area += a.ln();
                    count += 1.0;
                }
            }
            if count == 0.0 {
                continue;
            }
            let a_ref = (sum_log_area / count).exp();

            // 2. Enforce strict volume conservation det S = 1
            for &oi in &members {
                if let Some(pose) = poses_per_obs[oi].get_mut(p) {
                    let curr_area = (pose.radii[0] * pose.radii[1]).max(1e-4);
                    let k = (a_ref / curr_area).sqrt();
                    pose.radii[0] = (pose.radii[0] * k).max(1.0);
                    pose.radii[1] = (pose.radii[1] * k).max(1.0);
                }
            }
        }
    }
}

/// Induce a unified Don Bluth Character Board Model Sheet (HolonModelSheet) for an autonomous process holon.
pub fn induce_holon_model_sheet(
    name: &str,
    observations: &[CharacterObservation],
    forced_parts: Option<usize>,
    forced_views: Option<usize>,
) -> HolonModelSheet {
    if observations.is_empty() {
        return HolonModelSheet {
            name: name.to_string(),
            constitutive: ConstitutiveTuple::default(),
            line_of_action: Mv301::ZERO,
            invariant_proportions: vec![1.0],
            turnaround_views: Vec::new(),
            parts: Vec::new(),
            cde_signature: CliffordDensityElement::new(1.0, 0.0, 0, 1, [255, 255, 255]),
        };
    }

    let decomp = factor_temporal_covariance_idempotents(observations, forced_views);
    let holon = fit_articulated_holon(observations, forced_parts);

    // Line of Action from central observation
    let mid_idx = observations.len() / 2;
    let loa = observations[mid_idx]
        .line_of_action()
        .or_else(|| observations[0].line_of_action())
        .unwrap_or(super::LineOfAction {
            spine_bivector: Mv301::ZERO,
            center_of_mass: [0.0, 0.0],
            direction: [1.0, 0.0],
            angle_rad: 0.0,
            spread_parallel: 1.0,
            spread_perpendicular: 1.0,
            curvature: 0.0,
            total_mass: 1.0,
        });

    let spread_ratio = loa.spread_parallel / loa.spread_perpendicular.max(0.1);
    let inv_props = vec![
        spread_ratio as f32,
        (observations[mid_idx].width() as f32) / (observations[mid_idx].height().max(1) as f32),
        loa.spread_parallel as f32,
        loa.spread_perpendicular as f32,
    ];

    // Articulated SubHolonPartSpec
    let mut parts = Vec::with_capacity(holon.num_parts);
    for p in 0..holon.num_parts {
        let part_name = if p == 0 {
            format!("{}_root", name)
        } else {
            format!("{}_part_{}", name, p)
        };
        let parent_part_id = if p == 0 { None } else { Some(holon.parent_indices[p]) };

        // Measure joint angle variance across observations
        let mut angles = Vec::new();
        let mut offsets_x = Vec::new();
        let mut offsets_y = Vec::new();
        for poses in &holon.poses_per_obs {
            if let Some(pose) = poses.get(p) {
                angles.push(pose.joint_angle_rad);
                offsets_x.push(pose.joint_offset_local[0]);
                offsets_y.push(pose.joint_offset_local[1]);
            }
        }
        let mean_angle = if !angles.is_empty() {
            angles.iter().sum::<f64>() / angles.len() as f64
        } else {
            0.0
        };
        let var_angle = if angles.len() > 1 {
            angles.iter().map(|a| (a - mean_angle).powi(2)).sum::<f64>() / (angles.len() - 1) as f64
        } else {
            0.0
        };
        let mean_ox = if !offsets_x.is_empty() {
            offsets_x.iter().sum::<f64>() / offsets_x.len() as f64
        } else {
            0.0
        };
        let mean_oy = if !offsets_y.is_empty() {
            offsets_y.iter().sum::<f64>() / offsets_y.len() as f64
        } else {
            0.0
        };

        let (flow_compliance, spring_stiffness, damping) = if p == 0 {
            (0.0f32, 100.0f32, 10.0f32)
        } else {
            let comp = (var_angle * 10.0).clamp(0.0, 10.0) as f32;
            let stiff = (1.0 / (var_angle + 0.05)).clamp(1.0, 100.0) as f32;
            let damp = 0.5 * stiff.sqrt();
            (comp, stiff, damp)
        };

        parts.push(SubHolonPartSpec {
            part_id: p,
            name: part_name,
            parent_part_id,
            flow_compliance,
            spring_stiffness,
            damping,
            anchor_local: [mean_ox as f32, mean_oy as f32],
            rest_angle: mean_angle as f32,
        });
    }

    // Turnaround views
    let mut turnaround_views = Vec::with_capacity(decomp.k_aspects);
    for v in 0..decomp.k_aspects {
        let view_regime = classify_view_regime(decomp.aspect_rotors[v], loa.angle_rad, spread_ratio);
        let idempotent = decomp.idempotents[v].clone();
        let reference_rotor = decomp.aspect_rotors[v];

        // Part poses for this aspect: average over observations assigned to aspect v
        let mut aspect_poses = Vec::with_capacity(holon.num_parts);
        for p in 0..holon.num_parts {
            let mut sum_center = [0.0f64; 2];
            let mut sum_radii = [0.0f64; 2];
            let mut sum_angle = 0.0f64;
            let mut cnt = 0.0f64;
            for (oi, poses) in holon.poses_per_obs.iter().enumerate() {
                if decomp.observation_assignments[oi] == v {
                    if let Some(pose) = poses.get(p) {
                        sum_center[0] += pose.center[0];
                        sum_center[1] += pose.center[1];
                        sum_radii[0] += pose.radii[0];
                        sum_radii[1] += pose.radii[1];
                        sum_angle += pose.angle_rad;
                        cnt += 1.0;
                    }
                }
            }
            if cnt > 0.0 {
                aspect_poses.push(SubHolonPose {
                    part_id: p,
                    parent_id: holon.parent_indices.get(p).copied().unwrap_or(0),
                    center: [sum_center[0] / cnt, sum_center[1] / cnt],
                    angle_rad: sum_angle / cnt,
                    radii: [sum_radii[0] / cnt, sum_radii[1] / cnt],
                    joint_angle_rad: parts[p].rest_angle as f64,
                    joint_offset_local: [parts[p].anchor_local[0] as f64, parts[p].anchor_local[1] as f64],
                    view_idempotent_idx: v,
                    phase_rad: 0.0,
                    heading_rad: (reference_rotor[1] as f64).atan2(reference_rotor[0] as f64),
                });
            } else if let Some(first_poses) = holon.poses_per_obs.first() {
                aspect_poses.push(first_poses[p]);
            }
        }

        // Part kernels
        let mut part_kernels = Vec::with_capacity(holon.num_parts);
        let kernel_cell_idx = v.min(holon.kernels.len().saturating_sub(1));
        for p in 0..holon.num_parts {
            if let Some(k) = holon.kernels.get(kernel_cell_idx).and_then(|vc| vc.get(p)) {
                part_kernels.push(ModelSheetPartKernel {
                    part_id: p,
                    canonical_res: [k.res_w, k.res_h],
                    cel_fill_rgb: k.dc_rgba.clone(),
                    ink_contour_blade: k.ink_contour_blade.clone(),
                    cft_deformation_coeffs: k.cft_rgba_coeffs.clone(),
                });
            } else {
                part_kernels.push(ModelSheetPartKernel {
                    part_id: p,
                    canonical_res: [16, 16],
                    cel_fill_rgb: vec![128; 16 * 16 * 4],
                    ink_contour_blade: vec![0.0; 16 * 16],
                    cft_deformation_coeffs: vec![[0.0; CFT_WAVEPACKET_MODES]; 16 * 16 * 4],
                });
            }
        }

        turnaround_views.push(TurnaroundViewAspect {
            view_index: v,
            view_regime,
            idempotent,
            reference_rotor,
            part_poses: aspect_poses,
            part_kernels,
        });
    }

    let cde_signature = holon
        .poses_per_obs
        .first()
        .and_then(|poses| poses.first())
        .map(|pose| pose.to_cde(1.0, holon.part_colors.first().copied().unwrap_or([255, 255, 255])))
        .unwrap_or_else(|| CliffordDensityElement::new(1.0, 0.0, 0, 1, [255, 255, 255]));

    let constitutive = ConstitutiveTuple {
        cohesion: 1.0,
        density: 1.0,
        bulk_modulus: 1.0,
        luminescence: [
            cde_signature.signature[0].s as f32,
            cde_signature.signature[1].s as f32,
            cde_signature.signature[2].s as f32,
        ],
        _pad: [0.0, 0.0],
    };

    HolonModelSheet {
        name: name.to_string(),
        constitutive,
        line_of_action: loa.spine_bivector,
        invariant_proportions: inv_props,
        turnaround_views,
        parts,
        cde_signature,
    }
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
            heading_rad: 0.0,
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
            heading_rad: 0.0,
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
        heading_rad: 0.0,
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
    contributes: &[bool],
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
        if pose.view_idempotent_idx != view_cell || !contributes.get(oi).copied().unwrap_or(true) {
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

    let n_obs_cell = cell_obs_indices.len();
    let min_consensus_hits: u32 = if n_obs_cell >= 2 {
        (n_obs_cell as f64 * 0.40).ceil() as u32
    } else {
        1
    };

    let mut dc_rgba = vec![0u8; n_cells * 4];
    let harmonic_rgba = vec![128u8; n_cells * 4];
    let mut cft_rgba_coeffs = vec![[0.0f32; CFT_WAVEPACKET_MODES]; n_cells * 4];
    let mut gaussian_envelope = vec![1.0f32; n_cells];
    let mut mode_rgb = vec![[0u8; 3]; n_cells];

    for k_idx in 0..n_cells {
        if fg_hits[k_idx] >= min_consensus_hits {
            if let Some((c, _)) = buckets[k_idx].mode() {
                mode_rgb[k_idx] = c;
                dc_rgba[k_idx * 4..k_idx * 4 + 3].copy_from_slice(&c);
            }
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

    let mut rgb_slice = vec![0u8; n_cells * 3];
    let mut mask_slice = vec![false; n_cells];
    for i in 0..n_cells {
        rgb_slice[i * 3] = mode_rgb[i][0];
        rgb_slice[i * 3 + 1] = mode_rgb[i][1];
        rgb_slice[i * 3 + 2] = mode_rgb[i][2];
        mask_slice[i] = fg_hits[i] >= min_consensus_hits;
    }
    let dual = decompose_masked_clifford_dual_product(&rgb_slice, Some(&mask_slice), res_w, res_h);

    for ky in 0..res_h {
        let v = (ky as f32 / (res_h as f32 - 1.0).max(1.0)) * 2.0 - 1.0;
        for kx in 0..res_w {
            let u = (kx as f32 / (res_w as f32 - 1.0).max(1.0)) * 2.0 - 1.0;
            let k_idx = ky * res_w + kx;
            let r2 = u * u + v * v;
            gaussian_envelope[k_idx] = (-0.08 * r2 * r2).exp();
            if fg_hits[k_idx] >= min_consensus_hits {
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
        ink_contour_blade: dual.ink_contour,
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
