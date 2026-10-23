//! Wording-space and visual feature projection into 16-channel $(\mathbb{G}_{2,0}^+)^{16}$ spinors
//! via thin symmetric SVD of the normalized Gram matrix.

use serde::{Deserialize, Serialize};

use crate::Mv2;

/// Number of even-spinor channels in `Spinor16` ($2 \times 16 = 32$ real degrees of freedom).
pub const SPINOR_CHANNELS: usize = 16;

/// A 16-channel even spinor $\psi \in (\mathbb{G}_{2,0}^+)^{16}$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Spinor16 {
    pub channels: [Mv2; SPINOR_CHANNELS],
}

impl Default for Spinor16 {
    fn default() -> Self {
        Self {
            channels: [Mv2::ZERO; SPINOR_CHANNELS],
        }
    }
}

impl Spinor16 {
    /// Construct a normalized `Spinor16` from 32 real spectral coordinates,
    /// pairing `(c[2k], c[2k+1])` into `c[2k] + c[2k+1] e12`.
    pub fn from_real32(coords: &[f64]) -> Self {
        let mut channels = [Mv2::ZERO; SPINOR_CHANNELS];
        let mut norm_sq = 0.0;
        for k in 0..SPINOR_CHANNELS {
            let re = coords.get(2 * k).copied().unwrap_or(0.0);
            let im = coords.get(2 * k + 1).copied().unwrap_or(0.0);
            channels[k] = Mv2::even(re, im);
            norm_sq += re * re + im * im;
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for ch in &mut channels {
                *ch = ch.scale(1.0 / norm);
            }
        }
        Self { channels }
    }

    /// Squared norm $\sum_k \langle \psi_k \widetilde{\psi}_k \rangle_0$.
    pub fn norm_sqr(&self) -> f64 {
        self.channels.iter().map(|c| c.even_norm_sqr()).sum()
    }

    /// Coordinate-free bilinear scalar inner product:
    /// $$\mathcal{S}(\psi_A, \psi_B) = \frac{\sum_k \langle \psi_{A,k} \widetilde{\psi}_{B,k} \rangle_0}{\|\psi_A\|\,\|\psi_B\|} \in [-1, 1]$$
    pub fn coherence(&self, other: &Self) -> f64 {
        let mut dot = 0.0;
        for k in 0..SPINOR_CHANNELS {
            // < (s_a + b_a e12)(s_b - b_b e12) >_0 = s_a s_b + b_a b_b
            let prod = self.channels[k].gp(other.channels[k].reverse());
            dot += prod.scalar_part();
        }
        let denom = (self.norm_sqr() * other.norm_sqr()).sqrt();
        if denom <= 1e-12 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    /// Apply a uniform phase rotor $e^{\theta e_{12}}$ across all channels.
    pub fn rotate_phase(&self, theta: f64) -> Self {
        let rotor = Mv2::even(theta.cos(), theta.sin());
        let mut out = *self;
        for ch in &mut out.channels {
            *ch = ch.gp(rotor);
        }
        out
    }
}

/// Project a batch of high-dimensional embedding vectors $\mathbf{e}_1, \dots, \mathbf{e}_N \in \mathbb{R}^D$
/// into $(\mathbb{G}_{2,0}^+)^{16}$ spinors via thin Jacobi SVD of their normalized Gram matrix
/// $G_{ij} = \langle \hat{\mathbf{e}}_i, \hat{\mathbf{e}}_j \rangle$.
pub fn project_gram_svd(embeddings: &[Vec<f64>]) -> Vec<Spinor16> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. L2-normalize input embeddings
    let normed: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|v| {
            let l2 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            v.iter().map(|x| x / l2).collect()
        })
        .collect();

    // 2. Build symmetric N x N Gram matrix G_{ij} = <e_i, e_j>
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let dot: f64 = normed[i].iter().zip(&normed[j]).map(|(x, y)| x * y).sum();
            a[i * n + j] = dot;
            a[j * n + i] = dot;
        }
    }

    // 3. Symmetric Jacobi eigendecomposition: G = V * diag(lambda) * V^T
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_sweeps = 60;
    for _sweep in 0..max_sweeps {
        let mut max_off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                max_off = max_off.max(a[p * n + q].abs());
            }
        }
        if max_off < 1e-12 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-13 {
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

    // 4. Sort eigenmodes in descending order of singular value lambda_k >= 0
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        a[j * n + j]
            .partial_cmp(&a[i * n + i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let rank = (2 * SPINOR_CHANNELS).min(n);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut coords = [0.0; 2 * SPINOR_CHANNELS];
        for (k, &col) in order.iter().take(rank).enumerate() {
            let sigma = a[col * n + col].max(0.0).sqrt();
            coords[k] = v[i * n + col] * sigma;
        }
        out.push(Spinor16::from_real32(&coords));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_svd_preserves_inner_products_in_spinor_space() {
        let e1 = vec![1.0, 0.8, 0.1, 0.0, -0.2, 0.5];
        let e2 = vec![0.95, 0.85, 0.15, 0.05, -0.18, 0.48]; // close synonym to e1
        let e3 = vec![-0.1, 0.05, 0.9, -0.8, 0.6, 0.1]; // orthogonal/unrelated
        let spinors = project_gram_svd(&[e1, e2, e3]);
        let sim_syn = spinors[0].coherence(&spinors[1]);
        let sim_orth = spinors[0].coherence(&spinors[2]).abs();
        assert!(sim_syn > 0.98, "expected high synonym coherence, got {sim_syn}");
        assert!(sim_orth < 0.15, "expected low orthogonal coherence, got {sim_orth}");
    }
}
