//! 6-Block $\text{Pin}(3,0,1)$-Equivariant Geometric Algebra Transformer (`GATr`)
//! Multi-Body Verb Binder (Track R3.2, Whitepaper §3.2 Stratum 2 & §5.3).
//!
//! Implements the `GATr` relational architecture (Brehmer et al. 2023, `reference_cliffordgatr.pdf`)
//! operating over $K \le 32$ scene Holons with:
//! - `16` multivector channels in $\mathbb{G}_{3,0,1}$ (`Mv301`),
//! - `64` auxiliary scalar channels encoding `Spinor16` noun/archetype/emotion and verb spinors,
//! - Projective Join $\operatorname{EquiJoin}(X, Y; Z) = Z_{0123}(X^* \wedge Y^*)^*$ and
//!   geometric bilinear product $XY$,
//! - Distance-aware $E(3)$-invariant dot-product attention.

use implicate_math::{Mv301, Spinor16, mat3_vec};
use serde::{Deserialize, Serialize};

pub const GATR_BLOCKS: usize = 6;
pub const GATR_MV_CHANNELS: usize = 16;
pub const GATR_SCALAR_CHANNELS: usize = 64;
const BASIS_DIM: usize = 3;

/// Input scene token for a single Holon $i \in \{0..K-1\}$ in `GATr`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GatrHolonToken {
    /// World 3D position $\mathbf{p}_i \in \mathbb{R}^3$ (embedded as grade-3 trivector $P_i$).
    pub position: [f32; 3],
    /// Linear velocity $\mathbf{v}_i \in \mathbb{R}^3$ (embedded in grade-2 ideal bivector $B_i$).
    pub velocity: [f32; 3],
    /// Angular velocity $\boldsymbol{\omega}_i \in \mathbb{R}^3$ (embedded in grade-2 Euclidean bivector $B_i$).
    pub angular_velocity: [f32; 3],
    /// Dominant support/boundary plane normal $\hat{\mathbf{n}}_i \in \mathbb{R}^3$ (grade-1 plane $\Pi_i$).
    pub support_normal: [f32; 3],
    /// Signed plane offset $d_i$.
    pub support_dist: f32,
    /// Intrinsic noun/emotion/archetype spinor $\psi_{\text{noun},i} \in (\mathbb{G}_{2,0}^+)^{16}$.
    pub noun_spinor: Spinor16,
    /// Active verb spinor $\psi_{\text{verb},i} \in (\mathbb{G}_{2,0}^+)^{16}$.
    pub verb_spinor: Spinor16,
    /// Primary relational target Holon index $j \in \{0..K-1\}$.
    pub target_idx: usize,
    /// Normalized action phase $\tau \in [0, 1]$.
    pub phase_tau: f32,
}

impl GatrHolonToken {
    /// Apply an arbitrary 3D rigid transformation $(R \in SO(3), \mathbf{t} \in \mathbb{R}^3) \in E(3)$
    /// to this Holon token.
    pub fn transform_e3(&self, rot: [[f32; 3]; 3], trans: [f32; 3]) -> Self {
        let rp = mat3_vec(rot, self.position);
        let pos = [rp[0] + trans[0], rp[1] + trans[1], rp[2] + trans[2]];
        let vel = mat3_vec(rot, self.velocity);
        let ang = mat3_vec(rot, self.angular_velocity);
        let norm = mat3_vec(rot, self.support_normal);
        Self {
            position: pos,
            velocity: vel,
            angular_velocity: ang,
            support_normal: norm,
            support_dist: self.support_dist,
            noun_spinor: self.noun_spinor,
            verb_spinor: self.verb_spinor,
            target_idx: self.target_idx,
            phase_tau: self.phase_tau,
        }
    }
}

/// Output predicted by `GATr` for each scene Holon $i$.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GatrHolonOutput {
    /// Predicted relative linear motor impulse $\Delta \mathbf{v}_i \in \mathbb{R}^3$ (grade-2 ideal bivector).
    pub delta_v: [f32; 3],
    /// Predicted relative rotational motor impulse $\Delta \boldsymbol{\omega}_i \in \mathbb{R}^3$ (grade-2 Euclidean bivector).
    pub delta_omega: [f32; 3],
    /// Predicted $\mathcal{Q}_{\text{Cl}}$ beacon parameters `(k_r, m_ang, depth_kappa, phase_offset)`.
    pub q_beacon_params: [f32; 4],
}

/// Learned equivariant readout weights for a specific verb in `GATr`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerbReadoutRule {
    pub verb_spinor: Spinor16,
    /// Scalar weights combining the 6 $E(3)$-equivariant vector channels of the final `GATr` block.
    pub linear_weights: [f32; BASIS_DIM],
    pub angular_weights: [f32; BASIS_DIM],
    pub beacon_params: [f32; 4],
}

/// 6-Block $\text{Pin}(3,0,1)$-Equivariant `GATr` Relational Operator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatrVerbBinder {
    pub block_scales: [f32; GATR_BLOCKS],
    pub grade_mix: [[f32; 5]; GATR_BLOCKS],
    pub rules: Vec<VerbReadoutRule>,
}

impl Default for GatrVerbBinder {
    fn default() -> Self {
        Self::new()
    }
}

impl GatrVerbBinder {
    pub fn new() -> Self {
        let block_scales = [0.22, 0.20, 0.18, 0.16, 0.14, 0.12];
        let grade_mix = [
            [1.0, 0.9, 1.0, 1.0, 0.8],
            [0.95, 1.0, 1.05, 1.0, 0.85],
            [1.0, 0.92, 0.98, 1.0, 0.9],
            [1.02, 0.95, 1.0, 1.0, 0.88],
            [0.98, 1.0, 1.02, 1.0, 0.92],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ];
        Self {
            block_scales,
            grade_mix,
            rules: Vec::new(),
        }
    }

    /// Run the 6-block $\text{Pin}(3,0,1)$-equivariant `GATr` backbone on $K$ scene Holons,
    /// returning the final 16 multivector channels, 64 scalar channels, and 6 equivariant 3D basis vectors per holon.
    pub fn forward_backbone(
        &self,
        tokens: &[GatrHolonToken],
    ) -> (
        Vec<[Mv301; GATR_MV_CHANNELS]>,
        Vec<[f32; GATR_SCALAR_CHANNELS]>,
        Vec<[[f32; 3]; BASIS_DIM]>,
    ) {
        let k_len = tokens.len();
        let mut mv = vec![[Mv301::ZERO; GATR_MV_CHANNELS]; k_len];
        let mut sc = vec![[0.0f32; GATR_SCALAR_CHANNELS]; k_len];
        let mut eq_basis = vec![[[0.0f32; 3]; BASIS_DIM]; k_len];

        if k_len == 0 {
            return (mv, sc, eq_basis);
        }

        // Stem: populate 64 scalar channels from Spinor16 noun + Spinor16 verb,
        // and 16 multivector channels from Point P_i, Motor B_i, Plane \Pi_i, and Projective Join P_j \vee P_i.
        for (i, tok) in tokens.iter().enumerate() {
            for c in 0..16 {
                sc[i][2 * c] = tok.noun_spinor.channels[c].s as f32;
                sc[i][2 * c + 1] = tok.noun_spinor.channels[c].e12 as f32;
                sc[i][32 + 2 * c] = tok.verb_spinor.channels[c].s as f32;
                sc[i][32 + 2 * c + 1] = tok.verb_spinor.channels[c].e12 as f32;
            }

            let tgt = &tokens[tok.target_idx.min(k_len - 1)];
            let p_i = Mv301::from_point(tok.position);
            let p_j = Mv301::from_point(tgt.position);
            let b_i = Mv301::from_motor_bivector(tok.velocity, tok.angular_velocity);
            let b_j = Mv301::from_motor_bivector(tgt.velocity, tgt.angular_velocity);
            let pi_i = Mv301::from_plane(tok.support_normal, 0.0);

            // Equivariant Projective Join: extracts translation-invariant displacement (p_j - p_i)
            let join_ij = p_j.equi_join(&p_i, 1.0);
            let r_vec = join_ij.ideal_bivector();
            let u_rel = unit3(r_vec);
            // Gram-Schmidt orthogonalized support normal and tangent bivector dual
            let n_raw = unit3(tok.support_normal);
            let dot_nr = n_raw[0] * u_rel[0] + n_raw[1] * u_rel[1] + n_raw[2] * u_rel[2];
            let n_up = unit3([
                n_raw[0] - dot_nr * u_rel[0],
                n_raw[1] - dot_nr * u_rel[1],
                n_raw[2] - dot_nr * u_rel[2],
            ]);
            let u_tang = cross3(n_up, u_rel);
            let v_rel = [
                tok.velocity[0] - tgt.velocity[0],
                tok.velocity[1] - tgt.velocity[1],
                tok.velocity[2] - tgt.velocity[2],
            ];

            eq_basis[i] = [u_rel, n_up, u_tang];

            mv[i][0] = p_i;
            mv[i][1] = b_i;
            mv[i][2] = pi_i;
            mv[i][3] = Mv301::from_motor_bivector(u_rel, u_tang);
            mv[i][4] = b_j;
            mv[i][5] = Mv301::from_motor_bivector(n_up, u_tang);
            mv[i][6] = Mv301::from_motor_bivector(v_rel, tok.angular_velocity);
            mv[i][7] = join_ij;
            for c in 8..GATR_MV_CHANNELS {
                let g = sc[i][c * 2].tanh();
                mv[i][c] = mv[i][c - 7].scale(g);
            }
        }

        // 6 Pin(3,0,1)-equivariant GATr transformer blocks
        let mut attn_mv = [Mv301::ZERO; GATR_MV_CHANNELS];
        for b in 0..GATR_BLOCKS {
            let b_scale = self.block_scales[b];
            let g_mix = self.grade_mix[b];

            for i in 0..k_len {
                // 1. E(3)-invariant RMS LayerNorm over MV channels
                let mut norm_sq = 1e-5f32;
                for c in 0..GATR_MV_CHANNELS {
                    norm_sq += mv[i][c].invariant_norm_sq();
                }
                let inv_rms = (GATR_MV_CHANNELS as f32 / norm_sq).sqrt().clamp(0.1, 10.0);

                // 2. Distance-aware E(3)-invariant Attention over active neighbor set
                let tgt = tokens[i].target_idx.min(k_len - 1);
                let pi_coords = mv[i][0].point_coords();
                let pj_coords = mv[tgt][0].point_coords();
                let dx = pj_coords[0] - pi_coords[0];
                let dy = pj_coords[1] - pi_coords[1];
                let dz = pj_coords[2] - pi_coords[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                let mut dot_sc = 0.0f32;
                for s_idx in 0..16 {
                    dot_sc += sc[i][s_idx] * sc[tgt][s_idx + 16];
                }
                let attn_gate = sigmoid(0.25 * dot_sc - 0.05 * dist_sq);

                for c in 0..GATR_MV_CHANNELS {
                    let normed = mv[i][c].scale(inv_rms).scale_grades(g_mix);
                    let partner = mv[tgt][c].scale(attn_gate * 0.25);
                    attn_mv[c] = normed.add(partner);
                }

                // 3. Geometric Bilinear + Scalar-Gated Residual (GGR)
                for c in 0..GATR_MV_CHANNELS {
                    let next_c = (c + 1) & (GATR_MV_CHANNELS - 1);
                    let geo = attn_mv[c].geometric_bilinear(&attn_mv[next_c]);
                    let gate = silu(sc[i][c * 4]);
                    let update = attn_mv[c].scale(gate).add(geo.scale(0.15));
                    mv[i][c] = mv[i][c].add(update.scale(b_scale));
                    sc[i][c] += b_scale * 0.05 * mv[i][c].c[0];
                }
            }
        }

        // Re-orthonormalize the equivariant readout triad from the 6-block refined ideal bivectors
        for i in 0..k_len {
            let m3 = unit3(mv[i][3].ideal_bivector());
            let m5 = unit3(mv[i][5].ideal_bivector());
            let u0 = unit3([
                0.9 * eq_basis[i][0][0] + 0.1 * m3[0],
                0.9 * eq_basis[i][0][1] + 0.1 * m3[1],
                0.9 * eq_basis[i][0][2] + 0.1 * m3[2],
            ]);
            let n_cand = [
                0.9 * eq_basis[i][1][0] + 0.1 * m5[0],
                0.9 * eq_basis[i][1][1] + 0.1 * m5[1],
                0.9 * eq_basis[i][1][2] + 0.1 * m5[2],
            ];
            let dot = n_cand[0] * u0[0] + n_cand[1] * u0[1] + n_cand[2] * u0[2];
            let u1 = unit3([
                n_cand[0] - dot * u0[0],
                n_cand[1] - dot * u0[1],
                n_cand[2] - dot * u0[2],
            ]);
            let u2 = cross3(u1, u0);
            eq_basis[i] = [u0, u1, u2];
        }

        (mv, sc, eq_basis)
    }

    /// Execute the full 6-block `GATr` forward pass for $K$ scene Holons, predicting
    /// $\Delta B_i = (\Delta \mathbf{v}_i, \Delta \boldsymbol{\omega}_i)$ and $\mathcal{Q}_{\text{Cl}}$ beacon parameters.
    pub fn forward(&self, tokens: &[GatrHolonToken]) -> Vec<GatrHolonOutput> {
        let (_, _, eq_basis) = self.forward_backbone(tokens);
        let mut out = Vec::with_capacity(tokens.len());

        for (i, tok) in tokens.iter().enumerate() {
            let basis = &eq_basis[i];
            let mut best_w_lin = [0.5f32, 0.2, 0.1];
            let mut best_w_ang = [0.0f32; BASIS_DIM];
            let mut best_beacon = [1.0f32, 1.0, 1.0, 0.0];
            let mut best_sim = -1.0f64;

            for rule in &self.rules {
                let sim = tok.verb_spinor.coherence(&rule.verb_spinor);
                if sim > best_sim {
                    best_sim = sim;
                    best_w_lin = rule.linear_weights;
                    best_w_ang = rule.angular_weights;
                    best_beacon = rule.beacon_params;
                }
            }

            let mut dv = [0.0f32; 3];
            let mut dw = [0.0f32; 3];
            for m in 0..BASIS_DIM {
                for d in 0..3 {
                    dv[d] += best_w_lin[m] * basis[m][d];
                    dw[d] += best_w_ang[m] * basis[m][d];
                }
            }
            out.push(GatrHolonOutput {
                delta_v: dv,
                delta_omega: dw,
                q_beacon_params: best_beacon,
            });
        }
        out
    }

    /// Train/fit the equivariant readout weights for a verb from $N_{\text{train}} \in \{1, 5, 20\}$
    /// scene examples via closed-form equivariant normal equations:
    /// $\min_{\mathbf{w}} \sum_{n} \|\sum_m w_m \mathbf{F}_m^{(n)} - \Delta \mathbf{v}_{\text{gt}}^{(n)}\|^2 + \lambda \|\mathbf{w}\|^2$.
    pub fn fit_verb_from_examples(
        &mut self,
        verb_spinor: Spinor16,
        examples: &[(Vec<GatrHolonToken>, Vec<GatrHolonOutput>)],
    ) {
        let mut ata = [[0.0f64; BASIS_DIM]; BASIS_DIM];
        let mut at_v = [0.0f64; BASIS_DIM];
        let mut at_w = [0.0f64; BASIS_DIM];
        let mut beacon_acc = [0.0f32; 4];
        let mut count = 0usize;

        for (tokens, targets) in examples {
            let (_, _, eq_basis) = self.forward_backbone(tokens);
            // Fit on the primary subject holon (token 0) of each labeled verb clip
            if let (Some(basis), Some(tgt)) = (eq_basis.first(), targets.first()) {
                for i in 0..BASIS_DIM {
                    let dot_vi = (basis[i][0] * tgt.delta_v[0]
                        + basis[i][1] * tgt.delta_v[1]
                        + basis[i][2] * tgt.delta_v[2]) as f64;
                    let dot_wi = (basis[i][0] * tgt.delta_omega[0]
                        + basis[i][1] * tgt.delta_omega[1]
                        + basis[i][2] * tgt.delta_omega[2]) as f64;
                    at_v[i] += dot_vi;
                    at_w[i] += dot_wi;
                    for j in 0..BASIS_DIM {
                        let dot_ij = (basis[i][0] * basis[j][0]
                            + basis[i][1] * basis[j][1]
                            + basis[i][2] * basis[j][2]) as f64;
                        ata[i][j] += dot_ij;
                    }
                }
                for b_idx in 0..4 {
                    beacon_acc[b_idx] += tgt.q_beacon_params[b_idx];
                }
                count += 1;
            }
        }

        for i in 0..BASIS_DIM {
            ata[i][i] += 1e-4;
        }
        let w_lin_f64 = solve_sym_6(ata, at_v);
        let w_ang_f64 = solve_sym_6(ata, at_w);
        let mut linear_weights = [0.0f32; BASIS_DIM];
        let mut angular_weights = [0.0f32; BASIS_DIM];
        for i in 0..BASIS_DIM {
            linear_weights[i] = w_lin_f64[i] as f32;
            angular_weights[i] = w_ang_f64[i] as f32;
        }
        let inv_c = 1.0 / (count.max(1) as f32);
        let beacon_params = [
            beacon_acc[0] * inv_c,
            beacon_acc[1] * inv_c,
            beacon_acc[2] * inv_c,
            beacon_acc[3] * inv_c,
        ];

        self.rules.push(VerbReadoutRule {
            verb_spinor,
            linear_weights,
            angular_weights,
            beacon_params,
        });
    }
}

/// Non-equivariant standard coordinate MLP/linear baseline operating directly on raw world coordinates
/// $(\mathbf{p}_i, \mathbf{p}_j, \mathbf{v}_i, \hat{\mathbf{n}}_i, 1) \in \mathbb{R}^{13}$ without
/// `GATr`'s $\mathbb{G}_{3,0,1}$ projective join or $E(3)$-equivariant basis.
#[derive(Clone, Debug, Default)]
pub struct NonEquivariantMlpBaseline {
    pub rules: Vec<(Spinor16, [[f64; 13]; 3])>,
}

impl NonEquivariantMlpBaseline {
    pub fn fit_verb_from_examples(
        &mut self,
        verb_spinor: Spinor16,
        examples: &[(Vec<GatrHolonToken>, Vec<GatrHolonOutput>)],
    ) {
        let mut ata = [[0.0f64; 13]; 13];
        let mut at_y = [[0.0f64; 13]; 3];

        for (tokens, targets) in examples {
            let k_len = tokens.len();
            for (tok, tgt) in tokens.iter().zip(targets.iter()) {
                let partner = &tokens[tok.target_idx.min(k_len - 1)];
                let feat = Self::raw_features(tok, partner);
                for i in 0..13 {
                    for d in 0..3 {
                        at_y[d][i] += feat[i] * (tgt.delta_v[d] as f64);
                    }
                    for j in 0..13 {
                        ata[i][j] += feat[i] * feat[j];
                    }
                }
            }
        }
        for i in 0..13 {
            ata[i][i] += 1e-2;
        }
        let wx = solve_sym_13(ata, at_y[0]);
        let wy = solve_sym_13(ata, at_y[1]);
        let wz = solve_sym_13(ata, at_y[2]);
        self.rules.push((verb_spinor, [wx, wy, wz]));
    }

    pub fn forward(&self, tokens: &[GatrHolonToken]) -> Vec<GatrHolonOutput> {
        let k_len = tokens.len();
        let mut out = Vec::with_capacity(k_len);
        for tok in tokens {
            let partner = &tokens[tok.target_idx.min(k_len - 1)];
            let feat = Self::raw_features(tok, partner);
            let mut best_w = [[0.0f64; 13]; 3];
            let mut best_sim = -1.0f64;
            for (sp, w) in &self.rules {
                let sim = tok.verb_spinor.coherence(sp);
                if sim > best_sim {
                    best_sim = sim;
                    best_w = *w;
                }
            }
            let mut dv = [0.0f32; 3];
            for d in 0..3 {
                let mut s = 0.0;
                for i in 0..13 {
                    s += best_w[d][i] * feat[i];
                }
                dv[d] = s as f32;
            }
            out.push(GatrHolonOutput {
                delta_v: dv,
                delta_omega: [0.0; 3],
                q_beacon_params: [1.0, 1.0, 1.0, 0.0],
            });
        }
        out
    }

    #[inline]
    fn raw_features(tok: &GatrHolonToken, partner: &GatrHolonToken) -> [f64; 13] {
        [
            tok.position[0] as f64,
            tok.position[1] as f64,
            tok.position[2] as f64,
            partner.position[0] as f64,
            partner.position[1] as f64,
            partner.position[2] as f64,
            tok.velocity[0] as f64,
            tok.velocity[1] as f64,
            tok.velocity[2] as f64,
            tok.support_normal[0] as f64,
            tok.support_normal[1] as f64,
            tok.support_normal[2] as f64,
            1.0,
        ]
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-15.0, 15.0)).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn unit3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / n, v[1] / n, v[2] / n]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn solve_sym_6(mut a: [[f64; BASIS_DIM]; BASIS_DIM], mut b: [f64; BASIS_DIM]) -> [f64; BASIS_DIM] {
    for k in 0..BASIS_DIM {
        let diag = a[k][k];
        if diag.abs() < 1e-12 {
            continue;
        }
        for i in (k + 1)..BASIS_DIM {
            let f = a[i][k] / diag;
            b[i] -= f * b[k];
            for j in k..BASIS_DIM {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    let mut x = [0.0; BASIS_DIM];
    for i in (0..BASIS_DIM).rev() {
        let mut s = b[i];
        for j in (i + 1)..BASIS_DIM {
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

fn solve_sym_13(mut a: [[f64; 13]; 13], mut b: [f64; 13]) -> [f64; 13] {
    for k in 0..13 {
        let diag = a[k][k];
        if diag.abs() < 1e-12 {
            continue;
        }
        for i in (k + 1)..13 {
            let f = a[i][k] / diag;
            b[i] -= f * b[k];
            for j in k..13 {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    let mut x = [0.0; 13];
    for i in (0..13).rev() {
        let mut s = b[i];
        for j in (i + 1)..13 {
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
