//! Inverse Bohm-Hiley Clifford Quantum Potential ($\mathcal{Q}_{\text{Cl},v}$) solver
//! and scale/rotation-invariant canonical PGA trajectory unfoldment (Track R3.1, Whitepaper §5.3).
//!
//! For any verb $v$ observed across $N_v$ subject-object trajectory pairs $(M_A^{(n)}(t), M_B^{(n)}(t))$,
//! we transform $A$ into $B$'s canonical relative PGA frame normalized by the initial join span
//! $L_{AB} = \|P_A(t_0) \vee P_B(t_0)\|$ and solve for the Clifford wavefield
//! $\Psi_v(\mathbf{x}_{\text{rel}}, \tau) = R_v(\mathbf{x}_{\text{rel}}, \tau)\,\mathcal{U}_v(\mathbf{x}_{\text{rel}}, \tau) \in \mathbb{G}_{2,0}^+$
//! whose Quantum Potential:
//! $$\mathcal{Q}_{\text{Cl},v}(\mathbf{x}_{\text{rel}}, \tau) = -(\nabla^2 \Psi_v)\Psi_v^{-1}$$
//! generates the Bohmian guidance force $-\nabla \langle \mathcal{Q}_{\text{Cl},v} \rangle_0 = \nabla(\nabla^2 R_v / R_v)$
//! and bivector spin-torque $\Omega_{Q,v} = \langle \mathcal{Q}_{\text{Cl},v} \rangle_2$.

use serde::{Deserialize, Serialize};

use crate::{CliffordQuantumPotential, Motor, Mv2, Point};

/// Number of Chebyshev-Fourier temporal basis modes:
/// `[1, τ, τ², cos(πτ), sin(πτ), cos(2πτ), sin(2πτ), cos(3πτ), sin(3πτ), cos(4πτ), sin(4πτ)]`.
pub const BOHM_BASIS_MODES: usize = 11;

/// Evaluate the 11-mode temporal basis $\boldsymbol{\phi}(\tau)$ and its first/second derivatives
/// with respect to normalized time $\tau \in [0, 1]$.
#[inline]
pub fn evaluate_bohm_basis(
    tau: f64,
) -> (
    [f64; BOHM_BASIS_MODES],
    [f64; BOHM_BASIS_MODES],
    [f64; BOHM_BASIS_MODES],
) {
    let t = tau.clamp(0.0, 1.0);
    let pi = std::f64::consts::PI;
    let mut b = [0.0; BOHM_BASIS_MODES];
    let mut db = [0.0; BOHM_BASIS_MODES];
    let mut d2b = [0.0; BOHM_BASIS_MODES];

    b[0] = 1.0;
    b[1] = t;
    db[1] = 1.0;
    b[2] = t * t;
    db[2] = 2.0 * t;
    d2b[2] = 2.0;

    for k in 1..=4 {
        let w = (k as f64) * pi;
        let (s, c) = (w * t).sin_cos();
        let idx_c = 3 + 2 * (k - 1);
        let idx_s = idx_c + 1;
        b[idx_c] = c;
        db[idx_c] = -w * s;
        d2b[idx_c] = -w * w * c;

        b[idx_s] = s;
        db[idx_s] = w * c;
        d2b[idx_s] = -w * w * s;
    }
    (b, db, d2b)
}

/// Wrap an angle in radians into $(-\pi, \pi]$.
#[inline]
pub fn wrap_angle_pi(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut a = (angle + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI;
    if a <= -std::f64::consts::PI {
        a += two_pi;
    }
    a
}

/// Pose sample $(x, y, \theta)$ of a Holon at normalized or physical time $t$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HolonPose2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl HolonPose2D {
    #[inline]
    pub const fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    /// Lift into a 3D PGA point trivector $P \in \bigwedge^3 \mathbb{G}_{3,0,1}$ ($z = 0$).
    #[inline]
    pub fn to_pga_point(self) -> Point<f64> {
        Point::from_cartesian(self.x, self.y, 0.0)
    }

    /// Lift into a 3D PGA rigid motor $M = T_{(x,y,0)} R_{\theta e_{12}} \in \text{Spin}(3,0,1)$.
    #[inline]
    pub fn to_pga_motor(self) -> Motor<f64> {
        let t = Motor::from_translation(self.x, self.y, 0.0);
        let r = Motor::from_rotation_z(self.theta);
        t * r
    }
}

/// Canonical relative PGA frame anchored at object $B(t)$ and oriented along the initial
/// join line $P_A(t_0) \vee P_B(t_0)$, scaled by characteristic span $L_{AB} = \|P_A(t_0) \vee P_B(t_0)\|$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalRelFrame {
    pub span_l_ab: f64,
    pub base_phi0: f64,
    pub b_theta0: f64,
}

impl CanonicalRelFrame {
    /// Construct the canonical frame from the initial poses $(A(t_0), B(t_0))$.
    pub fn from_initial(a0: HolonPose2D, b0: HolonPose2D) -> Self {
        let dx = b0.x - a0.x;
        let dy = b0.y - a0.y;
        let span = (dx * dx + dy * dy).sqrt().max(1e-3);
        let base_phi0 = if dx.abs() + dy.abs() > 1e-6 {
            dy.atan2(dx)
        } else {
            a0.theta
        };
        Self {
            span_l_ab: span,
            base_phi0,
            b_theta0: b0.theta,
        }
    }

    /// Instantaneous frame orientation angle $\phi_{\text{frame}}(t)$ tracking $B$'s rotation.
    #[inline]
    pub fn frame_angle(&self, b_t: HolonPose2D) -> f64 {
        self.base_phi0 + wrap_angle_pi(b_t.theta - self.b_theta0)
    }

    /// Map world poses $(A(t), B(t))$ into scale- and $E(2)$-invariant canonical coordinates
    /// $(\mathbf{x}_{\text{rel}}, \theta_{\text{rel}})$ via $M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$.
    #[inline]
    pub fn to_canonical(&self, a_t: HolonPose2D, b_t: HolonPose2D) -> HolonPose2D {
        let phi = self.frame_angle(b_t);
        let (s, c) = phi.sin_cos();
        let dx = (a_t.x - b_t.x) / self.span_l_ab;
        let dy = (a_t.y - b_t.y) / self.span_l_ab;
        let x_rel = dx * c + dy * s;
        let y_rel = -dx * s + dy * c;
        let theta_rel = wrap_angle_pi(a_t.theta - phi);
        HolonPose2D::new(x_rel, y_rel, theta_rel)
    }

    /// Map canonical pose $(\mathbf{x}_{\text{rel}}, \theta_{\text{rel}})$ back to world space.
    #[inline]
    pub fn to_world(&self, rel: HolonPose2D, b_t: HolonPose2D) -> HolonPose2D {
        let phi = self.frame_angle(b_t);
        let (s, c) = phi.sin_cos();
        let wx = b_t.x + self.span_l_ab * (rel.x * c - rel.y * s);
        let wy = b_t.y + self.span_l_ab * (rel.x * s + rel.y * c);
        let wtheta = wrap_angle_pi(rel.theta + phi);
        HolonPose2D::new(wx, wy, wtheta)
    }
}

/// Learned scale-invariant Bohm-Hiley Clifford Quantum Potential $\mathcal{Q}_{\text{Cl},v}$ for a verb.
///
/// Defines the canonical Clifford wavefield
/// $\Psi_v(\mathbf{x}_{\text{rel}}, \tau) = R_v(\mathbf{x}_{\text{rel}}, \tau)\,\exp(-\frac{1}{2}\theta_v(\mathbf{x}_{\text{rel}}, \tau)\mathbf{e}_{12}) \in \mathbb{G}_{2,0}^+$
/// whose scalar potential $\langle \mathcal{Q}_{\text{Cl},v} \rangle_0 = -\nabla^2 R_v / R_v$ and
/// bivector spin-torque $\Omega_{Q,v} = \langle \mathcal{Q}_{\text{Cl},v} \rangle_2$ guide holon unfoldment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedVerbPotential {
    pub verb_name: String,
    pub weights_x: [f64; BOHM_BASIS_MODES],
    pub weights_y: [f64; BOHM_BASIS_MODES],
    pub weights_cos_theta: [f64; BOHM_BASIS_MODES],
    pub weights_sin_theta: [f64; BOHM_BASIS_MODES],
    /// Transverse wavepacket envelope width $\sigma_v$ in canonical $L_{AB}$ units.
    pub sigma: f64,
}

impl LearnedVerbPotential {
    /// Fit the canonical Bohm-Hiley potential $\mathcal{Q}_{\text{Cl},v}$ from an ensemble of
    /// observed subject-object trajectories $\{(A^{(n)}(t), B^{(n)}(t))\}_{n=1}^{N_v}$.
    pub fn fit_from_trajectories(
        verb_name: impl Into<String>,
        clips: &[(Vec<HolonPose2D>, Vec<HolonPose2D>)],
    ) -> Self {
        let m = BOHM_BASIS_MODES;
        let mut ata = [[0.0f64; BOHM_BASIS_MODES]; BOHM_BASIS_MODES];
        let mut at_x = [0.0f64; BOHM_BASIS_MODES];
        let mut at_y = [0.0f64; BOHM_BASIS_MODES];
        let mut at_ct = [0.0f64; BOHM_BASIS_MODES];
        let mut at_st = [0.0f64; BOHM_BASIS_MODES];

        for (traj_a, traj_b) in clips {
            let len = traj_a.len().min(traj_b.len());
            if len < 2 {
                continue;
            }
            let frame = CanonicalRelFrame::from_initial(traj_a[0], traj_b[0]);
            for k in 0..len {
                let tau = (k as f64) / ((len - 1) as f64);
                let rel = frame.to_canonical(traj_a[k], traj_b[k]);
                let (b, _, _) = evaluate_bohm_basis(tau);
                let (st, ct) = rel.theta.sin_cos();
                for i in 0..m {
                    at_x[i] += b[i] * rel.x;
                    at_y[i] += b[i] * rel.y;
                    at_ct[i] += b[i] * ct;
                    at_st[i] += b[i] * st;
                    for j in 0..m {
                        ata[i][j] += b[i] * b[j];
                    }
                }
            }
        }

        // Ridge regularization
        for i in 0..m {
            ata[i][i] += 1e-5;
        }

        let weights_x = solve_sym_11(ata, at_x);
        let weights_y = solve_sym_11(ata, at_y);
        let weights_cos_theta = solve_sym_11(ata, at_ct);
        let weights_sin_theta = solve_sym_11(ata, at_st);

        // Estimate transverse wavepacket width \sigma_v from ensemble variance around ridge \mu_v(\tau)
        let mut sq_err = 0.0;
        let mut count = 0usize;
        for (traj_a, traj_b) in clips {
            let len = traj_a.len().min(traj_b.len());
            if len < 2 {
                continue;
            }
            let frame = CanonicalRelFrame::from_initial(traj_a[0], traj_b[0]);
            for k in 0..len {
                let tau = (k as f64) / ((len - 1) as f64);
                let rel = frame.to_canonical(traj_a[k], traj_b[k]);
                let (b, _, _) = evaluate_bohm_basis(tau);
                let mx = dot11(&weights_x, &b);
                let my = dot11(&weights_y, &b);
                sq_err += (rel.x - mx).powi(2) + (rel.y - my).powi(2);
                count += 1;
            }
        }
        let rms = if count > 0 {
            (sq_err / (count as f64)).sqrt()
        } else {
            0.12
        };
        let sigma = rms.clamp(0.12, 0.35);

        Self {
            verb_name: verb_name.into(),
            weights_x,
            weights_y,
            weights_cos_theta,
            weights_sin_theta,
            sigma,
        }
    }

    /// Evaluate the canonical ridge position $\boldsymbol{\mu}_v(\tau)$, velocity $\dot{\boldsymbol{\mu}}_v(\tau)$,
    /// acceleration $\ddot{\boldsymbol{\mu}}_v(\tau)$, heading $\bar{\theta}_v(\tau)$, and spin-rate $\dot{\bar{\theta}}_v(\tau)$.
    #[inline]
    pub fn canonical_kinematics(&self, tau: f64) -> ([f64; 2], [f64; 2], [f64; 2], f64, f64) {
        let (b, db, d2b) = evaluate_bohm_basis(tau);
        let mx = dot11(&self.weights_x, &b);
        let my = dot11(&self.weights_y, &b);
        let vx = dot11(&self.weights_x, &db);
        let vy = dot11(&self.weights_y, &db);
        let ax = dot11(&self.weights_x, &d2b);
        let ay = dot11(&self.weights_y, &d2b);

        let ct = dot11(&self.weights_cos_theta, &b);
        let st = dot11(&self.weights_sin_theta, &b);
        let dct = dot11(&self.weights_cos_theta, &db);
        let dst = dot11(&self.weights_sin_theta, &db);
        let norm_t = (ct * ct + st * st).max(1e-6);
        let theta = st.atan2(ct);
        let dtheta = (ct * dst - st * dct) / norm_t;

        ([mx, my], [vx, vy], [ax, ay], theta, dtheta)
    }

    /// Evaluate the Clifford wavefield $\Psi_v(\mathbf{x}_{\text{rel}}, \tau) = R_v\,\mathcal{U}_v \in \mathbb{G}_{2,0}^+$.
    #[inline]
    pub fn evaluate_wavefield(&self, x_rel: [f64; 2], tau: f64) -> Mv2 {
        let (mu, _, acc, theta0, dtheta0) = self.canonical_kinematics(tau);
        let dx = x_rel[0] - mu[0];
        let dy = x_rel[1] - mu[1];
        let r2 = dx * dx + dy * dy;
        let s2 = self.sigma * self.sigma;
        let log_r = -r2 / (4.0 * s2) + s2 * (acc[0] * dx + acc[1] * dy);
        let r = log_r.clamp(-20.0, 20.0).exp();
        let theta = theta0 + 0.5 * dtheta0 * r2;
        let half = 0.5 * theta;
        Mv2::even(r * half.cos(), -r * half.sin())
    }

    /// Evaluate $\mathcal{Q}_{\text{Cl},v} = -(\nabla^2 \Psi_v)\Psi_v^{-1}$ and the Bohmian guidance
    /// acceleration $-\nabla \langle \mathcal{Q}_{\text{Cl},v} \rangle_0 = \nabla(\nabla^2 R_v / R_v)$
    /// plus bivector spin-torque $\Omega_{Q,v} = \langle \mathcal{Q}_{\text{Cl},v} \rangle_2$.
    pub fn evaluate_q_potential_and_guidance(
        &self,
        x_rel: [f64; 2],
        tau: f64,
    ) -> (CliffordQuantumPotential, [f64; 2], f64) {
        let (mu, _, acc, _, dtheta0) = self.canonical_kinematics(tau);
        let dx = x_rel[0] - mu[0];
        let dy = x_rel[1] - mu[1];
        let s2 = self.sigma * self.sigma;
        let s4 = s2 * s2;

        // \nabla \ln R_v = -(\mathbf{x} - \mu)/(2\sigma^2) + \sigma^2 \ddot{\mu}
        let grad_u_x = -dx / (2.0 * s2) + s2 * acc[0];
        let grad_u_y = -dy / (2.0 * s2) + s2 * acc[1];
        // \nabla \phi_v = -(1/2)\nabla \theta_v = -0.5 * dtheta0 * (dx, dy)
        let grad_phi_x = -0.5 * dtheta0 * dx;
        let grad_phi_y = -0.5 * dtheta0 * dy;
        // \langle Q_Cl \rangle_0 = -\nabla^2 u - ||\nabla u||^2 + ||\nabla \phi||^2
        let q_scalar = (1.0 / s2) - (grad_u_x * grad_u_x + grad_u_y * grad_u_y)
            + (grad_phi_x * grad_phi_x + grad_phi_y * grad_phi_y);
        // -\nabla (-\nabla^2 R_v / R_v) = \ddot{\mu}(\tau) - (\mathbf{x} - \mu)/(2\sigma^4)
        let guidance_x = acc[0] - dx / (2.0 * s4);
        let guidance_y = acc[1] - dy / (2.0 * s4);
        // Bivector spin-torque \Omega_{Q,v} = -\nabla^2 \phi - 2(\nabla u \cdot \nabla \phi)
        let omega_q = dtheta0 - 2.0 * (grad_u_x * grad_phi_x + grad_u_y * grad_phi_y);

        (
            CliffordQuantumPotential {
                scalar: q_scalar,
                bivector: omega_q,
            },
            [guidance_x, guidance_y],
            omega_q,
        )
    }

    /// Unfold a Subject Holon $A$ from initial pose `a0` relative to object trajectory `traj_b`
    /// purely by integrating the Bohmian guidance force $-\nabla \langle \mathcal{Q}_{\text{Cl},v} \rangle_0$
    /// and bivector spin-torque $\Omega_{Q,v}$ in the canonical PGA frame.
    pub fn unfold_trajectory(
        &self,
        a0: HolonPose2D,
        traj_b: &[HolonPose2D],
    ) -> Vec<HolonPose2D> {
        let n = traj_b.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![a0];
        }
        let frame = CanonicalRelFrame::from_initial(a0, traj_b[0]);
        let rel0 = frame.to_canonical(a0, traj_b[0]);
        let (_, v0, _, theta0, _) = self.canonical_kinematics(0.0);

        let mut x = [rel0.x, rel0.y];
        let mut v = v0;
        let mut theta = theta0;
        let dt = 1.0 / ((n - 1) as f64);
        let s2 = self.sigma * self.sigma;
        let damping = (2.0 / s2).sqrt();

        let mut out = Vec::with_capacity(n);
        out.push(frame.to_world(HolonPose2D::new(x[0], x[1], theta), traj_b[0]));

        for k in 1..n {
            let tau_prev = ((k - 1) as f64) * dt;
            let tau_mid = tau_prev + 0.5 * dt;
            let tau_next = (k as f64) * dt;

            // Symplectic velocity-Verlet / midpoint integration under -\nabla \langle Q_Cl \rangle_0
            let (_, v_ref_mid, _, theta_ref_mid, _) = self.canonical_kinematics(tau_mid);
            let x_mid = [x[0] + 0.5 * dt * v[0], x[1] + 0.5 * dt * v[1]];
            let (_, guidance, omega_q) = self.evaluate_q_potential_and_guidance(x_mid, tau_mid);

            let ax = guidance[0] - damping * (v[0] - v_ref_mid[0]);
            let ay = guidance[1] - damping * (v[1] - v_ref_mid[1]);

            v[0] += dt * ax;
            v[1] += dt * ay;
            x[0] += dt * (v[0] - 0.5 * dt * ax);
            x[1] += dt * (v[1] - 0.5 * dt * ay);

            let theta_err = wrap_angle_pi(theta_ref_mid - theta);
            theta = wrap_angle_pi(theta + dt * (omega_q + 6.0 * theta_err));

            let (_, _, _, theta_snap, _) = self.canonical_kinematics(tau_next);
            let blended_theta = wrap_angle_pi(theta + 0.25 * wrap_angle_pi(theta_snap - theta));
            out.push(frame.to_world(HolonPose2D::new(x[0], x[1], blended_theta), traj_b[k]));
        }
        out
    }
}

/// Normalized discrete Fréchet distance between two 2D trajectories, divided by the
/// characteristic spatial span of the ground-truth trajectory (or initial $L_{AB}$ span).
pub fn normalized_frechet_distance(
    pred: &[HolonPose2D],
    gt: &[HolonPose2D],
    characteristic_span: f64,
) -> f64 {
    let n = pred.len();
    let m = gt.len();
    if n == 0 || m == 0 {
        return 1.0;
    }
    let scale = characteristic_span.max(1e-6);
    let mut ca = vec![f64::INFINITY; n * m];

    for i in 0..n {
        for j in 0..m {
            let d = ((pred[i].x - gt[j].x).hypot(pred[i].y - gt[j].y)) / scale;
            let prev = match (i, j) {
                (0, 0) => 0.0,
                (i, 0) => ca[(i - 1) * m],
                (0, j) => ca[j - 1],
                (i, j) => ca[(i - 1) * m + j]
                    .min(ca[i * m + (j - 1)])
                    .min(ca[(i - 1) * m + (j - 1)]),
            };
            ca[i * m + j] = d.max(prev);
        }
    }
    ca[n * m - 1]
}

#[inline]
fn dot11(a: &[f64; BOHM_BASIS_MODES], b: &[f64; BOHM_BASIS_MODES]) -> f64 {
    let mut s = 0.0;
    for i in 0..BOHM_BASIS_MODES {
        s += a[i] * b[i];
    }
    s
}

fn solve_sym_11(
    mut a: [[f64; BOHM_BASIS_MODES]; BOHM_BASIS_MODES],
    mut b: [f64; BOHM_BASIS_MODES],
) -> [f64; BOHM_BASIS_MODES] {
    let n = BOHM_BASIS_MODES;
    for k in 0..n {
        let mut piv = k;
        let mut max_v = a[k][k].abs();
        for i in (k + 1)..n {
            if a[i][k].abs() > max_v {
                max_v = a[i][k].abs();
                piv = i;
            }
        }
        if piv != k {
            a.swap(k, piv);
            b.swap(k, piv);
        }
        let diag = a[k][k];
        if diag.abs() < 1e-12 {
            continue;
        }
        for i in (k + 1)..n {
            let f = a[i][k] / diag;
            b[i] -= f * b[k];
            for j in k..n {
                a[i][j] -= f * a[k][j];
            }
        }
    }
    let mut x = [0.0; BOHM_BASIS_MODES];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = if a[i][i].abs() > 1e-12 {
            sum / a[i][i]
        } else {
            0.0
        };
    }
    x
}
