//! Core Clifford algebra kernel for `ImplicateModel`.
//!
//! Unifies:
//! 1. $\mathbb{G}_{2,0}$ multivectors (`Mv2`) and 16-channel $(\mathbb{G}_{2,0}^+)^{16}$ spinors (`Spinor16`)
//!    with thin Gram-SVD projection (`spinor_svd`).
//! 2. 3D Projective Geometric Algebra $\mathbb{G}_{3,0,1}$ (`Point`, `Line`, `Plane`, `Motor`, `pga`).
//! 3. Basil Hiley's `CliffordDensityElement`, primitive state idempotents (`StateIdempotent`),
//!    and the scale-invariant Clifford Quantum Potential $\mathcal{Q}_{\text{Cl}} = -(\nabla^2 \Psi)\Psi^{-1}$.

use std::ops::{Add, Mul, Neg, Sub};

use clifford::specialized::projective::dim3;
use serde::{Deserialize, Serialize};

pub use dim3::{Line, Motor, Plane, Point};

pub mod g301;
pub mod holon;
pub mod idempotent;
pub mod pga;
pub mod q_potential;
pub mod spinor_svd;

pub use g301::{Mv301, mat3_vec};
pub use holon::CliffordDensityElement;
pub use idempotent::{StateBasis, StateIdempotent, TransitionOperator};
pub use pga::{
    Camera2D, OrientedLine2, RayHit, SegmentHit, integrate_2d_camera, integrate_3d_ego,
    plane_through_point, ray_hit_plane, ray_hit_plane_pga, segment_hit_oriented_line,
    snap_manhattan_normal, snap_manhattan_plane,
};
pub use q_potential::{
    BOHM_BASIS_MODES, CanonicalRelFrame, HolonPose2D, LearnedVerbPotential, evaluate_bohm_basis,
    normalized_frechet_distance, wrap_angle_pi,
};
pub use spinor_svd::{SPINOR_CHANNELS, Spinor16, project_gram_svd};

/// Multivector of `G(2,0)`.
///
/// Basis: `1, e1, e2, e12`, with `e1² = e2² = 1` and `e12² = -1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Mv2 {
    pub s: f64,
    pub e1: f64,
    pub e2: f64,
    pub e12: f64,
}

impl Mv2 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    pub const E1: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    pub const E2: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    pub const E12: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    pub const fn new(s: f64, e1: f64, e2: f64, e12: f64) -> Self {
        Self { s, e1, e2, e12 }
    }

    pub const fn scalar(s: f64) -> Self {
        Self::new(s, 0.0, 0.0, 0.0)
    }

    /// Map a complex number to the even subalgebra: `re + im e12`.
    pub const fn from_complex(re: f64, im: f64) -> Self {
        Self::new(re, 0.0, 0.0, im)
    }

    /// Even multivector `s + b e12`.
    pub const fn even(s: f64, b: f64) -> Self {
        Self::new(s, 0.0, 0.0, b)
    }

    pub const fn complex(self) -> (f64, f64) {
        (self.s, self.e12)
    }

    pub fn is_even(self) -> bool {
        self.e1 == 0.0 && self.e2 == 0.0
    }

    pub fn plus(self, rhs: Self) -> Self {
        Self::new(
            self.s + rhs.s,
            self.e1 + rhs.e1,
            self.e2 + rhs.e2,
            self.e12 + rhs.e12,
        )
    }

    pub fn minus(self, rhs: Self) -> Self {
        Self::new(
            self.s - rhs.s,
            self.e1 - rhs.e1,
            self.e2 - rhs.e2,
            self.e12 - rhs.e12,
        )
    }

    pub fn scale(self, k: f64) -> Self {
        Self::new(self.s * k, self.e1 * k, self.e2 * k, self.e12 * k)
    }

    /// Geometric product. Signs follow `e1 e2 = e12` and `e2 e1 = -e12`.
    pub fn gp(self, rhs: Self) -> Self {
        let s = self.s * rhs.s + self.e1 * rhs.e1 + self.e2 * rhs.e2 - self.e12 * rhs.e12;
        let e1 = self.s * rhs.e1 + self.e1 * rhs.s - self.e2 * rhs.e12 + self.e12 * rhs.e2;
        let e2 = self.s * rhs.e2 + self.e1 * rhs.e12 + self.e2 * rhs.s - self.e12 * rhs.e1;
        let e12 = self.s * rhs.e12 + self.e1 * rhs.e2 - self.e2 * rhs.e1 + self.e12 * rhs.s;
        Self::new(s, e1, e2, e12)
    }

    /// Grade reversal: `s + e1 + e2 - e12`.
    pub fn reverse(self) -> Self {
        Self::new(self.s, self.e1, self.e2, -self.e12)
    }

    /// Clifford conjugate: reverse plus grade involution.
    pub fn conjugate(self) -> Self {
        Self::new(self.s, -self.e1, -self.e2, -self.e12)
    }

    /// Sum of squared coefficients.
    pub fn norm_sqr(self) -> f64 {
        self.s * self.s + self.e1 * self.e1 + self.e2 * self.e2 + self.e12 * self.e12
    }

    /// Squared modulus of the even part.
    pub fn even_norm_sqr(self) -> f64 {
        self.s * self.s + self.e12 * self.e12
    }

    /// Amplitude `R` of an even multivector `Ψ = R U`.
    pub fn amplitude(self) -> f64 {
        self.even_norm_sqr().sqrt()
    }

    /// Inverse of an even multivector `s + b e12`.
    pub fn inverse_even(self) -> Option<Self> {
        let n = self.even_norm_sqr();
        if n <= f64::MIN_POSITIVE {
            None
        } else {
            Some(Self::even(self.s / n, -self.e12 / n))
        }
    }

    /// Exponential of an even multivector: `exp(s + b e12)`.
    pub fn exp_even(self) -> Self {
        let r = self.s.exp();
        Self::even(r * self.e12.cos(), r * self.e12.sin())
    }

    pub fn scalar_part(self) -> f64 {
        self.s
    }

    pub fn e12_part(self) -> f64 {
        self.e12
    }
}

impl Add for Mv2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.plus(rhs)
    }
}

impl Sub for Mv2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.minus(rhs)
    }
}

impl Neg for Mv2 {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl Mul for Mv2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.gp(rhs)
    }
}

impl Mul<f64> for Mv2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl Mul<Mv2> for f64 {
    type Output = Mv2;
    fn mul(self, rhs: Mv2) -> Mv2 {
        rhs.scale(self)
    }
}

/// `Q_Cl = -(∇²Ψ) Ψ^{-1}`, decomposed into scalar and `e12` parts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CliffordQuantumPotential {
    /// `⟨Q_Cl⟩_0`: scale-invariant valley field used for guidance.
    pub scalar: f64,
    /// `⟨Q_Cl⟩_2`: bivector spin torque (`e12` coefficient).
    pub bivector: f64,
}

impl CliffordQuantumPotential {
    /// Evaluate `Q_Cl` from an even field and its geometric Laplacian.
    pub fn from_field(psi: Mv2, laplacian: Mv2) -> Option<Self> {
        let inv = psi.inverse_even()?;
        let q = -laplacian.gp(inv);
        Some(Self {
            scalar: q.s,
            bivector: q.e12,
        })
    }

    /// Gradient-flow acceleration `-∇⟨Q_Cl⟩_0`.
    pub fn guidance(self, grad_scalar: [f64; 2]) -> [f64; 2] {
        [-grad_scalar[0], -grad_scalar[1]]
    }
}
