//! 16-blade Projective Geometric Algebra $\mathbb{G}_{3,0,1}$ multivector (`Mv301`)
//! for the $\text{Pin}(3,0,1)$-equivariant `GATr` relational operator (Track R3.2, Whitepaper §3.2).
//!
//! Basis layout (`[f32; 16]`):
//! - `0`: `1` (Grade 0 scalar)
//! - `1`: `e0`, `2`: `e1`, `3`: `e2`, `4`: `e3` (Grade 1 planes / vectors)
//! - `5`: `e01`, `6`: `e02`, `7`: `e03` (Grade 2 ideal / translation bivectors)
//! - `8`: `e12`, `9`: `e23`, `10`: `e31` (Grade 2 Euclidean / rotation bivectors)
//! - `11`: `e123`, `12`: `e032` ($x$), `13`: `e013` ($y$), `14`: `e021` ($z$) (Grade 3 points)
//! - `15`: `e0123` (Grade 4 pseudoscalar)

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// 16-blade multivector in $\mathbb{G}_{3,0,1}$ stored in SIMD-friendly `f32` layout.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct Mv301 {
    pub c: [f32; 16],
}

impl Default for Mv301 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl Mv301 {
    pub const ZERO: Self = Self { c: [0.0; 16] };

    /// Embed a finite 3D point $(x, y, z)$ as a grade-3 trivector $P = e_{123} + x e_{032} + y e_{013} + z e_{021}$.
    #[inline]
    pub fn from_point(p: [f32; 3]) -> Self {
        let mut c = [0.0; 16];
        c[11] = 1.0;
        c[12] = p[0];
        c[13] = p[1];
        c[14] = p[2];
        Self { c }
    }

    /// Extract Euclidean point coordinates $(x, y, z)$ from the grade-3 trivector blades.
    #[inline]
    pub fn point_coords(&self) -> [f32; 3] {
        let w = if self.c[11].abs() > 1e-6 {
            self.c[11]
        } else {
            1.0
        };
        [self.c[12] / w, self.c[13] / w, self.c[14] / w]
    }

    /// Embed a linear velocity $\mathbf{v}$ (ideal bivector $e_{01}, e_{02}, e_{03}$) and angular
    /// velocity $\boldsymbol{\omega}$ (Euclidean bivector $e_{12}, e_{23}, e_{31}$) as a grade-2 motor generator $B$.
    #[inline]
    pub fn from_motor_bivector(v_lin: [f32; 3], omega: [f32; 3]) -> Self {
        let mut c = [0.0; 16];
        c[5] = v_lin[0];
        c[6] = v_lin[1];
        c[7] = v_lin[2];
        c[8] = omega[2]; // e12
        c[9] = omega[0]; // e23
        c[10] = omega[1]; // e31
        Self { c }
    }

    /// Extract the translational bivector components $(v_x, v_y, v_z)$ from $(e_{01}, e_{02}, e_{03})$.
    #[inline]
    pub fn ideal_bivector(&self) -> [f32; 3] {
        [self.c[5], self.c[6], self.c[7]]
    }

    /// Extract the rotational bivector components $(\omega_x, \omega_y, \omega_z)$ from $(e_{23}, e_{31}, e_{12})$.
    #[inline]
    pub fn euclidean_bivector(&self) -> [f32; 3] {
        [self.c[9], self.c[10], self.c[8]]
    }

    /// Embed an oriented plane with normal $\hat{\mathbf{n}}$ and signed offset $d$ as a grade-1 vector.
    #[inline]
    pub fn from_plane(normal: [f32; 3], dist: f32) -> Self {
        let mut c = [0.0; 16];
        c[1] = dist;
        c[2] = normal[0];
        c[3] = normal[1];
        c[4] = normal[2];
        Self { c }
    }

    /// Extract grade-1 Euclidean vector components $(e_1, e_2, e_3)$.
    #[inline]
    pub fn euclidean_vector(&self) -> [f32; 3] {
        [self.c[2], self.c[3], self.c[4]]
    }

    /// Embed a 3D Plücker line passing through `point` with direction `dir` as a grade-2 bivector $L \in \mathbb{G}_{3,0,1}$.
    ///
    /// Plücker line representation: direction $\mathbf{d} = (d_x, d_y, d_z)$ in $(e_{23}, e_{31}, e_{12})$
    /// and moment $\mathbf{m} = \mathbf{p} \times \mathbf{d}$ in $(e_{01}, e_{02}, e_{03})$.
    #[inline]
    pub fn from_line(point: [f32; 3], dir: [f32; 3]) -> Self {
        let mut c = [0.0; 16];
        let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-9);
        let d = [dir[0] / norm, dir[1] / norm, dir[2] / norm];
        // Direction vector in (e23, e31, e12)
        c[9] = d[0];  // e23
        c[10] = d[1]; // e31
        c[8] = d[2];  // e12
        // Moment vector m = p × d in (e01, e02, e03)
        c[5] = point[1] * d[2] - point[2] * d[1]; // e01
        c[6] = point[2] * d[0] - point[0] * d[2]; // e02
        c[7] = point[0] * d[1] - point[1] * d[0]; // e03
        Self { c }
    }

    /// Embed a 2D line passing through `point` $(x, y)$ with direction `dir` $(dx, dy)$ in the $z=0$ plane.
    #[inline]
    pub fn from_line_2d(point: [f32; 2], dir: [f32; 2]) -> Self {
        Self::from_line([point[0], point[1], 0.0], [dir[0], dir[1], 0.0])
    }

    /// Extract unit line direction vector $(d_x, d_y, d_z)$ from $(e_{23}, e_{31}, e_{12})$.
    #[inline]
    pub fn line_direction(&self) -> [f32; 3] {
        self.euclidean_bivector()
    }

    /// Extract line moment vector $(m_x, m_y, m_z)$ from $(e_{01}, e_{02}, e_{03})$.
    #[inline]
    pub fn line_moment(&self) -> [f32; 3] {
        self.ideal_bivector()
    }

    /// Extract the 6-component bivector line blade $[d_x, d_y, d_z, m_x, m_y, m_z]$ in $\mathbb{G}_{3,0,1}$.
    #[inline]
    pub fn line_bivector_blade6(&self) -> [f32; 6] {
        [
            self.c[9],  // e23 = dx
            self.c[10], // e31 = dy
            self.c[8],  // e12 = dz
            self.c[5],  // e01 = mx
            self.c[6],  // e02 = my
            self.c[7],  // e03 = mz
        ]
    }

    /// Construct a line bivector from a 6-component blade $[d_x, d_y, d_z, m_x, m_y, m_z]$.
    #[inline]
    pub fn from_bivector_blade6(blade: [f32; 6]) -> Self {
        let mut c = [0.0; 16];
        c[9] = blade[0];
        c[10] = blade[1];
        c[8] = blade[2];
        c[5] = blade[3];
        c[6] = blade[4];
        c[7] = blade[5];
        Self { c }
    }

    /// Distance from a 3D point to this line bivector: $\|\mathbf{p} \times \mathbf{d} - \mathbf{m}\|$.
    #[inline]
    pub fn distance_to_line(&self, point: [f32; 3]) -> f32 {
        let d = self.line_direction();
        let m = self.line_moment();
        let px = point[1] * d[2] - point[2] * d[1] - m[0];
        let py = point[2] * d[0] - point[0] * d[2] - m[1];
        let pz = point[0] * d[1] - point[1] * d[0] - m[2];
        (px * px + py * py + pz * pz).sqrt()
    }

    /// Closest point on this line bivector to a given 3D point:
    /// $\mathbf{p}_{\text{proj}} = (\mathbf{d} \times \mathbf{m}) + (\mathbf{p} \cdot \mathbf{d})\mathbf{d}$.
    #[inline]
    pub fn closest_point_on_line(&self, point: [f32; 3]) -> [f32; 3] {
        let d = self.line_direction();
        let m = self.line_moment();
        let d_cross_m = [
            d[1] * m[2] - d[2] * m[1],
            d[2] * m[0] - d[0] * m[2],
            d[0] * m[1] - d[1] * m[0],
        ];
        let p_dot_d = point[0] * d[0] + point[1] * d[1] + point[2] * d[2];
        [
            d_cross_m[0] + p_dot_d * d[0],
            d_cross_m[1] + p_dot_d * d[1],
            d_cross_m[2] + p_dot_d * d[2],
        ]
    }

    /// Check if this multivector represents a valid PGA simple line blade (satisfying $\mathbf{d} \cdot \mathbf{m} = 0$).
    #[inline]
    pub fn is_valid_line(&self, eps: f32) -> bool {
        let d = self.line_direction();
        let m = self.line_moment();
        let dot = d[0] * m[0] + d[1] * m[1] + d[2] * m[2];
        let norm_d = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        dot.abs() <= eps && (norm_d - 1.0).abs() <= eps
    }

    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        let mut c = [0.0; 16];
        for i in 0..16 {
            c[i] = self.c[i] + rhs.c[i];
        }
        Self { c }
    }

    #[inline]
    pub fn scale(self, s: f32) -> Self {
        let mut c = [0.0; 16];
        for i in 0..16 {
            c[i] = self.c[i] * s;
        }
        Self { c }
    }

    /// Apply a 5-parameter grade-equivariant linear scaling $\sum_{g=0}^4 w_g \langle X \rangle_g$.
    #[inline]
    pub fn scale_grades(&self, w: [f32; 5]) -> Self {
        let mut c = [0.0; 16];
        c[0] = self.c[0] * w[0];
        for i in 1..5 {
            c[i] = self.c[i] * w[1];
        }
        for i in 5..11 {
            c[i] = self.c[i] * w[2];
        }
        for i in 11..15 {
            c[i] = self.c[i] * w[3];
        }
        c[15] = self.c[15] * w[4];
        Self { c }
    }

    /// Poincaré right-complement dual $X^*$ in $\mathbb{G}_{3,0,1}$ ($b \wedge b^* = e_{0123}$).
    #[inline]
    pub fn dual(&self) -> Self {
        let x = &self.c;
        let mut d = [0.0; 16];
        d[0] = x[15];
        d[15] = x[0];
        // Grade 1 <-> Grade 3
        d[11] = x[1];
        d[12] = -x[2];
        d[13] = -x[3];
        d[14] = -x[4];

        d[1] = -x[11];
        d[2] = x[12];
        d[3] = x[13];
        d[4] = x[14];
        // Grade 2 ideal <-> Grade 2 Euclidean
        d[8] = x[7]; // e03* = e12
        d[9] = x[5]; // e01* = e23
        d[10] = x[6]; // e02* = e31

        d[5] = x[9]; // e23* = e01
        d[6] = x[10]; // e31* = e02
        d[7] = x[8]; // e12* = e03
        Self { c: d }
    }

    /// Equivariant Projective Join $\operatorname{EquiJoin}(X, Y; z_{0123}) = z_{0123}(X^* \wedge Y^*)^*$
    /// (Proposition 7 of Brehmer et al. 2023, `reference_cliffordgatr.pdf`).
    /// Extracts relative line/displacement bivectors between grade-3 points and intersections of planes.
    #[inline]
    pub fn equi_join(&self, rhs: &Self, orientation_sign: f32) -> Self {
        let px = self.point_coords();
        let py = rhs.point_coords();
        let wx = self.c[11];
        let wy = rhs.c[11];

        let mut out = [0.0f32; 16];
        // Relative translation bivector (wy * px - wx * py) in (e01, e02, e03)
        let rx = wy * px[0] - wx * py[0];
        let ry = wy * px[1] - wx * py[1];
        let rz = wy * px[2] - wx * py[2];
        out[5] = orientation_sign * rx;
        out[6] = orientation_sign * ry;
        out[7] = orientation_sign * rz;

        // Moment bivector px \times py in (e23, e31, e12)
        let mx = px[1] * py[2] - px[2] * py[1];
        let my = px[2] * py[0] - px[0] * py[2];
        let mz = px[0] * py[1] - px[1] * py[0];
        out[9] = orientation_sign * mx;
        out[10] = orientation_sign * my;
        out[8] = orientation_sign * mz;

        Self { c: out }
    }

    /// Grade-preserving equivariant geometric bilinear product combining inner dot products
    /// and cross/commutator bivectors on $O(3)$-covariant sub-blades.
    #[inline]
    pub fn geometric_bilinear(&self, rhs: &Self) -> Self {
        let mut out = [0.0f32; 16];
        let va = self.ideal_bivector();
        let vb = rhs.ideal_bivector();
        let na = self.euclidean_vector();
        let nb = rhs.euclidean_vector();
        let wa = self.euclidean_bivector();
        let wb = rhs.euclidean_bivector();

        // Grade 0 scalar inner product
        let dot_v = va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2];
        let dot_n = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
        let dot_w = wa[0] * wb[0] + wa[1] * wb[1] + wa[2] * wb[2];
        out[0] = self.c[0] * rhs.c[0] + dot_v + dot_n - dot_w;

        // Grade 1 normal vector modulation
        for k in 0..3 {
            out[2 + k] = self.c[0] * nb[k] + rhs.c[0] * na[k];
        }

        // Grade 2 ideal bivector: scalar scaling + cross interaction with rotation bivector (wb \times va)
        let cross_wv = [
            wa[1] * vb[2] - wa[2] * vb[1],
            wa[2] * vb[0] - wa[0] * vb[2],
            wa[0] * vb[1] - wa[1] * vb[0],
        ];
        for k in 0..3 {
            out[5 + k] = self.c[0] * vb[k] + rhs.c[0] * va[k] + cross_wv[k];
        }

        // Grade 2 Euclidean bivector: wedge product of velocities (va \wedge vb) + commutator
        let wedge_v = [
            va[1] * vb[2] - va[2] * vb[1],
            va[2] * vb[0] - va[0] * vb[2],
            va[0] * vb[1] - va[1] * vb[0],
        ];
        out[9] = self.c[0] * wb[0] + rhs.c[0] * wa[0] + wedge_v[0];
        out[10] = self.c[0] * wb[1] + rhs.c[0] * wa[1] + wedge_v[1];
        out[8] = self.c[0] * wb[2] + rhs.c[0] * wa[2] + wedge_v[2];

        out[15] = self.c[15] * rhs.c[0] + self.c[0] * rhs.c[15];
        Self { c: out }
    }

    /// $E(3)$-invariant inner product on the translation-invariant grades (grades 0, 1-normal, 2, 4).
    #[inline]
    pub fn invariant_inner(&self, rhs: &Self) -> f32 {
        let mut s = self.c[0] * rhs.c[0] + self.c[15] * rhs.c[15];
        for i in 2..11 {
            s += self.c[i] * rhs.c[i];
        }
        s
    }

    /// $E(3)$-invariant squared norm of the translation-invariant grades.
    #[inline]
    pub fn invariant_norm_sq(&self) -> f32 {
        self.invariant_inner(self)
    }

    /// Transform a multivector under a 3D rigid motion $(R \in SO(3), \mathbf{t} \in \mathbb{R}^3)$.
    pub fn transform_e3(&self, rot: [[f32; 3]; 3], trans: [f32; 3]) -> Self {
        let mut out = *self;
        // Grade 1 normal (e1, e2, e3)
        let n = self.euclidean_vector();
        let rn = mat3_vec(rot, n);
        out.c[2] = rn[0];
        out.c[3] = rn[1];
        out.c[4] = rn[2];

        // Grade 2 ideal bivector (e01, e02, e03) rotates via R
        let v = self.ideal_bivector();
        let rv = mat3_vec(rot, v);
        out.c[5] = rv[0];
        out.c[6] = rv[1];
        out.c[7] = rv[2];

        // Grade 2 Euclidean bivector (e23, e31, e12) rotates via R
        let w = self.euclidean_bivector();
        let rw = mat3_vec(rot, w);
        out.c[9] = rw[0];
        out.c[10] = rw[1];
        out.c[8] = rw[2];

        // Grade 3 point (e123, e032, e013, e021) rotates and translates if weight e123 != 0
        if self.c[11].abs() > 1e-6 {
            let p = self.point_coords();
            let rp = mat3_vec(rot, p);
            let w_pt = self.c[11];
            out.c[12] = w_pt * (rp[0] + trans[0]);
            out.c[13] = w_pt * (rp[1] + trans[1]);
            out.c[14] = w_pt * (rp[2] + trans[2]);
        }
        out
    }
}

#[inline]
pub fn mat3_vec(r: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_bivector_plucker_properties() {
        let pt = [10.0f32, 20.0, 5.0];
        let dir = [0.0f32, 1.0, 0.0]; // Y axis line through (10, 20, 5)
        let line = Mv301::from_line(pt, dir);

        assert!(line.is_valid_line(1e-5), "Line must satisfy Plücker condition d · m = 0");
        let d = line.line_direction();
        assert!((d[0] - 0.0).abs() < 1e-6);
        assert!((d[1] - 1.0).abs() < 1e-6);
        assert!((d[2] - 0.0).abs() < 1e-6);

        // Point on line: (10, 50, 5)
        let test_pt_on_line = [10.0, 50.0, 5.0];
        assert!(line.distance_to_line(test_pt_on_line) < 1e-5);
        let closest = line.closest_point_on_line(test_pt_on_line);
        assert!((closest[0] - 10.0).abs() < 1e-5);
        assert!((closest[1] - 50.0).abs() < 1e-5);
        assert!((closest[2] - 5.0).abs() < 1e-5);

        // Point off line: (13, 20, 5) -> dist = 3.0
        let test_pt_off = [13.0, 20.0, 5.0];
        let dist = line.distance_to_line(test_pt_off);
        assert!((dist - 3.0).abs() < 1e-5);
        let closest_off = line.closest_point_on_line(test_pt_off);
        assert!((closest_off[0] - 10.0).abs() < 1e-5);
        assert!((closest_off[1] - 20.0).abs() < 1e-5);
        assert!((closest_off[2] - 5.0).abs() < 1e-5);

        // Blade6 round-trip
        let b6 = line.line_bivector_blade6();
        let reconstructed = Mv301::from_bivector_blade6(b6);
        assert_eq!(line.c, reconstructed.c);
    }
}
