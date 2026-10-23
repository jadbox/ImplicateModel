//! Articulated `CliffordDensityElement` Holon coupling to the Wave-Matter Lattice
//! (Track R4.1, Whitepaper §6.3).
//!
//! Couples root `CliffordDensityElement` ($\Phi_L = R\,U\,\epsilon \in \mathbb{G}_{2,0}$)
//! and articulated `SubHolonLink` child parts (compliant capes, wings, plumage, limbs)
//! to the GPU-reduced radiation pressure $-\nabla p$, Bohmian guidance $-\nabla \langle \mathcal{Q}_{\text{Cl}} \rangle_0$,
//! local lattice velocity $\mathbf{v}(\mathbf{x}, t)$, chromatic luminescence gradient $\nabla \mathbf{L}(\mathbf{x}, t)$,
//! and exact PGA boundary incidences $P_i \wedge B_k = 0$.

use bytemuck::{Pod, Zeroable};
use implicate_math::{
    CliffordDensityElement, OrientedLine2, StateBasis, segment_hit_oriented_line, wrap_angle_pi,
};
use serde::{Deserialize, Serialize};

/// Active Holon wake & luminescence emitter uploaded to the GPU (`32 bytes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct GpuHolonSource {
    /// `(x, y, vx, vy)` in lattice grid coordinates.
    pub kinematics: [f32; 4],
    /// `(L_r, L_g, L_b, radius)`.
    pub emission: [f32; 4],
}

/// Holon or binaural ear probe query uploaded to the GPU (`16 bytes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct GpuHolonQuery {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub yield_pressure: f32,
}

/// Zero-readback reduction record returned by `reduce_holons` (`32 bytes`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct GpuHolonReduction {
    /// `(fx, fy, tau_z, max_pressure)`.
    pub forces: [f32; 4],
    /// `(flow_vx, flow_vy, grad_lum_x, grad_lum_y)`.
    pub field_env: [f32; 4],
}

/// An articulated child sub-holon part (e.g., compliant cape, wing, head, limb) attached to a root Holon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubHolonLink {
    pub name: String,
    /// Attachment anchor in the root holon's local frame.
    pub anchor_local: [f32; 2],
    /// Rest orientation angle relative to root heading (radians).
    pub rest_angle: f32,
    /// Current articulated joint angle relative to root heading (radians).
    pub theta_rel: f32,
    /// Current joint angular velocity (rad/s).
    pub omega_rel: f32,
    /// Coupling gain to local lattice flow velocity $\mathbf{v}(\mathbf{x}, t)$ and root slipstream.
    pub flow_compliance: f32,
    /// Restoring harmonic spring stiffness $k_{\text{spring}}$.
    pub spring_stiffness: f32,
    /// Joint damping coefficient $c_{\text{damp}}$.
    pub damping: f32,
    /// 2.5D/3D surface normal $\hat{\mathbf{n}}_p$ in local part coordinates for $\nabla \mathbf{L}$ facet lighting.
    pub facet_normal_local: [f32; 3],
    /// Dynamically computed chromatic irradiance from $\nabla \mathbf{L}(\mathbf{x}, t)$.
    pub facet_irradiance: [f32; 3],
}

impl SubHolonLink {
    pub fn new_compliant(
        name: impl Into<String>,
        anchor_local: [f32; 2],
        rest_angle: f32,
        flow_compliance: f32,
        facet_normal_local: [f32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            anchor_local,
            rest_angle,
            theta_rel: rest_angle,
            omega_rel: 0.0,
            flow_compliance,
            spring_stiffness: 18.0,
            damping: 4.5,
            facet_normal_local,
            facet_irradiance: [0.0; 3],
        }
    }
}

/// A physical Holon coupling `CliffordDensityElement` ($\Phi_L = R\,U\,\epsilon$), constitutive
/// material parameters $(\rho, K)$, chromatic emission $\mathbf{E}_{\text{rgb}}$, and articulated `SubHolonLink` parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArticulatedFieldHolon {
    pub id: u32,
    pub name: String,
    pub cde: CliffordDensityElement,
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub radius: f32,
    /// Constitutive density $\rho > 0$.
    pub rho: f32,
    /// Constitutive bulk modulus $K > 0$ (modal wave speed $c = \sqrt{K/\rho}$).
    pub bulk_modulus_k: f32,
    /// Griffith yield pressure $p_{\text{yield}}$.
    pub yield_pressure: f32,
    /// Intrinsic chromatic luminescence emission `(L_r, L_g, L_b)` (e.g., Red Amulet or plasma bolt).
    pub luminescence_emission: [f32; 3],
    /// Articulated child parts ($P \ge 1$).
    pub parts: Vec<SubHolonLink>,
    /// Most recent GPU reduction record.
    pub last_reduction: GpuHolonReduction,
}

impl ArticulatedFieldHolon {
    pub fn new(
        id: u32,
        name: impl Into<String>,
        position: [f32; 2],
        velocity: [f32; 2],
        heading: f64,
        radius: f32,
        rho: f32,
        bulk_modulus_k: f32,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            cde: CliffordDensityElement::new(1.0, heading, 0, 2, [255, 255, 255]),
            position,
            velocity,
            radius,
            rho: rho.max(0.05),
            bulk_modulus_k: bulk_modulus_k.max(0.05),
            yield_pressure: 1.2,
            luminescence_emission: [0.0; 3],
            parts: Vec::new(),
            last_reduction: GpuHolonReduction::default(),
        }
    }

    /// Constitutive acoustic wave speed $c = \sqrt{K/\rho}$.
    #[inline]
    pub fn wave_speed(&self) -> f32 {
        (self.bulk_modulus_k / self.rho).sqrt()
    }

    /// Fundamental modal frequency $f_0 = 440\sqrt{K/\rho}$ Hz (Whitepaper §7.2).
    #[inline]
    pub fn modal_fundamental_hz(&self) -> f32 {
        440.0 * self.wave_speed()
    }

    #[inline]
    pub fn to_gpu_source(&self) -> GpuHolonSource {
        GpuHolonSource {
            kinematics: [
                self.position[0],
                self.position[1],
                self.velocity[0],
                self.velocity[1],
            ],
            emission: [
                self.luminescence_emission[0],
                self.luminescence_emission[1],
                self.luminescence_emission[2],
                self.radius,
            ],
        }
    }

    #[inline]
    pub fn to_gpu_query(&self) -> GpuHolonQuery {
        GpuHolonQuery {
            x: self.position[0],
            y: self.position[1],
            radius: self.radius,
            yield_pressure: self.yield_pressure,
        }
    }

    /// Advance the root `CliffordDensityElement`, articulated child parts, and PGA boundary contacts
    /// for timestep `dt` using the GPU-reduced field forces and luminescence gradients.
    pub fn integrate_step(
        &mut self,
        red: GpuHolonReduction,
        external_accel: [f32; 2],
        boundaries: &[OrientedLine2],
        bounds_wh: [f32; 2],
        sim_time: f64,
        frame_idx: u64,
        substep_idx: u32,
        dt: f32,
        events_out: &mut Vec<CollisionImpulseEvent>,
    ) {
        self.last_reduction = red;
        let inv_mass = 1.0 / (self.rho * self.radius.max(1.0));
        let ax = external_accel[0] + red.forces[0] * inv_mass;
        let ay = external_accel[1] + red.forces[1] * inv_mass;

        let p0 = [self.position[0] as f64, self.position[1] as f64];
        self.velocity[0] += ax * dt;
        self.velocity[1] += ay * dt;

        let mut p1 = [
            p0[0] + (self.velocity[0] * dt) as f64,
            p0[1] + (self.velocity[1] * dt) as f64,
        ];

        // Resolve exact PGA oriented-line boundary incidences (P_i \wedge B_k = 0)
        for (b_idx, line) in boundaries.iter().enumerate() {
            if let Some(hit) = segment_hit_oriented_line(p0, p1, line) {
                let n = line.normal;
                let vn = (self.velocity[0] as f64) * n[0] + (self.velocity[1] as f64) * n[1];
                if vn < 0.0 {
                    let restitution = 0.82;
                    let impulse_mag = (-(1.0 + restitution) * vn * (self.rho as f64)) as f32;
                    self.velocity[0] -= ((1.0 + restitution) * vn * n[0]) as f32;
                    self.velocity[1] -= ((1.0 + restitution) * vn * n[1]) as f32;
                    // Push out slightly along normal to guarantee 100% boundary compliance
                    p1 = [hit.point[0] + n[0] * 0.15, hit.point[1] + n[1] * 0.15];
                    // Execute idempotent branch flip on collision
                    let next_s = (self.cde.state() + 1) % 2;
                    self.cde.idempotent =
                        StateBasis::new(2).get(next_s).expect("valid state").clone();
                    events_out.push(CollisionImpulseEvent {
                        holon_id: self.id,
                        boundary_idx: b_idx,
                        sim_time,
                        frame_idx,
                        substep_idx,
                        contact_pos: [hit.point[0] as f32, hit.point[1] as f32],
                        normal: [n[0] as f32, n[1] as f32],
                        impulse_magnitude: impulse_mag.max(0.05),
                        wave_speed_c: self.wave_speed(),
                    });
                }
            } else {
                // Also enforce half-space non-penetration within radius margin
                let signed_d = line.side(p1);
                let prev_d = line.side(p0);
                if prev_d >= 0.0 && signed_d < 0.0 {
                    let n = line.normal;
                    let vn = (self.velocity[0] as f64) * n[0] + (self.velocity[1] as f64) * n[1];
                    if vn < 0.0 {
                        let restitution = 0.82;
                        let _impulse_mag = (-(1.0 + restitution) * vn * (self.rho as f64)) as f32;
                        self.velocity[0] -= ((1.0 + restitution) * vn * n[0]) as f32;
                        self.velocity[1] -= ((1.0 + restitution) * vn * n[1]) as f32;
                    }
                    p1[0] -= (signed_d - 0.12) * n[0];
                    p1[1] -= (signed_d - 0.12) * n[1];
                }
            }
        }

        // Clamp within outer domain box [margin, W - margin] x [margin, H - margin]
        let margin = self.radius.clamp(1.0, 4.0) as f64;
        let max_x = (bounds_wh[0] as f64 - margin).max(margin);
        let max_y = (bounds_wh[1] as f64 - margin).max(margin);
        if p1[0] < margin {
            p1[0] = margin;
            if self.velocity[0] < 0.0 {
                self.velocity[0] = -0.8 * self.velocity[0];
            }
        } else if p1[0] > max_x {
            p1[0] = max_x;
            if self.velocity[0] > 0.0 {
                self.velocity[0] = -0.8 * self.velocity[0];
            }
        }
        if p1[1] < margin {
            p1[1] = margin;
            if self.velocity[1] < 0.0 {
                self.velocity[1] = -0.8 * self.velocity[1];
            }
        } else if p1[1] > max_y {
            p1[1] = max_y;
            if self.velocity[1] > 0.0 {
                self.velocity[1] = -0.8 * self.velocity[1];
            }
        }

        self.position = [p1[0] as f32, p1[1] as f32];

        // Update CliffordDensityElement rotor heading from velocity and quantum spin-torque
        let speed = self.velocity[0].hypot(self.velocity[1]);
        if speed > 0.05 {
            self.cde
                . apply_quantum_torque([self.velocity[0] as f64, self.velocity[1] as f64], 0.25);
        }
        let root_theta = self.cde.heading_angle() as f32;
        let (sr, cr) = root_theta.sin_cos();

        // Advance articulated child parts (compliant capes/wings deform with local fluid flow v - v_root)
        let rel_flow_x = red.field_env[0] - 0.35 * self.velocity[0];
        let rel_flow_y = red.field_env[1] - 0.35 * self.velocity[1];
        let grad_lx = red.field_env[2];
        let grad_ly = red.field_env[3];
        let grad_l_mag = grad_lx.hypot(grad_ly);

        for part in &mut self.parts {
            let world_angle = root_theta + part.theta_rel;
            let (sw, cw) = world_angle.sin_cos();
            // Transverse aerodynamic/hydrodynamic torque from local lattice velocity field v(x, t)
            let flow_cross = cw * rel_flow_y - sw * rel_flow_x;
            let spring_restore =
                -part.spring_stiffness * (wrap_angle_pi((part.theta_rel - part.rest_angle) as f64) as f32);
            let alpha_part =
                part.flow_compliance * flow_cross + spring_restore - part.damping * part.omega_rel;
            part.omega_rel += alpha_part * dt;
            part.theta_rel = part.rest_angle
                + (part.theta_rel + part.omega_rel * dt - part.rest_angle).clamp(-1.1, 1.1);

            // 2.5D/3D normal facet lighting from chromatic luminescence gradient \nabla L
            let nx_world = part.facet_normal_local[0] * cr - part.facet_normal_local[1] * sr;
            let ny_world = part.facet_normal_local[0] * sr + part.facet_normal_local[1] * cr;
            let nz_world = part.facet_normal_local[2];
            if grad_l_mag > 1e-5 {
                // Light direction points from the part toward the luminescence peak (+\nabla L)
                let lx = grad_lx / grad_l_mag;
                let ly = grad_ly / grad_l_mag;
                let ndotl = (nx_world * lx + ny_world * ly + nz_world * 0.35).max(0.0);
                let irr = (grad_l_mag * (0.35 + 0.65 * ndotl)).min(4.0);
                part.facet_irradiance = [irr, irr * 0.92, irr * 0.85];
            } else {
                part.facet_irradiance = [0.0; 3];
            }
        }
    }
}

/// Record of an exact PGA collision impulse ($\Delta \mathbf{P}_B$ at $P_i \wedge B_k = 0$)
/// used to excite both lattice shockwaves and synchronized $48\text{ kHz}$ `@rolpa` modal audio.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollisionImpulseEvent {
    pub holon_id: u32,
    pub boundary_idx: usize,
    pub sim_time: f64,
    pub frame_idx: u64,
    pub substep_idx: u32,
    pub contact_pos: [f32; 2],
    pub normal: [f32; 2],
    pub impulse_magnitude: f32,
    pub wave_speed_c: f32,
}
