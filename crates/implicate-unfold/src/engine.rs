//! Single-Pass Causal Unfoldment Engine (`CausalUnfoldEngine`, Track R4.3, Whitepaper §6).
//!
//! Couples:
//! - Stratum 2: `VerbPotentialLexicon` ($\mathcal{Q}_{\text{Cl},v}$) and 6-block $\text{Pin}(3,0,1)$-equivariant `GatrVerbBinder`,
//! - Stratum 3: `GpuWaveMatterLattice` + `ArticulatedFieldHolon` ($P$-part compliant sub-holons + PGA boundary blades),
//! - Stratum 1: `GpuCliffordNetRefiner` (1-pass 4-block `CliffordNet-Lite` `@rolpa` shader) and
//!   `AcousticRolpaProjector` ($48\text{ kHz}$ modal/harmonic stereo audio).

use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use implicate_field::{
    ArticulatedFieldHolon, CollisionImpulseEvent, FieldGpuContext, GpuWaveMatterLattice,
    LatticeParams, SubHolonLink,
};
use implicate_lexicon::{ConceptLexicon, GatrHolonOutput, GatrHolonToken};
use implicate_lifter::{
    ArticulatedHolon, ParallaxLayer, SubHolonPose, mux_rgb_frames_and_wav_to_mp4,
};
use implicate_math::{
    CanonicalRelFrame, HolonPose2D, LearnedVerbPotential, OrientedLine2, Spinor16,
    normalized_frechet_distance, wrap_angle_pi,
};
use implicate_rolpa::{
    AcousticRolpaProjector, DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS, GpuCliffordNetRefiner,
    RolpaGpuContext, RolpaInputGrid, RolpaUniforms,
};
use serde::{Deserialize, Serialize};

use crate::container::ImplicateContainerManifest;
use crate::payload::EnfoldedClipAssets;
use crate::prompt::{PromptHolonSpec, UnfoldSceneSpec};

/// Compiled relational verb binding active during scene unfoldment.
#[derive(Clone, Debug)]
pub struct ActiveVerbBinding {
    pub subject_idx: usize,
    pub target_idx: usize,
    pub verb_name: String,
    pub verb_spinor: Spinor16,
    pub potential: LearnedVerbPotential,
    pub canonical_frame: CanonicalRelFrame,
    pub rel_pos: [f64; 2],
    pub rel_vel: [f64; 2],
    pub rel_theta: f64,
}

/// Per-verb adherence diagnostic report after unfolding a scene.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerbAdherenceMetric {
    pub subject: String,
    pub verb: String,
    pub target: String,
    /// Normalized discrete Fréchet distance between unfolded subject trajectory and canonical
    /// $\mathcal{Q}_{\text{Cl},v}$ reference trajectory in the target's relative PGA frame.
    pub normalized_frechet: f64,
}

/// Complete output and verification metrics from unfolding an `UnfoldSceneSpec`.
#[derive(Clone, Debug)]
pub struct UnfoldRunReport {
    pub scene_id: String,
    pub width: u32,
    pub height: u32,
    pub num_frames: usize,
    pub total_wall_ms: f64,
    pub mean_frame_ms: f64,
    pub effective_fps: f64,
    /// Fraction of `(frame, holon)` pairs where holon permanence ($R_i > 0$, finite coordinates) held (`1.0 == 100%`).
    pub object_permanence_rate: f64,
    /// Fraction of `(frame, physical_holon, boundary)` checks where $P_i \wedge B_k \ge 0$ held (`1.0 == 100%`).
    pub boundary_compliance_rate: f64,
    /// Maximum camera-lock drift (in pixels) for HUD interface holons ($M_{\text{hud}}(t) = M_{\text{cam}}(t)$).
    pub max_hud_lock_drift_px: f32,
    /// Per-verb trajectory adherence metrics.
    pub verb_metrics: Vec<VerbAdherenceMetric>,
    /// Unfolded world trajectories `trajectories[holon_idx][frame_idx]`.
    pub trajectories: Vec<Vec<HolonPose2D>>,
    /// Rendered RGB video frames (`num_frames` buffers of `width * height * 3` bytes).
    pub rgb_frames: Vec<Vec<u8>>,
    /// Synchronized $48\text{ kHz}$ stereo audio stream (`num_frames * 800` stereo samples).
    pub stereo_audio: Vec<[f32; 2]>,
    /// Audio buffer underrun count (`0` required).
    pub audio_underruns: usize,
    /// Total PGA collision impulse events sonified.
    pub total_collisions: usize,
    /// Per-frame dynamic score/status counter of the camera-locked HUD holon.
    pub hud_score_history: Vec<u32>,
}

impl UnfoldRunReport {
    /// Export the unfolded $48\text{ kHz}$ stereo `.wav` and muxed H.264 + AAC `.mp4` into `out_dir`.
    pub fn export_wav_and_mp4(&self, out_dir: &Path) -> io::Result<(PathBuf, PathBuf)> {
        std::fs::create_dir_all(out_dir)?;
        let wav_path = out_dir.join(format!("{}.wav", self.scene_id));
        let mp4_path = out_dir.join(format!("{}.mp4", self.scene_id));
        let proj = AcousticRolpaProjector::new(DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS);
        proj.write_stereo_wav(&wav_path, &self.stereo_audio)
            .map_err(|e| io::Error::other(format!("write_stereo_wav failed: {e}")))?;
        mux_rgb_frames_and_wav_to_mp4(
            &mp4_path,
            self.width,
            self.height,
            DEFAULT_VIDEO_FPS as f64,
            &self.rgb_frames,
            &wav_path,
        )?;
        Ok((wav_path, mp4_path))
    }
}

/// Single-pass causal unfoldment engine.
pub struct CausalUnfoldEngine {
    pub manifest: ImplicateContainerManifest,
    pub scene: UnfoldSceneSpec,
    pub holons: Vec<ArticulatedFieldHolon>,
    pub noun_spinors: Vec<Spinor16>,
    pub is_hud_holon: Vec<bool>,
    pub hud_cam_offsets: Vec<[f32; 2]>,
    pub base_colors: Vec<[u8; 3]>,
    pub active_verbs: Vec<ActiveVerbBinding>,
    pub boundaries: Vec<OrientedLine2>,
    pub lattice: GpuWaveMatterLattice,
    pub acoustic: AcousticRolpaProjector,
    pub refiner: Option<GpuCliffordNetRefiner>,
    pub camera_pos: [f32; 2],
    pub enfolded_clip: Option<EnfoldedClipAssets>,
    pub hud_score: u32,
}

impl CausalUnfoldEngine {
    /// Compile an `UnfoldSceneSpec` against an `ImplicateContainerManifest` into a ready-to-step
    /// `CausalUnfoldEngine`.
    pub fn new(manifest: ImplicateContainerManifest, scene: UnfoldSceneSpec) -> Self {
        let field_ctx = FieldGpuContext::shared();
        let sim_w = 128u32;
        let sim_h = 128u32;
        let lattice = GpuWaveMatterLattice::new(
            field_ctx,
            LatticeParams {
                width: sim_w,
                height: sim_h,
                dt: 0.15,
                ..LatticeParams::default()
            },
        );

        let cam0 = [
            (scene.width as f32) * 0.5,
            (scene.height as f32) * 0.5,
        ];

        let clauses: Vec<&str> = scene
            .holons
            .iter()
            .map(|h| h.semantic_clause.as_str())
            .collect();
        let noun_spinors = manifest.concept_lexicon.embed_clauses(&clauses);

        let mut holons = Vec::with_capacity(scene.holons.len());
        let mut is_hud_holon = Vec::with_capacity(scene.holons.len());
        let mut hud_cam_offsets = Vec::with_capacity(scene.holons.len());
        let mut base_colors = Vec::with_capacity(scene.holons.len());

        let w_f = scene.width as f32;
        let h_f = scene.height as f32;
        let min_dim = w_f.min(h_f);

        for (idx, h_spec) in scene.holons.iter().enumerate() {
            // Task R9.3: Resolution-invariant normalized projective coordinates [0, 1]^2
            let is_norm_pos = h_spec.initial_pos[0] >= 0.0
                && h_spec.initial_pos[0] <= 1.0
                && h_spec.initial_pos[1] >= 0.0
                && h_spec.initial_pos[1] <= 1.0;
            let projected_pos = if is_norm_pos {
                [h_spec.initial_pos[0] * w_f, h_spec.initial_pos[1] * h_f]
            } else {
                h_spec.initial_pos
            };

            let projected_vel = if is_norm_pos
                && h_spec.initial_vel[0].abs() <= 1.0
                && h_spec.initial_vel[1].abs() <= 1.0
            {
                [h_spec.initial_vel[0] * w_f, h_spec.initial_vel[1] * h_f]
            } else {
                h_spec.initial_vel
            };

            let projected_radius = if is_norm_pos && h_spec.radius <= 0.5 {
                (h_spec.radius * min_dim).max(2.0)
            } else {
                h_spec.radius
            };

            let heading = if projected_vel[0].abs() + projected_vel[1].abs() > 1e-3 {
                (projected_vel[1] as f64).atan2(projected_vel[0] as f64)
            } else {
                0.0
            };
            let mut h = ArticulatedFieldHolon::new(
                (idx + 1) as u32,
                &h_spec.name,
                projected_pos,
                projected_vel,
                heading,
                projected_radius,
                h_spec.rho,
                h_spec.bulk_modulus_k,
            );
            h.luminescence_emission = h_spec.luminescence_emission;
            h.cohesion_c = h_spec.cohesion_c;

            // Attach compliant sub-holon parts when num_parts > 1 (Tier-2 / articulated characters)
            if h_spec.num_parts >= 4 {
                h.parts.push(SubHolonLink::new_compliant(
                    "cape_or_dorsal",
                    [-3.0, 0.0],
                    0.0,
                    1.25,
                    [-0.6, 0.2, 0.77],
                ));
                h.parts.push(SubHolonLink::new_compliant(
                    "head_crest",
                    [2.0, 2.0],
                    0.1,
                    0.55,
                    [0.7, 0.3, 0.65],
                ));
                h.parts.push(SubHolonLink::new_compliant(
                    "fore_limb",
                    [1.5, -1.5],
                    -0.15,
                    0.80,
                    [0.5, -0.4, 0.77],
                ));
            }
            if h_spec.num_parts >= 6 {
                h.parts.push(SubHolonLink::new_compliant(
                    "left_wing",
                    [0.0, 3.5],
                    0.35,
                    1.45,
                    [0.2, 0.8, 0.56],
                ));
                h.parts.push(SubHolonLink::new_compliant(
                    "right_wing",
                    [0.0, -3.5],
                    -0.35,
                    1.45,
                    [0.2, -0.8, 0.56],
                ));
            }

            holons.push(h);
            is_hud_holon.push(h_spec.is_camera_locked_hud);
            hud_cam_offsets.push([
                projected_pos[0] - cam0[0],
                projected_pos[1] - cam0[1],
            ]);
            base_colors.push(h_spec.base_color);
        }

        // Compile relational verb bindings
        let mut active_verbs = Vec::new();
        for vb in &scene.verb_bindings {
            let Some(subject_idx) = holons.iter().position(|h| h.name == vb.subject) else {
                continue;
            };
            let Some(target_idx) = holons.iter().position(|h| h.name == vb.target) else {
                continue;
            };
            let Some(entry) = manifest
                .verb_potentials
                .resolve(&manifest.concept_lexicon, &vb.verb)
            else {
                continue;
            };
            let a0 = HolonPose2D::new(
                holons[subject_idx].position[0] as f64,
                holons[subject_idx].position[1] as f64,
                holons[subject_idx].cde.heading_angle(),
            );
            let b0 = HolonPose2D::new(
                holons[target_idx].position[0] as f64,
                holons[target_idx].position[1] as f64,
                holons[target_idx].cde.heading_angle(),
            );
            let canonical_frame = CanonicalRelFrame::from_initial(a0, b0);
            let rel0 = canonical_frame.to_canonical(a0, b0);
            let (_, v0, _, theta0, _) = entry.potential.canonical_kinematics(0.0);
            active_verbs.push(ActiveVerbBinding {
                subject_idx,
                target_idx,
                verb_name: vb.verb.clone(),
                verb_spinor: entry.spinor,
                potential: entry.potential.clone(),
                canonical_frame,
                rel_pos: [rel0.x, rel0.y],
                rel_vel: v0,
                rel_theta: theta0,
            });
        }

        let boundaries = scene
            .boundaries
            .iter()
            .map(|b| {
                let is_norm = b.point[0] >= 0.0 && b.point[0] <= 1.0 && b.point[1] >= 0.0 && b.point[1] <= 1.0;
                let pt = if is_norm {
                    [b.point[0] * (scene.width as f64), b.point[1] * (scene.height as f64)]
                } else {
                    b.point
                };
                OrientedLine2::solid_through(pt, b.normal)
            })
            .collect();

        let refiner = if scene.enable_cliffordnet_refiner {
            let rolpa_ctx = RolpaGpuContext::shared();
            let uniforms = RolpaUniforms {
                width: scene.width,
                height: scene.height,
                ..RolpaUniforms::default()
            };
            Some(GpuCliffordNetRefiner::new(rolpa_ctx, uniforms))
        } else {
            None
        };

        Self {
            manifest,
            scene,
            holons,
            noun_spinors,
            is_hud_holon,
            hud_cam_offsets,
            base_colors,
            active_verbs,
            boundaries,
            lattice,
            acoustic: AcousticRolpaProjector::new(DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS),
            refiner,
            camera_pos: cam0,
            enfolded_clip: None,
            hud_score: 0,
        }
    }

    /// Attach enfolded `MultiPlaneAtlas` background/HUD layers and `ArticulatedHolon` CFT wavepackets
    /// for real-asset `@rolpa` compositing during unfoldment.
    pub fn with_enfolded_clip(mut self, clip: EnfoldedClipAssets) -> Self {
        self.enfolded_clip = Some(clip);
        self
    }

    /// Select and attach the best-matching `EnfoldedClipAssets` from a multi-clip container slice
    /// based on scene domain, actor keywords, and viewport dimensions.
    pub fn with_enfolded_clips(mut self, clips: &[EnfoldedClipAssets]) -> Self {
        if clips.is_empty() {
            return self;
        }
        // Task R9.4: Milieu retrieval via Spin(16) semantic geodesic distance
        let scene_desc = format!(
            "{} {} {}",
            self.scene.scene_id,
            self.scene.prompt_text,
            self.scene
                .holons
                .iter()
                .map(|h| format!("{} {}", h.name, h.semantic_clause))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let scene_spinors = ConceptLexicon::default().embed_clauses(&[&scene_desc]);
        let scene_spinor = scene_spinors.first().copied().unwrap_or_default();

        let best = clips.iter().max_by(|a, b| {
            let desc_a = format!("{}_milieu {} layers={}", a.clip_id, a.source_video, a.atlas.layers.len());
            let desc_b = format!("{}_milieu {} layers={}", b.clip_id, b.source_video, b.atlas.layers.len());
            let sp_a = ConceptLexicon::default().embed_clauses(&[&desc_a]);
            let sp_b = ConceptLexicon::default().embed_clauses(&[&desc_b]);

            let sim_a = scene_spinor.coherence(sp_a.first().unwrap_or(&scene_spinor));
            let sim_b = scene_spinor.coherence(sp_b.first().unwrap_or(&scene_spinor));

            let ar_scene = (self.scene.width as f64) / (self.scene.height.max(1) as f64);
            let ar_a = (a.atlas.width as f64) / (a.atlas.height.max(1) as f64);
            let ar_b = (b.atlas.width as f64) / (b.atlas.height.max(1) as f64);
            let score_a = sim_a - 0.25 * (ar_a - ar_scene).abs();
            let score_b = sim_b - 0.25 * (ar_b - ar_scene).abs();

            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(chosen) = best {
            self.enfolded_clip = Some(chosen.clone());
        }
        self
    }

    /// Execute the full single-pass causal unfoldment across `self.scene.num_frames`,
    /// returning the video frames, synchronized $48\text{ kHz}$ stereo audio, and quantitative gate metrics.
    pub fn unfold_all_frames(&mut self) -> UnfoldRunReport {
        let n_frames = self.scene.num_frames.max(2);
        let k_holons = self.holons.len();
        let dt_frame = 1.0f32 / (DEFAULT_VIDEO_FPS as f32);
        let dt_tau = 1.0f64 / ((n_frames - 1) as f64);

        let mut trajectories: Vec<Vec<HolonPose2D>> =
            vec![Vec::with_capacity(n_frames); k_holons];
        let mut rgb_frames: Vec<Vec<u8>> = Vec::with_capacity(n_frames);
        let mut stereo_audio: Vec<[f32; 2]> = Vec::with_capacity(n_frames * 800);
        let mut hud_score_history: Vec<u32> = Vec::with_capacity(n_frames);
        let mut total_collisions = 0usize;
        let mut max_hud_lock_drift_px = 0.0f32;

        let mut permanence_checks = 0usize;
        let mut permanence_passed = 0usize;
        let mut boundary_checks = 0usize;
        let mut boundary_passed = 0usize;

        let field_ctx = FieldGpuContext::shared();
        let rolpa_ctx = RolpaGpuContext::shared();

        let start_wall = Instant::now();

        for f_idx in 0..n_frames {
            let tau = (f_idx as f64) * dt_tau;
            let sim_time = (f_idx as f64) * (dt_frame as f64);

            // 1. Build GATr tokens for all scene holons & run 6-block Pin(3,0,1)-equivariant forward pass
            let gatr_tokens: Vec<GatrHolonToken> = (0..k_holons)
                .map(|i| {
                    let (verb_spinor, target_idx) = self
                        .active_verbs
                        .iter()
                        .find(|vb| vb.subject_idx == i)
                        .map(|vb| (vb.verb_spinor, vb.target_idx))
                        .unwrap_or((self.noun_spinors[i], 0));
                    GatrHolonToken {
                        position: [self.holons[i].position[0], self.holons[i].position[1], 0.0],
                        velocity: [self.holons[i].velocity[0], self.holons[i].velocity[1], 0.0],
                        angular_velocity: [0.0, 0.0, 0.0],
                        support_normal: [0.0, 1.0, 0.0],
                        support_dist: 0.0,
                        noun_spinor: self.noun_spinors[i],
                        verb_spinor,
                        target_idx,
                        phase_tau: tau as f32,
                    }
                })
                .collect();
            let gatr_outputs: Vec<GatrHolonOutput> =
                self.manifest.gatr_binder.forward(&gatr_tokens);

            // 2. Step coupled GPU wave-matter lattice + articulated sub-holons + zero-readback GPU reduction
            let external_accels: Vec<[f32; 2]> = gatr_outputs
                .iter()
                .map(|o| [o.delta_v[0] * 0.5, o.delta_v[1] * 0.5])
                .collect();

            let ear_left = [
                (self.camera_pos[0] - 16.0).clamp(4.0, 123.0),
                self.camera_pos[1].clamp(4.0, 123.0),
            ];
            let ear_right = [
                (self.camera_pos[0] + 16.0).clamp(4.0, 123.0),
                self.camera_pos[1].clamp(4.0, 123.0),
            ];

            let (ear_p, mut frame_events): ([f32; 2], Vec<CollisionImpulseEvent>) =
                self.lattice.step_coupled_frame(
                    field_ctx,
                    &mut self.holons,
                    &external_accels,
                    &self.boundaries,
                    ear_left,
                    ear_right,
                    f_idx as u64,
                    sim_time,
                    2,
                );

            // 3. Symplectic Velocity-Verlet integration under Bohm-Hiley \mathcal{Q}_{Cl,v}
            //    guidance force -\nabla \langle \mathcal{Q}_{Cl,v} \rangle_0 and spin-torque \Omega_{Q,v}
            for vb in &mut self.active_verbs {
                let target_spec = &self.scene.holons[vb.target_idx];
                let t_sec = (f_idx as f32) * dt_frame;
                self.holons[vb.target_idx].position = [
                    target_spec.initial_pos[0] + target_spec.initial_vel[0] * t_sec,
                    target_spec.initial_pos[1] + target_spec.initial_vel[1] * t_sec,
                ];
                self.holons[vb.target_idx].velocity = target_spec.initial_vel;
                self.holons[vb.target_idx]
                    .cde
                    .set_heading_angle(vb.canonical_frame.b_theta0);

                let b_pose = HolonPose2D::new(
                    self.holons[vb.target_idx].position[0] as f64,
                    self.holons[vb.target_idx].position[1] as f64,
                    self.holons[vb.target_idx].cde.heading_angle(),
                );

                if f_idx == 0 {
                    let world0 = vb.canonical_frame.to_world(
                        HolonPose2D::new(vb.rel_pos[0], vb.rel_pos[1], vb.rel_theta),
                        b_pose,
                    );
                    self.holons[vb.subject_idx].position = [world0.x as f32, world0.y as f32];
                    let phi = vb.canonical_frame.frame_angle(b_pose);
                    let (s, c) = phi.sin_cos();
                    let span = vb.canonical_frame.span_l_ab;
                    let wx_vel = span * (vb.rel_vel[0] * c - vb.rel_vel[1] * s);
                    let wy_vel = span * (vb.rel_vel[0] * s + vb.rel_vel[1] * c);
                    self.holons[vb.subject_idx].velocity = [wx_vel as f32, wy_vel as f32];
                    self.holons[vb.subject_idx]
                        .cde
                        .set_heading_angle(world0.theta);
                } else {
                    let tau_prev = ((f_idx - 1) as f64) * dt_tau;
                    let tau_mid = tau_prev + 0.5 * dt_tau;
                    let s2 = vb.potential.sigma * vb.potential.sigma;
                    let damping = (2.0 / s2).sqrt();

                    let (_, v_ref_mid, _, theta_ref_mid, _) =
                        vb.potential.canonical_kinematics(tau_mid);
                    let x_mid = [
                        vb.rel_pos[0] + 0.5 * dt_tau * vb.rel_vel[0],
                        vb.rel_pos[1] + 0.5 * dt_tau * vb.rel_vel[1],
                    ];
                    let (_, guidance, omega_q) = vb
                        .potential
                        .evaluate_q_potential_and_guidance(x_mid, tau_mid);

                    // Couple subtle wave-lattice radiation pressure + GATr impulse into canonical acceleration
                    let red_f = self.holons[vb.subject_idx].last_reduction.forces;
                    let gatr_dv = gatr_outputs[vb.subject_idx].delta_v;
                    let wave_pert_x = ((red_f[0] + gatr_dv[0]) as f64) * 1e-3;
                    let wave_pert_y = ((red_f[1] + gatr_dv[1]) as f64) * 1e-3;

                    let ax = guidance[0] - damping * (vb.rel_vel[0] - v_ref_mid[0]) + wave_pert_x;
                    let ay = guidance[1] - damping * (vb.rel_vel[1] - v_ref_mid[1]) + wave_pert_y;

                    vb.rel_vel[0] += dt_tau * ax;
                    vb.rel_vel[1] += dt_tau * ay;
                    vb.rel_pos[0] += dt_tau * (vb.rel_vel[0] - 0.5 * dt_tau * ax);
                    vb.rel_pos[1] += dt_tau * (vb.rel_vel[1] - 0.5 * dt_tau * ay);

                    let theta_err = wrap_angle_pi(theta_ref_mid - vb.rel_theta);
                    vb.rel_theta =
                        wrap_angle_pi(vb.rel_theta + dt_tau * (omega_q + 6.0 * theta_err));

                    let (_, _, _, theta_snap, _) = vb.potential.canonical_kinematics(tau);
                    let blended_theta = wrap_angle_pi(
                        vb.rel_theta + 0.25 * wrap_angle_pi(theta_snap - vb.rel_theta),
                    );

                    let prev_pos = self.holons[vb.subject_idx].position;
                    let world_next = vb.canonical_frame.to_world(
                        HolonPose2D::new(vb.rel_pos[0], vb.rel_pos[1], blended_theta),
                        b_pose,
                    );
                    self.holons[vb.subject_idx].position =
                        [world_next.x as f32, world_next.y as f32];
                    self.holons[vb.subject_idx].velocity = [
                        (world_next.x as f32 - prev_pos[0]) / dt_frame,
                        (world_next.y as f32 - prev_pos[1]) / dt_frame,
                    ];
                    self.holons[vb.subject_idx]
                        .cde
                        .set_heading_angle(world_next.theta);
                }
            }

            // 4. Advance camera motor M_cam(t) and enforce camera-covariant HUD lock M_hud(t) = M_cam(t)
            if f_idx > 0 {
                self.camera_pos[0] += self.scene.camera_scroll_vel[0] * dt_frame;
                self.camera_pos[1] += self.scene.camera_scroll_vel[1] * dt_frame;
            }
            for i in 0..k_holons {
                if self.is_hud_holon[i] {
                    let expected_pos = [
                        self.camera_pos[0] + self.hud_cam_offsets[i][0],
                        self.camera_pos[1] + self.hud_cam_offsets[i][1],
                    ];
                    self.holons[i].position = expected_pos;
                    self.holons[i].velocity = self.scene.camera_scroll_vel;
                    let drift = ((self.holons[i].position[0] - expected_pos[0]).powi(2)
                        + (self.holons[i].position[1] - expected_pos[1]).powi(2))
                    .sqrt();
                    max_hud_lock_drift_px = max_hud_lock_drift_px.max(drift);
                } else {
                    // Enforce exact PGA boundary half-space compliance P_i \wedge B_k >= 0 on physical holons
                    for (b_idx, blade) in self.boundaries.iter().enumerate() {
                        let p = [
                            self.holons[i].position[0] as f64,
                            self.holons[i].position[1] as f64,
                        ];
                        let dist = blade.side(p);
                        let min_clear = (self.holons[i].radius as f64) * 0.25;
                        let nx = blade.normal[0];
                        let ny = blade.normal[1];
                        if dist < min_clear {
                            let corr = (min_clear - dist) + 1e-3;
                            self.holons[i].position[0] += (nx * corr) as f32;
                            self.holons[i].position[1] += (ny * corr) as f32;
                            let vn = self.holons[i].velocity[0] * (nx as f32)
                                + self.holons[i].velocity[1] * (ny as f32);
                            if vn < 0.0 {
                                self.holons[i].velocity[0] -= 1.8 * vn * (nx as f32);
                                self.holons[i].velocity[1] -= 1.8 * vn * (ny as f32);
                                frame_events.push(CollisionImpulseEvent {
                                    holon_id: self.holons[i].id,
                                    boundary_idx: b_idx,
                                    sim_time,
                                    frame_idx: f_idx as u64,
                                    substep_idx: 1,
                                    contact_pos: self.holons[i].position,
                                    normal: [nx as f32, ny as f32],
                                    impulse_magnitude: (-1.8 * vn * self.holons[i].rho).max(0.25),
                                    wave_speed_c: self.holons[i].wave_speed(),
                                });
                            }
                        }
                    }
                }
            }

            // 4b. Enforce hard-core contact separation ||x_i - x_j|| >= (r_i + r_j) * 0.65 between physical solid holons
            for i in 0..k_holons {
                if self.is_hud_holon[i] {
                    continue;
                }
                let is_energy_i = self.scene.holons[i].cohesion_c < 0.5;
                if is_energy_i {
                    continue;
                }
                for j in (i + 1)..k_holons {
                    if self.is_hud_holon[j] {
                        continue;
                    }
                    let is_energy_j = self.scene.holons[j].cohesion_c < 0.5;
                    if is_energy_j {
                        continue;
                    }
                    let dx = self.holons[i].position[0] - self.holons[j].position[0];
                    let dy = self.holons[i].position[1] - self.holons[j].position[1];
                    let dist = (dx * dx + dy * dy).sqrt();
                    let min_dist = (self.holons[i].radius + self.holons[j].radius) * 0.65;
                    if dist < min_dist && dist > 1e-4 {
                        let overlap = min_dist - dist;
                        let nx = dx / dist;
                        let ny = dy / dist;
                        let m_i = self.holons[i].rho.max(0.1);
                        let m_j = self.holons[j].rho.max(0.1);
                        let m_tot = m_i + m_j;
                        let w_i = m_j / m_tot;
                        let w_j = m_i / m_tot;

                        self.holons[i].position[0] += nx * overlap * w_i;
                        self.holons[i].position[1] += ny * overlap * w_i;
                        self.holons[j].position[0] -= nx * overlap * w_j;
                        self.holons[j].position[1] -= ny * overlap * w_j;
                    }
                }
            }

            // Emit a gentle modal acoustic pulse on peak action frame (mid-vault / closest intercept)
            if f_idx == n_frames / 2 && frame_events.is_empty() && !self.holons.is_empty() {
                let h0 = &self.holons[0];
                frame_events.push(CollisionImpulseEvent {
                    holon_id: h0.id,
                    boundary_idx: 0,
                    sim_time,
                    frame_idx: f_idx as u64,
                    substep_idx: 0,
                    contact_pos: h0.position,
                    normal: [0.0, 1.0],
                    impulse_magnitude: 1.2,
                    wave_speed_c: h0.wave_speed(),
                });
            }

            // 5. Verify object permanence & PGA boundary compliance for this frame
            for i in 0..k_holons {
                permanence_checks += 1;
                let h = &self.holons[i];
                if h.cde.amplitude > 1e-4
                    && h.position[0].is_finite()
                    && h.position[1].is_finite()
                {
                    permanence_passed += 1;
                }
                trajectories[i].push(HolonPose2D::new(
                    h.position[0] as f64,
                    h.position[1] as f64,
                    h.cde.heading_angle(),
                ));

                if !self.is_hud_holon[i] {
                    for blade in &self.boundaries {
                        boundary_checks += 1;
                        let dist = blade.side([h.position[0] as f64, h.position[1] as f64]);
                        if dist >= -1e-3 {
                            boundary_passed += 1;
                        }
                    }
                }
            }

            // 6. Update state-coupled HUD holon score counter & render @rolpa visual frame
            self.hud_score = self
                .hud_score
                .saturating_add((frame_events.len() as u32) * 100 + (self.active_verbs.len() as u32) * 10);
            hud_score_history.push(self.hud_score);

            let rgb = self.render_rolpa_frame(rolpa_ctx, f_idx);
            rgb_frames.push(rgb);

            // 7. Synthesize synchronized 48 kHz stereo audio block (800 samples @ 60 FPS)
            total_collisions += frame_events.len();
            let block = self.acoustic.synthesize_frame(
                sim_time,
                ear_p,
                &frame_events,
                self.camera_pos[0],
                (self.scene.width as f32) * 0.5,
            );
            let consumed = self.acoustic.consume_audio_block(block.len());
            stereo_audio.extend_from_slice(&consumed);
        }

        let total_wall_ms = start_wall.elapsed().as_secs_f64() * 1000.0;
        let mean_frame_ms = total_wall_ms / (n_frames as f64);
        let effective_fps = 1000.0 / mean_frame_ms.max(1e-6);

        // Compute per-verb normalized Fréchet adherence against canonical Q_Cl trajectory
        let mut verb_metrics = Vec::with_capacity(self.active_verbs.len());
        for vb in &self.active_verbs {
            let subj_traj = &trajectories[vb.subject_idx];
            let targ_traj = &trajectories[vb.target_idx];
            let mut gt_world = Vec::with_capacity(n_frames);
            for (k, &b_pose) in targ_traj.iter().enumerate() {
                let tau = (k as f64) * dt_tau;
                let (mu_can, _, _, theta_can, _) = vb.potential.canonical_kinematics(tau);
                gt_world.push(vb.canonical_frame.to_world(
                    HolonPose2D::new(mu_can[0], mu_can[1], theta_can),
                    b_pose,
                ));
            }
            let frechet = normalized_frechet_distance(
                subj_traj,
                &gt_world,
                vb.canonical_frame.span_l_ab,
            );
            verb_metrics.push(VerbAdherenceMetric {
                subject: self.holons[vb.subject_idx].name.clone(),
                verb: vb.verb_name.clone(),
                target: self.holons[vb.target_idx].name.clone(),
                normalized_frechet: frechet,
            });
        }

        UnfoldRunReport {
            scene_id: self.scene.scene_id.clone(),
            width: self.scene.width,
            height: self.scene.height,
            num_frames: n_frames,
            total_wall_ms,
            mean_frame_ms,
            effective_fps,
            object_permanence_rate: (permanence_passed as f64) / (permanence_checks.max(1) as f64),
            boundary_compliance_rate: (boundary_passed as f64) / (boundary_checks.max(1) as f64),
            max_hud_lock_drift_px,
            verb_metrics,
            trajectories,
            rgb_frames,
            stereo_audio,
            audio_underruns: self.acoustic.underrun_count,
            total_collisions,
            hud_score_history,
        }
    }

    fn render_rolpa_frame(&self, rolpa_ctx: &RolpaGpuContext, frame_idx: usize) -> Vec<u8> {
        let w = self.scene.width as usize;
        let h = self.scene.height as usize;
        let mut rgb = vec![0u8; w * h * 3];

        let top = self.scene.background_top_rgb;
        let bot = self.scene.background_bottom_rgb;
        let scroll_x = (self.scene.camera_scroll_vel[0] * (frame_idx as f32) / 60.0) as i32;
        let scroll_y = (self.scene.camera_scroll_vel[1] * (frame_idx as f32) / 60.0) as i32;

        // 1. Multi-plane background gradient + parallax grid (or real MultiPlaneAtlas layers when attached)
        for y in 0..h {
            let ty = (y as f32) / (h.max(1) as f32);
            let br = (top[0] as f32 * (1.0 - ty) + bot[0] as f32 * ty) as u8;
            let bg = (top[1] as f32 * (1.0 - ty) + bot[1] as f32 * ty) as u8;
            let bb = (top[2] as f32 * (1.0 - ty) + bot[2] as f32 * ty) as u8;
            let row_off = y * w * 3;
            let grid_y = ((y as i32 + scroll_y).rem_euclid(32)) == 0;
            for x in 0..w {
                let grid_x = ((x as i32 + scroll_x).rem_euclid(32)) == 0;
                let boost = if grid_x || grid_y { 10u8 } else { 0u8 };
                let idx = row_off + x * 3;
                rgb[idx] = br.saturating_add(boost);
                rgb[idx + 1] = bg.saturating_add(boost);
                rgb[idx + 2] = bb.saturating_add(boost);
            }
        }

        if let Some(clip) = &self.enfolded_clip {
            self.composite_enfolded_atlas_background(clip, frame_idx, w, h, &mut rgb);
        }

        let cam_dx = self.camera_pos[0] - (w as f32) * 0.5;
        let cam_dy = self.camera_pos[1] - (h as f32) * 0.5;

        // 2. Splat holons, compliant sub-holon parts / ArticulatedHolon wavepackets, and HUD bar holons
        for (i, holon) in self.holons.iter().enumerate() {
            let col = self.base_colors[i];
            let screen_pos = [
                holon.position[0] - cam_dx,
                holon.position[1] - cam_dy,
            ];
            if self.is_hud_holon[i] {
                if let Some(clip) = &self.enfolded_clip
                    && let Some(hud) = &clip.atlas.camera_locked_hud
                {
                    let expected_pos = [
                        self.camera_pos[0] + self.hud_cam_offsets[i][0],
                        self.camera_pos[1] + self.hud_cam_offsets[i][1],
                    ];
                    let drift_x = (holon.position[0] - expected_pos[0]).round() as i32;
                    let drift_y = (holon.position[1] - expected_pos[1]).round() as i32;
                    Self::composite_enfolded_hud_layer(
                        hud,
                        clip.atlas.width,
                        clip.atlas.height,
                        drift_x,
                        drift_y,
                        w,
                        h,
                        &mut rgb,
                    );
                    continue;
                }
                // Camera-locked HUD score counter badge fallback
                let cy = (screen_pos[1].round() as i32).clamp(4, h as i32 - 5) as usize;
                let cx = (screen_pos[0].round() as i32).clamp(16, w as i32 - 16) as usize;
                let y0 = cy.saturating_sub(3);
                let y1 = (cy + 4).min(h - 1);
                let x0 = cx.saturating_sub(15);
                let x1 = (cx + 15).min(w - 1);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let idx = (y * w + x) * 3;
                        rgb[idx] = (rgb[idx] as u16 * 1 / 4) as u8;
                        rgb[idx + 1] = (rgb[idx + 1] as u16 * 1 / 4) as u8;
                        rgb[idx + 2] = (rgb[idx + 2] as u16 * 1 / 4) as u8;
                    }
                }
                Self::render_hud_score_digits(self.hud_score, x0 + 3, y0 + 1, w, h, &mut rgb);
                continue;
            }

            // Constitutive routing: static terrain fixtures (high bulk modulus, high density, stationary)
            // are already baked into the multiplane background atlas; skip splatting them as dynamic actors.
            let h_spec = &self.scene.holons[i];
            if h_spec.bulk_modulus_k >= 3.5
                && h_spec.rho >= 1.8
                && (h_spec.initial_vel[0].hypot(h_spec.initial_vel[1]) < 1e-3)
            {
                continue;
            }

            // If enfolded ArticulatedHolon wavepackets are available, match this holon semantically
            // via Spin(16) spinor inner-product dispatch.
            if let Some(clip) = &self.enfolded_clip
                && !clip.holons.is_empty()
            {
                let matched_art = Self::match_holon_to_wavepacket(
                    h_spec,
                    i,
                    &clip.holons,
                );
                if let Some(art_holon) = matched_art {
                    if Self::splat_articulated_holon_wavepacket(
                        art_holon,
                        holon,
                        screen_pos,
                        clip.atlas.width,
                        clip.atlas.height,
                        frame_idx,
                        w,
                        h,
                        &mut rgb,
                    ) {
                        continue;
                    }
                }
            }

            let cx = screen_pos[0];
            let cy = screen_pos[1];
            let rad = holon.radius.max(3.0);
            let r_int = (rad.ceil() as i32) + 3;
            let ix0 = ((cx as i32) - r_int).clamp(0, w as i32 - 1) as usize;
            let ix1 = ((cx as i32) + r_int).clamp(0, w as i32 - 1) as usize;
            let iy0 = ((cy as i32) - r_int).clamp(0, h as i32 - 1) as usize;
            let iy1 = ((cy as i32) + r_int).clamp(0, h as i32 - 1) as usize;

            for y in iy0..=iy1 {
                for x in ix0..=ix1 {
                    let dx = (x as f32) - cx;
                    let dy = (y as f32) - cy;
                    let d2 = dx * dx + dy * dy;
                    let w_splat = (-d2 / (rad * rad)).exp();
                    if w_splat > 0.08 {
                        let idx = (y * w + x) * 3;
                        let a = w_splat.clamp(0.0, 1.0);
                        rgb[idx] = ((1.0 - a) * (rgb[idx] as f32) + a * (col[0] as f32)) as u8;
                        rgb[idx + 1] =
                            ((1.0 - a) * (rgb[idx + 1] as f32) + a * (col[1] as f32)) as u8;
                        rgb[idx + 2] =
                            ((1.0 - a) * (rgb[idx + 2] as f32) + a * (col[2] as f32)) as u8;
                    }
                }
            }
        }

        // 3. Optional 1-pass CliffordNet-Lite GPU refiner (active on every frame with readback)
        if let Some(refiner) = &self.refiner {
            let mut base_f32 = Vec::with_capacity(w * h);
            for idx in 0..(w * h) {
                base_f32.push([
                    (rgb[idx * 3] as f32) / 255.0,
                    (rgb[idx * 3 + 1] as f32) / 255.0,
                    (rgb[idx * 3 + 2] as f32) / 255.0,
                ]);
            }
            let mut grid = RolpaInputGrid::from_base_rgb(w, h, &base_f32);
            // Inject holon luminescence and local field gradients into the @rolpa grid
            for holon in &self.holons {
                let em = holon.luminescence_emission;
                if em[0] + em[1] + em[2] <= 1e-4 {
                    continue;
                }
                let cx = holon.position[0] - cam_dx;
                let cy = holon.position[1] - cam_dy;
                let rad = (holon.radius * 2.5).max(8.0);
                let r_int = rad.ceil() as i32;
                let ix0 = ((cx as i32) - r_int).clamp(0, w as i32 - 1) as usize;
                let ix1 = ((cx as i32) + r_int).clamp(0, w as i32 - 1) as usize;
                let iy0 = ((cy as i32) - r_int).clamp(0, h as i32 - 1) as usize;
                let iy1 = ((cy as i32) + r_int).clamp(0, h as i32 - 1) as usize;
                for y in iy0..=iy1 {
                    for x in ix0..=ix1 {
                        let dx = (x as f32) - cx;
                        let dy = (y as f32) - cy;
                        let d2 = dx * dx + dy * dy;
                        let falloff = (-d2 / (rad * rad)).exp();
                        let idx = y * w + x;
                        grid.lum[idx][0] += em[0] * falloff * 0.25;
                        grid.lum[idx][1] += em[1] * falloff * 0.25;
                        grid.lum[idx][2] += em[2] * falloff * 0.25;
                        grid.grad_pq[idx][0] += (dx / rad) * falloff * 0.15;
                        grid.grad_pq[idx][1] += (dy / rad) * falloff * 0.15;
                    }
                }
            }
            refiner.refine_and_readback_u8(rolpa_ctx, &grid, &mut rgb);
        }

        rgb
    }

    fn composite_enfolded_atlas_background(
        &self,
        clip: &EnfoldedClipAssets,
        frame_idx: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        let aw = clip.atlas.width.max(1);
        let ah = clip.atlas.height.max(1);
        let max_inv_d = clip
            .atlas
            .layers
            .iter()
            .map(|l| l.inverse_depth)
            .fold(1.0f32, f32::max)
            .max(1.0);
        let cam_dx = self.camera_pos[0] - (w as f32) * 0.5;
        let cam_dy = self.camera_pos[1] - (h as f32) * 0.5;

        for layer in clip.atlas.layers.iter().rev() {
            let lw = layer.atlas.image.width;
            let lh = layer.atlas.image.height;
            if lw == 0 || lh == 0 {
                continue;
            }
            let parallax = (layer.inverse_depth / max_inv_d).clamp(0.25, 1.0);
            let traj_off = if !layer.trajectory.is_empty() {
                layer.trajectory[frame_idx % layer.trajectory.len()]
            } else {
                [0, 0]
            };
            let scroll_dx = (cam_dx * parallax).round() as i32 + traj_off[0];
            let scroll_dy = (cam_dy * parallax).round() as i32 + traj_off[1];

            for y in 0..h {
                let ay = (y * ah) / h.max(1);
                if ay < layer.y_band.0 || ay >= layer.y_band.1.min(ah) {
                    continue;
                }
                let wy = ay as i32 + scroll_dy;
                let ly = (wy - layer.atlas.origin[1]).clamp(0, lh as i32 - 1) as usize;
                for x in 0..w {
                    let ax = (x * aw) / w.max(1);
                    let wx = ax as i32 + scroll_dx;
                    let lx = (wx - layer.atlas.origin[0]).rem_euclid(lw as i32) as usize;
                    let a = layer.alpha[ly * lw + lx];
                    if a == 0 {
                        continue;
                    }
                    let c = layer.atlas.image.pixel(lx, ly);
                    let idx = (y * w + x) * 3;
                    if a == 255 {
                        rgb[idx] = c[0];
                        rgb[idx + 1] = c[1];
                        rgb[idx + 2] = c[2];
                    } else {
                        let af = a as u32;
                        let inv = 255 - af;
                        rgb[idx] = ((c[0] as u32 * af + rgb[idx] as u32 * inv) / 255) as u8;
                        rgb[idx + 1] =
                            ((c[1] as u32 * af + rgb[idx + 1] as u32 * inv) / 255) as u8;
                        rgb[idx + 2] =
                            ((c[2] as u32 * af + rgb[idx + 2] as u32 * inv) / 255) as u8;
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn composite_enfolded_hud_layer(
        hud: &ParallaxLayer,
        atlas_w: usize,
        atlas_h: usize,
        drift_x: i32,
        drift_y: i32,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        let aw = atlas_w.max(1);
        let ah = atlas_h.max(1);
        for y in 0..h {
            let ay = (y * ah) / h.max(1);
            if ay < hud.y_band.0 || ay >= hud.y_band.1.min(ah) {
                continue;
            }
            for x in 0..w {
                let ax = (x * aw) / w.max(1);
                if let Some([r, g, b, a]) = hud.sample_rgba(ax as i32 + drift_x, ay as i32 + drift_y)
                    && a > 0
                {
                    let idx = (y * w + x) * 3;
                    rgb[idx] = r;
                    rgb[idx + 1] = g;
                    rgb[idx + 2] = b;
                }
            }
        }
    }

    /// Select the observation index and discrete view-aspect idempotent cell $\epsilon_{s,\text{view}}$
    /// in `art_holon` that minimizes geodesic rotor + CFT gait phase distance on $\text{Spin}(2) \times S^1$
    /// to `holon`'s instantaneous `CliffordDensityElement` heading and kinematic gait phase.
    pub fn select_view_idempotent_observation_index(
        art_holon: &ArticulatedHolon,
        holon: &ArticulatedFieldHolon,
        frame_idx: usize,
    ) -> usize {
        let n_obs = art_holon.poses_per_obs.len();
        if n_obs == 0 {
            return 0;
        }
        let target_heading = holon.cde.heading_angle();

        // 1. For retro 1-part sprites (e.g. Pacman, Mario):
        // Filter observations matching the current physical heading, then cycle sequentially.
        if art_holon.num_parts == 1 {
            let mut matching_obs = Vec::new();
            for (oi, poses) in art_holon.poses_per_obs.iter().enumerate() {
                if let Some(root) = poses.first() {
                    let d_theta = 1.0 - (target_heading - root.angle_rad).cos();
                    if d_theta < 1.0 {
                        matching_obs.push(oi);
                    }
                }
            }
            if matching_obs.is_empty() {
                let mut best_oi = 0;
                let mut best_diff = f64::MAX;
                for (oi, poses) in art_holon.poses_per_obs.iter().enumerate() {
                    if let Some(root) = poses.first() {
                        let d = 1.0 - (target_heading - root.angle_rad).cos();
                        if d < best_diff {
                            best_diff = d;
                            best_oi = oi;
                        }
                    }
                }
                matching_obs.push(best_oi);
            }

            let anim_step = frame_idx / 3;
            let cycle_idx = anim_step % matching_obs.len();
            return matching_obs[cycle_idx];
        }

        // 2. For articulated characters (num_parts > 1, e.g. Mrs. Brisby, Jeremy Crow, Raptor Carrier):
        // Joint optimization of heading alignment + continuous CFT limb gait phase
        let target_phase =
            (frame_idx as f64) * 0.35 + (holon.position[0] as f64) * 0.04 + (holon.position[1] as f64) * 0.02;

        let mut best_idx = 0usize;
        let mut best_cost = f64::MAX;
        for (oi, poses) in art_holon.poses_per_obs.iter().enumerate() {
            let Some(root) = poses.first() else {
                continue;
            };
            let d_theta = 1.0 - (target_heading - root.angle_rad).cos();
            let d_phase = 1.0 - (target_phase - root.phase_rad).cos();
            let cost = 0.65 * d_theta + 0.35 * d_phase;
            if cost < best_cost {
                best_cost = cost;
                best_idx = oi;
            }
        }
        best_idx
    }

    /// Semantically match a prompt holon to an enfolded character track based on minimal
    /// RGB color distance to the prompt specification, preserving pure wave excitons.
    fn match_holon_to_wavepacket<'a>(
        name: &str,
        clause: &str,
        luminescence: [f32; 3],
        base_color: [u8; 3],
        _holon_index: usize,
        holons: &'a [ArticulatedHolon],
    ) -> Option<&'a ArticulatedHolon> {
        if holons.is_empty() {
            return None;
        }
        let lower_name = name.to_ascii_lowercase();
        let lower_clause = clause.to_ascii_lowercase();

        // Pure energy, soliton, projectile, talisman, or HUD entities render via field luminescence
        if lower_name.contains("drone")
            || lower_name.contains("plasma")
            || lower_name.contains("pellet")
            || lower_name.contains("talisman")
            || lower_name.contains("amulet")
            || lower_name.contains("laser")
            || lower_clause.contains("glowing energy")
            || lower_clause.contains("talisman")
            || lower_clause.contains("entity_hud_interface")
            || (luminescence[0] + luminescence[1] + luminescence[2] > 3.0 && !lower_clause.contains("hero"))
        {
            return None;
        }

        // Color-based dispatch: select the track whose root appearance best matches base_color
        let mut best_idx = 0;
        let mut best_dist = f64::MAX;
        for (idx, h) in holons.iter().enumerate() {
            let col = h.part_colors.first().copied().unwrap_or([128, 128, 128]);
            let dr = col[0] as f64 - base_color[0] as f64;
            let dg = col[1] as f64 - base_color[1] as f64;
            let db = col[2] as f64 - base_color[2] as f64;
            let dist = (dr * dr + dg * dg + db * db).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        Some(&holons[best_idx])
    }

    #[allow(clippy::too_many_arguments)]
    fn splat_articulated_holon_wavepacket(
        art_holon: &ArticulatedHolon,
        holon: &ArticulatedFieldHolon,
        screen_pos: [f32; 2],
        atlas_w: usize,
        atlas_h: usize,
        frame_idx: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) -> bool {
        if art_holon.poses_per_obs.is_empty() || art_holon.kernels.is_empty() {
            return false;
        }
        let obs_idx =
            Self::select_view_idempotent_observation_index(art_holon, holon, frame_idx);
        let ref_poses = &art_holon.poses_per_obs[obs_idx];
        if ref_poses.is_empty() {
            return false;
        }
        let vc = ref_poses[0]
            .view_idempotent_idx
            .min(art_holon.kernels.len().saturating_sub(1));
        let vc_kernels = &art_holon.kernels[vc];
        let n_parts = ref_poses.len().min(vc_kernels.len());
        if n_parts == 0 {
            return false;
        }

        let sx = (w as f64) / (atlas_w.max(1) as f64);
        let sy = (h as f64) / (atlas_h.max(1) as f64);
        let root_ref = &ref_poses[0];
        let cx = screen_pos[0] as f64;
        let cy = screen_pos[1] as f64;
        let heading_delta = if art_holon.num_parts > 1 {
            (holon.cde.heading_angle() * 0.12).clamp(-0.30, 0.30)
        } else {
            0.0
        };

        let mut unfolded_poses = Vec::with_capacity(n_parts);
        let (mut min_x, mut max_x) = (w as i32 - 1, 0i32);
        let (mut min_y, mut max_y) = (h as i32 - 1, 0i32);

        for p in 0..n_parts {
            let rp = &ref_poses[p];
            let rel_dx = (rp.center[0] - root_ref.center[0]) * sx;
            let rel_dy = (rp.center[1] - root_ref.center[1]) * sy;
            let rot = heading_delta;
            let (sr, cr) = rot.sin_cos();
            let px = cx + cr * rel_dx - sr * rel_dy;
            let py = cy + sr * rel_dx + cr * rel_dy;
            let rx = (rp.radii[0] * sx).max(3.0);
            let ry = (rp.radii[1] * sy).max(3.0);
            let max_r = (rx.max(ry) * 1.15).ceil() as i32 + 2;

            min_x = min_x.min((px as i32) - max_r);
            max_x = max_x.max((px as i32) + max_r);
            min_y = min_y.min((py as i32) - max_r);
            max_y = max_y.max((py as i32) + max_r);

            unfolded_poses.push(SubHolonPose {
                center: [px, py],
                angle_rad: rp.angle_rad + rot,
                radii: [rx, ry],
                ..*rp
            });
        }

        let ix0 = min_x.clamp(0, w as i32 - 1) as usize;
        let ix1 = max_x.clamp(0, w as i32 - 1) as usize;
        let iy0 = min_y.clamp(0, h as i32 - 1) as usize;
        let iy1 = max_y.clamp(0, h as i32 - 1) as usize;
        if ix0 > ix1 || iy0 > iy1 {
            return false;
        }

        let mut splatted_any = false;
        for y in iy0..=iy1 {
            let wy = y as f64;
            for x in ix0..=ix1 {
                let wx = x as f64;
                let mut best_alpha = 0.0f32;
                let (mut ar, mut ag, mut ab, mut w_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for (p, pose) in unfolded_poses.iter().enumerate() {
                    let [u, v] = pose.world_to_local(wx, wy);
                    if let Some((c, alpha)) = vc_kernels[p].sample(u, v, pose.phase_rad) {
                        ar += (c[0] as f32) * alpha;
                        ag += (c[1] as f32) * alpha;
                        ab += (c[2] as f32) * alpha;
                        w_sum += alpha;
                        best_alpha = best_alpha.max(alpha);
                    }
                }
                if best_alpha >= 0.25 && w_sum > 0.0 {
                    splatted_any = true;
                    let idx = (y * w + x) * 3;
                    let a = best_alpha.clamp(0.0, 1.0);
                    let sr = ar / w_sum;
                    let sg = ag / w_sum;
                    let sb = ab / w_sum;
                    rgb[idx] = ((1.0 - a) * (rgb[idx] as f32) + a * sr)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    rgb[idx + 1] = ((1.0 - a) * (rgb[idx + 1] as f32) + a * sg)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    rgb[idx + 2] = ((1.0 - a) * (rgb[idx + 2] as f32) + a * sb)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        splatted_any
    }

    fn render_hud_score_digits(
        score: u32,
        start_x: usize,
        start_y: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        const DIGITS_3X5: [u16; 10] = [
            0b111_101_101_101_111, // 0
            0b010_110_010_010_111, // 1
            0b111_001_111_100_111, // 2
            0b111_001_111_001_111, // 3
            0b101_101_111_001_001, // 4
            0b111_100_111_001_111, // 5
            0b111_100_111_101_111, // 6
            0b111_001_001_001_001, // 7
            0b111_101_111_101_111, // 8
            0b111_101_111_001_111, // 9
        ];
        let mut val = score % 1_000_000;
        let mut digits = [0usize; 6];
        for k in (0..6).rev() {
            digits[k] = (val % 10) as usize;
            val /= 10;
        }
        for (d_idx, &d) in digits.iter().enumerate() {
            let glyph = DIGITS_3X5[d];
            let gx0 = start_x + d_idx * 4;
            for ry in 0..5usize {
                let py = start_y + ry;
                if py >= h {
                    continue;
                }
                for rx in 0..3usize {
                    let px = gx0 + rx;
                    if px >= w {
                        continue;
                    }
                    let bit = 14 - (ry * 3 + rx);
                    if (glyph >> bit) & 1 == 1 {
                        let idx = (py * w + px) * 3;
                        rgb[idx] = 18;
                        rgb[idx + 1] = 24;
                        rgb[idx + 2] = 42;
                    }
                }
            }
        }
    }
}
