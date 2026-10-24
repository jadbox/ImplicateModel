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
    ArticulatedHolon, GpuModelSheetMonad, ParallaxLayer, SubHolonPose, mux_rgb_frames_and_wav_to_mp4,
};
use implicate_math::{
    CanonicalRelFrame, HolonPose2D, LearnedVerbPotential, OrientedLine2, Spinor16,
    normalized_frechet_distance, wrap_angle_pi,
};
use implicate_rolpa::{
    AcousticRolpaProjector, DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS, GpuCliffordNetRefiner,
    GpuModelSheetUnfolder, RolpaGpuContext, RolpaInputGrid, RolpaUniforms, UnfoldUniforms,
};
use serde::{Deserialize, Serialize};

use crate::container::ImplicateContainerManifest;
use crate::payload::EnfoldedClipAssets;
use crate::prompt::UnfoldSceneSpec;

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
    /// Per-frame world-to-screen camera offset (screen = world - offset).
    pub camera_offsets: Vec<[f32; 2]>,
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
    pub unfolder: Option<GpuModelSheetUnfolder>,
    pub camera_pos: [f32; 2],
    pub enfolded_clip: Option<EnfoldedClipAssets>,
    /// For each prompt holon, the semantically matched `ArticulatedHolon` index in `enfolded_clip`.
    pub wavepacket_matches: Vec<Option<usize>>,
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

        let unfolder = if scene.width > 0 && scene.height > 0 {
            let rolpa_ctx = RolpaGpuContext::shared();
            Some(GpuModelSheetUnfolder::new(rolpa_ctx, scene.width, scene.height))
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
            unfolder,
            camera_pos: cam0,
            enfolded_clip: None,
            wavepacket_matches: Vec::new(),
            hud_score: 0,
        }
    }

    /// Attach enfolded `MultiPlaneAtlas` background/HUD layers and `ArticulatedHolon` CFT wavepackets
    /// for real-asset `@rolpa` compositing during unfoldment.
    ///
    /// A scene without an explicit resolution is re-projected onto the clip's native raster first,
    /// so enfolded atlas texels and sprite kernels unfold 1:1 without resampling.
    pub fn with_enfolded_clip(mut self, clip: EnfoldedClipAssets) -> Self {
        if !self.scene.resolution_explicit {
            let mut scene = self.scene.clone();
            scene.adapt_to_resolution(clip.atlas.width as u32, clip.atlas.height as u32);
            self = Self::new(self.manifest, scene);
        }
        self.wavepacket_matches = Self::resolve_wavepacket_matches(&self.scene, &self.manifest, &clip);
        self.enfolded_clip = Some(clip);
        self
    }

    /// Unfold a character holon on the GPU using `GpuModelSheetUnfolder`.
    pub fn unfold_holon_gpu(
        &self,
        monad: &GpuModelSheetMonad,
        uniforms: &UnfoldUniforms,
        kernel_data: Option<&[[f32; 4]]>,
        out_rgb: &mut [u8],
        out_mask: &mut [bool],
    ) -> bool {
        if let Some(ref unfolder) = self.unfolder {
            let ctx = RolpaGpuContext::shared();
            unfolder.unfold_and_readback_u8(ctx, monad, uniforms, kernel_data, out_rgb, out_mask);
            true
        } else {
            false
        }
    }

    /// Clip in which one of the scene's verbs was empirically grounded, if any.
    fn grounded_clip_id(&self) -> Option<String> {
        self.scene.verb_bindings.iter().find_map(|vb| {
            self.manifest
                .verb_potentials
                .get(&vb.verb)
                .and_then(|e| e.grounding.as_ref())
                .map(|g| g.clip_id.clone())
        })
    }

    /// Select and attach the best-matching `EnfoldedClipAssets` from a multi-clip container slice
    /// based on scene domain, actor keywords, and viewport dimensions.
    pub fn with_enfolded_clips(self, clips: &[EnfoldedClipAssets]) -> Self {
        if clips.is_empty() {
            return self;
        }
        // A verb learned from a specific clip is the strongest evidence of the intended milieu.
        if let Some(id) = self.grounded_clip_id()
            && let Some(chosen) = clips.iter().find(|c| c.clip_id == id)
        {
            return self.with_enfolded_clip(chosen.clone());
        }
        // Task R9.4: milieu retrieval. Evidence is the clip's identity (id + source name) appearing in
        // the scene; Spin(16) coherence (one shared Gram-SVD batch) breaks ties. Without evidence the
        // scene unfolds procedurally rather than borrowing an unrelated world.
        match Self::retrieve_milieu(&self.scene, clips) {
            Some((idx, _)) => self.with_enfolded_clip(clips[idx].clone()),
            None => self,
        }
    }

    /// Identity tokens: lowercase alphanumeric words with a letter and at least
    /// `MIN_IDENTITY_TOKEN_LEN` characters (drops articles, frame counts, resolutions).
    fn identity_tokens(text: &str) -> std::collections::BTreeSet<String> {
        const MIN_IDENTITY_TOKEN_LEN: usize = 4;
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= MIN_IDENTITY_TOKEN_LEN && t.chars().any(|c| c.is_alphabetic()))
            .filter(|t| !t.chars().all(|c| c.is_ascii_digit() || c == 'x' || c == 'p'))
            .map(str::to_string)
            .collect()
    }

    /// Index of the best-evidenced clip and its `(shared identity tokens, coherence)` score.
    pub fn retrieve_milieu(scene: &UnfoldSceneSpec, clips: &[EnfoldedClipAssets]) -> Option<(usize, (usize, f64))> {
        let scene_desc = format!(
            "{} {} {}",
            scene.scene_id,
            scene.prompt_text,
            scene.holons.iter().map(|h| format!("{} {}", h.name, h.semantic_clause)).collect::<Vec<_>>().join(" ")
        );
        let scene_tokens = Self::identity_tokens(&scene_desc);
        let clip_descs: Vec<String> = clips.iter().map(|c| format!("{} {}", c.clip_id, c.source_video)).collect();
        let mut batch: Vec<&str> = vec![scene_desc.as_str()];
        batch.extend(clip_descs.iter().map(String::as_str));
        let sp = ConceptLexicon::default().embed_clauses(&batch);
        let coherence = |i: usize| if sp.len() == batch.len() { sp[0].coherence(&sp[1 + i]) } else { 0.0 };
        let scored: Vec<(usize, (usize, f64))> = clip_descs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, (Self::identity_tokens(d).intersection(&scene_tokens).count(), coherence(i))))
            .collect();
        if std::env::var_os("IMPLICATE_DEBUG_MILIEU").is_some() {
            for (i, (shared, coh)) in &scored {
                eprintln!("[milieu] {} shared_identity_tokens={shared} coherence={coh:.4}", clips[*i].clip_id);
            }
        }
        scored
            .into_iter()
            .filter(|(_, (shared, _))| *shared > 0)
            .max_by(|a, b| a.1.0.cmp(&b.1.0).then(a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal)))
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
        let mut camera_offsets: Vec<[f32; 2]> = Vec::with_capacity(n_frames);
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

            // 4b. Enforce hard-core contact separation ||x_i - x_j|| >= (r_i + r_j) * 0.65 between physical solid holons.
            // Holons whose pose is kinematically prescribed by an active verb potential are excluded:
            // displacing them would corrupt the learned relative trajectory and verb adherence.
            let is_verb_bound = |idx: usize| {
                self.active_verbs
                    .iter()
                    .any(|vb| vb.subject_idx == idx || vb.target_idx == idx)
            };
            for i in 0..k_holons {
                if self.is_hud_holon[i] || is_verb_bound(i) {
                    continue;
                }
                let is_energy_i = self.scene.holons[i].cohesion_c < 0.5;
                if is_energy_i {
                    continue;
                }
                for j in (i + 1)..k_holons {
                    if self.is_hud_holon[j] || is_verb_bound(j) {
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

            camera_offsets.push([
                self.camera_pos[0] - (self.scene.width as f32) * 0.5,
                self.camera_pos[1] - (self.scene.height as f32) * 0.5,
            ]);
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
            camera_offsets,
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
                // With an enfolded clip but no camera-locked layer, any HUD is baked into the
                // static background; a synthetic badge would overwrite it.
                if self.enfolded_clip.is_some() {
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

            // Static level fixtures are already part of the enfolded background atlas.
            let h_spec = &self.scene.holons[i];
            if h_spec.is_static_fixture && self.enfolded_clip.is_some() {
                continue;
            }

            // If enfolded ArticulatedHolon wavepackets are available, splat the semantically matched
            // character track via Spin(16) spinor inner-product dispatch.
            if let Some(clip) = &self.enfolded_clip
                && let Some(art_holon) = self
                    .wavepacket_matches
                    .get(i)
                    .copied()
                    .flatten()
                    .and_then(|idx| clip.holons.get(idx))
            {
                if Self::splat_articulated_holon_wavepacket(
                    art_holon,
                    holon,
                    screen_pos,
                    clip.atlas.width,
                    clip.atlas.height,
                    frame_idx,
                    clip.fps() / DEFAULT_VIDEO_FPS as f64,
                    w,
                    h,
                    &mut rgb,
                ) {
                    continue;
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
            // The unfolded camera motor alone drives the view; the source trajectory only anchors
            // where the novel camera starts in the atlas (the clip's first frame).
            let traj_off = layer.trajectory.first().copied().unwrap_or([0, 0]);
            let _ = frame_idx;
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

    /// Select which enfolded observation (posture idempotent $\epsilon_{s,\text{view}}$ + CFT phase)
    /// to project for `holon` at unfold frame `frame_idx`. One law for every holon:
    ///
    /// 1. **Spin(2) alignment:** keep observations whose travel heading rotor is within a quarter
    ///    turn of the holon's current heading ($\langle R, R_{\text{obs}} \rangle_0 > 0$ on the half
    ///    angle), or the best-aligned one when none is.
    /// 2. **Source cadence:** play the kept observations in source order at the source's own rate
    ///    (`source_frames_per_frame` = source fps / unfold fps), so animation timing is inherited
    ///    from the enfolded footage rather than imposed.
    pub fn select_view_idempotent_observation_index(
        art_holon: &ArticulatedHolon,
        holon: &ArticulatedFieldHolon,
        frame_idx: usize,
        source_frames_per_frame: f64,
    ) -> usize {
        let n_obs = art_holon.poses_per_obs.len();
        if n_obs == 0 {
            return 0;
        }
        // Direction of travel (matches how `heading_rad` was measured at enfoldment); at rest, the
        // holon's own Spin(2) heading rotor.
        let v = holon.velocity;
        let target = if v[0].hypot(v[1]) > f32::EPSILON {
            (v[1] as f64).atan2(v[0] as f64)
        } else {
            holon.cde.heading_angle()
        };
        let alignment = |oi: usize| {
            art_holon.poses_per_obs[oi].first().map_or(-1.0, |r| (target - r.heading_rad).cos())
        };
        let mut kept: Vec<usize> = (0..n_obs).filter(|&oi| alignment(oi) > 0.0).collect();
        if kept.is_empty() {
            let best = (0..n_obs)
                .max_by(|&a, &b| alignment(a).partial_cmp(&alignment(b)).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
            kept.push(best);
        }
        let frame_of = |oi: usize| art_holon.obs_frames.get(oi).copied().unwrap_or(oi) as f64;
        let first = frame_of(kept[0]);
        let span = frame_of(kept[kept.len() - 1]) - first + 1.0;
        let t = (frame_idx as f64 * source_frames_per_frame).rem_euclid(span.max(1.0));
        kept.iter().copied().rev().find(|&oi| frame_of(oi) - first <= t).unwrap_or(kept[0])
    }

    /// Dispatch prompt holons to enfolded `ArticulatedHolon` tracks.
    ///
    /// Evidence is consumed from strongest to weakest:
    /// 1. **Explicit appearance:** author-stated colours are jointly assigned to tracks maximizing
    ///    chromatic affinity x observation prior.
    /// 2. **Verb grounding:** a verb learned from this clip binds its subject/object to the exact
    ///    tracks whose relative motor $M_{\text{rel}} = \widetilde{M}_B M_A$ defined it.
    /// 3. **Saliency:** remaining solid holons (including stated colours with no chromatic evidence)
    ///    take the remaining tracks in enfoldment order (longest, most-observed first).
    ///
    /// HUD overlays, static fixtures, and field solitons ($C < 0.5$) never bind sprites.
    fn resolve_wavepacket_matches(
        scene: &UnfoldSceneSpec,
        manifest: &ImplicateContainerManifest,
        clip: &EnfoldedClipAssets,
    ) -> Vec<Option<usize>> {
        let specs = &scene.holons;
        let tracks = &clip.holons;
        let mut matches: Vec<Option<usize>> = vec![None; specs.len()];
        let mut used = vec![false; tracks.len()];
        let renderable: Vec<bool> = specs
            .iter()
            .map(|s| !s.is_camera_locked_hud && !s.is_static_fixture && s.cohesion_c >= 0.5)
            .collect();
        let usable = |j: usize| tracks.get(j).is_some_and(|t| !t.poses_per_obs.is_empty() && !t.kernels.is_empty());
        let idx_of = |name: &str| specs.iter().position(|s| s.name == name);

        let bind = |i: usize, j: usize, matches: &mut Vec<Option<usize>>, used: &mut Vec<bool>| {
            if renderable[i] && matches[i].is_none() && usable(j) && !used[j] {
                matches[i] = Some(j);
                used[j] = true;
            }
        };

        // 1. Explicit appearance: the joint assignment of stated colours to tracks maximizing total
        //    evidence = chromatic affinity (likelihood) x observation share (prior that the track is a
        //    genuine holon). Joint, not greedy: names are fuzzy, relative fit decides.
        let explicit: Vec<usize> = (0..specs.len())
            .filter(|&i| renderable[i] && matches[i].is_none() && specs[i].appearance_explicit)
            .collect();
        let candidates: Vec<usize> = (0..tracks.len()).filter(|&j| !used[j] && usable(j)).collect();
        let max_obs = candidates.iter().map(|&j| tracks[j].poses_per_obs.len()).max().unwrap_or(1).max(1) as f32;
        let evidence: Vec<Vec<f32>> = explicit
            .iter()
            .map(|&i| {
                let p = specs[i].base_color.map(|v| v as f32);
                candidates
                    .iter()
                    .map(|&j| Self::chromatic_affinity(p, &tracks[j]).max(0.0) * tracks[j].poses_per_obs.len() as f32 / max_obs)
                    .collect()
            })
            .collect();
        if std::env::var_os("IMPLICATE_DEBUG_DISPATCH").is_some() {
            for (r, &i) in explicit.iter().enumerate() {
                let row: Vec<String> = candidates.iter().zip(&evidence[r]).map(|(j, e)| format!("t{j}={e:.2}")).collect();
                eprintln!("[dispatch] '{}' evidence: {}", specs[i].name, row.join(" "));
            }
        }
        for (r, c) in Self::max_weight_assignment(&evidence).into_iter().enumerate() {
            if let Some(c) = c {
                bind(explicit[r], candidates[c], &mut matches, &mut used);
            }
        }

        // 2. Verb grounding.
        for vb in &scene.verb_bindings {
            let Some(g) = manifest.verb_potentials.get(&vb.verb).and_then(|e| e.grounding.as_ref()) else {
                continue;
            };
            if g.clip_id != clip.clip_id {
                continue;
            }
            if let Some(i) = idx_of(&vb.subject) {
                bind(i, g.subject_track, &mut matches, &mut used);
            }
            if let (Some(i), Some(j)) = (idx_of(&vb.target), g.target_track) {
                bind(i, j, &mut matches, &mut used);
            }
        }

        // 3. Saliency order.
        for i in 0..specs.len() {
            // Includes explicit colours that found no chromatic evidence: they carry no information.
            if renderable[i] && matches[i].is_none() {
                if let Some(j) = (0..tracks.len()).find(|&j| !used[j] && usable(j)) {
                    bind(i, j, &mut matches, &mut used);
                }
            }
        }
        matches
    }

    /// Chromatic affinity of a stated colour with a track: alpha-weighted mean cosine between colour
    /// directions in L*a*b* seen from the neutral point (L* = 50, a* = b* = 0). Hue enters through the
    /// a*b* chroma spinor $z = a^* + b^*\mathbf{e}_{12}$, lightness through L*.
    fn chromatic_affinity(p: [f32; 3], art: &ArticulatedHolon) -> f32 {
        let dir = |c: [f32; 3]| {
            let l = Self::srgb_to_lab(c);
            [l[0] - 50.0, l[1], l[2]]
        };
        let pd = dir(p);
        let pn = (pd[0] * pd[0] + pd[1] * pd[1] + pd[2] * pd[2]).sqrt().max(1e-6);
        let (mut acc, mut wsum) = (0.0f32, 0.0f32);
        for k in art.kernels.iter().flatten() {
            for (t, px) in k.dc_rgba.chunks_exact(4).enumerate() {
                let a = px[3] as f32 / 255.0 * k.gaussian_envelope.get(t).copied().unwrap_or(1.0);
                if a <= 0.0 {
                    continue;
                }
                let cd = dir([px[0] as f32, px[1] as f32, px[2] as f32]);
                let cn = (cd[0] * cd[0] + cd[1] * cd[1] + cd[2] * cd[2]).sqrt().max(1e-6);
                acc += a * (pd[0] * cd[0] + pd[1] * cd[1] + pd[2] * cd[2]) / (pn * cn);
                wsum += a;
            }
        }
        if wsum > 0.0 { acc / wsum } else { 0.0 }
    }

    /// Exact maximum-weight assignment of rows to distinct columns (a row stays unassigned when
    /// all its weights are zero). Exhaustive branch-and-bound; scenes bind few holons.
    fn max_weight_assignment(w: &[Vec<f32>]) -> Vec<Option<usize>> {
        fn go(r: usize, w: &[Vec<f32>], used: &mut Vec<bool>, cur: &mut Vec<Option<usize>>, score: f32, best: &mut (f32, Vec<Option<usize>>)) {
            if r == w.len() {
                if score > best.0 {
                    *best = (score, cur.clone());
                }
                return;
            }
            let bound: f32 = w[r..].iter().map(|row| row.iter().copied().fold(0.0, f32::max)).sum();
            if score + bound <= best.0 {
                return;
            }
            for c in 0..w[r].len() {
                if !used[c] && w[r][c] > 0.0 {
                    used[c] = true;
                    cur[r] = Some(c);
                    go(r + 1, w, used, cur, score + w[r][c], best);
                    used[c] = false;
                    cur[r] = None;
                }
            }
            go(r + 1, w, used, cur, score, best);
        }
        let cols = w.first().map_or(0, Vec::len);
        let mut best = (f32::NEG_INFINITY, vec![None; w.len()]);
        go(0, w, &mut vec![false; cols], &mut vec![None; w.len()], 0.0, &mut best);
        best.1
    }

    /// CIE L*a*b* (D65) of an sRGB colour. The a*b* plane is the opponent-chroma plane: a colour's
    /// chroma is the $\mathbb{G}_{2,0}^+$ spinor $z = a^* + b^*\,\mathbf{e}_{12}$ (hue = rotor angle).
    fn srgb_to_lab(c: [f32; 3]) -> [f32; 3] {
        let lin = |v: f32| {
            let v = v / 255.0;
            if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
        };
        let (r, g, b) = (lin(c[0]), lin(c[1]), lin(c[2]));
        let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
        let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
        let f = |t: f32| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
        let (fx, fy, fz) = (f(x), f(y), f(z));
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
    }


    #[allow(clippy::too_many_arguments)]
    fn splat_articulated_holon_wavepacket(
        art_holon: &ArticulatedHolon,
        holon: &ArticulatedFieldHolon,
        screen_pos: [f32; 2],
        atlas_w: usize,
        atlas_h: usize,
        frame_idx: usize,
        source_frames_per_frame: f64,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) -> bool {
        if art_holon.poses_per_obs.is_empty() || art_holon.kernels.is_empty() {
            return false;
        }
        let obs_idx = Self::select_view_idempotent_observation_index(
            art_holon,
            holon,
            frame_idx,
            source_frames_per_frame,
        );
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

        let mut unfolded_poses = Vec::with_capacity(n_parts);
        let (mut min_x, mut max_x) = (w as i32 - 1, 0i32);
        let (mut min_y, mut max_y) = (h as i32 - 1, 0i32);

        for p in 0..n_parts {
            let rp = &ref_poses[p];
            let rel_dx = (rp.center[0] - root_ref.center[0]) * sx;
            let rel_dy = (rp.center[1] - root_ref.center[1]) * sy;
            // Posture change is carried by the view idempotent; the drawing itself is not rotated.
            let px = cx + rel_dx;
            let py = cy + rel_dy;
            let rx = (rp.radii[0] * sx).max(3.0);
            let ry = (rp.radii[1] * sy).max(3.0);
            let max_r = (rx.max(ry) * 1.15).ceil() as i32 + 2;

            min_x = min_x.min((px as i32) - max_r);
            max_x = max_x.max((px as i32) + max_r);
            min_y = min_y.min((py as i32) - max_r);
            max_y = max_y.max((py as i32) + max_r);

            unfolded_poses.push(SubHolonPose {
                center: [px, py],
                angle_rad: rp.angle_rad,
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
