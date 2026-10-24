# ImplicateModel — Engine Status & Production Roadmap

## Current Status
All 19 research and production milestones (`R1.1`–`R8.5`) and `42/42` workspace tests pass across the 7 crates (`implicate-math`, `implicate-field`, `implicate-lexicon`, `implicate-lifter`, `implicate-rolpa`, `implicate-unfold`, `implicate-cli`).

- **Enfoldment (`R1`–`R2`):** Streaming Dirac-Spinor 2D CFT shot/regime segmentation (`> 1,350 FPS` @ 1080p), multi-plane background atlases (`32.4–40.3 dB` PSNR) with camera-locked HUD holons, multi-actor articulated $P$-part Clifford wavepackets (`IoU > 0.968`), and `Qwen2.5-VL` (`llama-cli` + GBNF) semantic harvesting into `Spinor16`.
- **Relational & Surface Strata (`R3`):** Scale/$E(2)$-invariant Chebyshev-Fourier $\mathcal{Q}_{\text{Cl},v}$ verb potentials (`0.24%–0.69%` Fréchet error), 6-block $\text{Pin}(3,0,1)$ `GATr` binder (`0.096 ms/pass`), and 1-pass 4-block `CliffordNet-Lite` WGSL refiner (`0.46 ms/frame` @ 1080p, `+5.37 dB` PSNR).
- **Unfoldment, Audio & Production Pipeline (`R4`–`R7`):** GPU Yee wave-matter lattice + articulated holon reduction (`1.01 ms/frame`), sample-accurate $48\text{ kHz}$ modal/harmonic stereo synthesis (`0.0000 ms` sync error), multi-clip `.implicate` v1 binary asset serialization (`ENFD`), multi-actor pairwise PGA trajectory $\mathcal{Q}_{\text{Cl},v}$ assimilation ($M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$), bottom-up CFT kinematic impulse snapping, $\text{Spin}(2) \times S^1$ view-idempotent selection ($\epsilon_{s,\text{view}}$), `Spinor16` open-vocabulary verb resolution, state-coupled HUD score progression, and muxed `.mp4` + `.wav` output at `132–346+ FPS`.

---

## Production Milestones (Tracks R5 & R6 — Completed)

- [x] **R5.1 — MP4 Video + 48 kHz Audio Muxing & Per-Frame Refiner Readback (`COMPLETED`):**
  - Updated `CausalUnfoldEngine::render_rolpa_frame` (`crates/implicate-unfold/src/engine.rs`) and `GpuCliffordNetRefiner::refine_and_readback_u8` (`crates/implicate-rolpa/src/gpu_refiner.rs`) so every frame uploads holon luminescence/field gradients, runs the 4-block `CliffordNet-Lite` WGSL shader, and reads back the refined RGB8 buffer (`158–239 FPS` on `320x180`).
  - Added `mux_rgb_frames_and_wav_to_mp4` and `probe_audio_stream` (`crates/implicate-lifter/src/video_io.rs`) plus `UnfoldRunReport::export_wav_and_mp4`, wired into `implicate unfold` (`crates/implicate-cli/src/main.rs`) and verified by `r5_1_mp4_audio_video_mux_and_refiner_readback_gate` (`crates/implicate-unfold/tests/r4_3_gate.rs`).
- [x] **R5.2 — Enfold-to-Container Serialization (`COMPLETED`):**
  - Added binary payload encoder/decoder (`crates/implicate-unfold/src/payload.rs`) for `EnfoldedClipAssets` (`MultiPlaneAtlas` layers, `camera_locked_hud` layer, bit-packed actor masks, and `ArticulatedHolon` CFT wavepacket kernels) inside the `.implicate` v1 container (`IMPLICATE_FLAG_ENFOLDED_ASSETS = 0x02`).
  - Wired `implicate enfold --video <video.mp4> [--output <file.implicate>]` and `implicate inspect` (`crates/implicate-cli/src/main.rs`), and verified bit-exact round-trip reconstruction in `r5_2_enfold_to_container_serialization_gate` (`crates/implicate-unfold/tests/r4_3_gate.rs`).
- [x] **R5.3 — Real Atlas & Wavepacket Compositing in `CausalUnfoldEngine` (`COMPLETED`):**
  - Upgraded `CausalUnfoldEngine::render_rolpa_frame` (`crates/implicate-unfold/src/engine.rs`) with `with_enfolded_clip` (`composite_enfolded_atlas_background`, `splat_articulated_holon_wavepacket`, `composite_enfolded_hud_layer`) to composite enfolded `MultiPlaneAtlas` layers, camera-locked HUD holons (`0.0000 px` drift), and `ArticulatedHolon` sub-holon wavepackets at their unfolded PGA poses (`327.6–338.7 FPS` in `r5_3_real_atlas_and_wavepacket_compositing_gate`).
- [x] **R5.4 — Text / `DzogLang` Prompt Parser (`COMPLETED`):**
  - Added `UnfoldSceneSpec::parse_prompt` (`crates/implicate-unfold/src/prompt.rs`) to compile free-form text or `@scene` `DzogLang` scripts (actors, $P$-part counts, emotions, archetypes, relational verbs, camera-locked HUD holons, and `@rolpa` refiner flags) into `UnfoldSceneSpec`, wired `--prompt "..."` and `--dzog <file.dzog>` into `implicate unfold` (`crates/implicate-cli/src/main.rs`), and verified in `r5_4_text_and_dzoglang_prompt_parser_gate`.
- [x] **R6.1 — Multi-Clip Corpus Assimilation, Open-Vocab `Spinor16` Verb Resolution, & State-Coupled HUD Score (`COMPLETED`):**
  - Added `upsert_enfolded_clip_to_container` and `assimilate_clip_into_manifest` (`crates/implicate-unfold/src/payload.rs`) to accumulate multiple enfolded video clips into a single `.implicate` container while fitting real `LearnedVerbPotential` ($\mathcal{Q}_{\text{Cl},v}$) fields from tracked `ArticulatedHolon` root trajectories.
  - Added `VerbPotentialLexicon::resolve` (`crates/implicate-lexicon/src/verb_potential.rs`), `CausalUnfoldEngine::with_enfolded_clips` domain matching, and state-coupled HUD holon score progression (`hud_score_history` + 3x5 bitmap score digit rendering in `crates/implicate-unfold/src/engine.rs`), verified in `r6_multi_clip_corpus_enfold_trajectory_assimilation_and_dynamic_hud_gate` (`265.2 FPS`, `0.16%–0.21%` Fréchet error).
- [x] **R7.1 — Multi-Actor Track Extraction, Pairwise PGA Trajectory Assimilation, & $\text{Spin}(2) \times S^1$ View-Idempotent Selection (`COMPLETED`):**
  - Added `extract_character_observation_tracks` (`crates/implicate-lifter/src/articulated_wavepacket/mod.rs`) and `detect_cft_kinematic_impulses` (`crates/implicate-lexicon/src/cft_snap.rs`) so `enfold_video_clip` extracts multiple distinct `ArticulatedHolon` actors per clip, grounds verb windows in empirical Bohmian momentum discontinuities $\|\Delta \mathbf{P}_B(t)\|$, and fits true two-body pairwise relative PGA verb potentials ($M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$, `enfolded_pair_<clip_id>`).
  - Upgraded `CausalUnfoldEngine::splat_articulated_holon_wavepacket` with `select_view_idempotent_observation_index` (`crates/implicate-unfold/src/engine.rs`) to select each sub-holon's discrete view-aspect idempotent cell $\epsilon_{s,\text{view}}$ via geodesic rotor and CFT gait-phase coherence on $\text{Spin}(2) \times S^1$, verified in `r7_multi_actor_cft_impulse_and_view_idempotent_gate` (`216.5 FPS`, `1.39% / 0.48%` Fréchet error).

---

## Completed Milestones: Track R8 — Visual Fidelity & Artifact Elimination (`5/5` Gates Verified)

- [x] **R8.1:** Static camera arena preservation (`active_y = (0, H)`, `hud_band = None`), full playing field restored, actors un-occluded.
- [x] **R8.2:** SMB3 World 1-1 enfoldment at $20.0\text{s}$, raster coordinate realignment ($y_{\text{ground}} = 192$), upward normal $[0, -1]$, Mario vaults cleanly over terrain.
- [x] **R8.3:** Semantic role matching (`match_holon_to_wavepacket`) prevents pure energy drones (`Plasma_Drone`, pellets, talismans) from adopting arbitrary enemy sprites.
- [x] **R8.4:** Single continuous shot isolation via `StreamingShotSegmenter`, temporal median fallback in `layer_accum.rs`, refiner weights tuned $[0.10, -0.04, 0.05, 0.05]$ eliminating 16x16 tile block seams.
- [x] **R8.5:** End-to-end visual regression gate suite (`tests/r8_gates.rs`) verified across all benchmark scenes.

---

## Core Engineering Principles & Architectural Mandate

1. **Do More With Less via Powerful Clifford Abstractions:**
   - Solve kinematics, visual dispatch, and inpainting with unified mathematical operators ($\text{Spin}(n)$ rotors, PGA motors $M \in \mathbb{G}_{3,0,1}$, Poisson boundary value PDEs, and idempotent projectors $\Pi_s$) rather than ad-hoc heuristics.
2. **Zero Magic Numbers & Patchwork Solutions:**
   - Eliminate hardcoded raster coordinates, arbitrary thresholds, and string branch checks. System behavior must derive strictly from constitutive physical parameters ($C, \rho, K, \mathbf{L}$) and normalized coordinates $[0, 1]^2$.
3. **Rigorous Visual & Diagnostic Tooling:**
   - Build automated diagnostic inspection tools (`implicate inspect --tracks --dump-tracks`, strip extraction, and pixel-level assertion suites) to isolate root causes with mathematical certainty.
4. **Universal Media Invariance:**
   - The engine must make zero assumptions about input video domain: NES sprite games, vertical shmups, 2.5D games, and feature-length hand-drawn cinema (*The Secret of NIMH*) run through the exact same Clifford pipelines.

---

## Active Bug Fix Journey: Animation Coherence & Level Fidelity (Track R8.6)

### Visual Findings & Diagnoses
1. **Pac-Man Animation Incoherence:**
   - *Root Cause:* $P=1$ sprites hardcoded `angle_rad = 0.0`. In `select_view_idempotent_observation_index`, rotor distance $\langle \widetilde{R}_{\text{obs}}, R_{\text{target}} \rangle$ degenerated to a constant, causing ungrounded phase sampling that randomly hopped across incompatible view cells (facing left, right, down, closed mouth) every frame.
   - *Clifford Resolution:* Bind the observation's physical trajectory heading to its $\text{Spin}(2)$ rotor $R = \cos(\theta/2) + \mathbf{e}_{12}\sin(\theta/2)$. Filter view cells by directional rotor alignment ($\langle R, R_{\text{obs}} \rangle > 0$), and cycle through matching directional frames in continuous temporal sequence. Disable fractional angle rotation for $P=1$ sprites to preserve pixel-crisp raster art.
2. **SMB3 Level Brick Corruption & Background Tears:**
   - *Root Cause A:* Background inpainting previously used isotropic 2D diffusion, which leaked vertical sky color downwards across horizontal scanlines into the green bushes and ground.
   - *Root Cause B:* `Koopa_Block` (a static level prop) was defined as a discrete holon actor. Color dispatch matched it to a cropped question-mark block wavepacket from the training video, splatting a duplicate, deformed block on top of the already complete level atlas at an offset position.
   - *Unified Resolution:* Formulate background reconstruction using temporal observation ratio $\rho_{\text{occl}} = N_{\text{occl}} / N_{\text{obs}}$. Pixels with $\rho_{\text{occl}} < 0.65$ (moving actors) are restored directly from the unmasked temporal median (eliminating all tears in bushes/ground). Pixels with $\rho_{\text{occl}} \ge 0.65$ (permanently occluded stationary actors like Jeremy in NIMH) are filled via horizontal neighbor inpainting. Static terrain fixtures ($Z \to \infty, \mathbf{v} = \mathbf{0}$) are handled as boundary blades rather than radiative splatted sprites.
3. **Contact Velocity Shock:**
   - *Root Cause:* Early elastic contact resolution applied impulsive velocity kicks $(1 + e) v_{\text{rel}}$, disrupting Bohmian guidance potentials and causing Frechet divergence.
   - *Resolution:* Pure normal positional overlap separation without artificial kinetic energy injection, maintaining exact Bohmian verb fidelity.

### Status Tracker: Track R8.6
- [x] **R8.6a (CLI Inspection Tool):** Added `--dump-tracks` / `--tracks` to `implicate inspect` in `crates/implicate-cli/src/main.rs`.
- [x] **R8.6b (Track Extraction & Color-Based Dispatch):** Expanded `max_tracks = 8`, discarded stationary tracks, minimal RGB distance dispatch in `crates/implicate-unfold/src/engine.rs`.
- [x] **R8.6c (Projective Camera Kinematics):** World-to-screen coordinate transformation applied to wavepacket splatting, field discs, and luminescence injection.
- [x] **R8.6d (Hard-Core Inter-Holon Separation):** Non-impulsive positional separation $\delta \mathbf{x} = \frac{1}{2}(r_i + r_j - d)\hat{\mathbf{n}}$ preserving continuous verb guidance.
- [x] **R8.6e (Scanline Inpainting & Refiner Stabilization):** Horizon-preserving scanline inpainting in `layer_accum.rs` and clamped discrete Laplacian context in `cliffordnet_rolpa.wgsl`.

---

## Next Steps for Fresh Session Resume

1. **Verify Pac-Man Directional Gait Consistency:**
   - Confirm Pac-Man's mouth chomping animation cycles at a steady 10–15 Hz cadence along his instantaneous velocity vector across all directional transitions.
2. **Verify SMB3 Level Cleanliness:**
   - Confirm that the SMB3 level atlas has zero vertical tears in bushes/ground and that the duplicate floating brick block is fully purged.
3. **Extract Verification Strips:**
   - Run `implicate unfold --scene all` and generate fresh strips in `/tmp/opencode/human_test/anim_verified_strips/` for final human inspection.
4. **Transition to Track R9 (Generalization & Zero-Heuristics):**
   - Implement $\text{Spin}(16)$ semantic spinor inner-product dispatch, physical constitutive routing ($C, \rho, K, \mathbf{L}$), and normalized projective coordinates $[0, 1]^2$.

---

## Active Milestone: Track R9 — General-Purpose Semantic Generalization & Rigorous Clifford Abstraction

- [ ] **R9.1:** Semantic spinor inner-product dispatch $\langle \psi_{\text{prompt}}, \psi_{\text{asset}} \rangle$ replacing character name heuristics.
- [ ] **R9.2:** Physical constitutive routing ($C \ge 0.5$ matter vs $C < 0.5, L > 0$ field solitons).
- [ ] **R9.3:** Resolution-invariant normalized projective coordinates $[0, 1]^2$.
- [ ] **R9.4:** Milieu retrieval via $\text{Spin}(16)$ geodesic distance.
- [ ] **R9.5:** Zero-heuristic static audit and generalization gate suite (`tests/r9_gates.rs`).

