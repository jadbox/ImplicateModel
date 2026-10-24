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

## Completed Bug Fix: Animation Fidelity, Inter-Holon Contacts, & Atlas Inpainting (Track R8.6 — `5/5` Verified)

Visual animation evaluation of frame strips (`/tmp/opencode/human_test/anim_strips/`) identified and resolved 5 animation defects:
1. **Camera Scroll vs Holon Motion Disconnect:** Holons transformed to screen coordinates via $\mathbf{x}_{\text{screen}} = \mathbf{x}_{\text{world}} - (\mathbf{x}_{\text{cam}} - \mathbf{x}_{\text{cam},0})$, locking stationary obstacles (`Koopa_Block`) to scrolling terrain and giving moving actors proper traction.
2. **Actor Track Dropping & Inverted Binding:** Expanded `max_tracks = 8`, filtered static HUD tracks ($\max \|\Delta \mathbf{x}\| \le 2.0\text{ px}$), and bound prompt entities to candidate wavepackets by minimal Euclidean RGB distance $\|\mathbf{c}_{\text{spec}} - \mathbf{c}_{\text{track}}\|$. Pac-Man correctly binds to yellow track; Blinky and Inky bind to red and cyan tracks.
3. **Ghosting & Overlap:** Added hard-core contact separation $\|\mathbf{x}_i - \mathbf{x}_j\| \ge r_i + r_j$ with elastic impulse restitution between solid physical holons, preventing entity merging.
4. **Baked-in Stationary Actors in Atlas:** Replaced unmasked actor copying with multi-directional inverse-distance ray search and 16-iteration harmonic Laplacian diffusion across unobserved occlusions in `layer_accum.rs`.
5. **Tile Seam Banding in Refiner:** Clamped discrete Laplacian context $\nabla^2 H \in [-0.05, 0.05]$ and geometric product accumulations in `cliffordnet_rolpa.wgsl`, eliminating tile border seams.

- [x] **R8.6a (CLI Inspection Tool):** Added `--dump-tracks` / `--tracks` to `implicate inspect` in `crates/implicate-cli/src/main.rs`.
- [x] **R8.6b (Track Extraction & Color-Based Dispatch):** Expanded `max_tracks = 8`, discarded stationary tracks, minimal RGB distance dispatch in `crates/implicate-unfold/src/engine.rs`.
- [x] **R8.6c (Projective Camera Kinematics):** World-to-screen coordinate transformation applied to wavepacket splatting, field discs, and luminescence injection.
- [x] **R8.6d (Hard-Core Inter-Holon Elastic Separation):** Pairwise normal separation and elastic impulse resolution in `CausalUnfoldEngine`.
- [x] **R8.6e (Bi-Harmonic Background Inpainting & Refiner Stabilization):** Harmonic Laplacian background diffusion in `crates/implicate-lifter/src/multiplane_atlas/layer_accum.rs` and bounded context in `crates/implicate-rolpa/src/cliffordnet_rolpa.wgsl`.

---

## Active Milestone: Track R9 — General-Purpose Semantic Generalization & Rigorous Clifford Abstraction

- [ ] **R9.1:** Semantic spinor inner-product dispatch $\langle \psi_{\text{prompt}}, \psi_{\text{asset}} \rangle$ replacing character name heuristics.
- [ ] **R9.2:** Physical constitutive routing ($C \ge 0.5$ matter vs $C < 0.5, L > 0$ field solitons).
- [ ] **R9.3:** Resolution-invariant normalized projective coordinates $[0, 1]^2$.
- [ ] **R9.4:** Milieu retrieval via $\text{Spin}(16)$ geodesic distance.
- [ ] **R9.5:** Zero-heuristic static audit and generalization gate suite (`tests/r9_gates.rs`).

