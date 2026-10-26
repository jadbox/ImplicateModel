# ImplicateModel — Research & Experiment Tasks

> **Status:** Track S1 & S2 — COMPLETE (`58/58` Workspace Tests Passing, 0 Warnings). Active Milestone: Track S3 — Intrinsic Clifford Platforming & Controller Motor Injection.
> **Note (Visual QA Tooling):** Visual and temporal inspection CLIs available at `./tools/review_img.sh <IMAGE_FILE>` (example runner: `./tools/review_img_example.sh`) and `./tools/review_video.sh <VIDEO_FILE>` (short video temporal defect & trajectory auditing). Analyzes scene/character traits, motion continuity, and rendering glitches with pixel bounding boxes via local `llama-server` (Qwen 3 VL). Supports reference-counted multi-worker execution.

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs into a unified Clifford spacetime field, and enables:
1. **Mode A (Generative Storybook):** Remixing game worlds and animated scenes into novel narrative sequences from natural-language prompts.
2. **Mode B (Playable Game Engine):** Directly playing the video in real-time ($\ge 60\text{ FPS}$) via controller motor injection without an external Cartesian collision engine.

---

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)`
- **Benchmark Corpus:**
  1. `pacman_240x308.mp4` (`/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4`, 240x308, discrete 4-direction arcade turns)
  2. `smb3_w1_1_256x240.mp4` (`/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4`, 256x240, NES platformer gameplay)
  3. `raptor_320x240.mp4` (`/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4`, 320x240, 550s PC DOS shaded sprites)

## Core Architectural Directives (The Carmack–Hiley Synthesis)
1. **Zero Magic Number Constants:**
   - Eliminate hardcoded pixel intensity cutoffs, arbitrary aspect clamps, and manual reach bounds.
   - Use self-consistent statistical lattice noise floor: $\tau = \text{median}(\mathcal{I}) + 3.0 \times 1.4826 \times \text{MAD}(\mathcal{I})$.
2. **Clifford Process Solitons & Model Sheets:**
   - Characters are not 2D pixel boxes; they are minimal left ideals with invariant Line of Action $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ and canonical turnaround idempotents $\{\epsilon_v\}$.
   - Dual geometric product decomposition: ink contour blades ($u \wedge \nabla^2 u$) and cel paint fills ($u \cdot \nabla^2 u$).
3. **Intrinsic Clifford Physics (No External Collision Engine):**
   - Platforms and terrain are oriented planes $\Pi \in \mathbb{G}_{3,0,1}$.
   - Actors are trivector points $P \in \mathbb{G}_{3,0,1}$.
   - Ground contact and normal reaction force are governed by the geometric product boundary condition $P \wedge \Pi = 0$ and the scale-invariant Clifford Quantum Potential $\mathcal{Q}_{\text{Cl}}$.
4. **Mechanical Sympathy & Real-Time Performance:**
   - Hardware-aligned 64-byte multivector monad ([`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19)) matching AVX-512 registers and GPU cache lines.
   - Single-pass WGSL compute shader unfolding at $\ge 120\text{ FPS}$ on RTX 4090 ($< 0.5\text{ ms/frame}$).
5. **Empirical Verification Only:**
   - Evaluate against real video benchmarks; review exported artifacts directly in `./generative_output/`.
   - Avoid spawning loops of `ffmpeg` subprocesses (`for t in ...; do ffmpeg`); rely on streaming Rust readers (`FrameReader` / `read_frames`) to conserve system resources.

---

## Current Status & Immediate Next Steps to Resume

### Completed in Recent Session:
1. **Hiley–Carmack Bilinear Soliton Coherence Trace (`enforce_track_comoving_soliton_coherency`):**
   - Replaced ad-hoc color mode heuristics in [`enforce_track_comoving_soliton_coherency`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L2290-L2490) with the unified bilinear trace $\text{Tr}(\rho_S \rho_C) = \rho_S(\mathbf{r}) \cdot \rho_C(c) \ge 1 / M^2$ orthogonal to the local exterior vacuum state $\mathbf{b}_{\text{ext}}$, plus 1–2px contour gap closure between the primary body and chromatically coherent limbs ($\rho_C \ge 1/M$).
   - Completely eliminated Pac-Man's detached $3\times 3$ white corridor pellet and dark mouth/ringing artifacts, bridged Super Mario's 2px ankle contour line, and preserved 100% of the Raptor fighter jet nose cone and wings across the 550s longitudinal timeline.
2. **Single-Image $1.0\text{s}$-Apart Stability Pair Exports & Visual/Temporal QA (`r10_model_sheet_gates.rs`):**
   - Replaced split-frame duplicate-sprite contact sheets with single centered frame pairs $1.0\text{s}$ apart (`*_t0s.png` at $t=0.0\text{s}$ and `*_t1s.png` at $t=1.0\text{s}$) alongside the 60-frame `.mp4` unfold videos in `./generative_output/`.
   - Audited exported images and videos with `./tools/review_img.sh` and `./tools/review_video.sh`.
3. **Full Workspace Test Suite Verification (`58/58` Passing, 0 Warnings):**
   - Removed remaining `NIMH_PATH` references from `crates/implicate-lifter/tests/r2_gates.rs` and `crates/implicate-rolpa/tests/r3_3_gate.rs`.
   - Verified `cargo test --workspace --tests` passes 100% (`58/58` tests) with zero compiler warnings.

### Next Steps to Execute Immediately Upon Resuming (Track S3):
1. **Step 1 — Oriented Ground Planes in $\mathbb{G}_{3,0,1}$ (`implicate-field`):**
   - Represent platform ground bricks and obstacle walls as oriented Grade-1 planes $\Pi_{\text{ground}} \in \bigwedge^1 \mathbb{G}_{3,0,1}$.
2. **Step 2 — Controller Motor Injection (`implicate-field`, `implicate-unfold`):**
   - Map keyboard/gamepad inputs ($v_x, v_y, \text{jump}$) to Lie motor velocity generators $\Omega_{\text{player}} = v_x e_{01} + v_y e_{02} \in \mathfrak{spin}(3,0,1)$.
3. **Step 3 — Intrinsic Ground Contact & Fresnel Momentum Reflection (`pga_platforming_gates`):**
   - Couple actor trivector position $P(t)$ with ground plane $\Pi$ via $P \wedge \Pi = 0$ and reflect vertical momentum via Fresnel rotor $v' = \Pi v \widetilde{\Pi}$.

---

## Active Milestone Roadmap: Track S

### Track S1 — Universal Spatiotemporal Plenum Inpainting & Coherent Character Lifting (COMPLETE)

> **Objective:** Fix the background bake-in contamination during stationary holds, eliminate character limb amputation, and purge medium shadows/wakes and foreign corridor satellites.

#### 🎯 Milestone S1 Showcase Example: 100% Character Mass & Clean Soliton Extraction
- **Showcase Command:** `cargo test -p implicate-lifter --test r10_model_sheet_gates`
- **Output Artifacts:**
  1. [`generative_output/smb3_mario_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_t0s.png) | [`smb3_mario_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_t1s.png): Super Mario renders with complete brown boots, black overalls, white gloves, mustache, and red cap ($\text{IoU} = 0.9943$, $\text{LPIPS} = 0.0012$).
  2. [`generative_output/pacman_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_t0s.png) | [`pacman_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_t1s.png): Yellow Pac-Man renders cleanly at 8x arcade scale ($14\times 13$ native bbox) across discrete turnaround directions with 0 maze wall ringing and 0 detached corridor pellets ($\text{IoU} = 0.9752$, $\text{LPIPS} = 0.0065$).
  3. [`generative_output/raptor_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_t0s.png) | [`raptor_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_t1s.png): Metallic shaded fighter jet with intact nose cone and wings induced across the 550s timeline ($\text{IoU} = 1.0000$, $\text{LPIPS} = 0.0000$).

#### Status & Implemented Foundational Components:
- [x] **S1.1 — Spatiotemporal Atlas Inpainting & Bake-In Purge (`multiplane_atlas/mod.rs`, `layer_accum.rs`)**
  - Actor-dominated background hole detection implemented in [`accumulate_band_atlas_masked`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/multiplane_atlas/layer_accum.rs#L260-L370) with harmonic-guided empirical mode selection.
- [x] **S1.2 — Topological Cavity Disocclusion & $\mathbb{G}_{2,0}^+$ Chromatic Medium Shadow/Wake Annihilation (`morphology.rs`, `articulated_wavepacket/mod.rs`)**
  - Implemented [`detect_enclosed_cavities`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/morphology.rs#L143-L225) and [`disocclude_vacuum_cavities`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1512-L1556).
  - Implemented [`is_chromatic_medium_modulation`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1830-L1862) evaluating the Clifford geometric product $\psi_c \widetilde{\psi}_b$ in $\mathbb{G}_{2,0}^+$ to strip cast shadows and water wakes on chromatically polarized media ($|\psi_b| > 3\tau$).
- [x] **S1.3 — Boundary Ringing Specks & Co-Moving Soliton Coherency (`articulated_wavepacket/mod.rs`)**
  - Enforced `comp_sz >= 4` and harmonic reach $M$ in [`envelope_silhouette`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1667-L1819).
  - Implemented Hiley–Carmack bilinear soliton coherence trace $\text{Tr}(\rho_S \rho_C) \ge 1/M^2$ in [`enforce_track_comoving_soliton_coherency`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L2290-L2490).
- [x] **S1.4 — Mario Complete Character Mass Restoration (`r10_model_sheet_gates.rs`)**
  - Enforced $r_B = 0$ for thin sprites ($d_{\text{max}} \le 4$) in [`dimensional_opening_radius`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1599-L1616).
  - Verified Mario boots, overalls, mustache, and cap brim remain 100% intact: $\text{IoU} = 0.9943$ (target $\ge 0.985$), $\text{LPIPS} = 0.0012$ (target $\le 0.005$).

#### Verification Gates & Success Matrix:
| Benchmark | Target Metric | Visual Acceptance Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Super Mario Bros. 3** | $\text{IoU} \ge 0.985$, $\text{LPIPS} \le 0.005$ | Complete boots, overalls, cap brim; connected ankle contour | ✅ PASS ($\text{IoU}=0.9943, \text{LPIPS}=0.0012$) |
| **Pac-Man** | $\text{IoU} \ge 0.850$, $\text{LPIPS} \le 0.050$ | Yellow Pac-Man extracted; 0 detached white pellet dot; 0 dark mouth ringing | ✅ PASS ($\text{IoU}=0.9752, \text{LPIPS}=0.0065$) |
| **Raptor: Call of Shadows**| $\text{IoU} \ge 0.800$, $\text{LPIPS} \le 0.040$ | Metallic shaded fighter jet hull & nose cone intact across 550s windows | ✅ PASS ($\text{IoU}=1.0000, \text{LPIPS}=0.0000$) |
| **Workspace Test Suite** | 100% Pass (`cargo test --workspace --tests`) | Zero test regressions across all crates (`58/58` tests, 0 warnings) | ✅ PASS (`58/58`) |

---

### Track S2 — Model Sheet Induction & Multi-Second Presentation Viewports (COMPLETE)

> **Objective:** Upgrade video presentation from brief sub-second flashes to crisp, looped, multi-second character demonstrations and $1.0\text{s}$-apart stability frame pairs.

#### 🎯 Milestone S2 Showcase Example: The 60-Frame Looping Turnaround Reel & $1.0\text{s}$ Stability Pairs
- **Showcase Command:** `cargo test -p implicate-lifter --test r10_model_sheet_gates -- r10_5_generative_video_export_and_user_validation_gate`
- **Output Artifacts:**
  1. [`generative_output/smb3_mario_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4) | [`smb3_mario_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_t0s.png) | [`smb3_mario_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_t1s.png) (60 frames @ $8\times$ NES scaling, $192 \times 256$).
  2. [`generative_output/pacman_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4) | [`pacman_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_t0s.png) | [`pacman_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_t1s.png) (60 frames @ $8\times$ arcade scaling, $240 \times 240$).
  3. [`generative_output/raptor_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_generative_unfold.mp4) | [`raptor_t0s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_t0s.png) | [`raptor_t1s.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_t1s.png) (60 frames @ $4\times$ VGA scaling, $240 \times 240$).

- [x] **S2.1 — Integer Presentation Scaling (`r10_model_sheet_gates.rs`)**
- [x] **S2.2 — Multi-Second Generative Video Export Loops (`r10_model_sheet_gates.rs`)**
- [x] **S2.3 — Generative Video & $1.0\text{s}$ Stability Pair Export & Visual QA (`generative_output/`)**
  - Verified with `./tools/review_img.sh` and `./tools/review_video.sh`.

---

### Track S2.5 — Full-Screen World Composite & Initial State Restitution ($t_0$ Diorama Gate)
Objective: Reconstruct and render complete, pixel-accurate initial game screens ($t = t_0$) for SMB3 (256×240), Pac-Man (240×308), and Raptor (320×240) by compositing the inpainted background plenum, camera-locked HUD, induced actor model sheets at $P_i(t_0) \in \bigwedge^3 \mathbb{G}_{3,0,1}$, and extracted world constraint geometry ($\Pi_k, L_k$).
- S2.5.1 — Multi-Layer World Restitution (implicate-lifter, implicate-unfold):
- Composite MultiPlaneAtlas background + camera_locked_hud + all active HolonModelSheet actors at their exact $t_0$ PGA motors $M_i(t_0) \in \text{Spin}(3,0,1)$ with integer presentation scaling.
- S2.5.2 — Algebraic Constraint Alignment Audit:
- Extract initial world boundaries ($\Pi_{\text{ground}}, \Pi_{\text{block}}$ for SMB3; corridor lines $L_k = \Pi_1 \wedge \Pi_2$ for Pac-Man; viewport bounds for Raptor) and verify zero initial incidence error ($|P_{\text{mario}}(t_0) \vee \Pi_{\text{ground}}| \le 0.5\text{ px}$, $d(P_{\text{pacman}}(t_0), L_k) \le 0.5\text{ px}$).
- S2.5.3 — Full-Screen Visual & Quantitative Gate (s2_5_world_screen_gates):
- Export full-screen composite PNGs (smb3_world_t0.png, pacman_world_t0.png, raptor_world_t0.png) and diagnostic constraint overlays (*_constraints_t0.png).
- Gate: Full-screen $\text{PSNR} \ge 34.0\text{ dB}$ against ground-truth frame $t_0$, $100\%$ actor placement accuracy, verified via ./tools/review_img.sh.

### Track S3 — Universal Clifford World Interaction & Lie Motor Agency (implicate-math, implicate-field, implicate-unfold)
Objective: Encode all world interactions across 2D, 2.5D, and 3D regimes into four closed-form $\mathbb{G}_{3,0,1}$ algebraic operators on Mv301, with zero external physics engine dependencies.
- S3.1 — Ambient Vacuum Potential $\mathcal{Q}_{\text{Cl, env}}$ & Grade-Polymorphic Constraints:
- Upgrade ArticulatedFieldHolon to native 64-byte Mv301.
- Unify gravity, zero-G rail lock, and scroll drift under $-\nabla \langle \mathcal{Q}_{\text{Cl, env}} \rangle_0 - \Gamma B_{\text{vel}}$.
- Implement Grade-1 chiral planes $\Pi$ (solid, one-way semisolid, elastic) and Grade-2 Plücker corridor rails $L = \Pi_1 \wedge \Pi_2$ with junction turns $J = L_1 \wedge L_2$ and portal motors $M_{\text{portal}} = \Pi_{\text{out}}\widetilde{\Pi}_{\text{in}}$.
- S3.2 — Universal Lie Motor Agency ($\Omega_{\text{ctrl}} \in \mathfrak{spin}(3,0,1)$):
- Map human controller inputs and AI verb beacons into the same 6-DOF Lie generator $\Omega_{\text{ctrl}}$ filtered by regime sub-algebra masks (GraviticPlatform, CorridorRail, InertialFlight, Sector6DOF).
- S3.3 — The 4 Bilinear Holon Couplings:
1. Chiral Fresnel Reflection/Slide: $\Pi B_{\text{vel}} \widetilde{\Pi}$ (platform landing, wall stop, top-stomp vs. lateral damage).
2. Soliton Absorption & Idempotent Flip: $\rho_B \to 0$ + $T_{u \to v} = \epsilon_v M \epsilon_u$ (Pac-Man dots/power pellets, Mario coins).
3. Griffith Cohesion Fracture: Overpressure $|p| > p_{\text{yield}} \implies C \to 0$ (Raptor enemy destruction, brick smashing).
4. Wavepacket Emission: Spawning transient projectile solitons $\Phi_{\text{proj}}$ (Raptor lasers, Mario fireballs).
- S3.4 — Cross-Genre Zero-Physics-Engine Verification Gate (pga_world_interaction_gates):
- Deterministic kinematic proofs for SMB3 (jump, one-way platform pass-through, landing, stomp), Pac-Man (corridor rail lock, $90^\circ$ junction turn, pellet absorption, ghost state flip), and Raptor (banked 2D flight, laser emission, hull fracture).
Track S4 — The Universal Playable World Engine (implicate-cli play)
Objective: Launch any enfolded .implicate world (SMB3, Pac-Man, Raptor) as an interactive $60\text{--}120\text{ FPS}$ real-time game starting from the validated S2.5 initial layout and driven by S3 Lie motor agency.
- S4.1 — Interactive winit/wgpu Runtime (implicate play --input <world.implicate>).
- S4.2 — Decoupled $1,000\text{ Hz}$ Symplectic PGA Stepping + $60\text{--}120\text{ FPS}$ @rolpa Display & $0.0\text{ ms}$ Foley Audio.
- S4.3 — Multi-Genre Playability Verification (SMB3, Pac-Man, Raptor).

---

### Track S4 — 3D Projective Sector-Plane Lifter & Inverse-BSP World Compilation (e.g. *Doom*) (PLANNED)

> **Objective:** Ramp the unified $\mathbb{G}_{3,0,1}$ architecture from 2D/2.5D planar slices ($e_3 = 0$) to full 3D perspective environments ($e_3 \neq 0$), reconstructing navigable 3D sector manifolds and 8-view actor turnarounds from 3D gameplay footage (e.g. $320 \times 240$ *Doom*).
> **Design Mandate for S1–S5:** All new features across S1–S5 must assume general 2D or 3D environments in $\mathbb{G}_{3,0,1}$ ([`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19))—never hardcoding 2D-only coordinate assumptions—so that S1–S5 modules compose directly into Track S6.

#### 🎯 Milestone S6 Showcase Example: Inverse-BSP *Doom* E1M1 Lifting & 6-DOF Playable Unfoldment
- **Showcase Command:** `cargo run -p implicate-cli -- play --input ./tests/fixtures/doom_e1m1_320x240.implicate`
- **Output Artifacts & Geometric Proof:**
  - Enfolds $320 \times 240$ *Doom* gameplay footage into oriented 3D wall planes $\Pi_w = n_x e_1 + n_z e_3 + d e_0$ and floor/ceiling planes $\Pi_h = \pm e_2 + h e_0 \in \bigwedge^1 \mathbb{G}_{3,0,1}$.
  - Partially observed corridor corners are analytically closed via the exterior wedge product ($L_{\text{corner}} = \Pi_1 \wedge \Pi_2$, $P_{\text{vertex}} = \Pi_1 \wedge \Pi_2 \wedge \Pi_{\text{floor}}$), extrapolating periodic CFT wall textures across unseen corners.
  - Monsters and projectiles are lifted into the exact same [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/PLAN.md#L169) 8-view turnaround idempotents $\{\epsilon_v\}$ used in S1–S5, enabling 6-DOF playable navigation with intrinsic wall collision ($P \wedge \Pi_w = 0$).

- [ ] **S6.1 — 6-DOF Projective Camera Motor Odometry (`implicate-lifter`)**
  - Extend camera trajectory estimation from 2D translation $(t_x, t_y)$ to 3D PGA screw motors $M_{\text{cam}}(t) = \exp(-\frac{1}{2}(\theta e_{31} + \phi e_{23} + \mathbf{v}\cdot\mathbf{e}_{0i})) \in \text{Spin}(3,0,1)$ capturing radial perspective divergence ($\propto 1/Z$) and yaw/pitch homographies.
- [ ] **S6.2 — Inverse-BSP Plane Fitting & Algebraic Corner Closure (`implicate-lifter`, `implicate-field`)**
  - Fit Grade-1 planes $\Pi \in \bigwedge^1 \mathbb{G}_{3,0,1}$ to perspective depth/flow partitions, unwarp perspective trapezoids into canonical CFT texture atlases, and close partially observed corners via $\Pi_1 \wedge \Pi_2$.
- [ ] **S6.3 — 3D Perspective Actor Lifting & 6-DOF Playable Unfoldment (`implicate-unfold`, `implicate-cli`)**
  - Subtract perspective sector planes to lift 3D billboard/polyhedral actors into [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/PLAN.md#L169) turnarounds and execute 6-DOF WASD + mouse turn gameplay at $\ge 120\text{ FPS}$.
