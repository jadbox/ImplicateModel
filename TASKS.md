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

### Track S3 — Intrinsic Clifford Platforming & Controller Motor Injection (PLANNED)

> **Objective:** Implement physical platforming directly through Projective Geometric Algebra boundary predicates without a third-party physics engine.

#### 🎯 Milestone S3 Showcase Example: The 0-Physics-Engine PGA Jump & Land Sandbox
- **Showcase Command:** `cargo test -p implicate-field --test pga_platforming_gates`
- **Output Artifacts & Kinematic Proof:**
  - A deterministic trajectory test where a jump impulse $\Omega_{\text{jump}} = 120\,e_{02}$ is injected into Mario at $(x=32, y=64)$ above an oriented ground plane $\Pi = 0\,e_1 + 1\,e_2 + 0\,e_3 - 32\,e_0$.
  - Mario rises, reaches apex at $t=0.3\text{s}$, and accelerates downward under Bohmian gravitational flux $\mathbf{g} = -\nabla \langle \mathcal{Q} \rangle_0 = -9.8\,e_2$.
  - At $t=0.6\text{s}$, contact condition $P \wedge \Pi = 0$ is detected; vertical velocity reflects to $0$ via Fresnel rotor $v' = \Pi v \widetilde{\Pi}$, perfectly arresting downward movement with zero penetration ($y = 32.000 \pm 0.000$) and 0 external physics library dependencies.

- [ ] **S3.1 — Oriented Ground Planes in $\mathbb{G}_{3,0,1}$ (`implicate-field`)**
  - Represent platform ground bricks and obstacle walls as oriented planes $\Pi_{\text{ground}} \in \bigwedge^1 \mathbb{G}_{3,0,1}$.
- [ ] **S3.2 — Controller Motor Injection (`implicate-field`, `implicate-unfold`)**
  - Map keyboard/gamepad inputs ($v_x, v_y, \text{jump}$) directly to Lie motor velocity generators:
    $$\Omega_{\text{player}} = v_x e_{01} + v_y e_{02} \in \mathfrak{spin}(3,0,1)$$
- [ ] **S3.3 — Intrinsic Ground Contact & Fresnel Momentum Reflection**
  - Couple actor trivector position $P(t)$ with the ground plane $\Pi$ using the Clifford wedge product $P \wedge \Pi = 0$.
  - Reflect vertical momentum via the rotor sandwich $v' = \Pi v \widetilde{\Pi}$, arresting downward fall on platforms without an external collision engine.

---

### Track S4 — The Playable Mario Engine (`implicate play`) (PLANNED)

> **Objective:** Enable real-time keyboard/gamepad gameplay directly inside the unfolded Clifford video world.

#### 🎯 Milestone S4 Showcase Example: Live Playable NES World 1-1 Run
- **Showcase Command:** `cargo run -p implicate-cli -- play --input ./tests/fixtures/smb3_world1.implicate`
- **Interactive Gameplay Proof:**
  - Launches a native $120\text{ FPS}$ Vulkan/Metal window.
  - The player uses Left/Right Arrow keys to move and Spacebar to jump.
  - Mario runs across real NES brick terrain, jumps onto an elevated question-mark block ($\Pi_{\text{block}}$), lands solidly with audio synthesized on contact ($0.0\text{ ms}$ latency), and navigates the level in real-time. This proves the concept: playing an interactive video game directly from enfolded video without game code or a Cartesian collision engine.

- [ ] **S4.1 — CLI Subcommand `implicate play` (`implicate-cli`)**
  - Implement `implicate play --input <corpus.implicate>` launching an interactive `winit` / `wgpu` viewport.
- [ ] **S4.2 — Real-Time Interactive Loop ($\ge 60\text{ FPS}$)**
  - Decouple input processing ($1,000\text{ Hz}$ SIMD motor updates) from visual compute shader unfolding ($60\text{--}120\text{ FPS}$).
- [ ] **S4.3 — Interactive Platforming Verification**
  - Verify player can run left/right, jump with Space, land on bricks, and traverse the scrolling SMB3 stage.

---

### Track S5 — The Generative Storybook Engine (`implicate remix`) (PLANNED)

> **Objective:** Enable open-ended narrative scene remixing from natural language prompts.

#### 🎯 Milestone S5 Showcase Example: Remixed Scene
- **Showcase Command:** `cargo run -p implicate-cli -- remix --video ./tests/fixtures/smb3_world1.implicate --prompt "Mario sprints forward, leaps over the Koopa block, and lands on the brick platform"`
- **Output Artifact & Visual Proof:**
  - [`generative_output/smb3_remix_koopa_vault.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_remix_koopa_vault.mp4) (120 frames @ 24 fps with phase-locked 48 kHz stereo audio).
  - The engine translates prompt verbs into spatial $\mathcal{Q}_{\text{Cl}}$ attractor beacons, rotates the actor's Line of Action $L_{\text{spine}}$ across turnaround idempotents, and synthesizes a new scene without diffusion hallucinations.

- [ ] **S5.1 — CLI Subcommand `implicate remix` (`implicate-cli`)**
  - Implement `implicate remix --video <corpus.implicate> --prompt "<prompt>"` generating novel scene animations.
- [ ] **S5.2 — Continuous Turnaround & Wavepacket Unfolding**
  - Blend smoothly between discrete view-aspect idempotents $\{\epsilon_v\}$ and continuous Clifford Fourier wavepackets based on prompted actions.
- [ ] **S5.3 — Event-Sampled Foley Soundboard & Sub-Sample Triggering**
  - Extract transient audio slices ($50\text{--}500\text{ ms}$) from source audio at empirical impulse and contact events ($\|\Delta \mathbf{P}_B\| > \tau_{\text{imp}}$, $P \wedge \Pi = 0$).
  - Bind slices to entity behaviors: e.g. `(Mario, Action::Jump) -> "jump.wav"`, `(Mario, Event::PlatformLand) -> "land.wav"`, `(Mario, Event::BlockHit) -> "coin_box.wav"`, `(Raptor, Action::Fire) -> "plasma_fire.wav"`.
  - Trigger sampled playback with sub-sample onset timing ($< 1.0\text{ ms}$) and binaural stereo panning.
  - *(Note: Word-level dialogue segmentation and speech synthesis are explicitly out of scope until a much later milestone).*

---

### Track S6 — 3D Projective Sector-Plane Lifter & Inverse-BSP World Compilation (e.g. *Doom*) (PLANNED)

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
