# ImplicateModel — Research & Experiment Tasks

> **Status:** Track S (Storybook & Playable World Engine) Active Engineering & Verification.

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to hand-drawn feature animation) into a unified Clifford spacetime field, and enables:
1. **Mode A (Generative Storybook):** Remixing animated films and game worlds into novel narrative sequences from natural-language prompts.
2. **Mode B (Playable Game Engine):** Directly playing the video in real-time ($\ge 60\text{ FPS}$) via controller motor injection without an external Cartesian collision engine.

---

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)`
- **Benchmark Corpus (`./generative_output/sources/`):**
  1. `pacman_240x308.mp4` (240x308, discrete 4-direction arcade turns)
  2. `smb3_w1_1_256x240.mp4` & `smb3_w1_1.mp4` (256x240 & native, NES platformer gameplay @ $t \ge 20\text{s}$)
  3. `raptor_320x240.mp4` & `raptor.mp4` (320x240 & native, 10.3 min PC DOS shaded sprites)
  4. `nimh_1982_1080p.mp4` (1920x1040, 23.976 fps, 1h 22m 1080p hand-drawn feature animation)

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

---

## Active Milestone Roadmap: Track S

### Track S1 — Universal Spatiotemporal Plenum Inpainting & Coherent Character Lifting (ACTIVE)

> **Objective:** Fix the background bake-in contamination during stationary holds and eliminate character limb amputation, achieving 100% complete character mass and zero trapped sky cavities.

- [ ] **S1.1 — Spatiotemporal Atlas Inpainting & Bake-In Purge (`multiplane_atlas/mod.rs`, `layer_accum.rs`)**
  - Mask out active actor bounding tracks from temporal-mode accumulators in `build_multiplane_atlas`.
  - When actor presence exceeds $50\%$ of temporal frames at a locus, inpaint the occluded background using harmonic minimal-gradient continuation ($\nabla^2 I = 0$) from ambient background boundaries.
  - Verify that `bg_crop` behind Brisby's stationary hold locus contains pure rock/sky plenum with 0 fur texels.

- [ ] **S1.2 — Topological Cavity Disocclusion (`morphology.rs`, `articulated_wavepacket/mod.rs`)**
  - Implement `detect_enclosed_cavities`: flood-fill from raster borders to isolate all interior cavities $\{C_k\}$ enclosed by character silhouette boundaries.
  - Implement `disocclude_vacuum_cavities`: evaluate chromatic likelihood ratio $\Delta(C_k) = \frac{1}{|C_k|} \sum [d(p, \mathcal{H}) - d(p, \mathcal{V})]$. If $\Delta(C_k) > 0$, punch out $C_k$ from the silhouette mask.
  - Verify that closed appendage loops (e.g. Brisby arm holding rope) cleanly disocclude trapped yellow sunset sky ($[225, 135, 52]$) while preserving solid interior bodies.

- [ ] **S1.3 — Boundary Ringing Specks Purge (`articulated_wavepacket/mod.rs`)**
  - In `envelope_silhouette`, enforce `comp_sz >= 4` for contiguous satellites (`min_dist <= 2.5`), permanently eliminating isolated 1–2 px compression ringing specks around character perimeters.

- [ ] **S1.4 — Mario Complete Character Mass Restoration (`r10_model_sheet_gates.rs`)**
  - Verify Mario's boots, overalls, face, mustache, and red cap remain 100% intact across all 23 frames without limb amputation.
  - Achieve SMB3 Mario $\text{IoU} \ge 0.985$ and $\text{LPIPS} \le 0.005$.

- [ ] **S1.5 — Empirical Verification Gate across All 4 Benchmarks**
  - Verify zero regressions across full workspace test suite (`cargo test --workspace --tests`).
  - NIMH Mrs. Brisby achieves $\text{IoU} \ge 0.820$ and $\text{LPIPS} \le 0.050$ with zero yellow sky.
  - Pac-Man ($\text{IoU} = 1.000$) and Raptor ($\text{IoU} \ge 0.800$) remain green.

---

### Track S2 — Model Sheet Induction & Multi-Second Presentation Viewports (PLANNED)

> **Objective:** Upgrade video presentation from brief sub-second flashes to crisp, looped, multi-second character demonstrations.

- [ ] **S2.1 — Feature Animation Cel Presentation Scaling (`r10_model_sheet_gates.rs`)**
  - Enforce minimum $2\times$ integer scaling for feature animation cels ($236 \times 360$ canvas), eliminating cramped $1\times$ fallbacks.
- [ ] **S2.2 — Multi-Second Generative Video Export Loops (`r10_model_sheet_gates.rs`)**
  - Extend generative video export duration from 7 frames ($0.29\text{s}$) to full multi-second turnaround loops ($24\text{--}60\text{ frames}$, $\ge 2.0\text{s}$) with smooth cyclic playback.
- [ ] **S2.3 — Generative Video & Contact Sheet Re-Export (`generative_output/`)**
  - Re-export all 4 generative comparison MP4s and PNG contact sheets to `./generative_output/`:
    1. SMB3 Mario: [`smb3_mario_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4) | [`smb3_mario_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_contact_sheet.png)
    2. Pac-Man: [`pacman_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4) | [`pacman_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_contact_sheet.png)
    3. Raptor Jet: [`raptor_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_generative_unfold.mp4) | [`raptor_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_contact_sheet.png)
    4. NIMH Mrs. Brisby: [`nimh_brisby_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4) | [`nimh_brisby_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_contact_sheet.png)
  - Perform visual inspection of exported frames: verify 0 trapped sunset sky, 0 perimeter specks, crisp $2\times$ rendering, and complete body/limbs.

---

### Track S3 — Intrinsic Clifford Platforming & Controller Motor Injection (PLANNED)

> **Objective:** Implement physical platforming directly through Projective Geometric Algebra boundary predicates without a third-party physics engine.

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

- [ ] **S4.1 — CLI Subcommand `implicate play` (`implicate-cli`)**
  - Implement `implicate play --input <corpus.implicate>` launching an interactive `winit` / `wgpu` viewport.
- [ ] **S4.2 — Real-Time Interactive Loop ($\ge 60\text{ FPS}$)**
  - Decouple input processing ($1,000\text{ Hz}$ SIMD motor updates) from visual compute shader unfolding ($60\text{--}120\text{ FPS}$).
- [ ] **S4.3 — Interactive Platforming Verification**
  - Verify player can run left/right, jump with Space, land on bricks, and traverse the scrolling SMB3 stage.

---

### Track S5 — The Generative Storybook Engine (`implicate remix`) (PLANNED)

> **Objective:** Enable open-ended narrative animated film remixing from natural language prompts.

- [ ] **S5.1 — CLI Subcommand `implicate remix` (`implicate-cli`)**
  - Implement `implicate remix --video <corpus.implicate> --prompt "<prompt>"` generating novel scene animations.
- [ ] **S5.2 — Continuous Turnaround & Aerodynamic Cape Unfolding**
  - Blend smoothly between discrete view-aspect idempotents $\{\epsilon_v\}$ and continuous Clifford Fourier cape wavepackets based on prompted actions.
- [ ] **S5.3 — Synchronized 48 kHz Stereo Audio Synthesis**
  - Synthesize material-tuned modal audio impulses at character contact events.
