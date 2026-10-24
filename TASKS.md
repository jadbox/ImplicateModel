# ImplicateModel — Research & Experiment Tasks

> **Status:** Active Step-by-Step Engineering & Verification.

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to feature-length hand-drawn cinema) into a compact, symbolic Clifford world-and-verb lexicon, and unfolds new, controllable videos in a single causal wave pass at **real-time framerates ($\ge 60\text{ FPS}$ standard, $120+\text{ FPS}$ GPU target)**.

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)`
- **Benchmark Corpus:**
  1. `pacman_240x308.mp4` (`240x308`, discrete 4-direction arcade turns)
  2. `smb3_w1_1_256x240.mp4` (`256x240`, NES platformer gameplay @ $t \ge 20\text{s}$)
  3. `raptor_320x240.mp4` (`320x240`, PC DOS shaded sprites)
  4. `The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4` (`1920x1040`, `23.976 fps`, 1080p hand-drawn feature animation)

## Core Architectural Directives (The Hiley–Carmack Synthesis)
1. **Zero Magic Number Constants:**
   - Eliminate hardcoded pixel intensity cutoffs (`FOREGROUND_RESIDUAL = 26`, `MIN_OBSERVATION_PIXELS = 12`).
   - Use self-consistent statistical lattice noise floor: $\tau = \text{median}(\mathcal{I}) + 3.0 \times 1.4826 \times \text{MAD}(\mathcal{I})$.
   - Eliminate bounding-box aspect ratio clamping (`MAX_EXTENT_RATIO = 1.6`) and component amputations (`MIN_COMPONENT_FRACTION = 4`).
   - Eliminate stationary actor purge (`spread <= 0.5 * mean_extent`).
2. **Clifford Idempotent Soliton (CIS) & Don Bluth Model Sheets:**
   - Entities are not pixel boxes; they are minimal left ideals with invariant Line of Action $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ and canonical turnaround idempotents $\{\epsilon_v\}$.
   - Motion is governed by closed-form PGA Lie automorphisms $M = \exp(-\frac{\theta}{2} L)$ on $\text{Spin}(3,0,1)$.
   - Appendages and compliant structures (capes, cloaks, hair, wings, flags, tails) are generalized sub-holon parts with continuous `flow_compliance` ($\kappa_{\text{flow}} \ge 0$) and `spring_stiffness`.
3. **Mechanical Sympathy & Cache Coherence:**
   - 64-byte aligned SIMD multivector monads ([`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19)) matching AVX-512 registers and GPU cache lines.
   - Single-pass WGSL compute shader unfolding at $\ge 120\text{ FPS}$ on RTX 4090.
4. **Empirical Verification Only:**
   - Purge synthetic procedural mocks (`build_brisby_cape_turn_sequence`); all gates evaluate against real video footage.

---

## Track R10: The Clifford Idempotent Soliton & Don Bluth Model Sheet Engine (ACTIVE)

- [ ] **R10.1 — Hiley CDE Invariant Tracking & Active Residual Field (`implicate-lifter`)**
  - **Target Files:**
    - [`crates/implicate-lifter/src/articulated_wavepacket/mod.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs)
    - [`crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs)
  - **Objectives:**
    - Replace `FOREGROUND_RESIDUAL = 26` with statistical noise floor $\tau = \text{median}(\mathcal{I}) + 3.0 \times 1.4826 \times \text{MAD}(\mathcal{I})$ derived from residual multivector $\Delta \Psi$.
    - Replace screen-space constant velocity and box aspect clamping (`MAX_EXTENT_RATIO = 1.6`) with Hiley CDE bilinear geodesic affinity $\mathcal{A}(\rho_{c,1}, \rho_{c,2})$.
    - Remove `spread <= 0.5 * mean_extent` filter, preserving stationary, talking, and idling actors with 100% permanence.
  - **Gate Criteria:**
    - Recovers 100% of frames for Mario in real SMB3 World 1-1 gameplay (`smb3_w1_1_256x240.mp4` @ $t=25\text{--}28\text{s}$), Pac-Man and ghosts in `pacman_240x308.mp4`, and Mrs. Brisby in *The Secret of NIMH* without dropping stationary hold frames.

- [ ] **R10.2 — Clifford Dual Contour/Fill Projection & Line of Action (`implicate-lifter`, `implicate-math`)**
  - **Target Files:**
    - [`crates/implicate-lifter/src/articulated_wavepacket/clifford_features.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/clifford_features.rs)
    - [`crates/implicate-math/src/g301.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs)
  - **Objectives:**
    - Implement dual geometric product decomposition ($u \nabla^2 u = u \cdot \nabla^2 u + u \wedge \nabla^2 u$).
    - Eliminate connected component area cutoffs (`MIN_COMPONENT_FRACTION = 4`), preserving detached props (amulet, string), billowing cloaks, and trailing tails.
    - Extract 3D PGA Line of Action bivector $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ from spatial mass inertia tensor using 64-byte `Mv301`.
  - **Gate Criteria:**
    - Detached props and compliant appendages are preserved in extracted holons with zero component amputations. Line of Action bivector dynamically tracks spinal curvature through jumping and turnaround sequences.

- [ ] **R10.3 — Algebraic Idempotent Model Sheet Induction (`implicate-lifter`, `implicate-unfold`)**
  - **Target Files:**
    - [`crates/implicate-lifter/src/articulated_wavepacket/mod.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs)
    - [`crates/implicate-unfold/src/container.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/container.rs)
    - [`crates/implicate-unfold/src/payload.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/payload.rs)
  - **Objectives:**
    - Implement spectral idempotent decomposition on the temporal density covariance operator $\Gamma = \sum \sigma_k \epsilon_k$ to discover canonical turnaround view aspects $\{\epsilon_v\}$ without hardcoding view counts.
    - Compile representation into [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/PLAN.md#3-the-don-bluth-model-sheet-specification), defining invariant projective proportions and canonical turnaround poses.
    - Formally serialize `SubHolonPartSpec` with continuous `flow_compliance` and `spring_stiffness`.
  - **Gate Criteria:**
    - Automatically discovers canonical turnaround aspects (Pac-Man 4 directions; Mario Run/Jump/Skew; Mrs. Brisby Walk/Cape Wrap/Turnaround) with view classification accuracy $> 95\%$ against ground-truth keys.

- [ ] **R10.4 — Volume-Conserving Clifford Fourier Wavepacket & GPU Compute (`implicate-lifter`, `implicate-rolpa`, `implicate-unfold`)**
  - **Target Files:**
    - [`crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs)
    - [`crates/implicate-rolpa/src/lib.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-rolpa/src/lib.rs)
    - [`crates/implicate-unfold/src/engine.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/engine.rs)
  - **Objectives:**
    - Upgrade `WavepacketPartKernel` to separate invariant ink contour blades ($W = u \wedge \nabla^2 u$) from cel paint fills ($D = u \cdot \nabla^2 u$).
    - Parameterize continuous gait deformations along phase $\phi \in S^1$ with volume conservation ($\det S = 1$), eliminating Gibbs color ringing and blurry outline mush.
    - Implement direct WGSL / GPU compute shader kernel evaluating `Mv301` solitons in a single pass.
  - **Gate Criteria:**
    - Unfolded character reconstruction achieves $\text{IoU} \ge 0.985$ and $\text{LPIPS} \le 0.005$ across all frames of continuous turnaround and walk cycles, maintaining sharp ink contours. GPU unfoldment throughput $\ge 120\text{ FPS}$ on RTX 4090.

- [ ] **R10.5 — Real-Video Empirical Gate Suite (`tests/r10_model_sheet_gates.rs`)**
  - **Target Files:**
    - `crates/implicate-lifter/tests/r10_model_sheet_gates.rs`
    - `crates/implicate-lifter/tests/r2_gates.rs`
  - **Objectives:**
    - Purge all procedural synthetic ellipse sequences (`build_brisby_cape_turn_sequence`) from `r2_gates.rs`.
    - Test end-to-end model sheet induction and generative unfoldment directly against real video benchmarks:
      1. *The Secret of NIMH* (BluRay 1080p hand-drawn feature animation).
      2. *Super Mario Bros. 3* (NES 2D platformer with variable camera scroll).
      3. *Pac-Man* (Arcade 2D multi-actor discrete turns).
      4. *Raptor: Call of the Shadows* (PC DOS vertical shoot-'em-up with shaded sprites).
  - **Gate Criteria:**
    - All 4 benchmark clips extract valid `HolonModelSheet` representations with zero hardcoded magic number constants. Real-time unfoldment maintains $\ge 60\text{ FPS}$ on RTX 4090 with zero temporal flicker and sample-accurate audio-visual synchronization.
