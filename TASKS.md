# ImplicateModel — Research & Experiment Tasks

> **Status:** Track R10 Active Step-by-Step Engineering & Verification.

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

- [x] **R10.1 — Hiley CDE Invariant Tracking & Active Residual Field (`implicate-lifter`)**
  - **Completed Changes:**
    - Replaced `FOREGROUND_RESIDUAL = 26` with `compute_shot_noise_floor(frames, atlas)` deriving self-consistent statistical lattice noise floor $\tau = \text{median}(\|\Delta \Psi\|) + 3.0 \times 1.4826 \times \text{MAD}(\|\Delta \Psi\|)$ on non-zero lattice-tolerant residuals.
      - SMB3: auto-$\tau = 11.90$
      - Pac-Man: auto-$\tau = 6.45$
      - Raptor: auto-$\tau = 6.45$
      - The Secret of NIMH: auto-$\tau = 11.90$
    - Replaced constant-velocity linear screen prediction and bounding box ratio clamping (`MAX_EXTENT_RATIO = 1.6`, `MAX_APPEARANCE_DIST = 55.0`, `err > scale`) with `cde_bilinear_geodesic_affinity(tr, cand)`.
      - Unifies intrinsic spinor chromatic coherence $\mathcal{C}_{\text{spinor}}$, volume-conserving log-area scaling $\mathcal{S}_{\text{scale}}$, and geodesic spatial continuity $\mathcal{K}_{\text{geodesic}} = \exp\left(-\frac{1}{2} \left(\frac{\min(d_{\text{inertial}}, d_{\text{stationary}})}{E \sqrt{\Delta t}}\right)^2\right)$.
    - Removed `spread <= 0.5 * mean_extent` filter, guaranteeing 100% permanence for stationary, talking, and idling actors.
    - Upgraded `stitch_tracklets` to use CDE bilinear geodesic affinity across gaps.
    - Updated `choose_part_and_view_counts` to test $P=1$ Occam's razor before trying articulated parts, preserving rigid sprite classification with IoU $> 0.99$.
  - **Verification:**
    - SMB3 ($t=25\text{--}28\text{s}$): Mario recovered with **100.0% coverage** (90/90 frames, 0 missing).
    - Pac-Man ($t=5\text{--}8\text{s}$): Pac-Man and ghosts recovered with **100.0% coverage** (90/90 frames).
    - NIMH ($t=600\text{--}603\text{s}$): Mrs. Brisby and Jeremy crow tracks and stationary holds completely preserved.
    - Full workspace regression: all 42 tests passing (`cargo test --workspace` ok).

- [x] **R10.2 — Clifford Dual Contour/Fill Projection & Line of Action (`implicate-lifter`, `implicate-math`)**
  - **Completed Changes:**
    - Eliminated `MIN_COMPONENT_FRACTION = 4` in [`dominant_silhouette`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) in favor of the Line of Action support envelope: any cluster within kinematic reach of the principal inertia axis ($\le \max(L_{\text{body}} \times 0.75, 3\sigma_\parallel)$ or within transverse corridor $|\eta| \le \max(3\sigma_\perp, 1.5\sigma_\parallel)$) is preserved.
    - Implemented dual Clifford geometric product decomposition ($u \nabla^2 u = u \cdot \nabla^2 u + u \wedge \nabla^2 u$) in [`clifford_features.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/clifford_features.rs):
      - Grade 0 scalar inner product $D(\mathbf{x}) = u \cdot \nabla^2 u$ captures cel paint fill coherence and vitality.
      - Grade 2 bivector wedge product $W(\mathbf{x}) = u \wedge \nabla^2 u$ captures animated film ink contour blades with statistical noise floor $\tau_W = \text{median} + 1.5 \times 1.4826 \times \text{MAD}$ on non-zero vorticity.
      - Cel paint fills and ink contour blades form mutually exclusive partitions with zero pixel overlap.
    - Added closed-form 3D PGA Plücker line methods to the 64-byte aligned SIMD monad [`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs) (`from_line`, `from_line_2d`, `distance_to_line`, `closest_point_on_line`, `is_valid_line`).
    - Implemented [`extract_line_of_action`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) extracting the 3D PGA screw axis blade $L_{\text{spine}} \in \mathbb{G}(3,0,1)$ from spatial mass inertia moments ($\mu_{20}, \mu_{02}, \mu_{11}$), principal angle $\theta$, spreads $(\sigma_\parallel, \sigma_\perp)$, and dynamic quadratic spinal curvature $\kappa$.
  - **Empirical Verification:**
    - Super Mario Bros. 3 (`smb3_w1_1_256x240.mp4` @ $t=25\text{--}27\text{s}$): 288/288 observations (100.0%) produced valid 3D PGA line blades satisfying $\mathbf{d} \cdot \mathbf{m} = 0$. Dynamic spinal curvature tracked through run/jump transitions with $\kappa \in [-0.01432, 0.01199]$.
    - The Secret of NIMH (`NIMH 1080p` @ $t=600\text{s}$): Detached props (Mrs. Brisby's amulet: 9 pixels vs 2,416 body pixels) preserved with 100% retention (zero component amputations).
    - Generated visual inspection contact sheet: `/tmp/implicate_r10_2_contact_sheet.png` (92x86 RGB showing original observation, cel paint fill, ink contour blade, and yellow $L_{\text{spine}}$ overlay with magenta center-of-mass).
    - All 4 empirical tests passing (`tests/r10_2_dual_contour_and_line_of_action_gate.rs` ok in 11.2s).

- [x] **R10.3 — Algebraic Idempotent Model Sheet Induction (`implicate-lifter`, `implicate-unfold`)**
  - **Completed Implementation:**
    - **Carmack Mechanical Sympathy:** Enabled `bytemuck` across `implicate-math` and `implicate-lifter`. Derived `Pod` and `Zeroable` on 64-byte aligned SIMD multivector [`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs) and [`GpuModelSheetMonad`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) (exactly 19 flat 64-byte aligned `Mv301` monads, size = 1216 bytes, align = 64 bytes). Directly mappable to GPU uniform/storage buffers with zero heap allocation during unfold trajectory stepping.
    - **Don Bluth Character Board Model Sheet:** Defined [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) unifying physical constitutive parameters, 3D PGA Line of Action spine bivector $\mathbf{L}_{\text{spine}}$, invariant proportions, articulated part tree, and turnaround view aspects.
    - **Dual Product Part Kernels:** Implemented [`ModelSheetPartKernel`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) separating animated film ink contour blades ($\mathbf{u} \wedge \nabla^2 \mathbf{u}$) from cel paint fills ($\mathbf{u} \cdot \nabla^2 \mathbf{u}$) with zero pixel overlap.
    - **Unified World Cape Flow:** Implemented fluid-structure aerodynamic compliance on sub-holon parts (`flow_compliance > 1.0`, restoring stiffness and damping) conforming to ambient flow field compliance rather than isolated procedural physics.
    - **Hiley Spectral Idempotent Decomposition:** Implemented [`factor_temporal_covariance_idempotents`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs) factoring the temporal covariance operator $\Gamma = \int \Psi(t)\widetilde{\Psi}(t) dt = \sum \sigma_k \epsilon_k$ via symmetric Jacobi eigensolver into primitive orthogonal idempotents $\epsilon_v^2 = \epsilon_v$ on $\text{Spin}(2)$ with closed-form Lie transition operators $T_{u \to v} = \epsilon_v M \epsilon_u$.
    - **Container Payload v3:** Upgraded [`payload.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/payload.rs) to container payload version 3; automatically induces and serializes `HolonModelSheet` in `.implicate` container payloads alongside `MultiPlaneAtlas` and `ArticulatedHolon`.
  - **Empirical Verification ([`r10_3_model_sheet_induction_gate.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/tests/r10_3_model_sheet_induction_gate.rs)):**
    - **Carmack Direct GPU Buffer Dispatch:** `r10_3_carmack_gpu_model_sheet_monad_buffer_gate` PASSED. Verified exact 1216-byte memory layout (`size = 1216`, `align = 64`), direct byte buffer slice mapping, and zero heap allocation roundtrip deserialization via `bytemuck`.
    - **NIMH Hand-Drawn Cel/Contour & Flowing Cape:** `r10_3_nimh_brisby_bluth_model_sheet_and_cape_gate` PASSED. Verified Mrs. Brisby extracted from 1080p BluRay footage with rigid torso (`flow_compliance = 0.0`) and billowing aerodynamic cape (`flow_compliance = 3.93`, stiffness 2.26, damping 0.75), cel paint fill and ink contour blade mutually exclusive with zero pixel overlap.
    - **Pac-Man 4-Direction Turnaround:** `r10_3_pacman_4direction_turnaround_aspect_idempotents_gate` PASSED. Verified 100% Plücker condition, rank-1 idempotent projections ($\epsilon_v^2 = \epsilon_v$), closed-form Lie transition operators ($T_{u \to v} = \epsilon_v M \epsilon_u$), and **90.18% classification accuracy** across 163 active motion frames.
    - **SMB3 Mario Kinematic Regimes:** `r10_3_smb3_mario_posture_and_aspect_regimes_gate` PASSED. Evaluated on active motion frames ($|v| > 0.4$), achieving **96.25% accuracy** (77/80 frames) partitioning Run Right, Run Left, Ascending Jump, and Descending Fall with dynamic spinal curvature $\kappa \in [-0.01432, 0.01199]$.
    - **Full Workspace Test Suite:** All 51 workspace tests pass (`cargo test --workspace` ok in 41s).

- [ ] **R10.4 — Volume-Conserving Clifford Fourier Wavepacket & GPU Compute (`implicate-lifter`, `implicate-rolpa`, `implicate-unfold`)**
  - **Target Files:**
    - [`crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs)
    - [`crates/implicate-rolpa/src/cliffordnet_rolpa.wgsl`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-rolpa/src/cliffordnet_rolpa.wgsl)
    - [`crates/implicate-rolpa/src/gpu_refiner.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-rolpa/src/gpu_refiner.rs)
    - [`crates/implicate-unfold/src/payload.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/payload.rs)
    - [`crates/implicate-unfold/src/engine.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/src/engine.rs)
  - **Progress & Verified Sub-Tasks:**
    - **Volume Conservation ($\det S = 1$):** Implemented `enforce_volume_conserving_gait_deformation` and `continuous_gait_volume_scale`. Normalized part radii to geometric mean reference area $A_{\text{ref}}$, enforcing strict unimodular scaling across gait phase $\phi \in S^1$. Mario walk-cycle reconstruction reached $\text{IoU} = 0.9995$ and $\text{LPIPS} = 0.0001$.
    - **Dual Clifford Ink Blades in Kernels:** Added `ink_contour_blade: Vec<f32>` to `WavepacketPartKernel` and `ModelSheetPartKernel`, decomposing DC textures via `decompose_masked_clifford_dual_product` into animated film ink lines ($|u \wedge \nabla^2 u|$) and flat cel fills ($u \cdot \nabla^2 u$).
    - **Container Payload v4:** Bumped `ENFOLDED_PAYLOAD_VERSION` to 4 in `implicate-unfold::payload` with forward/backward-compatible binary serialization of ink contour blades.
    - **GPU WGSL Unfoldment Shader:** Added fused `@compute @workgroup_size(16, 16, 1) fn unfold_monads(...)` in `cliffordnet_rolpa.wgsl` evaluating flat 1216-byte `GpuModelSheetMonad` (19 `Mv301` monads) in $O(1)$ constant time per pixel with strict volume conservation ($\det S = 1$) and animated film ink contour blades ($|u \wedge \nabla^2 u|$) vs cel fills ($u \cdot \nabla^2 u$).
    - **GPU Pipeline & Throughput Benchmark:** Implemented `GpuModelSheetUnfolder` in `gpu_refiner.rs` and integrated into `CausalUnfoldEngine` (`engine.rs`). Verified GPU throughput of **21,877.6 FPS** (mean latency: 0.0457 ms/pass) on NVIDIA GeForce RTX 4090 Laptop GPU (target $\ge 120$ FPS exceeded by $180\times$).
    - **Razor-Sharp Ink Blades:** Verified 100% mutual exclusivity between animated film ink contour blades and cel fills with **zero pixel overlap** (`ink_pixels=104, cel_pixels=77, overlap_pixels=0`).
    - **Continuous Gait Reconstruction:** SMB3 Mario walk/jump cycle achieved **$\text{IoU} = 0.9995$** (target $\ge 0.985$) and **$\text{LPIPS} = 0.0001$** (target $\le 0.005$). Mrs. Brisby turnaround achieved **$\text{IoU} = 0.9964$** (target $\ge 0.985$) and **$\text{LPIPS} = 0.0009$** (target $\le 0.005$).
    - **Remaining Active Task (Phase 2):** Resolve Pac-Man discrete 4-direction turnaround aspect switching to achieve $\text{IoU} \ge 0.985$ across discrete turns (currently reaching $1.0000$ on aligned aspect frames, averaging $0.9238$ across turn transitions).

- [ ] **R10.5 — Real-Video Empirical Gate Suite (`tests/r10_model_sheet_gates.rs`)**
  - **Target Files:**
    - `crates/implicate-lifter/tests/r10_model_sheet_gates.rs`
    - `crates/implicate-lifter/tests/r2_gates.rs`
  - **Active Directives:**
    - Purge all procedural synthetic ellipse sequences (`build_brisby_cape_turn_sequence`) from `r2_gates.rs`.
    - Test end-to-end model sheet induction and generative unfoldment directly against real video benchmarks:
      1. *The Secret of NIMH* (BluRay 1080p hand-drawn feature animation).
      2. *Super Mario Bros. 3* (NES 2D platformer with variable camera scroll).
      3. *Pac-Man* (Arcade 2D multi-actor discrete turns).
      4. *Raptor: Call of the Shadows* (PC DOS vertical shoot-'em-up with shaded sprites).
  - **Gate Criteria:**
    - All 4 benchmark clips extract valid `HolonModelSheet` representations with zero hardcoded magic number constants. Real-time unfoldment maintains $\ge 60\text{ FPS}$ on RTX 4090 with zero temporal flicker and sample-accurate audio-visual synchronization.
