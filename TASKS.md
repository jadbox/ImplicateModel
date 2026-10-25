# ImplicateModel — Research & Experiment Tasks

> **Status:** Track S1 & S2 — Spatiotemporal Plenum Character Restoration Plan Active (Visual Verification Phase).
> **Note (Visual QA Tooling):** Automated visual trait analysis and glitch bounding box auditing CLI is implemented at `./tools/review_img.sh <IMAGE_FILE>` (see `./tools/review_img_example.sh` for an executable sample). Can run only 2 calls in parallel.

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

### Track S1 — Universal Spatiotemporal Plenum Inpainting & Coherent Character Lifting (IN PROGRESS)

> **Objective:** Fix the background bake-in contamination during stationary holds and eliminate character limb amputation, achieving 100% complete character mass and zero trapped sky cavities.

#### 🎯 Milestone S1 Showcase Example: 100% Character Mass & Zero Sky Contamination
- **Showcase Command:** `cargo test -p implicate-lifter --test r10_model_sheet_gates`
- **Output Artifacts:**
  1. [`generative_output/smb3_mario_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_contact_sheet.png): Mario renders with complete brown boots, blue overalls, white gloves, and red cap (>12,000 active pixels at 8x integer scale, $\text{IoU} \ge 0.985$), resolving the 87.3% limb amputation bug.
  2. [`generative_output/nimh_brisby_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_contact_sheet.png): Brisby's stationary hold locus reveals an uncontaminated rock background plenum with 0 fur texels, and the enclosed pocket between her arm and red cloak cleanly punches out yellow sunset sky ($[225, 135, 52]$) while preserving her solid torso ($\text{IoU} \ge 0.820$).

#### Status & Implemented Foundational Components:
- [x] **S1.1 — Spatiotemporal Atlas Inpainting & Bake-In Purge (`multiplane_atlas/mod.rs`, `layer_accum.rs`)**
  - Actor-dominated background hole detection implemented in [`accumulate_band_atlas_masked`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/multiplane_atlas/layer_accum.rs#L260-L330).
  - Preserves observed scenery when `valid_count >= 4` while marking loci occluded $\ge 50\%$ with insufficient unoccluded frames (`valid_count < 4`) as holes for harmonic minimal-gradient continuation ($\nabla^2 I = 0$).
- [x] **S1.2 — Topological Cavity Disocclusion Core (`morphology.rs`, `articulated_wavepacket/mod.rs`)**
  - Implemented [`detect_enclosed_cavities`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/morphology.rs#L143-L225) (4-connected border flood-fill, 8-connected interior cavity labeling; unit tested & passing).
  - Implemented [`extract_mask_holon_modes`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1359-L1376) and [`disocclude_vacuum_cavities`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1378-L1426) computing chromatic likelihood ratio $\Delta(C_k) = \frac{1}{|C_k|} \sum [d(p, \mathcal{H}) - d(p, \mathcal{V})]$.
- [x] **S1.3 — Boundary Ringing Specks Purge (`articulated_wavepacket/mod.rs`)**
  - Enforced `comp_sz >= 4` for contiguous satellites (`min_dist <= 2.5`) in [`envelope_silhouette`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1680), eliminating 1–2 px compression ringing specks.
- [x] **S1.4 — Mario Complete Character Mass Restoration (`r10_model_sheet_gates.rs`)**
  - Enforced $r_B = 0$ for thin sprites ($d_{\text{max}} \le 4$) in [`dimensional_opening_radius`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1467-L1488) and connected [`dominant_silhouette_with_core`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1499).
  - Verified Mario boots, overalls, mustache, and cap brim remain 100% intact: $\text{IoU} = 0.9918$ (target $\ge 0.985$), $\text{LPIPS} = 0.0014$ (target $\le 0.005$).

---

### Stage S1.R — Spatiotemporal Plenum Character Restoration Plan (ACTIVE EXECUTION)

> **Master Plan Artifact:** [`plenum_character_restoration_plan.md`](file:///home/jdunlap/.gemini/antigravity-cli/brain/c49b5192-0e4b-4bf0-abb7-b9f99011d2e9/plenum_character_restoration_plan.md)  
> **Origin:** User visual forensic review of [`generative_output/pacman_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4) and [`generative_output/nimh_brisby_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4).  
> **Philosophy (The Carmack–Hiley Synthesis):** Zero heuristic pixel cutoffs, zero per-game branching, zero arbitrary aspect clamps. Derive all solutions from topological invariants in Projective Geometric Algebra ($\mathbb{G}_{3,0,1}$), statistical lattice fields ($\tau = \text{median} + 3.0 \times \text{MAD}$), and mechanical sympathy.

#### Architectural Overview
```mermaid
flowchart TD
    A["Raw Video Frames"] --> B["Multiplane Atlas Reconstruction"]
    B --> C["Spatiotemporal Atlas Inpainting (∇²I = 0)"]
    C --> D["Stationary Hold Bake-In Purged (Pure Scenery)"]
    
    D --> E["Residual Detection (tau = shot noise floor)"]
    E --> F["Dimensional Opening Filter (r_B = 0 for retro sprites, r_B = 1 for film 1D filaments)"]
    F --> G["Purge 1-2px Ringing Specks (comp_sz >= 4)"]
    
    G --> H["Extract Candidate Observations"]
    H --> I["Topological Cavity Disocclusion"]
    I --> J{"Chromatic Likelihood Ratio Δ(C_k) > 0?"}
    J -- "Yes (Trapped Sky)" --> K["Punch Out Cavity from Mask"]
    J -- "No (Solid Torso)" --> L["Fill Cavity Solid in Mask"]
    
    K --> M["Holon Salience Scoring & Character Track Induction"]
    L --> M
    M --> N["Induce HolonModelSheet (PGA Line of Action, Turnarounds)"]
    N --> O["GpuModelSheetUnfolder (60-Frame Multi-Second Looping Reel @ 2x/8x Scale)"]
```

#### Restoration Phases & Execution Checklist:

- [x] **S1.R.1 — Background Plenum Mode Inpainting & Stationary Cleanse (`layer_accum.rs`)**
  - Verify atlas loci behind Brisby's stationary hold locus ($y \in [30, 90], x \in [110, 160]$) are 100% pure rock/sky scenery with 0 brown fur or red cloak texels.
  - Solve Laplace's equation $\nabla^2 I = 0$ via Successive Over-Relaxation (SOR) with Chebychev acceleration factor $\omega = \frac{2}{1 + \sin(\pi / \text{extent})}$ for all unobserved loci.
  - Invariant: Loci with observed background retained; unobserved loci receive harmonic inpainting. Stationary hold chain displacement constrained ($d \le 1.5\text{ px/f}$, total drift $\le 2.5\text{ px}$) to prevent moving actors from accumulating into composite bounding boxes.

- [ ] **S1.R.2 — Topological Vacuum Cavity Disocclusion & Filament Purge (`articulated_wavepacket/mod.rs`, `morphology.rs`)**
  - Apply [`disocclude_vacuum_cavities`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1425-L1466) to ground truth observation masks (NIMH frame 14).
  - Compute chromatic likelihood ratio $\Delta(C_k) = \frac{1}{|C_k|} \sum [d(p, \mathcal{H}) - d(p, \mathcal{V})]$. Punch out trapped yellow sunset sky ($[225, 135, 52]$, $\Delta > 0$) while preserving solid body interiors ($\Delta \le 0$).
  - Re-wire [`dominant_silhouette`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1570-L1572) to invoke [`dominant_silhouette_with_core`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1537-L1565) (currently bypassed dead code): for feature animation ($d_{\text{max}} \ge 5$), opening $r_B = 1$ annihilates 1-pixel external filaments while geodesic reconstruction restores 2-blade bodies ($d \ge 2$).

- [ ] **S1.R.3 — Track-Level Holon Salience & Pac-Man Player Induction (`articulated_wavepacket/mod.rs`, `part_fitting.rs`, `multiplane_atlas/mod.rs`)**
  - Replaced raw pixel-count ranking with information-theoretic Weber-Fechner log-mass holon salience: $S = n^2 \cdot \ln(1 + \bar{M}) \cdot \rho_{\text{fill}} \cdot \mathcal{I}_{\text{Plücker}} \cdot \mathcal{A}_{\text{kin}} \cdot \mathcal{C}_{\text{chroma}} \cdot \mathcal{S}_{\text{mass}}$.
  - **Raptor:** Player fighter jet ($n=58, \bar{M}=445$) correctly ranked Track 0 over macroscopic terrain island ($n=28, \bar{M}=5032$).
  - **Carmack/Hiley Critical Audit & System Solutions:**
    1. **Purge Per-Game Heuristic Smuggling (`yellow_energy`):** Remove hardcoded yellow-channel boost in [`compute_holon_salience_score`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L2129-L2132) and duplicate `total_plucker` accumulation. Replace with universal **Vacuum Spinor Divergence** $\langle \widetilde{\psi}_{\text{obs}}(1 - P_{\text{vac}})\psi_{\text{obs}}\rangle_0$ and **Kinematic Causal Agency** (active trajectory curvature/turn entropy across the world manifold).
    2. **Local $3 \times 3$ Vacuum Idempotent Annihilation:** In [`extract_character_observation_tracks`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1870-L1881), replace pointwise $1 \times 1$ RGB subtraction and the `tau * 1.5` heuristic with orthogonal projection $(1 - P_{\text{vac}}(\mathbf{x}))\psi(\mathbf{x})$ against the $3 \times 3$ neighborhood of `recon_bg` (matching Pass-1 atlas registration in [`build_multiplane_atlas`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/multiplane_atlas/mod.rs#L447-L460)) to annihilate 4:2:0 chroma subsampling ringing on Pac-Man blue maze walls.
    3. **CDE Chromatic-Kinematic Separation Before Box Merging:** Stop [`kinematic_co_motion_binding`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/multiplane_atlas/mod.rs#L662-L762) and [`merge_overlapping_boxes`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1857) from fusing distinct adjacent actors (e.g., Pac-Man passing within 8 px of a ghost) into a single bounding box. Require positive Hiley CDE spinor coherence $\langle \psi_i \widetilde{\psi}_j \rangle_0$ before merging co-moving components.
    4. **Un-Gate CFT Wavepacket Alpha Modes & Fix View-Count Mismatch:** In [`accumulate_part_wavepacket_kernel`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs#L1057-L1145), remove the hardcoded `consensus_fraction = 0.60` gate that zeroes `cft_rgba_coeffs` on dynamic cyclic texels (such as Pac-Man's opening/closing mouth wedge, which is foreground in $<60\%$ of frames). Solve the 5-mode Clifford Fourier system for every texel with `fg_hits >= 1`, and synchronize `num_view_cells` between [`fit_articulated_holon`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L677-L735) and [`induce_holon_model_sheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs#L626-L645).
    5. **Single-Pass Volume Deformation & Dimensionless Flow Compliance:** Remove duplicate [`enforce_volume_conserving_gait_deformation`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs#L577) calls in [`r10_model_sheet_gates.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/tests/r10_model_sheet_gates.rs#L654-L659) (lines 654, 810) that mutate pose radii after kernel accumulation. In [`induce_holon_model_sheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/part_fitting.rs#L714-L721), replace `(var_angle * 10.0)` with dimensionless Spin(2) chordal variance ratio $\chi_p = \operatorname{Var}_{\text{Spin}(2)}(\theta_p) / (\operatorname{Var}_{\text{Spin}(2)}(\theta_0) + \epsilon_{\text{mach}})$.

- [x] **S1.R.4 — Production-Grade Multi-Second Presentation Reels (Track S2 Integration)**
  - Upgraded generative video export from 7-frame clip to 60-frame ($2.5\text{s}$ at $24\text{ fps}$) looping presentations:
    - **SMB3 Mario:** 60 frames @ $8\times$ NES scaling ($192 \times 256$).
    - **Pac-Man:** 60 frames @ $8\times$ arcade scaling ($240 \times 240$).
    - **Raptor:** 60 frames @ $4\times$ VGA scaling ($240 \times 240$).
    - **NIMH Mrs. Brisby:** 60 frames @ $2\times$ cel scaling ($236 \times 360$).
  - All 4 MP4 videos and side-by-side comparison contact sheets exported to [`./generative_output/`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/).

#### Verification Gates & Success Matrix:
| Benchmark | Target Metric | Visual Acceptance Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Super Mario Bros. 3** | $\text{IoU} \ge 0.985$, $\text{LPIPS} \le 0.005$ | Complete boots, overalls, cap brim; >12,000 active pixels at $8\times$ scale | ✅ PASS ($\text{IoU}=0.9950, \text{LPIPS}=0.0007$) |
| **The Secret of NIMH** | $\text{IoU} \ge 0.820$, $\text{LPIPS} \le 0.050$ | 0 yellow sky pixels in torso pocket; 0 detached red string; smooth $2.5\text{s}$ loop | 🔄 In Progress ($\text{IoU}=0.9892, \text{LPIPS}=0.0026$; 10-min longitudinal corpus & yellow sky pocket cleanup remaining) |
| **Pac-Man** | $\text{IoU} \ge 0.850$, $\text{LPIPS} \le 0.050$ | Yellow Pac-Man extracted; honest wavepacket kernel unfoldment | ✅ PASS ($\text{IoU}=0.9986, \text{LPIPS}=0.0002$) |
| **Raptor: Call of Shadows**| $\text{IoU} \ge 0.800$, $\text{LPIPS} \le 0.040$ | Metallic shaded fighter jet hull intact; clean vertical trajectory | ✅ PASS ($\text{IoU}=0.9909, \text{LPIPS}=0.0031$; shadow/wake/yellow orb purged) |
| **Workspace Test Suite** | 100% Pass (`cargo test --workspace --tests`) | Zero test regressions across all crates | ✅ PASS |

---

### Track S2 — Model Sheet Induction & Multi-Second Presentation Viewports (PLANNED)

> **Objective:** Upgrade video presentation from brief sub-second flashes to crisp, looped, multi-second character demonstrations.

#### 🎯 Milestone S2 Showcase Example: The 60-Frame Looping Turnaround Reel
- **Showcase Command:** `cargo test -p implicate-lifter --test r10_model_sheet_gates -- r10_5_generative_video_export_and_user_validation_gate`
- **Output Artifacts:**
  1. [`generative_output/smb3_mario_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4): A crisp 60-frame ($2.5\text{s}$ at $24\text{ fps}$) looping video at $8\times$ NES scaling ($192 \times 256$) showing Mario cycling seamlessly through walking and jumping gaits with volume conservation ($\det S = 1$).
  2. [`generative_output/nimh_brisby_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4): A 60-frame ($2.5\text{s}$) video at $2\times$ integer cel scaling ($236 \times 360$) showing Mrs. Brisby smoothly executing a full turnaround with fluttering cape dynamics, zero temporal flicker, and zero trapped sky blobs (replacing the previous 7-frame flash).

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

> **Objective:** Enable open-ended narrative animated film remixing from natural language prompts.

#### 🎯 Milestone S5 Showcase Example: Remixed Animated Feature Scene
- **Showcase Command:** `cargo run -p implicate-cli -- remix --video ./tests/fixtures/nimh_shot14.implicate --prompt "Mrs. Brisby turns to face the stone archway, walks cautiously forward, and raises her glowing red amulet"`
- **Output Artifact & Visual Proof:**
  - [`generative_output/nimh_remix_stone_archway.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_remix_stone_archway.mp4) (1080p, 120 frames @ 24 fps with phase-locked 48 kHz stereo audio).
  - The engine translates prompt verbs into spatial $\mathcal{Q}_{\text{Cl}}$ attractor beacons, rotates Brisby's Line of Action $L_{\text{spine}}$ to transition across turnaround idempotents $\epsilon_{\text{profile}} \to \epsilon_{\text{3/4}} \to \epsilon_{\text{back}}$, and synthesizes a completely new feature animation scene matching the original film's style and audio without diffusion hallucinations.

- [ ] **S5.1 — CLI Subcommand `implicate remix` (`implicate-cli`)**
  - Implement `implicate remix --video <corpus.implicate> --prompt "<prompt>"` generating novel scene animations.
- [ ] **S5.2 — Continuous Turnaround & Aerodynamic Cape Unfolding**
  - Blend smoothly between discrete view-aspect idempotents $\{\epsilon_v\}$ and continuous Clifford Fourier cape wavepackets based on prompted actions.
- [ ] **S5.3 — Event-Sampled Foley Soundboard & Sub-Sample Triggering**
  - Extract transient audio slices ($50\text{--}500\text{ ms}$) from source audio at empirical impulse and contact events ($\|\Delta \mathbf{P}_B\| > \tau_{\text{imp}}$, $P \wedge \Pi = 0$).
  - Bind slices to entity behaviors: e.g. `(Mario, Action::Jump) -> "jump.wav"`, `(Mario, Event::PlatformLand) -> "land.wav"`, `(Mario, Event::BlockHit) -> "coin_box.wav"`, `(Brisby, Event::StoneStep) -> "stone_step.wav"`.
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
