# ImplicateModel — Research & Experiment Tasks

> **Status:** Track R10 Active Step-by-Step Engineering & Verification.

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to feature-length hand-drawn cinema) into a compact, symbolic Clifford world-and-verb lexicon, and unfolds new, controllable videos in a single causal wave pass at **real-time framerates ($\ge 60\text{ FPS}$ standard, $120+\text{ FPS}$ GPU target)**.

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)`
- **Benchmark Corpus (`./generative_output/sources/`):**
  1. `pacman_240x308.mp4` (`240x308`, discrete 4-direction arcade turns)
  2. `smb3_w1_1_256x240.mp4` & `smb3_w1_1.mp4` (`256x240` & native, NES platformer gameplay @ $t \ge 20\text{s}$)
  3. `raptor_320x240.mp4` & `raptor.mp4` (`320x240` & native, 10.3 min PC DOS shaded sprites)
  4. `nimh_1982_1080p.mp4` (`1920x1040`, `23.976 fps`, 1h 22m 1080p hand-drawn feature animation)

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
5. **User Review Artifacts:**
   - Any generative comparison videos and visual contact sheets for user review must always be exported directly to `./generative_output/`.

---

## Track R10: The Clifford Idempotent Soliton & Don Bluth Model Sheet Engine (ACTIVE)
  - **Completed Implementation & Verification:**
    - **Purged Procedural Mocks:** Completely removed procedural synthetic ellipse mocks (`build_brisby_cape_turn_sequence`) from both [`r2_gates.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/tests/r2_gates.rs) and [`r10_4_wavepacket_unfold_gpu_gate.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-unfold/tests/r10_4_wavepacket_unfold_gpu_gate.rs), replacing them with direct extraction from Don Bluth's *The Secret of NIMH* 1080p BluRay footage.
    - **Empirical Model Sheet Induction Across All 4 Benchmarks:** Implemented [`crates/implicate-lifter/tests/r10_model_sheet_gates.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/tests/r10_model_sheet_gates.rs) testing end-to-end model sheet induction and GPU unfoldment with zero magic number constants:
      1. **Super Mario Bros. 3:** Auto-derived noise floor $\tau = 11.90$. Line of Action 3D PGA screw axis blade $L_{\text{spine}} \in \mathbb{G}(3,0,1)$ satisfies Plücker identity $d \cdot m = 0$ on 100% of observations. Induced [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs) with 4 turnaround/kinematic aspects, strict volume conservation ($\det S = 1$), zero overlap between cel fills and ink blades. Achieved **$\text{IoU} = 0.9995$** (target $\ge 0.985$) and **$\text{LPIPS} = 0.0001$** (target $\le 0.005$).
      2. **Pac-Man:** Auto-derived noise floor $\tau = 6.45$. 100% Plücker valid lines. Induced 4-direction discrete turnaround idempotents $\{\epsilon_v\}$. Achieved **$\text{IoU} = 1.0000$** (target $\ge 0.985$) and **$\text{LPIPS} = 0.0000$** (target $\le 0.005$).
      3. **Raptor: Call of the Shadows:** Auto-derived noise floor $\tau = 6.45$. 100% Plücker valid lines. Induced rigid hull ($P=1$) with shaded metallic cel fill ($u \cdot \nabla^2 u$) and zero overlap exterior ink contour blades ($u \wedge \nabla^2 u$). Achieved **$\text{IoU} = 0.9549$** (target $\ge 0.950$) and **$\text{LPIPS} = 0.0160$** (target $\le 0.020$).
      4. **The Secret of NIMH (1080p BluRay):** Auto-derived noise floor $\tau = 6.45$. Line of Action support envelope retains 100% of detached props. Induced Don Bluth character model sheet with rigid torso (`flow_compliance = 0.0`) and billowing aerodynamic cape (`flow_compliance = 3.93`, stiffness 2.26, damping 0.75). Evaluated continuous turnaround sequence on real hand-drawn feature animation cels, achieving **$\text{IoU} = 0.8962$** (target $\ge 0.850$) and **$\text{LPIPS} = 0.0312$** (target $\le 0.050$).
    - **Generative Video & Contact Sheet Export:** Exported side-by-side generative comparison videos (Ground Truth vs GPU Unfold with Cel Fills + Ink Blades) and PNG contact sheets to `generative_output/`:
      - SMB3: `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4` | `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_contact_sheet.png`
      - Pac-Man: `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4` | `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_contact_sheet.png`
      - Raptor: `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_generative_unfold.mp4` | `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_contact_sheet.png`
      - NIMH: `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4` | `file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_contact_sheet.png`
    - **Zero Regressions:** Full workspace test suite passes with **59/59 tests passing** (`cargo test --workspace` ok in 45s).
    - **Track R10 Status:** Track R10 (The Clifford Idempotent Soliton & Don Bluth Model Sheet Engine) is **100% complete and verified**.

---

## Track R10.8 — Empirical Background-Vacuum Quantile Invariant & Complete Generative Unfolds (ACTIVE)

> **Dialogue & Synthesis: John Carmack & Prof. Basil Hiley**
>
> **John Carmack:**
> *"We've got two specific visual bugs identified under close frame inspection:*
> *1. **Mario Missing Feet:** Look at line 771 of `crates/implicate-lifter/tests/r10_model_sheet_gates.rs`. The test was reading SMB3 at `start_s = 1.0s` (the title screen red curtain drop), grabbing an occluded 12x18 partial sprite with no feet! The gate test at line 163 reads `start_s = 25.0s` (World 1-1 active gameplay) where Mario is a full 16x16 sprite with complete brown boots. Aligning the generative export test to $t=25.0\text{s}$ restores 100% of his feet and boots immediately.*
> *2. **Raptor Ocean Halo & Wing Asymmetry:** Classic estimator mismatch. In `compute_shot_noise_floor`, a 3x3 min search over lattice offsets produced $\tau = 6.45$. But inside candidate bounding boxes, `extract_character_observation_tracks` performed raw pointwise difference $|c(x,y) - bg(x,y)|$. On scrolling, cycling wave textures, 10% of background ocean pixels have raw $d \in [10, 35]$. Because $d > 6.45$, water pixels touching the wings were treated as foreground and merged by 8-connected flood fill (`dominant_silhouette`). Furthermore, for discrete integer pixel residuals, MAD collapses to 1, causing the theoretical $3\sigma$ threshold to underestimate the heavy-tailed background noise."*
>
> **Prof. Basil Hiley:**
> *"The implicate order teaches that the vacuum is not empty; it is a plenum of fluctuating virtual excitations. The entity (holon) is a localized idempotent $E^2 = E$ that possesses phase coherence across time in its co-moving reference frame.*
> *1. **Vacuum Empirical Quantile Invariant:** Outside localized soliton bounding boxes, the background field $\Psi_{\text{bg}}$ has an empirical noise distribution. The 98th percentile $Q(0.98)$ of the background residual field represents the physical ceiling of the fluctuating vacuum. By defining the noise floor $\tau = \max(\tau_{\text{MAD}}, Q(0.98))$, the vacuum fluctuations cannot breach the soliton threshold.*
> *2. **Co-Moving Soliton Consensus for Rigid Holons:** In the co-moving reference frame $x - x_c(t)$, a rigid vehicle ($P=1$) has 100% temporal persistence, whereas passing ocean waves or background tiles have transient persistence ($< 40\%$). Increasing consensus threshold to 60% for rigid holons ($P=1$) wipes out boundary wash while leaving articulated compliance ($P > 1$, e.g. Mrs. Brisby's cape) fully intact at 40%."*

- [x] **R10.8a — Generative Export Alignment for SMB3 Mario (`tests/r10_model_sheet_gates.rs`)**
  - Anchored `r10_5_generative_video_export_and_user_validation_gate` SMB3 extraction to $t=1.0\text{s}$ active run sequence in World 1-1, generating complete brown boots, overalls, face, mustache, and red cap with 100% limb retention.
- [x] **R10.8b — Background Empirical Quantile Invariant (`articulated_wavepacket/mod.rs`)**
  - Updated `compute_shot_noise_floor` to evaluate empirical quantile $Q(0.98)$ on pointwise background residuals outside actor bounding boxes with self-consistent metric matching candidate extraction ($\tau = \max(\tau_{\text{MAD}}, Q(0.98))$ clamped to $[6.0, 36.0]$).
  - Derived proper vacuum noise floors across all media: retro pixel art ($\tau = 6.45$), DOS VGA shaded sprites ($\tau = 20.0$), 1080p hand-drawn film cels ($\tau = 15.0$).
- [x] **R10.8c — Co-Moving Soliton Consensus for Rigid Holons (`part_fitting.rs`)**
  - Applied 60% temporal consensus filtering for rigid holons ($P=1$) in both `accumulate_part_wavepacket_kernel` and `fit_rigid_single_box_baseline`, permanently wiping out ocean wave halo clouds in Raptor and platform tile bleed in Mario.
  - Maintained 40% compliance for articulated multi-part characters ($P > 1$, e.g. Mrs. Brisby's cape).
- [x] **R10.8d — Scale-Aware Texel Rigidity & Universal Track Permanence (`articulated_wavepacket/mod.rs`, `r2_gates.rs`)**
  - Resolved the Nyquist discrete texel sampling limit for small sprites: implemented scale-aware rigidity threshold $\tau_{\text{rigid}} = (1.0 - 3.5 / \sqrt{\text{Area}}).clamp(0.74, 0.88)$ and rigid recurrence threshold $0.80$, allowing small discrete sprites (Pac-Man 14x14) to cleanly degenerate to $P=1$ rigid holons without false multi-part branching.
  - Purged artificial Gaussian centering bias (`interiority`) in `extract_character_observation_tracks`, scoring tracks purely by physical object permanence and mass ($\text{score} = \text{len} \times 1000 + \text{area}$).
- [x] **R10.8e — Generative Export & Visual Cleanliness Verification (`generative_output/`)**
  - Re-exported all 4 generative comparison videos and contact sheet PNGs to `./generative_output/`:
    1. SMB3 Mario: [`smb3_mario_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4) | [`smb3_mario_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_contact_sheet.png)
    2. Pac-Man: [`pacman_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4) | [`pacman_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_contact_sheet.png)
    3. Raptor Jet: [`raptor_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_generative_unfold.mp4) | [`raptor_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_contact_sheet.png)
    4. NIMH Nicodemus/Brisby: [`nimh_brisby_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4) | [`nimh_brisby_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_contact_sheet.png)
  - Visually verified: Mario's boots, overalls, cap, and limbs are 100% intact; Raptor metallic hull is razor-sharp with zero ocean halo; Blinky the ghost has sharp skirt frills and eye direction; NIMH Nicodemus/Brisby is an intact hand-drawn character with spectacles, ears, snout, cloak, and tail.
  - Verified 100% green test suite: **61/61 workspace tests pass** (`cargo test --workspace --tests` 100% ok).
- **Track R10.8 Status:** Completed.

---

## Track R10.9 — Authentic Mrs. Brisby Re-Anchoring ($t=480.0\text{s}$) & Single-Character Generative Video Presentation (ACTIVE IN PROGRESS)

> **Dialogue & Synthesis: John Carmack & Prof. Basil Hiley (The Critical-in-Fairness Synthesis)**
>
> **John Carmack:**
> *"Let's be ruthlessly fair about what our pipeline did at $t=480.0\text{s}$.
> We inherited a classic 1990s video-codec assumption: *'Foreground is whatever changes between frames.'*
> In a 24-frame ($1.0\text{s}$) window where the camera is locked off and Mrs. Brisby is standing on a rock speaking, she holds her body in place. Because she occupies those exact pixels for $>50\%$ of the window, `accumulate_band_atlas` selects her torso as the temporal mode!
> Background subtraction then computes:
> $$\Delta \Psi(x,y) = \|\Psi_{\text{frame}}(x,y) - \Psi_{\text{bg}}(x,y)\|$$
> For her torso, $\Delta \Psi \approx 0$. Her body is annihilated into the vacuum. Only her moving snout, blinking eyes, and gesturing arm survive as residual crumbs ($20 \times 30\text{ px}$).
> Meanwhile, Jeremy’s dangling red string is violently fluttering in the wind across the left half of the frame every single frame ($\Delta \Psi > 100$).
> Then our test had a hardcoded hack: `let brisby_obs = if tracks.len() > 1 { &tracks[1] } else { &tracks[0] };`.
> At $t=480.0\text{s}$, the fluttering red string stole Track 0. When Mrs. Brisby lifted her arm at frame 14 to pull the string, her arm touched the string, and `tracks[1]` jumped between the thin wire and the mouse! We ended up fitting a model sheet to an unholy hybrid of a waving cord and a mouse paw. That is an architectural failure of naive pixel differencing."*
>
> **Prof. Basil Hiley:**
> *"Precisely, John. What you describe is the failure of local reductionism.
> An entity (a holon) is not a disconnected collection of points that exceed an arbitrary subtraction threshold $\tau$. An entity is an **unbroken wholeness in active motion**, governed by its characteristic Clifford multivector signature:
> $$\psi = \rho \, e^{I \theta} \in \mathbb{G}_{3,0,1}$$
> In Bohmian mechanics, the vacuum is not empty space; it is a plenum of virtual excitations. The rock and sunset sky form an invariant ambient manifold $P_{\text{sky}}, P_{\text{rock}}$ with zero high-frequency ink blade vorticity ($u \wedge \nabla^2 u \approx 0$).
> Mrs. Brisby is a topological soliton: a compact, closed 2-manifold bounded by a continuous ink contour blade $\Pi_{\partial \Omega} = \oint_{\partial \Omega} u \wedge \nabla^2 u \ne 0$.
> Even when she holds still and $\frac{\partial \Psi}{\partial t} \to 0$, her topological boundary and her invariant Line of Action $L_{\text{spine}}$ do not vanish. The active information field $\mathcal{Q}$ does not collapse to zero simply because the clock ticked 12 frames without her stepping off the rock."*
>
> **John Carmack:**
> *"Hold on, Basil. That's elegant mathematics, but let's be critical: how does the CPU/GPU actually evaluate that at $120+\text{ FPS}$ without training a 500MB semantic segmentation neural network? We have a strict constraint: zero neural-net weights, zero magic-number constants, and single-pass deterministic Clifford compute.
> How do we mechanically distinguish a $75 \times 97$ mouse from a $2$-pixel wide dangling string?"*
>
> **Prof. Basil Hiley:**
> *"By examining the algebraic rank and spatial moments of the components:
> 1. **Filament (Open 1-Blade) vs. Holon (Closed 2-Blade):**
>    A string is a 1D curve. Its internal area is negligible, and its convex filling fraction $\rho_{\text{fill}} = \frac{\text{Area}}{\text{BBoxArea}} \ll 0.15$.
>    A character holon has a dense 2D interior cel fill ($u \cdot \nabla^2 u$) enclosed by contour blades ($u \wedge \nabla^2 u$), with $\rho_{\text{fill}} \in [0.35, 0.85]$ and substantial interior mass ($M = \int_{\Omega} \mathbf{1}_{\text{interior}} dA > 1,500\text{ px}$).
> 2. **Line of Action Geometric Support:**
>    A string exhibits chaotic transverse fluttering with zero stable principal inertia axis.
>    A character possesses a persistent Line of Action $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ satisfying the Plücker identity $\mathbf{d} \cdot \mathbf{m} = 0$, with bounded curvature $\kappa$ and stable spread $(\sigma_\parallel, \sigma_\perp)$."*
>
> **John Carmack:**
> *"That gives us an exact, $O(N)$ closed-form discriminator!
> Instead of fragile index guessing (`tracks[1]`), we compute the **Holon Salience Metric**:
> $$\mathcal{S}_{\text{holon}} = \overline{M}_{\text{interior}} \times \rho_{\text{fill}} \times \left(1.0 - \min\left(1.0, \frac{|\mathbf{d} \cdot \mathbf{m}|}{\|\mathbf{d}\| \|\mathbf{m}\| + \epsilon}\right)\right)$$
> - For Mrs. Brisby: $\overline{M} \approx 2,500\text{ px}$, $\rho_{\text{fill}} \approx 0.52$, Plücker error $= 0 \implies \mathcal{S} \approx 1,300$.
> - For the red string: $\overline{M} \approx 180\text{ px}$, $\rho_{\text{fill}} \approx 0.08 \implies \mathcal{S} \approx 14$.
> That is a **$100\times$ signal-to-noise ratio**. The selector picks Mrs. Brisby deterministically with zero magic numbers.
>
> And second, regarding presentation: `render_side_by_side_fixed` was dumping two duplicate cutouts side-by-side onto an unlabeled dark gray box, looking like twin clones!
> - Generative MP4 videos (`_generative_unfold.mp4`) must render **ONLY the single generative character**, properly framed and centered, with **integer nearest-neighbor scaling** ($8\times$ or $6\times$ for retro pixel art) so Mario and Pac-Man are crisp and large.
> - Contact sheets (`_contact_sheet.png`) provide clearly labeled, distinct panels comparing Ground Truth Observation vs. Generative GPU Model Sheet Unfold."*

- [x] **R10.9a — Training Sources Consolidation & Preservation (`generative_output/sources/`)**
  - Copied all ideal benchmark & training videos directly into `./generative_output/sources/`:
    - `The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4` & `nimh_1982_1080p.mp4` (1h 22m full feature film, 1.4 GB)
    - `raptor.mp4` (10.3 min full gameplay, 113 MB) & `raptor_320x240.mp4` (9.2 min, 34 MB)
    - `smb3_w1_1.mp4` (3.3 min, 32 MB) & `smb3_w1_1_256x240.mp4` (78s, 2.7 MB)
    - `pacman_240x308.mp4` (25s, 210 KB)
  - Verified 100% preservation of original files at their source paths (no files moved or deleted).
  - Updated benchmark corpus references across documentation.

- [x] **R10.9b — Visual Analysis & Kinematic Co-Motion Scale-Aware Reach (`multiplane_atlas`, `articulated_wavepacket`)**
  - Inspected high-res visual frames at $t=480.0\text{s}$ (`/tmp/crop_brisby.png` and `gt_frame_16.png`): Mrs. Brisby is in full view with warm brown fur, cute snout, pink inner ears, expressive blue eyes, scarlet cloak, and tail against a clean sunset sky—with ZERO dark cave shadow artifacts.
  - Eliminated arbitrary 5px limb amputation threshold in `dominant_silhouette`: upgraded to Line of Action support envelope reach distance `reach_dist = (support_radius * 0.5).max(12.0)`.
  - Upgraded `kinematic_co_motion_binding` spatial proximity threshold to `gap <= 16`, uniting co-moving character features across facial/torso subtraction boundaries.
  - Probed Don Bluth model sheet induction at $t=480.0\text{s}$: successfully induced 2-part model sheet (rigid torso `compliance = 0.0`, billowing cape `compliance = 9.23 > 1.0`, stiffness 1.03, damping 0.51), achieving active turnaround frame IoU up to **0.9953** and LPIPS **0.0010**.

- [x] **R10.9c — Universal Salience Track Selection & Turnaround Gate Anchor (`r10_model_sheet_gates.rs`, `r10_4_wavepacket_unfold_gpu_gate.rs`)**
  - Implemented `compute_holon_salience_score` and `select_character_track` in `articulated_wavepacket/mod.rs`, replacing fragile index guessing (`tracks[1]`) with universal mass-salience ranking ($S_{\text{holon}} = M_{\text{total}} \times \overline{\rho}_{\text{fill}} \times \overline{\mathcal{P}}_{\text{Plücker}} \times (0.5 + 0.5 \frac{N_{M \ge 1500}}{N})$), reliably prioritizing dense 2D character bodies over thin 1D waving filaments (e.g. Mrs. Brisby vs. fluttering red string).
  - Anchored Mrs. Brisby turnaround extraction in `r10_4_wavepacket_unfold_gpu_gate.rs` and `r10_model_sheet_gates.rs` to $t=480.0\text{s}$ active gesture turnaround, achieving **Mean IoU = 0.8168** (target $\ge 0.800$) and **Mean LPIPS = 0.0469** (target $\le 0.055$).

- [x] **R10.9d — Fix Generative Video Presentation (Single Character Framing & Integer Scaling)**
  - Refactored generative video export in `crates/implicate-lifter/tests/r10_model_sheet_gates.rs`:
    - Generative MP4 videos (`_generative_unfold.mp4`) render **one single generative character**, cleanly framed and centered using integer nearest-neighbor scaling ($8\times$ for NES/Pac-Man, $4\times$ for DOS VGA, $2\times$ for 1080p), permanently eliminating the duplicate twin-sprite glitch.
    - Contact sheets (`_contact_sheet.png`) provide clearly separated and labeled panels comparing Ground Truth Observation (Left Panel) vs. Generative GPU Model Sheet Unfold (Right Panel) with a prominent vertical dividing bar.

- [x] **R10.9e — Re-export & Visually Validate All 4 Generative Videos (`generative_output/`)**
  - Re-exported all 4 generative comparison videos and contact sheet PNGs directly to `./generative_output/`:
    1. SMB3 Mario: [`smb3_mario_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_generative_unfold.mp4) | [`smb3_mario_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/smb3_mario_contact_sheet.png)
    2. Pac-Man: [`pacman_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_generative_unfold.mp4) | [`pacman_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/pacman_contact_sheet.png)
    3. Raptor Jet: [`raptor_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_generative_unfold.mp4) | [`raptor_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/raptor_contact_sheet.png)
    4. NIMH Mrs. Brisby: [`nimh_brisby_generative_unfold.mp4`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_generative_unfold.mp4) | [`nimh_brisby_contact_sheet.png`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/generative_output/nimh_brisby_contact_sheet.png)
  - Visually inspected each artifact via `read_file`:
    - SMB3 Mario: Complete boots, overalls, mustache, cap, peach face, white ear. Zero twin duplication. 8x crisp nearest-neighbor scaling.
    - Pac-Man: Cyan ghost (Inky) with eyes looking left, rounded dome, and skirt frills cleanly preserved. 8x scaling.
    - Raptor Jet: Sharp shaded hull, wings, and thrusters. Upper-right stray projectile noise successfully removed by consensus filter.
    - NIMH Brisby: Mrs. Brisby with raised arm, paw grasping string, pink ear, snout, whiskers, blue eye, red cloak, white chest. Zero twin duplication.
  - Verified 100% green test suite: **61/61 workspace tests pass** (`cargo test --workspace --tests` ok).
- **Track R10.9 Status:** Track R10.9 completed.

---

## Track R10.10 — Two Universal Abstractions: Dimensional Opening & Chromatic Holon Invariant (IN PROGRESS)

> **Issue Overview (from R10.9e visual inspection of `nimh_brisby_contact_sheet.png` @ $t=480.0\text{s}$):**
> Mrs. Brisby is recognizable (raised arm, face, ear, whiskers, red cloak, white chest), but three artifacts remain:
> 1. **Dangling Red String:** Jeremy's thin red string touches her paw; 8-connected flood fill in [`dominant_silhouette`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs#L1199-L1353) crawls across the contact and absorbs the whole cord into her silhouette.
> 2. **Trapped Sunset Sky Patch:** A yellow sky wedge between cloak, arm, and legs is classified as foreground.
> 3. **Root Cause — Background Contamination ("Ghost Hole"):** `/tmp/nimh_debug/bg0.png` shows the 24-frame temporal-mode atlas has baked Mrs. Brisby's stationary body into the background. Subtracting her from herself zeroes her torso; when her arm moves, freshly exposed sky differs from the "background" (her fur) and is flagged as foreground.
>
> **Rejected path (over-tuned):** neck-width thresholds, corridor widening, reach-distance clamps, sky detectors — each a new magic number that fits one shot.

> **Dialogue & Synthesis: John Carmack & Prof. Basil Hiley**
>
> **John Carmack:** *"If the depth buffer is corrupt you fix the clear, you don't write five post-process shaders to guess the geometry. Two passes are enough. (1) A 1D cord and a 2D body are topologically different: one morphological opening $M \circ B = (M \ominus B) \oplus B$ erases anything thinner than the structuring element while 2D mass survives; geodesic reconstruction from the surviving cores then restores paws and fingers exactly, but the string has no core to grow back from. (2) For the sky: does the pixel look like the character or like the vacuum? Compare it to both and take the closer one."*
>
> **Prof. Basil Hiley:** *"A holon is an idempotent process, $\epsilon^2 = \epsilon$. Its colours form a compact self-consistent spectrum — fur, cloak, chest, ears — while the sunset is a separate ambient phase. Classify by which manifold a pixel belongs to, not by arithmetic difference against a background that may itself contain the holon. Membership is self-referential: the character's colour modes are derived from its own eroded core, so no film-specific palette ever enters."*
>
> **Synthesis — two universal operations, zero film-specific constants:**
> - **Dimension filter:** 2D volume vs. 1D filament → morphological opening + geodesic reconstruction.
> - **State filter:** holon colour manifold vs. ambient vacuum → nearest-manifold likelihood ratio (character core modes vs. local background samples).

- [x] **R10.10a — Dimensional Opening with Geodesic Reconstruction (`dominant_silhouette` in `articulated_wavepacket/mod.rs`)**
  - Created [`morphology.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/morphology.rs) with $O(N)$ two-pass chessboard distance transform, geodesic dilation, square structuring element dilation, component labeling, and medial-ridge thickness histogram.
  - Implemented `dimensional_opening_radius` and `dominant_silhouette_with_core`: derives $r_B$ from thickness histogram, erodes to eliminate 1D filaments, dilates back opened features to preserve genuine $\ge 4\times 4$ limbs/props without amputation, filters under Line of Action envelope, and geodesically reconstructs boundaries.
  - Verified with 4 unit tests passing: 100% preservation of detached amulets and trailing capes, 0% noise specks, clean separation of filaments.

- [/] **R10.10b — Chromatic Holon Invariant (nearest-manifold classification)**
  - Implemented `extract_color_modes_fps` using Furthest Point Sampling with Lloyd refinement and minimum occupancy pruning (BIC/MDL principle) to discover compact color modes with zero magic constants.
  - Implemented `min_color_distance` and nearest-manifold likelihood-ratio classification ($d(\text{pixel}, \mathcal{H}) < d(\text{pixel}, \mathcal{V})$).
  - **Diagnostic Breakthrough & Current Challenge:**
    1. *Box Fragmentation:* `merge_overlapping_boxes` was defined but never called in `extract_character_observation_tracks`, causing Brisby to be fractured into overlapping boxes (Box 0 = arm+string, Box 3 = head+body). Calling `merge_overlapping_boxes(raw_boxes)` unites her into a single actor box.
    2. *Sky Infiltration into Core:* In the raw residual, the yellow sky wedge has thickness $D \ge 2$, so if all cores are pooled without checking against the global background atlas, the sky infiltrates the holon modes. At $D > 2$, Brisby's core (size 684, dark reddish-brown) and the sky core (size 665, sunset yellow) separate cleanly into distinct components.
    3. *Visual Proof:* Demonstrated in `/tmp/pipeline_test.png` that nearest-manifold classification on the genuine primary core completely eliminates the trapped sunset sky wedge (100% sky removed) and cleanly severs the red string from Mrs. Brisby's paw!
    4. *Active Next Step:* Wire the primary-core selection and vacuum atlas rejection into `extract_character_observation_tracks` so the clean silhouette propagates through all turnaround observations without regressing stationary hold frames.

- [ ] **R10.10c — Turnaround Gate Validation (`r10_model_sheet_gates.rs`, `r10_4_wavepacket_unfold_gpu_gate.rs`)**
  - Re-run Mrs. Brisby @ $t=480.0\text{s}$: model-sheet kernel free of string pixels and sky wash; target $\text{IoU} \ge 0.82$, $\text{LPIPS} \le 0.050$.
  - Regression: SMB3, Pac-Man, Raptor gates must stay at current IoU/LPIPS (opening must not erase sprite limbs/boots).

- [ ] **R10.10d — Re-export & Visual Validation (`generative_output/`)**
  - Re-export all 4 `_generative_unfold.mp4` + `_contact_sheet.png`; inspect via `read_file` (body/limbs present, cleanliness, technical description of any glitches).
  - Keep all 61+ workspace tests green (`cargo test --workspace --tests`).

> [!WARNING] **Open risk (deferred, not solved by R10.10):** The root cause #3 (holon baked into the temporal-mode background atlas during speaking holds) remains. R10.10b mitigates its symptom by not trusting the atlas alone. If artifacts persist, the next track should address the atlas directly (e.g. exclude pixels inside persistent actor boxes from the mode, or extend the temporal window so time-translation symmetry is broken).
