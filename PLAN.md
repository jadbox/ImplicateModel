# ImplicateModel Engineering Proposal & Execution Roadmap

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to feature-length hand-drawn cinema) into a compact Clifford world-and-verb lexicon, and unfolds new, controllable videos in a single causal wave pass at **real-time frame rates ($\ge 24\text{ FPS}$ minimum, targeting $60+\text{ FPS}$ via native Rust and heavy GPU compute acceleration)**.

For project context, read the README, docs/whitepaper, docs/philosophy, and the PLAN.md proposal. The engine has bugs related to character/object/spinnors extractions. You are much smarter than the original developers. Let's fix character extraction by emulating what the old animation film masters like Don Bluth work by creating character board model sheets that are stored in higher symbolic representations. Investiage using CliffordNet or use Clifford Fourier Transforms or Clifford Density Elements (see PDFs for CliffordNet and GATR under ./docs). Our first step is just to create a rock solid plan into PLAN.md for review.
You may run quick experiments to explore hypothesis. Keep in mind our video generator is for ANY video game or animated movie. Our solution should be robust, flexible, and not rely on arbitrary magic number constants to work. Keep **Basil Hiley** in mind as mathematical thinker who understands that the future of system design is using Clifford mathematics (like Projective Geometric Algebra and Clifford Fourier Transforms) to find the inner unity of observations by knowing that all reality unfolds from the Implicate Order of Wholeness. Develop isolated tests as needed to formalize the ultimate pure perfect solution.

---

## I. Executive Philosophical & Systems Context

### 1. The Lineage of the Implicate Order
Following **David Bohm** and **Basil Hiley** (*The Undivided Universe*, 1993; *Process, Distinction, Groupoids and Clifford Algebras*, 2012), `ImplicateModel` is founded on the principle that the discrete, isolated objects of Cartesian coordinates and explicate screen pixels are secondary projections unfolded from an unbroken, continuous totality: the **Implicate Order** (the *holomovement*).

In conventional video generation (e.g. multi-step diffusion transformers such as Sora, Wan, and MiniMax-H3), motion is fractured into tens of thousands of static $(t, h, w)$ Cartesian tokens in dead Euclidean space. Because empty space has no intrinsic tension, impedance, or continuity, billions of parameters must be run over $20\text{--}50$ iterative denoising steps merely to prevent objects from dissolving between adjacent frames.

`ImplicateModel` repudiates this *explicate atomism*. Reality is modeled as an intensive wave-matter continuum (`implicate-field`) where entities are **algebraic process holons** inhabiting minimal left ideals in a Clifford algebra, actions are **Active Information** fields ($\mathcal{Q}_{\text{Cl}}$), and sight and sound unfold as inseparable projections (*Rolpa*) in a single forward pass.

### 2. Diagnosis: The Intern Drift into Heuristic Atomism
In recent development iterations, the project began to drift away from the foundational principles laid down in [`docs/implicate_model_philosophy.md`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/docs/implicate_model_philosophy.md) and [`docs/implicate_model_whitepaper.md`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/docs/implicate_model_whitepaper.md). Under pressure to pass tests, ad-hoc heuristics and arbitrary magic numbers crept into the character extraction pipeline (`crates/implicate-lifter/src/articulated_wavepacket/`):

1. **Magic Thresholds in Foreground Extraction:**
   - Hardcoded residual cutoffs (`FOREGROUND_RESIDUAL = 26`, `MIN_OBSERVATION_PIXELS = 12`) that fail on soft film grain, cel shading, and watercolor backgrounds in classic cinema.
2. **Brittle Tracking Heuristics:**
   - Constant-velocity Euclidean predictions in screen space with arbitrary bounding box ratio constraints (`MAX_EXTENT_RATIO = 1.6`) and color thresholds (`MAX_APPEARANCE_DIST = 55.0`). These violently break whenever an animated character undergoes classic *squash and stretch*, turns $90^\circ$ from profile to front (where projected width changes by $3\times$), or accelerates into an anticipation hold.
3. **Amputation of Limbs and Props:**
   - Component filtering (`dominant_silhouette` with `MIN_COMPONENT_FRACTION = 4`) that discards smaller connected components, amputating detached limbs, held props (such as Mrs. Brisby's amulet or Jeremy's string), billowing capes, and trailing tails.
4. **Motion-Only Spatial Scissoring:**
   - Filtering out any tracklet where `spread <= 0.5 * mean_extent`, falsely purging stationary, talking, blinking, or idling characters as "background debris."
5. **Synthetic Mocks Replacing Empirical Cinema:**
   - Rather than extracting Mrs. Brisby and Jeremy the Crow directly from the real 1080p BluRay footage of Don Bluth's *The Secret of NIMH*, the test suite in `r2_gates.rs` was bypassed with `build_brisby_cape_turn_sequence()`—procedural trigonometric ellipses drawn with `sin` and `cos`!

--- MILESTONG PLANNING SPACE BELOW ---

## II. The Animation Master Paradigm: The Don Bluth Model Sheet in Clifford Algebra

Classical animation masters like **Don Bluth**, **Milt Kahl**, and **Frank Thomas** never conceived of characters as bounding boxes of moving pixels. In classical feature animation, every character is designed and preserved through a **Character Board / Model Sheet**:

```text
+--------------------------------------------------------------------------------------------------------+
|                          DON BLUTH CHARACTER BOARD / MODEL SHEET ARCHITECTURE                         |
+--------------------------------------------------------------------------------------------------------+
|                                                                                                        |
|  1. THE LINE OF ACTION (Kinematic Spine Blade)                                                         |
|     • A primary 3D Projective Geometric Algebra (PGA) screw axis L_spine ∈ G(3,0,1)                      |
|     • Directs dynamic attitude, squash-and-stretch vectors, and momentum flow                          |
|                                                                                                        |
|  2. CANONICAL TURNAROUND IDEMPOTENTS ({ε_view})                                                        |
|     • Front, 3/4 Front, Profile, 3/4 Back, Back (Discrete view aspects)                                |
|     • Hiley primitive state idempotents: ε_v² = ε_v,  ε_u ε_v = 0,  Σ ε_v = I                          |
|     • View transitions governed by idempotent operators: T_{u → v} = ε_v M ε_u                         |
|                                                                                                        |
|  3. ARTICULATED VOLUMETRIC PART HIERARCHY ({H_p})                                                      |
|     • Kinematic tree: Cranium, Chest, Pelvis, Limbs, Cape/Tail                                         |
|     • Bound by relative Clifford motors M_{p ← parent} ∈ Spin(3,0,1) with volume conservation          |
|                                                                                                        |
|  4. CLIFFORDNET DUALITY: INK CONTOURS VS. CEL PAINT FILLS (uv = u · v + u ∧ v)                         |
|     • Exterior Wedge u ∧ ∇²u: Structural Line-of-Action Vorticity & Invariant Ink Boundary Blades (B_k) |
|     • Inner Product  u · ∇²u: Cel Paint Coherence, Flat Color Fill, and Active Vitality Density (ρ)    |
|                                                                                                        |
|  5. CONTINUOUS CLIFFORD FOURIER GAIT WAVEPACKETS (W_p(u, v, ϕ))                                        |
|     • Volume-conserving deformations parameterized by intrinsic phase ϕ ∈ S¹                           |
|     • Smooth continuous walk/run cycles, breathing, and viseme articulation without pixel Gibbs blur   |
+--------------------------------------------------------------------------------------------------------+
```

### 1. Mathematical Grounding in Basil Hiley's Clifford Pregeometry
In Hiley's Clifford process ontology (*The Clifford Algebra Approach to Quantum Mechanics*, Hiley & Callaghan 2010/2012):
1. **The Minimal Left Ideal ($\Phi_L$):**
   An entity is not an isolated point mass or Cartesian patch; it is an element of a minimal left ideal $\mathcal{I}_{i,v} = \mathcal{C}\ell\,\epsilon_v$:
   $$\Phi_{L,i}(\mathbf{x}, t) = R_i(\mathbf{x}, t)\, M_i(t)\, \epsilon_v$$
   where $R_i \ge 0$ is the real amplitude/vitality of the process, $M_i(t) \in \text{Spin}(3,0,1)$ is its continuous PGA motor (rigid translation and spatial rotation), and $\epsilon_v$ is the discrete view-aspect idempotent selecting its active turnaround posture.
2. **The Clifford Density Element ($\rho_c$):**
   All physical observables and geometric properties of the holon unfold bilinearly from its density operator:
   $$\rho_c = \Phi_L \widetilde{\Phi}_L = \rho\, U \epsilon_v \widetilde{U}, \qquad \rho = R^2$$
   - **Bilinear Invariant of the First Kind (Process Density / Mass):** $\rho = \text{tr}(\rho_c)$.
   - **Bilinear Invariant of the First Kind (Directed Spin / Heading Vector):** $\mathbf{s} = U \mathbf{e}_1 \widetilde{U} = (\cos\theta, \sin\theta)$.
   - **Bilinear Invariant of the Second Kind (Bohm Momentum & Kinetic Flux):** $\mathbf{P}_B = \rho v_0 \mathbf{s}$, evolving causally under the Bohm-Hiley Quantum Potential:
     $$\frac{d\mathbf{P}_B}{dt} = -\nabla \mathcal{Q}_{\text{Cl}} + \boldsymbol{\tau}_Q$$
   - **State Transition Operators:** Transitions between turnaround postures (e.g. from 3/4 Front to Profile) occur via idempotent boundary projection $T_{u \to v} = \epsilon_v M \epsilon_u$.

### 2. CliffordNet: The Algebraic Duality of Contours and Fills
In the CliffordNet architecture (*CliffordNet: All You Need is Geometric Algebra*, Ji 2026), deep learning modules (spatial mixers + MLPs) are replaced by the algebraic completeness of the **Clifford Geometric Product**:
$$\mathbf{u}\mathbf{v} = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \wedge \mathbf{v}$$
When evaluated over the discrete local Laplacian context $\mathcal{C}_{\text{loc}}(H) \approx \nabla^2 H$:
- **The Exterior Wedge ($\mathbf{u} \wedge \nabla^2 \mathbf{u}$):** Measures structural vorticity and topological curvature. In animation, this isolates the **ink line drawing**, edge boundaries, and silhouette creases without any arbitrary gradient threshold.
- **The Inner Product ($\mathbf{u} \cdot \nabla^2 \mathbf{u}$):** Measures regional alignment and coherence diffusion. In animation, this isolates the **flat cel paint fills**, tone shading, and material identity.

Because this decomposition is algebraically complete and linear in complexity ($O(N)$), it naturally unifies 8-bit flat pixel art (*Pac-Man*, *Super Mario Bros. 3*), shaded bitmap sprites (*Raptor*), and hand-inked feature animation (*The Secret of NIMH*).

---

## III. The Symbolic Holon Model Sheet Specification

We formalize the symbolic character representation as a persistent Rust data structure in `implicate-lifter` and `implicate-unfold`:

```rust
/// The complete symbolic Character Board Model Sheet for an autonomous process holon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HolonModelSheet {
    /// Unique semantic symbol (e.g., "Mrs_Brisby", "Mario", "Jeremy_Crow").
    pub name: String,
    /// Physical constitutive parameters: Cohesion C, Density ρ, Bulk Modulus K, Luminescence L.
    pub constitutive: ConstitutiveTuple,
    /// Line of Action: 3D PGA screw axis blade L = d + m e_0123.
    pub line_of_action: [f64; 6],
    /// Invariant projective proportions: cross-ratios and part dimension ratios.
    pub invariant_proportions: Vec<f64>,
    /// Turnaround model sheet: discrete view-aspect idempotents {ε_v}.
    pub turnaround_views: Vec<TurnaroundViewAspect>,
    /// Hierarchical articulated sub-holon kinematic tree.
    pub parts: Vec<SubHolonPartSpec>,
    /// Invariant Hiley Clifford Density Element signature.
    pub cde_signature: CliffordDensityElement,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnaroundViewAspect {
    pub view_index: usize,
    /// Canonical view label: Front, ThreeQuarterFront, Profile, ThreeQuarterBack, Back.
    pub view_regime: ViewRegime,
    /// Discrete primitive idempotent projector ε_v on Spin(2) × S¹.
    pub idempotent: StateIdempotent,
    /// Reference heading rotor on Spin(2).
    pub reference_rotor: [f64; 2],
    /// Articulated part poses for this canonical aspect.
    pub part_poses: Vec<SubHolonPose>,
    /// Invariant contour blade (wedge vorticity) and cel fill kernels per part.
    pub part_kernels: Vec<ModelSheetPartKernel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSheetPartKernel {
    pub part_id: usize,
    /// Normalized canonical coordinates [-1, 1]².
    pub canonical_res: [usize; 2],
    /// Invariant cel fill paint signature (inner product coherence u · ∇²u).
    pub cel_fill_rgb: Vec<u8>,
    /// Invariant ink line contour mask (wedge vorticity |u ∧ ∇²u|).
    pub ink_contour_blade: Vec<f32>,
    /// 5-mode Clifford Fourier deformation wavepacket coefficients (res_w * res_h * 4 * 5).
    pub cft_deformation_coeffs: Vec<[f32; 5]>,
}
```

---

## IV. The Zero-Magic-Number Extraction Pipeline

To achieve complete ontological univocity and purge all arbitrary constants, the enfoldment pipeline (`implicate-lifter`) is overhauled across four continuous Clifford stages:

```text
Raw Video Stream (Pac-Man / SMB3 / Raptor / The Secret of NIMH)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Active Information Residual Lattice Field              │
│   • Yee background equilibrium: B(x, y) = mode_t(x - X_cam(t))  │
│   • Residual multivector: ΔΨ(x, y, t) = Ψ_frame - Ψ_bg ∈ G(2,0) │
│   • Active Information: I_active = ||ΔΨ|| + ||ΔΨ ∧ ∇²ΔΨ||        │
│   • Noise-floor cutoff: τ = median(I) + 3.0 * MAD(I)            │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: CliffordNet Wedge-Vorticity Boundary Delineation       │
│   • Compute shifted geometric product: D ⊕ W over ∇²H           │
│   • Ink contour blades: Closed zero-crossings of ∇ ∧ W = 0      │
│   • Retains 100% of detached limbs, held props, and flowing capes│
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Hiley CDE Bilinear Geodesic Tracking                   │
│   • Bilinear density overlap: tr(ρ_c,1 ρ_c,2) / sqrt(...)       │
│   • Spin(3,0,1) motor continuity: ⟨M_pred, M_curr⟩_PGA          │
│   • Invariant under squash & stretch, turning, and holds        │
│   • Zero discarding of stationary/talking/gesticulating actors  │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Automated Character Board Model Sheet Induction        │
│   • 2D Clifford Fourier Transform posture descriptor (64D)      │
│   • Gram-SVD spectral clustering discovers turnaround {ε_v}     │
│   • Principal bivector axis yields Line of Action L_spine       │
│   • Vorticity saddles partition kinematic parts {H_p}           │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
            Persistent .implicate Container
```

### 1. Stage 1: Active Information Residual Lattice Field
- **Problem with Prior Code:** Fixed integer residual `FOREGROUND_RESIDUAL = 26` and pixel count threshold `12`.
- **Clifford Formulation:** The background is the stationary or camera-locked equilibrium state of the staggered Yee FDTD lattice. Foreground presence is the departure of the multivector wavefield from equilibrium:
  $$\Delta \Psi(\mathbf{x}, t) = \Psi_{\text{frame}}(\mathbf{x}, t) - \Psi_{\text{bg}}(\mathbf{x}, t) \in \mathbb{G}_{2,0}^+$$
  The **Active Information Density** is the sum of local amplitude disturbance and rotational gradient torque:
  $$\mathcal{I}_{\text{active}}(\mathbf{x}, t) = \|\Delta \Psi(\mathbf{x}, t)\| + \lambda\,\|\Delta \Psi(\mathbf{x}, t) \wedge \nabla^2 \Delta \Psi(\mathbf{x}, t)\|$$
  The detection threshold is computed self-consistently from the statistical noise floor of the frame using the Median Absolute Deviation (MAD):
  $$\tau_{\text{active}} = \text{median}(\mathcal{I}_{\text{active}}) + 3.0 \times 1.4826 \times \text{MAD}(\mathcal{I}_{\text{active}})$$
  This naturally adapts to crisp 8-bit NES palette buffers, noisy 1080p 35mm film grain, or low-light animated frames without tuning any constant.

### 2. Stage 2: CliffordNet Wedge-Vorticity Boundary Delineation
- **Problem with Prior Code:** Euclidean box expansion and discarding connected components smaller than $1/4$ of the largest component (`MIN_COMPONENT_FRACTION = 4`).
- **Clifford Formulation:** Instead of bounding boxes, boundaries are oriented **Clifford blades** $B_k \in \mathbb{G}_{3,0,1}$ extracted from the zero-crossings of the CliffordNet exterior wedge product:
  $$W(\mathbf{x}) = \mathbf{u}(\mathbf{x}) \wedge \nabla^2 \mathbf{u}(\mathbf{x})$$
  Because $W(\mathbf{x})$ vanishes at inflection ridges of structural vorticity, it traces closed contours around all anatomical masses. Detached items (an amulet, a staff, projectiles, droplets) form distinct closed loops that are preserved as subordinate members of the same holon field rather than being deleted.

### 3. Stage 3: Hiley CDE Bilinear Geodesic Tracking
- **Problem with Prior Code:** Screen-space constant-velocity extrapolation, bounding box aspect ratio clamping (`MAX_EXTENT_RATIO = 1.6`), and purging tracks that do not move across the screen (`spread <= 0.5 * mean_extent`).
- **Clifford Formulation:** Candidate observations are associated across time by maximizing the **geodesic affinity on the Hiley CDE manifold**:
  $$\mathcal{A}(\rho_{c,1}, \rho_{c,2}) = \underbrace{\frac{\text{tr}(\rho_{c,1} \rho_{c,2})}{\sqrt{\text{tr}(\rho_{c,1}^2)\,\text{tr}(\rho_{c,2}^2)}}}_{\text{Intrinsic Spinor & Palette Coherence}} \times \underbrace{\exp\left(-\frac{\|\log(M_1^{-1} M_2)\|^2}{2\sigma_{\text{motion}}^2}\right)}_{\text{PGA Geodesic Smoothness}}$$
  - **Squash and Stretch Invariance:** Volume-conserving deformations ($\det S = 1$) preserve the total process density $\rho = \text{tr}(\rho_c)$.
  - **Turnaround Invariance:** Even as projected width changes by $3\times$ when turning from profile to front, the chromatic spinor signature $\psi_{\text{id}}$ and continuous motor flow $M(t)$ remain maximal.
  - **Permanent Holon Support:** A character who stands in place, talks, gestures, or breathes is retained with 100% permanence because $\rho_c$ measures active vitality, not mandatory translation.

### 4. Stage 4: Automated Character Board Model Sheet Induction
- **Turnaround Aspect Clustering:**
  Each tracked observation is transformed via a 2D Clifford Fourier Transform into a 64D posture descriptor $D_{\text{CFT}} \in \mathbb{R}^{64}$. We construct the $N \times N$ Gram matrix:
  $$G_{ij} = \langle D_{\text{CFT},i}, D_{\text{CFT},j} \rangle$$
  Applying a thin symmetric Jacobi SVD reveals the discrete spectral clusters. The principal eigenvectors identify the canonical turnaround aspects (Front, 3/4 Front, Profile, etc.), which are formalized as primitive state idempotents $\epsilon_v$.
- **Line of Action & Part Decomposition:**
  The character's primary Line of Action $L_{\text{spine}}$ is extracted as the principal projective bivector axis of the mass distribution. Articulated sub-holon joints (neck, shoulder, hip, wing root) are identified at the saddle points of the wedge-vorticity field $W(\mathbf{x})$, constructing the kinematic tree $\{H_p\}$ attached via relative motors $M_{p \leftarrow \text{parent}} \in \text{Spin}(3,0,1)$.
- **5-Mode Continuous Clifford Fourier Wavepacket:**
  Within each canonical view idempotent cell, internal deformation is parameterized along the continuous gait phase $\phi \in [0, 2\pi)$ via regularized 5-mode normal equations, allowing the character to be unfolded at arbitrary speeds and framerates without visual stutter or blur.

---

## V. Milestone Execution Track: Track R10

We designate this work as **Track R10: The Don Bluth Character Board & Clifford Model Sheet Engine**, broken into 5 rigorous, measurable sub-milestones with automated verification gates.

```text
Track R10 Milestone Timeline:
  [R10.1] Hiley CDE Invariant Tracking & Active Residual Field
     │
     ▼
  [R10.2] CliffordNet Wedge-Vorticity Contour Delineation & Line of Action
     │
     ▼
  [R10.3] Automated Character Board Model Sheet Induction
     │
     ▼
  [R10.4] Volume-Conserving Clifford Fourier Wavepacket Part Refinement
     │
     ▼
  [R10.5] Real-Video Empirical Gate Suite (NIMH, SMB3, Pac-Man, Raptor)
```

### Detailed Sub-Milestones

- [ ] **R10.1 — Hiley CDE Invariant Tracking & Active Residual Field (`implicate-lifter`)**
  - **Objective:**
    - Replace the hardcoded `FOREGROUND_RESIDUAL = 26` with the statistical noise-floor threshold $\tau = \text{median}(\mathcal{I}) + 3.0 \times \text{MAD}(\mathcal{I})$ computed from the Clifford residual multivector $\Delta \Psi$.
    - Replace screen-space constant velocity and box aspect clamping with Hiley CDE bilinear geodesic affinity $\mathcal{A}(\rho_{c,1}, \rho_{c,2})$.
    - Eliminate the `spread <= 0.5 * mean_extent` filter, preserving stationary, talking, and idling actors.
  - **Gate Criteria:**
    - Recovers 100% of frames for Mario in real SMB3 World 1-1 gameplay (`smb3_w1_1_256x240.mp4` @ $t=25\text{--}28\text{s}$), Pac-Man and ghosts in `pacman_240x308.mp4`, and Mrs. Brisby in *The Secret of NIMH* without dropping frames during stationary holds.

- [ ] **R10.2 — CliffordNet Wedge-Vorticity Contour Delineation & Line of Action (`implicate-lifter`, `implicate-math`)**
  - **Objective:**
    - Implement contour extraction via zero-crossings of the CliffordNet exterior wedge product $W(\mathbf{x}) = \mathbf{u} \wedge \nabla^2 \mathbf{u}$, eliminating bounding-box scissoring and `MIN_COMPONENT_FRACTION = 4`.
    - Extract the 3D PGA Line of Action bivector $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ from the principal spatial inertia tensor of the holon mass.
  - **Gate Criteria:**
    - Detached limbs, flowing capes, and held props (e.g. Mrs. Brisby's cape and amulet) are preserved in the extracted holon silhouette with zero disconnected component amputations. Line of Action correctly tracks dynamic spinal curvature during jumping and turning.

- [ ] **R10.3 — Automated Character Board Model Sheet Induction (`implicate-lifter`, `implicate-unfold`)**
  - **Objective:**
    - Implement Gram-SVD spectral clustering on the 2D Clifford Fourier Transform posture descriptor $D_{\text{CFT}}$ to discover canonical turnaround view aspects $\{\epsilon_v\}$ without hardcoding view counts.
    - Compile the induced representation into `HolonModelSheet`, defining invariant projective proportions and canonical turnaround poses.
  - **Gate Criteria:**
    - Automatically discovers the canonical turnaround aspects (e.g., Left, Right, Up, Down for Pac-Man; Run, Skew, Jump for Mario; Walk, Profile Cape Wrap, Front Turn for Mrs. Brisby) with view classification accuracy $> 95\%$ against human-labeled turnaround keys.

- [ ] **R10.4 — Volume-Conserving Clifford Fourier Wavepacket Part Refinement (`implicate-lifter`, `implicate-rolpa`)**
  - **Objective:**
    - Upgrade `WavepacketPartKernel` to separate the invariant ink contour blade ($W = u \wedge \nabla^2 u$) from the interior cel paint fill ($D = u \cdot \nabla^2 u$).
    - Parameterize continuous gait deformations along phase $\phi \in S^1$ with volume conservation ($\det S = 1$), eliminating Gibbs color ringing and blurry outline mush.
  - **Gate Criteria:**
    - Unfolded character reconstruction achieves $\text{IoU} \ge 0.985$ and $\text{LPIPS} \le 0.005$ across all frames of continuous turnaround and walk cycles, maintaining sharp, pixel-crisp contours.

- [ ] **R10.5 — Real-Video Empirical Gate Suite (`tests/r10_model_sheet_gates.rs`)**
  - **Objective:**
    - Purge all procedural synthetic ellipse sequences (`build_brisby_cape_turn_sequence`, `build_jeremy_crow_wing_flap_sequence`) from the test suite.
    - Test end-to-end model sheet induction and generative unfoldment directly against real video benchmarks:
      1. *The Secret of NIMH* (BluRay 1080p hand-drawn feature animation).
      2. *Super Mario Bros. 3* (NES 2D platformer with variable camera scroll).
      3. *Pac-Man* (Arcade 2D multi-actor discrete turns).
      4. *Raptor: Call of the Shadows* (PC DOS vertical shoot-'em-up with shaded sprites).
  - **Gate Criteria:**
    - All 4 benchmark clips extract valid `HolonModelSheet` representations with zero hardcoded magic number constants. Real-time unfoldment maintains $\ge 60\text{ FPS}$ on RTX 4090 with zero temporal flicker and sample-accurate audio-visual synchronization.

---

## VI. Verification & Human Review Checklist

Before proceeding to implementation, please review the following architectural decisions:
1. **Model Sheet Formalism:** Does the `HolonModelSheet` data structure with canonical turnaround idempotents ($\{\epsilon_v\}$) and PGA Line of Action ($L_{\text{spine}}$) fully capture your intent for Don Bluth-style character representation?
2. **Elimination of Magic Constants:** Are you aligned with using statistical lattice noise floors ($\text{MAD}$) and CliffordNet wedge vorticity zero-crossings to eliminate all hardcoded thresholds?
3. **Purging Synthetic Mocks:** Confirming that we will evaluate solely against real video footage (NIMH BluRay, SMB3, Pac-Man, Raptor) in the new test suite.
