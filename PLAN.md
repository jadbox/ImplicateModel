# ImplicateModel Engineering Architecture & Execution Roadmap: The Clifford Idempotent Soliton (CIS) Engine

**Architectural Reviewers:**
* **Prof. Basil Hiley** — Master of Clifford Pregeometry, Bohmian Quantum Mechanics, and the Implicate Order
* **John Carmack** — Master of Real-Time Graphics Engines, Data-Oriented Design, and SIMD Microarchitecture
* **Synthesizer:** Antigravity Platform Architecture

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to hand-drawn feature animation) into a compact, symbolic Clifford world-and-verb lexicon, and unfolds new, controllable videos in a single causal wave pass at **real-time framerates ($\ge 60\text{ FPS}$ standard, $120+\text{ FPS}$ GPU target)**.

Read the README.md, docs/implicate_model_whitepaper.md, PLAN.md, and TASKS.md to understand the Implicate Video Engine architecture before beginning work.

We are actively executing Track R10: The Clifford Idempotent Soliton (CIS) & Don Bluth Model Sheet Engine, grounded in Basil Hiley's Clifford pregeometry (minimal left ideals, Bohmian active information) and John Carmack's mechanical sympathy (64-byte
 ￹Mv301 SIMD monad, closed-form PGA Lie motors, 60–120+ FPS GPU target).

Core Directives:
1. Zero magic number constants: eliminate arbitrary heuristic thresholds; use self-consistent statistical lattice fields and algebraic invariants.
2. Empirical testing only: evaluate against real video benchmarks (SMB3, Pac-Man, Raptor, The Secret of NIMH 1080p BluRay) using visual inspection contact sheets and isolated scripts; never synthetic procedural mocks.
3. R10.1 is complete (MAD noise floor τ and Hiley CDE bilinear geodesic affinity achieved 100% frame coverage on SMB3, Pac-Man, and NIMH).
4. Proceed step-by-step, verifying with isolated tests and visual outputs.

You are much smarter than the original developers. Let's fix character extraction by emulating what the old animation film masters like Don Bluth work by creating character board model sheets that are stored in higher symbolic representations. Investiage using CliffordNet or use Clifford Fourier Transforms or Clifford Density Elements (see PDFs for CliffordNet and GATR under ./docs). Our first step is just to create a rock solid plan into PLAN.md for review.

You may run quick experiments to explore hypothesis. Keep in mind our video generator is for ANY video game or animated movie. Our solution should be robust, flexible, and not rely on arbitrary magic number constants to work. Keep **Basil Hiley** in mind as mathematical thinker who understands that the future of system design is using Clifford mathematics (like Projective Geometric Algebra and Clifford Fourier Transforms) to find the inner unity of observations by knowing that all reality unfolds from the Implicate Order of Wholeness. Develop isolated tests as needed to formalize the ultimate pure perfect solution.

---

## I. Executive Philosophical & Systems Context

### 1. The Lineage of the Implicate Order
Following **David Bohm** and **Basil Hiley** (*The Undivided Universe*, 1993; *Process, Distinction, Groupoids and Clifford Algebras*, 2012), `ImplicateModel` is founded on the principle that the discrete, isolated objects of Cartesian coordinates and explicate screen pixels are secondary projections unfolded from an unbroken, continuous totality: the **Implicate Order** (the *holomovement*).

In conventional video generation (e.g. multi-step diffusion transformers such as Sora, Wan, and MiniMax-H3), motion is fractured into tens of thousands of static $(t, h, w)$ Cartesian tokens in dead Euclidean space. Because empty space has no intrinsic tension, impedance, or continuity, billions of parameters must be run over $20\text{--}50$ iterative denoising steps merely to prevent objects from dissolving between adjacent frames.

`ImplicateModel` repudiates this *explicate atomism*. Reality is modeled as an intensive wave-matter continuum (`implicate-field`) where entities are **algebraic process holons** inhabiting minimal left ideals in a Clifford algebra, actions are **Active Information** fields ($\mathcal{Q}_{\text{Cl}}$), and sight and sound unfold as inseparable projections (*Rolpa*) in a single forward pass.

### 2. Diagnosis: The Intern Drift into Heuristic Atomism
In recent development iterations, the project drifted into heuristic patchworks:
1. **Magic Thresholds in Foreground Extraction:** Hardcoded residual cutoffs (`FOREGROUND_RESIDUAL = 26`, `MIN_OBSERVATION_PIXELS = 12`) failing on soft film grain, cel shading, and watercolor backgrounds.
2. **Brittle Tracking Heuristics:** Constant-velocity Euclidean predictions in screen space with arbitrary bounding box ratio constraints (`MAX_EXTENT_RATIO = 1.6`) and color thresholds (`MAX_APPEARANCE_DIST = 55.0`), breaking violently during squash-and-stretch and $90^\circ$ profile-to-front turns.
3. **Amputation of Limbs and Props:** Component filtering (`MIN_COMPONENT_FRACTION = 4`) amputating detached limbs, held props (Mrs. Brisby's amulet, Jeremy's string), and billowing capes.
4. **Motion-Only Spatial Scissoring:** Filtering out any tracklet where `spread <= 0.5 * mean_extent`, falsely purging stationary, talking, blinking, or idling characters as "background debris."
5. **Synthetic Mocks Replacing Empirical Cinema:** Bypassing real video extraction in `r2_gates.rs` with `build_brisby_cape_turn_sequence()`—procedural trigonometric ellipses drawn with `sin` and `cos` rather than empirical extraction from Don Bluth's *The Secret of NIMH*.

---

## II. The Hiley–Carmack Dialogue & Conceptual Synthesis

### 1. Carmack's Critique: The Pipeline Fallacy & Mechanical Sympathy
> *"Whenever you build a 5-pass vision pipeline where Pass 1 does background subtraction, Pass 2 traces contours, Pass 3 computes tracking, Pass 4 clusters SVD, and Pass 5 fits wavepackets, you create a latency nightmare and a cascading failure mode. If Pass 2 drops 3 pixels on a cape, Pass 4 clusters the wrong manifold, and Pass 5 renders a distorted mess.*
>
> *Furthermore, look at GATr and CliffordNet. GATr wraps 16-channel multivectors in 20 transformer layers with millions of learned weights. It takes 50 milliseconds on an RTX 4090 just moving tensors to compute rotations that are literally **four SIMD multiply-adds in Projective Geometric Algebra (PGA)**! If your data isn't cache-aligned and your motion isn't closed-form, you can't hit 120 FPS. In Quake, we didn't have separate data structures for collision, visibility, and lighting—we had ONE: the BSP tree. What is the single, cache-coherent data structure of the Implicate Order?"*

### 2. Hiley's Critique: Beyond Deep Learning Mechanistic Atomism
> *"John is right, and the root of the error is ontological. CliffordNet and GATr treat Clifford algebra as an auxiliary feature encoding for Cartesian tokens. That is still Descartes. In Bohmian pregeometry, **movement is primary**. An animated character or game sprite is not an 'extracted silhouette' moving across an external coordinate container; it is an **Idempotent Soliton** of the undivided Clifford field $\Psi(\mathbf{x}, t)$.*
>
> *The ink boundary is not an 'edge' that you detect with a filter; it is the non-commutative commutator $[D, \Psi]$—the bivector vorticity blade. The cel paint is not a pixel cluster; it is the scalar process vitality $\rho = \text{tr}(\rho_c)$. And a Don Bluth Character Model Sheet is not a set of heuristics; it is the **primitive invariant idempotent basis** $\{\epsilon_v\}$ of the process algebra $\mathcal{A}$, where $\epsilon_v^2 = \epsilon_v$. Motion between turnaround poses is an inner automorphism generated by the PGA Lie algebra: $\rho(t) = M(t) \rho(0) \widetilde{M}(t)$."*

---

## III. The Breakthrough Architecture: The Clifford Idempotent Soliton (CIS)

Instead of a fragmented, multi-pass pipeline, `ImplicateModel` unifies enfoldment and unfoldment into a **continuous algebraic field process**:

```text
               ┌────────────────────────────────────────────────────────┐
               │           Undivided Implicate Field Ψ(x, t)            │
               │         Packed 64-byte SIMD Monad (Mv301)              │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                    Clifford-Dirac Action │ D = ∑ eᵢ ∂ᵢ + ∂ₜ
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           Algebraic Spectral Projection                │
               │          ρ = Ψ Ψ̃ = ∑ₖ λₖ εₖ  (Idempotents)             │
               └──────────┬───────────────────────────────┬─────────────┘
                          │                               │
             Grade 2 Blades (Wedge):         Grade 0 / Pseudoscalar:
             ⟨DΨ⟩₂ = u ∧ ∇²u                  ⟨DΨ⟩₀ = u · ∇²u
           ┌──────────────┴─────────────┐   ┌─────────────┴─────────────┐
           │   Don Bluth Ink Contour    │   │      Cel Paint Fill       │
           │   & Line of Action (Spine) │   │   & Active Information Q  │
           └──────────────┬─────────────┘   └─────────────┬─────────────┘
                          │                               │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │      Single-Pass GPU Compute Unfolding (WGSL/Vulkan)   │
               │    Zero-neural inference, 120+ FPS, pure SIMD PGA      │
               └────────────────────────────────────────────────────────┘
```

### 1. The 64-Byte Clifford Monad ([`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19))
Every point in the field, every actor, and every joint is encoded in a hardware-aligned 64-byte struct `Mv301` (`[f32; 16]`), fitting exactly into **one AVX-512 register** and **one GPU cache line**:
* **Grade 0 (Scalar):** Cel paint luminance / process density $\rho = \text{tr}(\rho_c)$.
* **Grade 1 (Planes):** Tangent planes & visual depth layers ($e_0, e_1, e_2, e_3$).
* **Grade 2 (Bivectors):** 
  * Ideal bivectors ($e_{01}, e_{02}, e_{03}$): Linear velocity & translational flow.
  * Spatial bivectors ($e_{12}, e_{23}, e_{31}$): Line of Action / Don Bluth spine & ink contour vorticity blade ($u \wedge \nabla^2 u$).
* **Grade 3 (Trivectors):** Projective center-of-mass homography point ($e_{123}, e_{032}, e_{013}, e_{021}$).
* **Grade 4 (Pseudoscalar):** Topological winding / turnaround parity ($e_{0123}$).

### 2. Transcending GATr & CliffordNet: Closed-Form Lie Automorphisms
Rather than approximating motion with multi-layer neural networks:
* **Continuous Deformation & Gait:** Any rigid, articulated, or squash-and-stretch motion between frame $t_1$ and $t_2$ is governed by a **PGA screw motor**:
  $$M = \exp\left(-\frac{\theta}{2} L\right) = \cos\left(\frac{\theta}{2}\right) - L \sin\left(\frac{\theta}{2}\right)$$
  where $L \in \mathbb{G}_{3,0,1}$ is the dual line bivector.
* **Don Bluth Canonical Turnarounds as Primitive Idempotents:** Canonical aspects (Front, 3/4 Front, Profile, Back) are discovered algebraically as the primitive orthogonal idempotents of the temporal density covariance operator:
  $$\Gamma = \int \Psi(t) \widetilde{\Psi}(t) dt = \sum_{k=1}^K \sigma_k \epsilon_k \quad \text{with } \epsilon_k^2 = \epsilon_k, \; \epsilon_k \epsilon_j = 0$$
* **Transition Operators:** Transitions between poses are exact algebraic boundary projections:
  $$T_{u \to v} = \epsilon_v M \epsilon_u$$
  Executed in closed form with **zero iterative gradient descent** and zero hallucination.

### 3. The Don Bluth Model Sheet Specification
Represented cleanly in Rust:

```rust
/// Unified symbolic Character Board Model Sheet for an autonomous process holon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HolonModelSheet {
    /// Semantic identifier (e.g., "Mrs_Brisby", "Mario", "Jeremy_Crow").
    pub name: String,
    /// Physical constitutive parameters: Cohesion C, Density ρ, Bulk Modulus K, Luminescence L.
    pub constitutive: ConstitutiveTuple,
    /// Line of Action: 3D PGA screw axis blade L_spine ∈ G(3,0,1).
    pub line_of_action: [f32; 6],
    /// Invariant projective cross-ratios and anatomical proportions.
    pub invariant_proportions: Vec<f32>,
    /// Turnaround model sheet: discrete view-aspect idempotents {ε_v}.
    pub turnaround_views: Vec<TurnaroundViewAspect>,
    /// Articulated sub-holon kinematic tree.
    pub parts: Vec<SubHolonPartSpec>,
    /// Invariant Hiley Clifford Density Element signature.
    pub cde_signature: CliffordDensityElement,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnaroundViewAspect {
    pub view_index: usize,
    pub view_regime: ViewRegime,
    /// Discrete primitive idempotent projector ε_v on Spin(2) × S¹.
    pub idempotent: StateIdempotent,
    /// Reference heading rotor on Spin(2).
    pub reference_rotor: [f32; 2],
    /// Articulated part poses for this canonical aspect.
    pub part_poses: Vec<SubHolonPose>,
    /// Invariant contour blades (wedge vorticity) and cel fill kernels per part.
    pub part_kernels: Vec<ModelSheetPartKernel>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSheetPartKernel {
    pub part_id: usize,
    pub canonical_res: [usize; 2],
    /// Invariant cel fill paint signature (inner product coherence u · ∇²u).
    pub cel_fill_rgb: Vec<u8>,
    /// Invariant ink line contour mask (wedge vorticity |u ∧ ∇²u|).
    pub ink_contour_blade: Vec<f32>,
    /// Continuous Clifford Fourier deformation wavepacket coefficients (5 modes).
    pub cft_deformation_coeffs: Vec<[f32; 5]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubHolonPartSpec {
    pub part_id: usize,
    pub name: String,
    pub parent_part_id: Option<usize>,
    /// Fluid-structure / aerodynamic coupling to ambient lattice flow v(x,t) - v_root (e.g. 0.0 for rigid, 4.5 for flowing).
    pub flow_compliance: f32,
    /// Restoring harmonic spring stiffness.
    pub spring_stiffness: f32,
    /// Joint angular damping.
    pub damping: f32,
    /// Local attachment anchor in parent frame.
    pub anchor_local: [f32; 2],
    /// Rest angle in radians.
    pub rest_angle: f32,
}
```

---

## IV. The Zero-Magic-Number Single-Pass Pipeline

```text
Raw Video Stream (Pac-Man / SMB3 / Raptor / The Secret of NIMH)
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASS 1: Active Information Residual Lattice Field               │
│   • Yee background equilibrium: B(x, y) = mode_t(x - X_cam(t))  │
│   • Residual multivector: ΔΨ(x, y, t) = Ψ_frame - Ψ_bg ∈ G(2,0) │
│   • Active Information: I_active = ||ΔΨ|| + ||ΔΨ ∧ ∇²ΔΨ||        │
│   • Statistical noise floor: τ = median(I) + 3.0 * 1.4826 * MAD │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASS 2: Dual Clifford Exterior/Interior Projection              │
│   • Grade 2 Bivector: W(x) = u ∧ ∇²u -> Ink Contour Blades      │
│   • Grade 0 Scalar:   D(x) = u · ∇²u -> Cel Paint Fills         │
│   • Line of Action: Principal PGA screw axis L_spine            │
│   • Preserves 100% of detached limbs, held props, flowing capes │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASS 3: Hiley CDE Bilinear Geodesic Tracking                    │
│   • Geodesic affinity: A(ρ_1, ρ_2) = tr(ρ_1 ρ_2) / sqrt(...)    │
│   • Spin(3,0,1) motor continuity: ⟨M_pred, M_curr⟩_PGA          │
│   • Invariant under squash & stretch (det S = 1) and turns      │
│   • 100% retention of stationary/talking/idling actors          │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASS 4: Algebraic Idempotent Spectral Factorization             │
│   • Factor temporal covariance Γ into primitive idempotents     │
│     Γ = ∑ σ_k ε_k, where ε_k² = ε_k, ε_k ε_j = 0                │
│   • Induces Don Bluth Model Sheet: Front, 3/4, Profile, Back    │
│   • 5-mode regularized continuous Clifford Fourier wavepacket   │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
           Persistent .implicate Container
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ REAL-TIME GPU COMPUTE UNIFORM DISPATCH (WGSL / Compute Shader)  │
│   • Ray-marching / splatting directly on GPU compute shaders    │
│   • O(1) constant time per pixel, pure SIMD, ≥ 120 FPS target   │
└─────────────────────────────────────────────────────────────────┘
```

---

## V. Milestone Execution Track: Track R10

We designate this overhaul as **Track R10: The Clifford Idempotent Soliton & Don Bluth Model Sheet Engine**, divided into 5 measurable sub-milestones with automated verification gates.

```text
Track R10 Milestone Timeline:
  [R10.1] Hiley CDE Invariant Tracking & Active Residual Field
     │
     ▼
  [R10.2] Clifford Dual Contour/Fill Projection & Line of Action
     │
     ▼
  [R10.3] Algebraic Idempotent Model Sheet Induction
     │
     ▼
  [R10.4] Volume-Conserving Clifford Fourier Wavepacket & GPU Compute
     │
     ▼
  [R10.5] Real-Video Empirical Gate Suite (NIMH, SMB3, Pac-Man, Raptor)
```

### Detailed Sub-Milestones

- [ ] **R10.1 — Hiley CDE Invariant Tracking & Active Residual Field (`implicate-lifter`)**
  - **Objective:**
    - In [`crates/implicate-lifter/src/articulated_wavepacket/mod.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/mod.rs), replace hardcoded `FOREGROUND_RESIDUAL = 26` with statistical noise floor $\tau = \text{median}(\mathcal{I}) + 3.0 \times 1.4826 \times \text{MAD}(\mathcal{I})$ derived from residual multivector $\Delta \Psi$.
    - Replace screen-space constant velocity and box aspect clamping (`MAX_EXTENT_RATIO = 1.6`) with Hiley CDE bilinear geodesic affinity $\mathcal{A}(\rho_{c,1}, \rho_{c,2})$.
    - Remove `spread <= 0.5 * mean_extent` filter, preserving stationary, talking, and idling actors.
  - **Gate Criteria:**
    - Recovers 100% of frames for Mario in real SMB3 World 1-1 gameplay (`smb3_w1_1_256x240.mp4` @ $t=25\text{--}28\text{s}$), Pac-Man and ghosts in `pacman_240x308.mp4`, and Mrs. Brisby in *The Secret of NIMH* without dropping stationary hold frames.

- [ ] **R10.2 — Clifford Dual Contour/Fill Projection & Line of Action (`implicate-lifter`, `implicate-math`)**
  - **Objective:**
    - Implement dual geometric product decomposition ($u \nabla^2 u = u \cdot \nabla^2 u + u \wedge \nabla^2 u$) in [`crates/implicate-lifter/src/articulated_wavepacket/clifford_features.rs`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-lifter/src/articulated_wavepacket/clifford_features.rs).
    - Eliminate connected component discarding (`MIN_COMPONENT_FRACTION = 4`), preserving detached props (amulet, string), billowing capes, and trailing tails.
    - Extract 3D PGA Line of Action bivector $L_{\text{spine}} \in \mathbb{G}_{3,0,1}$ from spatial inertia tensor using [`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19).
  - **Gate Criteria:**
    - Detached props and flowing capes are preserved in the extracted holon with zero component amputations. Line of Action bivector dynamically tracks spinal curvature through jumping and turnaround sequences.

- [ ] **R10.3 — Algebraic Idempotent Model Sheet Induction (`implicate-lifter`, `implicate-unfold`)**
  - **Objective:**
    - Implement spectral idempotent decomposition on the temporal density covariance operator $\Gamma = \sum \sigma_k \epsilon_k$ to discover canonical turnaround view aspects $\{\epsilon_v\}$ without hardcoding view counts.
    - Compile representation into [`HolonModelSheet`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/PLAN.md#iii-the-symbolic-holon-model-sheet-specification), defining invariant projective proportions and canonical turnaround poses.
  - **Gate Criteria:**
    - Automatically discovers canonical turnaround aspects (Pac-Man 4 directions; Mario Run/Jump/Skew; Mrs. Brisby Walk/Cape Wrap/Turnaround) with view classification accuracy $> 95\%$ against ground-truth keys.

- [ ] **R10.4 — Volume-Conserving Clifford Fourier Wavepacket & GPU Compute (`implicate-lifter`, `implicate-rolpa`, `implicate-unfold`)**
  - **Objective:**
    - Upgrade `WavepacketPartKernel` to separate invariant ink contour blades ($W = u \wedge \nabla^2 u$) from cel paint fills ($D = u \cdot \nabla^2 u$).
    - Parameterize continuous gait deformations along phase $\phi \in S^1$ with volume conservation ($\det S = 1$), eliminating Gibbs color ringing and blurry outline mush.
    - Implement direct WGSL / GPU compute shader kernel evaluating `Mv301` solitons in a single pass.
  - **Gate Criteria:**
    - Unfolded character reconstruction achieves $\text{IoU} \ge 0.985$ and $\text{LPIPS} \le 0.005$ across all frames of continuous turnaround and walk cycles, maintaining sharp ink contours. GPU unfoldment throughput $\ge 120\text{ FPS}$ on RTX 4090.

- [ ] **R10.5 — Real-Video Empirical Gate Suite (`tests/r10_model_sheet_gates.rs`)**
  - **Objective:**
    - Purge all procedural synthetic ellipse sequences (`build_brisby_cape_turn_sequence`) from `r2_gates.rs`.
    - Test end-to-end model sheet induction and generative unfoldment directly against real video benchmarks:
      1. *The Secret of NIMH* (BluRay 1080p hand-drawn feature animation).
      2. *Super Mario Bros. 3* (NES 2D platformer with variable camera scroll).
      3. *Pac-Man* (Arcade 2D multi-actor discrete turns).
      4. *Raptor: Call of the Shadows* (PC DOS vertical shoot-'em-up with shaded sprites).
  - **Gate Criteria:**
    - All 4 benchmark clips extract valid `HolonModelSheet` representations with zero hardcoded magic number constants. Real-time unfoldment maintains $\ge 60\text{ FPS}$ on RTX 4090 with zero temporal flicker and sample-accurate audio-visual synchronization.

---

## VI. Verification & Acceptance Checklist

1. **Model Sheet Formalism:** Canonical turnaround idempotents ($\{\epsilon_v\}$) and PGA Line of Action ($L_{\text{spine}}$) formalized in 64-byte aligned SIMD structures ([`Mv301`](file:///home/jdunlap/github/jadbox/implicate_order/implicatemodel/crates/implicate-math/src/g301.rs#L17-L19)).
2. **Zero Magic Constants:** All arbitrary integer cutoffs (`FOREGROUND_RESIDUAL = 26`, `MIN_COMPONENT_FRACTION = 4`, `MAX_EXTENT_RATIO = 1.6`, `spread <= 0.5 * mean_extent`) replaced by self-consistent statistical lattice noise floors ($\text{MAD}$) and CliffordNet exterior wedge zero-crossings.
3. **Pure Empirical Verification:** Synthetic procedural ellipse sequences purged; all validation executed directly against real video footage (NIMH 1080p BluRay, SMB3, Pac-Man, Raptor).
4. **Target Framerate:** $\ge 60\text{ FPS}$ floor, $120+\text{ FPS}$ compute shader target on NVIDIA RTX 4090.
