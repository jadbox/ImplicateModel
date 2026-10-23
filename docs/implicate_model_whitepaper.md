# ImplicateModel: The Clifford Process Video Engine
### Engineering Systems Whitepaper: Multi-Stratum Geometric Algebra, Bohm-Hiley Active Information, and Single-Pass Causal Video Unfoldment

<small>by [Jonathan Dunlap](https://www.linkedin.com/in/jonathandunlap/)</small>

**Companion Documents:**
* **Guiding Philosophy:** [`docs/implicate_model_philosophy.md`](./implicate_model_philosophy.md)
* **Prior Art & Foundations:** [`dzoglang/docs/field_first_whitepaper.md`](../../dzoglang/docs/field_first_whitepaper.md) · [`holonengine/docs/holon_whitepaper.md`](../../holonengine/docs/holon_whitepaper.md)
* **Execution Plan & Research Tasks:** [`PLAN.md`](../PLAN.md) · [`TASKS.md`](../TASKS.md)

---

## Abstract

Contemporary video generation architectures (such as MiniMax-H3, Sora, and Wan) model moving imagery by shattering continuous spacetime into discrete $(t, h, w)$ patch tokens and running $20\text{–}50$ iterative diffusion passes over $O(N^2)$ attention matrices. Because the latent space between tokens lacks physical continuity, conservation laws, or 3D geometric equivariance, these models require tens of billions of parameters and millions of training clips merely to prevent moving objects from dissolving between frames.

**`ImplicateModel`** introduces a fundamentally different architecture grounded in **David Bohm and Basil Hiley's** *Implicate Order* and *Clifford Pregeometry*, **Louis de Broglie's** *Nonlinear Soliton Mechanics*, and **Gilles Deleuze's** distinction between the *Diagram* and the *Cliché*. Rather than predicting pixel patches statistically from Gaussian noise, `ImplicateModel` compiles raw video and audio streams into an invariant **Implicate Lexicon & World Model** (`implicate enfold`) and generates new, prompt-controllable videos in a **single causal forward pass** (`implicate unfold`).

To achieve both mathematical rigor and hardware scalability on a single workstation GPU, `ImplicateModel` unifies three distinct Clifford regimes across three computational strata:
1. **Stratum 1 — Pixel Lifting & `@rolpa` Surface Refinement ($O(N)$ over 2D Grids):** Uses a **Shifted Clifford Geometric Product Network (`CliffordNet`)** operating over a factorized discrete Laplacian context ($\mathcal{C}_{\text{loc}} \approx \Delta H$) to extract sub-holon part masks and bivector edge flows during Enfoldment, and to refine projected wavepackets into 1080p frames in a single forward pass during Unfoldment.
2. **Stratum 2 — Semantic-to-Geometric Concept Binding ($O(K^2)$ over $K \ll 64$ Object Holons):** Uses a $\text{Pin}(3,0,1)$-equivariant **Geometric Algebra Transformer (`GATr`)** coupled with offline **`llama.cpp` Qwen-VL** semantic harvesting to bind natural-language nouns and verbs to $E(3)$-equivariant 3D Projective Geometric Algebra ($\mathbb{G}_{3,0,1}$) versor trajectories and scale-invariant Bohm-Hiley Quantum Potentials ($\mathcal{Q}_{\text{Cl}} = -(\nabla^2 \Psi)\Psi^{-1}$).
3. **Stratum 3 — Spatiotemporal Causality (The Implicate Spine):** Executes a deterministic, zero-AI GPU wave-matter lattice (`Yee` staggered FDTD + minimal left ideal `CliffordDensityElement` kinematics) that enforces object permanence, momentum conservation, Fresnel boundary reflection, Griffith fracture/annealing, and unified audio-visual projection (`@rolpa`) at $60\text{–}1,000+\text{ Hz}$.

---

## 1. System Architecture & End-to-End Pipeline

```text
====================================================================================================
                        PHASE I: ENFOLDMENT (Explicate Video -> Implicate Lexicon)
====================================================================================================

[ Input Video + Audio Stream ] (e.g., Raptor, SMB3, Pac-Man, or "The Secret of NIMH" 1080p)
       │
       ├──> 0. Streaming Shot & Camera Regime Segmenter (CFT Cross-Power + Divergence ∇·v)
       │       └── Splits stream into contiguous shots S_m; classifies camera motor M_cam(t) ∈ Spin(3,0,1)
       │           (Ortho2D-Static, Ortho2D-Scroll, MultiPlane-2.5D, Perspective-3D)
       │
       ├──> 1A. Semantic Stream (llama.cpp Qwen-VL + llama-embedding + Thin SVD)
       │       └── Extracts GBNF-constrained Scene Graphs:
       │           • Milieu (lighting, medium viscosity, atmosphere)
       │           • Holon Nouns (character/prop identity, material class, part tree) -> Spinors ψ_noun ∈ (G_2,0^+)^K
       │           • Process Verbs ((Holon A) --[Verb]--> (Holon B))                 -> Spinors ψ_verb ∈ (G_2,0^+)^K
       │
       └──> 1B. Geometric & Acoustic Stream (Multi-Plane CFT + CliffordNet Lifter)
               ├── Reconstructs depth-stratified ghost-free background atlases A_l(x) (1/z_l layers)
               ├── Tracks Articulated Holon Trees H_i = H_root ⋉ Π H_{i,p} as oriented Gaussian-Gabor wavepackets
               ├── Regresses relative PGA motor trajectories M_{i,p}(t) ∈ Spin(3,0,1) & empirical incidences (P ∧ B = 0)
               └── Extracts synchronized acoustic pressure envelopes E_audio(t, ω) at collision/impulse loci
                                                    │
                                                    ▼
       ┌───────────────────────────────────────────────────────────────────────────────────────┐
       │ 2. THE IMPLICATE LEXICON & WORLD STORE (.implicate / SQLite + Binary Tensor Store)    │
       │    • Noun Ideals:      ψ_noun  <~> Kinematic Part Tree + CFT Kernels K̂(k,ω) + (ρ,K,γ,C)│
       │    • Verb Operators:   ψ_verb  <~> Pin(3,0,1)-Equivariant GATr Operator -> (Q_Cl, ΔB) │
       │    • Milieu Grounds:   ψ_scene <~> Multi-Plane Boundary Blades B_k ∈ G_{3,0,1}        │
       └───────────────────────────────────────────────────────────────────────────────────────┘

====================================================================================================
                      PHASE II: UNFOLDMENT (Prompt / Script -> Single-Pass Video + Audio)
====================================================================================================

[ Text Prompt / DzogLang Script / Seed Frame ]
       │
       ▼
  3A. Scene & Lexicon Compiler (`implicate-unfold`)
       ├── Seeds Ground (Gzhi): initializes multi-plane boundaries B_k and constitutive lattice (ρ, K, γ, C, L_0)
       ├── Injects Holons (Nouns): instantiates articulated CliffordDensityElements Φ_{L,i,p} = R U ε_s
       └── Excites Potentials (Verbs): GATr maps active verbs to Q_Cl guidance beacons & motor impulses
       │
       ▼
  3B. Single-Pass Causal Lattice Runtime (`implicate-field` + `implicate-math`, 60–1000+ Hz)
       ├── Steps Yee-staggered wave-matter PDEs: pressure p, velocity v, Bohmian Q, luminescence L, cohesion C
       ├── Integrates articulated Holon motors M_{i,p}(t) under -∇⟨Q_Cl⟩_0 force & bivector spin-torque ⟨Q_Cl⟩_2
       └── Evaluates exact PGA incidence predicates (P ∧ B = 0) -> algebraic state transitions T_{s->s'} = ε_{s'} M ε_s
       │
       ▼
  4.  @rolpa Shadow-Manifold Projector (`implicate-rolpa`)
       ├── Visual Lens: Multi-plane CFT/Gaussian wavepacket splatting + ∇L facet lighting + 1-Pass CliffordNet Refiner
       └── Acoustic Lens: Binaural cockpit/camera pickups + Doppler shift + Material-tuned Modal/Harmonic Synth (48 kHz)
       │
       ▼
[ Output: 1080p @ 60 FPS Flicker-Free Video + Phase-Locked Stereo Audio ]
```

---

## 2. Core Mathematical Foundations (`implicate-math`)

### 2.1 The Algebraic Tower: $\mathcal{C}\ell_{0,1} \to \mathbb{G}_{2,0} \to \mathbb{G}_{3,0,1}$

`ImplicateModel` represents all spatial geometry, kinematic transformations, internal states, and semantic identities within a unified tower of Clifford algebras:

1. **Temporal / Phase Harmonics ($\mathcal{C}\ell_{0,1} \cong \mathbb{C}$):**
   Governs Continuous Fourier Transform (CFT) vitality kernels $\hat{K}(\mathbf{k}, \omega)$ and de Broglie phase-harmony locking $\exp(i\phi(t))$.
2. **Chromatic & Semantic Spinor Space ($(\mathbb{G}_{2,0}^+)^K$):**
   With basis $e_1^2 = e_2^2 = 1$ and bivector $e_{12} = e_1 e_2$ ($e_{12}^2 = -1$), a $K$-channel even multivector $\psi = (R_1 e^{\theta_1 e_{12}}, \dots, R_K e^{\theta_K e_{12}})$ encodes visual appearance and wording-space semantics. Similarity between any two signatures is evaluated via the coordinate-free bilinear scalar inner product:
   $$\mathcal{S}(\psi_A, \psi_B) = \frac{\langle \psi_A \widetilde{\psi}_B \rangle_0}{\|\psi_A\|\,\|\psi_B\|} \in [-1, 1]$$
3. **3D Projective Geometric Algebra ($\mathbb{G}_{3,0,1}$):**
   Uses basis $(e_0, e_1, e_2, e_3)$ with metric $e_0^2 = 0$ and $e_1^2 = e_2^2 = e_3^2 = 1$ ($2^4 = 16$ components). Following plane-based PGA duality (Dorst 2020; Brehmer et al. 2023), geometric primitives and $E(3)$ operators inhabit grades $0 \dots 4$:

| Grade | Basis Elements | Geometric Primitive | Transformation Operator ($\text{Pin}(3,0,1)$) |
| :--- | :--- | :--- | :--- |
| **0 (Scalar)** | $1$ | Invariant mass / vitality density $\rho$ | Identity scaling |
| **1 (Vector)** | $e_1, e_2, e_3;\; e_0$ | Oriented Plane $\Pi = n_x e_1 + n_y e_2 + n_z e_3 + d e_0$ | Reflection across plane $\Pi$ |
| **2 (Bivector)** | $e_{23}, e_{31}, e_{12};\; e_{01}, e_{02}, e_{03}$ | Oriented Line / Axis $L$ (rotation axis + translation moment) | Infinitesimal generator $B$ of a rigid body `Motor` $M = \exp(-\frac{1}{2}B)$ |
| **3 (Trivector)** | $e_{123};\; e_{023}, e_{031}, e_{012}$ | Euclidean Point $P = x e_{032} + y e_{013} + z e_{021} + e_{123}$ | Point reflection (inversion) through $P$ |
| **4 (Pseudoscalar)** | $I = e_{0123}$ | Oriented Volume / Chirality reference | Dualization / Chirality sign stabilizer |

Any rigid motion $M \in \text{Spin}(3,0,1)$ (combining 3D rotation around an arbitrary axis and 3D translation) acts universally on points, lines, and planes via the grade-preserving **sandwich product**:
$$X' = M\,X\,\widetilde{M}$$

### 2.2 Hierarchical Articulated Holons & Minimal Left Ideals

In `HolonEngine`, an actor was represented by a single bounding box. To handle general video—such as Mrs. Brisby's billowing cape, Jeremy the Crow's articulated wings, or multi-part bosses—`ImplicateModel` generalizes the **Algebraic Holon** into a **Kinematic Tree of Minimal Left Ideals**:

$$\mathcal{H}_i = \left(\mathcal{H}_{i,\text{root}},\; \left\{ \mathcal{H}_{i,p},\; \text{parent}(p),\; B_{\text{joint},p} \in \bigwedge^2 \mathbb{G}_{3,0,1} \right\}_{p=1}^{P_i}\right)$$

Each sub-holon node $p$ (where $p=0$ is the root) carries a **Clifford Density Element (CDE)** inhabiting a minimal left ideal $\mathcal{I}_{i,p} = \mathcal{C}\ell\,\epsilon_{i,s}$:

$$\Phi_{L,i,p}(\mathbf{x}, t) = R_{i,p}(\mathbf{x}, t)\, M_{i,p}(t)\, \epsilon_{i,s}, \qquad \rho_{c,i,p}(t) = \Phi_{L,i,p}\,\widetilde{\Phi}_{L,i,p} = \rho_{i,p}\, M_{i,p}\,\epsilon_{i,s}\,\widetilde{M}_{i,p}$$

where:
* **$R_{i,p} \ge 0$ ($\rho_{i,p} = R_{i,p}^2$):** Process vitality amplitude. Collapsing a part or consuming an excitation drives $\rho_{i,p} \to 0$.
* **$M_{i,p}(t) \in \text{Spin}(3,0,1)$:** World-space PGA motor of part $p$, factored recursively along the kinematic chain:
  $$M_{i,p}(t) = M_{i,\text{parent}(p)}(t)\, \exp\!\left(-\frac{1}{2}\,\theta_p(t)\, B_{\text{joint},p}\right)$$
  where $B_{\text{joint},p}$ is the unit PGA hinge/ball-joint bivector (or translational slider $e_0 \wedge \hat{\mathbf{u}}$) connecting part $p$ to its parent.
* **$\epsilon_{i,s}$ ($\epsilon_{i,s}^2 = \epsilon_{i,s},\; \epsilon_{i,s}\epsilon_{i,s'} = 0$):** Primitive state idempotent selecting the active behavioral or visual regime $s \in \{0, \dots, S-1\}$ (e.g., idle, walk/flight, attack/cast, recoiling, collapsed).

### 2.3 Oriented Clifford Gaussian-Gabor Wavepackets (Replacing Axis-Aligned Boxes)

To eliminate pixel-box quantization and support continuous 3D/2.5D rotation, perspective scaling ($1/z$), and non-rigid shear, each sub-holon $p$ represents its spatial occupancy and visual appearance as an **Oriented Clifford Wavepacket**:

$$\mathcal{W}_{i,p}(\mathbf{x}, t) = G\!\left(\mathbf{x}_{\text{loc}};\, \boldsymbol{\Sigma}_{i,p}\right) \cdot \mathcal{T}_{\phi_{i,p}(t)}\!\left[ \hat{K}_{i,p,0}\!\left(\mathbf{x}_{\text{loc}}, U_{\text{view}}\right) + \sum_{m=1}^{M_{\text{harm}}} \operatorname{Re}\!\left\{ \hat{K}_{i,p,m}\!\left(\mathbf{x}_{\text{loc}}, U_{\text{view}}\right) e^{i m \phi_{i,p}(t)} \right\} \right]$$

where:
* $\mathbf{x}_{\text{loc}} = \widetilde{M}_{i,p}(t)\,\mathbf{x}\,M_{i,p}(t)$ transforms world/screen coordinates into the canonical rest frame of part $p$,
* $G(\mathbf{x}_{\text{loc}}; \boldsymbol{\Sigma}_{i,p}) = \exp\!\left(-\frac{1}{2}\mathbf{x}_{\text{loc}}^\top \boldsymbol{\Sigma}_{i,p}^{-1} \mathbf{x}_{\text{loc}}\right)$ is the anisotropic spatial envelope defining the part's PGA bounding blade $B_{i,p}$,
* $\hat{K}_{i,p,0}$ and $\hat{K}_{i,p,m}$ are the DC and $m$-th temporal CFT harmonic appearance kernels indexed by relative camera viewing rotor $U_{\text{view}} = \widetilde{M}_{\text{cam}} M_{i,\text{root}}$,
* $\phi_{i,p}(t) = \int_0^t \omega_{i,p}(\tau)\,d\tau$ is the accumulated Bohmian action phase.

### 2.4 Active Information & The Clifford Quantum Potential ($\mathcal{Q}_{\text{Cl}}$)

Given the medium's multivector wavefield $\Psi(\mathbf{x}, t) = R(\mathbf{x}, t)\,\mathcal{U}(\mathbf{x}, t)$ (where $R = \sqrt{p^2 + \epsilon_0^2}$ is the regularized wave envelope and $\mathcal{U} \in \text{Spin}(3,0,1)$ is the local field rotor), the **Clifford Quantum Potential** (Hiley & Callaghan 2010) decomposes into a Grade-0 scalar potential and a Grade-2 bivector spin-torque:

$$\mathcal{Q}_{\text{Cl}}(\mathbf{x}, t) = -\left(\nabla^2 \Psi\right)\Psi^{-1} = \underbrace{-\frac{\nabla^2 R}{R}}_{\text{Scalar Guidance } \langle \mathcal{Q}_{\text{Cl}} \rangle_0} - \underbrace{\left(2(\nabla \ln R)\,\mathcal{U}\nabla\widetilde{\mathcal{U}} + (\nabla^2 \mathcal{U})\widetilde{\mathcal{U}}\right)}_{\text{Bivector Spin-Torque } \langle \mathcal{Q}_{\text{Cl}} \rangle_2}$$

Every holon responds to $\mathcal{Q}_{\text{Cl}}$ through two coupled bilinear equations of motion:
1. **Linear Bohmian Guidance Force:**
   $$\mathbf{F}_{Q,i} = -\kappa_Q \nabla \langle \mathcal{Q}_{\text{Cl}}(\mathbf{x}_i, t) \rangle_0$$
   Because $R$ cancels in $\nabla^2 R / R$, $\langle \mathcal{Q}_{\text{Cl}} \rangle_0$ is **scale-invariant**: a subtle goal beacon carves a guidance valley across the scene without requiring high-energy pressure shockwaves.
2. **Rotational Bivector Spin-Torque:**
   $$\Omega_{Q,i} = \mathbf{s}_i \wedge \left(-\nabla \langle \mathcal{Q}_{\text{Cl}}(\mathbf{x}_i, t) \rangle_0\right) + \eta_{\text{spin}}\langle \mathcal{Q}_{\text{Cl}}(\mathbf{x}_i, t) \rangle_2$$
   $$M_{i,\text{root}}(t + \Delta t) = M_{i,\text{root}}(t)\,\exp\!\left(-\frac{1}{2}\gamma_Q\,\Omega_{Q,i}\,\Delta t\right)$$
   This single algebraic law aligns an actor's heading $\mathbf{s}_i = M_{i,\text{root}} e_1 \widetilde{M}_{i,\text{root}}$ along open corridors, curves trajectories smoothly around high-impedance obstacles, and imparts rotational posture without A* search trees or hardcoded splines.

---

## 3. Resolution of the Clifford Neural Operator Question: The Three-Stratum Architecture

In [`docs/implicate_model_philosophy.md`](./implicate_model_philosophy.md), we posed the core architectural question:
> *What kind of Clifford Neural Operator do we build? Do we use `CliffordNet` (`reference_cliffordnet.pdf`), `GATr` (`reference_cliffordgatr.pdf`), or our own closed-form Clifford engine?*

### 3.1 Comparative Analysis of the Three Candidates

| Operator Family | Mathematical Core | Computational Complexity | Primary Strength | Fatal Weakness if Used Alone |
| :--- | :--- | :--- | :--- | :--- |
| **1. Closed-Form Spectral + PDE Engine** (`cc_math`, `dzog-wgpu`) | CFT cross-correlation + Staggered Yee FDTD wave-matter PDEs + PGA incidence ($P \wedge B = 0$). | $O(N_{\text{cells}})$ wave step; $O(N \log N)$ FFT. | Zero neural weights; exact momentum/energy conservation; $1,000+\text{ Hz}$ execution; zero temporal flicker. | Cannot map open-vocabulary natural language ("Mrs. Brisby cautiously approaches the glowing stone") to geometry, nor synthesize non-rigid cel/photorealistic occluded details. |
| **2. `CliffordNet`** (Ji 2026, `arXiv:2601.06793v2`) | Shifted Clifford Geometric Product ($D^{(s)} \oplus W^{(s)}$) between state $Z_{\text{det}}$ and factorized Laplacian context $Z_{\text{ctx}} \approx \Delta X$, with Gated Geometric Residual (GGR) and **zero FFN**. | Strict linear $O(N_{\text{pixels}} \cdot D \cdot |S|)$ with $|S| \in \{2, 5\}$. | Preserves native 2D grid topology; unifies anisotropic diffusion ($D^{(s)}$) and edge vorticity ($W^{(s)}$); beats ResNet/ViT with $4\text{–}8\times$ fewer parameters. | Operates on 2D feature channels via cyclic index shifts; lacks explicit 3D $E(3)$ / $\text{Pin}(3,0,1)$ rigid-motion group equivariance. |
| **3. `GATr`** (Brehmer et al. 2023, `arXiv:2305.18415v3`) | 16D $\mathbb{G}_{3,0,1}$ multivector tokens + $\text{Pin}(3,0,1)$-equivariant linear maps + Geometric Bilinear ($xy \oplus \text{EquiJoin}$) + distance-aware dot-product attention. | $O(K^2 \cdot D_{\text{MV}})$ for $K$ tokens. | Exact 3D $E(3)$ translation/rotation/reflection equivariance; computes exact point-to-plane/line distances via the Join; $10\text{–}100\times$ sample efficiency on dynamics. | $O(K^2)$ pairwise attention explodes in memory and compute if applied to pixel/patch grids ($K \sim 10^5$). |

### 3.2 The Three-Stratum Synthesis

Rather than forcing one tool to do all three jobs, `ImplicateModel` assigns each operator to the exact mathematical stratum where its inductive bias is optimal:

#### Stratum 1: Pixel-Grid Feature Lifting & `@rolpa` Surface Refinement $\implies$ `CliffordNet`
Whenever the engine operates on dense 2D image grids ($H \times W$ pixels or feature maps, $N \sim 10^5\text{–}10^6$), we use a **No-FFN `CliffordNet` Backbone**:
* **Why the math fits our ontology:** In `CliffordNet` (Eq. 4–5 of `reference_cliffordnet.pdf`), the local context field in Differential Mode ($\lambda = 1$) is the factorized discrete Laplacian:
  $$\mathcal{C}_{\text{loc}}(H) = \text{DWConv}_{3\times 3}\!\left(\text{DWConv}_{3\times 3}(H)\right) - H \approx \nabla^2 H$$
  Notice that $\mathcal{C}_{\text{loc}}(H)$ is the exact numerator of Bohm and Hiley's Quantum Potential $Q = -\nabla^2 R / R$. When `CliffordNet` computes the shifted inner and wedge products for shift offsets $s \in \mathcal{S} = \{1, 2, 4, 8, 16\}$:
  $$D^{(s)}(H, \mathcal{C}) = \text{SiLU}\!\left(H \odot \mathcal{T}_s(\mathcal{C})\right), \qquad W^{(s)}(H, \mathcal{C}) = H \odot \mathcal{T}_s(\mathcal{C}) - \mathcal{C} \odot \mathcal{T}_s(H)$$
  the scalar branch $D^{(s)}$ measures local phase coherence ($H \cdot \nabla^2 H$), while the bivector branch $W^{(s)}$ measures structural orthogonality and boundary curl ($H \wedge \nabla^2 H$).
* **Where it runs:**
  1. During **Enfoldment (`implicate-lifter`)**: Extracts boundary contours, sub-holon part segmentation masks, and dense optical flow bivectors in $O(N)$ time.
  2. During **Unfoldment (`implicate-rolpa`)**: Acts as a lightweight (2.6M–4.8M parameter) **1-Pass Local Refiner** that takes the composited CFT/Gaussian wavepackets and lattice gradients $(\nabla p, \nabla Q, C, \nabla L)$ and synthesizes crisp 1080p textures in a single forward pass ($< 4\text{ ms}$ on RTX 4090).

#### Stratum 2: Semantic-to-Geometric Verb & Relation Binding $\implies$ $\text{Pin}(3,0,1)$-Equivariant `GATr`
Whenever the engine reasons over **discrete Entities / Holons** and their semantic relationships ($K \sim 4\text{–}64$ holons per scene), we use a compact **`GATr` Relational Operator** (~1.5M–3.5M parameters, 6–10 blocks, 16 multivector channels + 64 auxiliary scalar channels):
* **Why `GATr` is required here:** While `HolonEngine` solved 2D arcade collisions (`StompTop`, `HitBlockBottom`) using closed-form PGA incidence, open-domain video generation (*The Secret of NIMH*) requires learning complex, continuous multi-body verbs from `Qwen-VL` labels—such as `"swoops_under"`, `"hands_amulet_to"`, `"Circles_predator"`, or `"backs_away_from"`.
* **How `GATr` embeds our Holons:**
  For each active holon or sub-holon $i \in \{1, \dots, K\}$:
  - Its world position $P_i \in \bigwedge^3 \mathbb{G}_{3,0,1}$ is embedded as a **trivector**,
  - Its current velocity / motor generator $B_i \in \bigwedge^2 \mathbb{G}_{3,0,1}$ is embedded as a **bivector**,
  - Its dominant boundary / support surface $\Pi_i \in \bigwedge^1 \mathbb{G}_{3,0,1}$ is embedded as a **grade-1 plane**,
  - Its semantic noun spinor $\psi_{\text{noun},i}$ and the active prompt verb spinor $\psi_{\text{verb}}$ are fed into `GATr`'s **auxiliary scalar channels** $q^s, k^s$.
* **Why the Projective Join is decisive:** As proven in Proposition 7 and Lemma 7 of `reference_cliffordgatr.pdf`, the geometric product $xy$ alone cannot extract Euclidean distance from null vectors ($e_0$) in $\mathbb{G}_{3,0,1}$. By combining $xy$ with the equivariant join $\text{EquiJoin}(x, y; z) = z_{0123}(x^* \wedge y^*)^*$ and distance-aware dot-product attention:
  $$\phi(q) \cdot \psi(k) = -\omega(q_{\setminus 0})\omega(k_{\setminus 0})\|k_{\setminus 0}\vec{q} - q_{\setminus 0}\vec{k}\|_{\mathbb{R}^3}^2$$
  `GATr` computes exact, $E(3)$-equivariant relative vectors, collision planes, and steering motors between any pair of holons. Because $K \le 64$, `GATr` attention over $K^2 \le 4096$ pairs executes in $< 0.3\text{ ms}$.

#### Stratum 3: Continuous Spatiotemporal Evolution $\implies$ Closed-Form `Dzog` + `Holon` Lattice
Time stepping never passes through a neural network. Once `GATr` emits the verb's active-information beacon parameters ($\mathcal{Q}_{\text{Cl}}$ attractors/repellers) and motor impulses ($\Delta B_i$), the closed-form GPU lattice (`implicate-field`) integrates the wave equations and holon kinematics deterministically.

---

## 4. Subsystem I: Multi-Shot Video Enfoldment & The Clifford Lifter (`implicate-lifter`)

To compile both 30-second gameplay videos (`raptor.mp4`, `smb3_w1_1_320x240.mp4`, `pacman_240x308.mp4`) and an 82-minute feature film (*The Secret of NIMH*) through the same pipeline, `implicate-lifter` executes four streaming stages:

### 4.1 Stage 0: Streaming Shot Segmentation & Camera Regime Discovery (`VERIFIED` in Milestone R2.1)
A single streaming pass over the `128x64` (`ANALYSIS_W x ANALYSIS_H`) `fast_bilinear` pipe evaluates the 2D CFT cross-power spectrum $C_{t,t+1}(u, v) = \mathcal{F}_{\text{Cl}}[\Psi_{t+1}]\mathcal{F}_{\text{Cl}}^*[\Psi_t]$ and frame-to-frame 64-bin luminance-chroma histogram divergence $D_{\text{KL}}(H_t \parallel H_{t+1})$:
1. **Hard Cut, Flash Rejection, & Linear-Convex Dissolve Partitioning:** Candidate hard cuts require local-maximum cut score, sustained post-cut divergence ($t-2$ vs. $t+2$, rejecting 1–2 frame lightning/explosion flashes), and post-cut internal stability. Gradual cross-dissolves are detected over a $2w$-frame window ($w = 5$) by verifying linear convex combination $I_t \approx (1-\alpha)I_{t-w} + \alpha I_{t+w}$ with $\alpha \in [0.15, 0.85]$ and bounded step L1 jump.
   * **Verified Benchmark (`R2.1`):** Achieves **`1,356–1,392 FPS`** full streaming segmentation on a 10-minute 1080p slice of *The Secret of NIMH* (`14,386 frames` in `10.3–10.6s`, partitioning `153 shots`: `43 Ortho2DStatic`, `65 Ortho2DScroll`, `40 MultiPlane25D`, `5 Perspective3D`), and **`Precision = 0.9667`**, **`Recall = 1.0000`**, **`F1 = 0.9831`** on a 30-segment ground-truth cut/dissolve/flash reel.
2. **Stationary-Origin HUD Exclusion & Per-Active-Frame Strip Shear:** When the primary CFT peak translates ($\|\mathbf{d}_1\| > 1$), secondary peaks at the stationary origin $\mathbf{d}_2 = (0, 0)$ are recognized as camera-locked HUD/bezel overlays rather than moving parallax planes, and strip velocities are normalized per active signal frame so sparse sky bands never create false depth shear.
3. **Camera Motor Classification ($M_{\text{cam}}(t) \in \text{Spin}(3,0,1)$):**
   From the CFT flow field $\mathbf{v}_{\text{flow}}(x, y, t)$, the lifter classifies camera ego-motion into `Ortho2DStatic` (*Pac-Man*), `Ortho2DScroll` (*SMB3*, *Raptor*), `MultiPlane25D` (*The Secret of NIMH* multi-plane pans), and `Perspective3D`.

### 4.2 Stage 1: Multi-Plane Depth-Stratified Panorama & Camera-Locked HUD Holon Reconstruction (`VERIFIED` in Milestone R2.2)
Single-layer temporal mode accumulation works for flat 2D games (`HolonEngine`), but blurs multi-plane animated scenes or scenes with camera-locked HUDs unless reference frames are separated algebraically:
* **Dirac-Spinor CFT Odometry (`ScrollEstimator`):** Each viewport band is lifted into a Hann-windowed 2D Clifford spinor field combining zero-mean luminance with the Dirac spatial gradient spinor:
  $$\Psi(x, y) = w_x(x)w_y(y)\Big[\big(0.35(L - \bar{L}) + \partial_x L\big) + \partial_y L\,\mathbf{e}_{12}\Big] \in \mathbb{G}_{2,0}^+$$
  Eliminating zero-padding boundary cliffs and uniform-sky DC bias.
* **Camera-Locked HUD Holon Separation (`camera_locked_hud`):** Following the principle that *nothing is a special case*, any border band whose CFT trajectory remains stationary in screen space ($\max_t \|\mathbf{X}_b(t)\| \le 2.5\text{ px}$ with low static registration error) while the interior world translates ($\max_t \|\mathbf{X}_{\text{world}}(t)\| \ge 8\text{ px}$) is promoted to a **Camera-Locked HUD Holon Layer** ($M_{\text{hud}}(t) = M_{\text{cam}}(t)$, $1/z \to \infty$), preserving HUDs for generative unfoldment without corrupting world odometry.
* **Minimum-Registration-Error Reference Selection & Pass-2 Dilated Inpainting:** Among world bands, the reference trajectory is selected by minimizing inter-frame L1 registration residual across the active world viewport; a band splits into a separate parallax plane only if it has sufficient gradient energy ($\|\nabla L\| \ge 8.0$) and reduces local registration error by $>20\%$.
  * **Verified Benchmark (`R2.2`):** Reconstructs held-out backgrounds at **`32.44 dB`** (*SMB3*, $L=1$ + HUD holon, `0.0000` ghosting), **`33.67 dB`** (*Raptor*, $L=1$, `<0.005` ghosting), and **`40.32 dB` / `34.82 dB` / `39.98 dB`** across 3 contiguous shots of *The Secret of NIMH* (`0.0000` ghosting).

### 4.3 Stage 2: Articulated Holon Extraction & 5-Mode Clifford Fourier Wavepackets (`VERIFIED` in Milestones R2.3 & R7.1)
To support both 2D arcade games and hand-inked animated films (*The Secret of NIMH*) without assuming discrete game sprite sheets:
1. **Spinor-Coherent Multi-Body Tracking (`extract_character_observation_tracks`):** Links foreground residual boxes across frames by joint centroid proximity and chromatic spinor coherence, extracting up to $M$ distinct moving `ArticulatedHolon` tracks per shot so multi-actor pairwise PGA interactions ($M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$) can be assimilated directly from raw video.
2. **2D CFT Spinor Posture Descriptors (`compute_cft_posture_descriptor`):** Each character observation in canonical $(u, v) \in [-1, 1]^2$ coordinates is transformed via a low-frequency 2D Clifford Fourier Transform ($k_u, k_v \in \{-1..2\}$) into a 64D posture spinor descriptor, clustering observations into discrete view-aspect idempotent cells $\epsilon_{s,\text{view}}$ ($S_{\text{view}} \in \{1..8\}$) and extracting the intrinsic continuous deformation phase $\phi_t \in [0, 2\pi)$. During unfoldment, `select_view_idempotent_observation_index` queries the observation pose minimizing geodesic rotor and CFT gait-phase distance on $\text{Spin}(2) \times S^1$.
3. **5-Mode Clifford Fourier Series Wavepackets (`WavepacketPartKernel`):**
   Within each view-aspect idempotent cell $\epsilon_{s,\text{view}}$ and sub-holon part $p \in \{1..P\}$ (segmented via `CliffordNet` Laplacian wedge-vorticity $D^{(s)} \oplus W^{(s)}$), the canonical RGBA field is projected onto the 5-mode Clifford Fourier basis $\mathbf{b}(\phi) = [1, \cos\phi, \sin\phi, \cos 2\phi, \sin 2\phi]$ via regularized $5 \times 5$ normal equations $(\mathbf{B}^\top \mathbf{B} + \lambda \mathbf{I}_5)\mathbf{w}_{k,c} = \mathbf{B}^\top \mathbf{y}_{k,c}$.
   * **Verified Benchmark (`R2.3`):** Tier-1 characters automatically degenerate to $P=1$ with **`IoU = 0.9997 / LPIPS = 0.0002`** (*Pac-Man*), **`IoU = 0.9766 / LPIPS = 0.0087`** (*SMB3*), and **`IoU = 0.9685 / LPIPS = 0.0119`** (*Raptor*). Tier-2 articulated characters decompose into multi-part kinematic trees with **`IoU = 0.9748 / LPIPS = 0.0057`** (Mrs. Brisby cape turn, $P=4$, vs. single-box baseline `IoU = 0.8732 / LPIPS = 0.0354`) and **`IoU = 0.9875 / LPIPS = 0.0024`** (Jeremy the Crow wing flap, $P=6$, vs. single-box baseline `IoU = 0.7552 / LPIPS = 0.0602`).

### 4.4 Stage 3: Empirical Incidence & Acoustic Envelope Extraction
* **PGA Boundary & Incidence Detection:** Static obstacle contours in each plane $l$ are compiled into `BoundaryBlade`s ($B_k \in \mathbb{G}_{3,0,1}$) with directional permeability `[left, right, top, bottom]`. Frame-exact trajectory discontinuities ($\|\Delta \mathbf{P}_{B,i}\| > \tau_{\text{imp}}$ coinciding with projective contact $P_i \wedge B_j = 0$) record empirical state transition operators $T_{s \to s'} = \epsilon_{s'} M \epsilon_s$.
* **Acoustic Event Binding:** The audio stream is decomposed via Short-Time Continuous Fourier Transform (STFT/CFT) into a transient impulse stream (onsets correlated with visual $\Delta \mathbf{P}_B$ collisions), a harmonic/modal stream (correlated with continuous motion speed $\|\mathbf{P}_B\|$), and an ambient/vocal stream. Each collision event binds the struck object's inferred acoustic wave speed $c = f_0 / 440 = \sqrt{K/\rho}$ to its material constitutive record.

---

## 5. Subsystem II: Semantic Harvester & The Implicate Lexicon (`implicate-lexicon`)

### 5.1 Local VLM Transduction via `llama.cpp` (`Qwen2.5-VL-7B-Instruct` @ `Q4_K_M`)

* **Model & Quantization Spec (`VERIFIED` in Milestone R1.1–R1.3):** `Qwen2.5-VL-7B-Instruct` at **`Q4_K_M`** quantization (`/mnt/storage/models/llama.cpp/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + multimodal vision projector `/mnt/storage/models/llama.cpp/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf`, invoked via `llama-cli --grammar-file` on `CUDA0` RTX 4090 at `30.5–48.0 ms/frame` vision encode, `785–1,064 tok/s` prompt eval, and `33–36 tok/s` generation). Implemented in `crates/implicate-lexicon`.
* **Two-Pass Semantic Harvesting Protocol:**
  1. **Pass A — Temporal Action & Entity Discovery Prompt (Clip-Level):**
     Given a sampled shot clip or timestamped frame sequence, `Qwen2.5-VL-7B-Instruct` (`Q4_K_M`) is queried with a temporal action-grounding prompt (adapted from standard temporal grounding to open-domain entities and verbs):
     > *"Analyze this video clip and identify all instances of entity actions and interactions (e.g., running, jumping, flying, colliding, casting light, or fleeing). For each instance, detect the action and output the start and end timestamps in the format `[HH:MM:SS.mmm - HH:MM:SS.mmm]`, the canonical verb, and a brief visual description of the subject and target (e.g., color, posture, material, and location in the frame)."*
  2. **Pass B — GBNF-Constrained Holon Binding (Shot Contact Sheet):**
     Each segmented shot $S_m$ is paired with numbered bounding contours drawn around the CFT-tracked Holons $\mathcal{H}_1 \dots \mathcal{H}_K$ and passed to `llama-mtmd-cli` (`--temp 0 --seed 42`) with a strict GBNF grammar (`implicate_lexicon.gbnf`) to bind the discovered temporal actions directly to track IDs:

```json
{
  "shot_id": 42,
  "milieu": {
    "description": "dark hollow inside cinder block, glowing magical blue-gold light",
    "viscosity": 0.12,
    "luminescence_decay": 0.85
  },
  "holons": [
    {
      "track_id": 1,
      "canonical_noun": "Mrs_Brisby",
      "visual_description": "small brown mouse in red cloak, lower-left frame",
      "category": "character_biped_mouse",
      "material": { "cohesion": 0.95, "stiffness": 1.2, "has_compliant_cape": true }
    },
    {
      "track_id": 2,
      "canonical_noun": "Red_Amulet",
      "visual_description": "glowing red-gold medallion suspended at center",
      "category": "artifact_rigid_luminous",
      "material": { "cohesion": 1.0, "stiffness": 2.4, "luminescence_source": 3.2 }
    }
  ],
  "verbs": [
    {
      "timestamp_window": "[00:02:14.200 - 00:02:15.700]",
      "subject_track": 1,
      "verb": "reaches_toward",
      "object_track": 2,
      "frame_start": 12,
      "frame_end": 48
    }
  ]
}
```
All VLM outputs are cached by SHA-256 content hash under `.implicate_cache/` so repeated enfoldment runs are deterministic and zero-latency.

### 5.2 Wording-Space Spinor Projection & Dynamic Open-Ended `ConceptLexicon` (`VERIFIED` in Milestone R1.3)
Every noun, verb, emotion, archetype, and milieu clause is embedded into a dynamic concept vector $\mathbf{e}_m \in \mathbb{R}^{D_{\text{dyn}}}$ via `ConceptLexicon` (`crates/implicate-lexicon/src/schema.rs`):
* **Seeded Ontological Strata (`ConceptKind`):** Includes `KinematicAction`, `PhysicalEntity` (including camera-locked `entity_hud_interface`), `AffectiveEmotion` (`fear_dread`, `courage_resolve`, `grief_sorrow`, `joy_wonder`, `rage_menace`, `awe_numinous`), and `JungianArchetype` (`self_talisman`, `shadow_beast`, `hero_protector`, `sage_hierophant`, `trickster_fool`, `threshold_guardian`, `anima_guide`, `great_mother_sanctuary`).
* **Open-Ended Online Concept Assimilation (`learn_from_shot`):** Because the thin symmetric Jacobi SVD of the normalized $N \times N$ Gram matrix $G_{mn} = \hat{\mathbf{e}}_m^\top \hat{\mathbf{e}}_n$ depends only on the number of clauses $N$ rather than the feature dimension $D_{\text{dyn}}$, novel objects, visual appearances, and action verbs discovered by `Qwen2.5-VL` dynamically append new `LearnedVisualConcept` clusters without bound while projecting cleanly into fixed 16-channel even spinors in $(\mathbb{G}_{2,0}^+)^{16}$:
$$\mathbf{U}_K \boldsymbol{\Sigma}_K \mathbf{U}_K^\top \approx \mathbf{G}, \qquad \psi_{m,k} = r_{m,k}\cos\theta_{m,k} + r_{m,k}\sin\theta_{m,k}\,e_{12}, \quad k \in \{1, \dots, 16\}$$
This guarantees that semantic similarity between any prompt word and any stored Lexicon entry reduces to the fast bilinear spinor inner product $\langle \psi_{\text{query}} \widetilde{\psi}_{\text{entry}} \rangle_0$.

### 5.3 Learning Verbs as Scale-Invariant $\mathcal{Q}_{\text{Cl}}$ Potentials & `GATr` Operators (`VERIFIED` in Milestones R3.1 & R3.2)

How does `ImplicateModel` learn the physical law of a verb (e.g., `"orbit"`, `"jump_over"`, `"patrol"`, `"pursue"`, `"flee_from"`) from 1–20 labeled video examples?

1. **Canonical Relative Frame Normalization (`CanonicalRelFrame`):**
   For every clip segment $[t_0, t_1]$ labeled with tuple $(\text{Subject } A, \text{Verb } v, \text{Object } B)$, we transform $A$'s trajectory into $B$'s instantaneous PGA rest frame aligned with the initial join line $P_A(t_0) \vee P_B(t_0)$:
   $$M_{\text{rel}}(t) = \widetilde{M}_B(t)\, M_A(t) \in \text{Spin}(3,0,1)$$
   and normalize spatial distance by the characteristic blade span $L_{AB} = \|P_A(t_0) \vee P_B(t_0)\|$.
2. **Inverse Bohm-Hiley Potential Extraction (`LearnedVerbPotential`, Verified in `R3.1`):**
   From the ensemble of normalized trajectories $\{M_{\text{rel}}^{(n)}(t)\}_{n=1}^{N_v}$ for verb $v$, we solve via 11-mode Chebyshev-Fourier ridge regression for the canonical amplitude envelope $R_v(\mathbf{x}_{\text{rel}}, \tau)$ and rotor field $\mathcal{U}_v(\mathbf{x}_{\text{rel}}, \tau)$ whose Clifford Quantum Potential:
   $$\mathcal{Q}_{\text{Cl}, v}(\mathbf{x}_{\text{rel}}, \tau) = -(\nabla^2 \Psi_v)\Psi_v^{-1} = -\frac{\nabla^2 R_v}{R_v} - \left(2(\nabla \ln R_v)\mathcal{U}_v\nabla\widetilde{\mathcal{U}}_v + (\nabla^2 \mathcal{U}_v)\widetilde{\mathcal{U}}_v\right)$$
   yields the exact Bohmian guidance force $-\nabla \langle \mathcal{Q}_{\text{Cl}, v} \rangle_0 = \ddot{\boldsymbol{\mu}}_v(\tau) - \frac{1}{2\sigma_v^4}(\mathbf{x}_{\text{rel}} - \boldsymbol{\mu}_v(\tau))$ and bivector spin-torque $\Omega_{Q,v} = \dot{\bar{\theta}}_v(\tau)$.
   * **Verified Benchmark (`R3.1`):** Reproduces `"patrol"` (`0.58%`), `"jump_over"` (`0.24%`), `"pursue"` (`0.30%`), `"flee_from"` (`0.27%`), and `"orbit"` (`0.69%`) across $2\times$ scale changes and $360^\circ$ rotations (`< 8%` Fréchet gate).
3. **`GATr` Multi-Body Conditioning (`GatrVerbBinder`, Verified in `R3.2`):**
   To handle scenes where multiple verbs and obstacles interact simultaneously, the 6-block $\text{Pin}(3,0,1)$-equivariant `GATr` Relational Operator (`16` $\mathbb{G}_{3,0,1}$ multivector channels + `64` scalar channels) takes the scene's active Holon multivectors $\{X_i \in \mathbb{G}_{3,0,1}\}_{i=1}^K$ and verb spinors $\{\psi_{\text{verb}}\}$ and predicts the per-holon motor impulses $\Delta B_i$ and $\mathcal{Q}_{\text{Cl}}$ beacon parameters.
   * **Verified Benchmark (`R3.2`):** Executes in **`0.0958 ms`** for $K = 32$ scene Holons (`< 0.5 ms` gate) and achieves **`0.00%–0.02%`** out-of-domain $E(3)$-transformed trajectory error at $N_{\text{train}} \in \{1, 5, 20\}$ (vs. `104%–156%` for a non-equivariant MLP).

---

## 6. Subsystem III: The Implicate Runtime & Causal Unfoldment (`implicate-unfold` + `implicate-field`)

At generation time, there is **no diffusion noise schedule** and **no $O(N^2)$ spacetime attention**. Generation is a single causal wave simulation coupled to articulated holon kinematics:

### 6.1 Lattice Channel Layout (`implicate-field`)
Evolving `dzog-wgpu`, the active medium is discretized onto a staggered Yee GPU lattice (`RGBA32Float` ping-pong textures, e.g., $480 \times 270$ field grid driving $1920 \times 1080$ `@rolpa` output):

| Texture Buffer | Channels (`.rgba`) | Mathematical Role in Unfoldment |
| :--- | :--- | :--- |
| **`State`** | `(p, vx_east, vy_north, Q_scalar)` | Compressible wave pressure $p$, staggered face velocities $\mathbf{v}$, and scalar Bohm-Hiley potential $\langle \mathcal{Q}_{\text{Cl}} \rangle_0 = -\nabla^2 R / R$. |
| **`SpinTorque`** | `(omega_xy, omega_yz, omega_zx, R_env)` | Grade-2 bivector spin-torque components $\langle \mathcal{Q}_{\text{Cl}} \rangle_2$ and smoothed wave envelope $R$. |
| **`Material`** | `(rho, K, gamma, cohesion_C)` | Local density $\rho$, bulk stiffness $K$, exponential damping $\gamma$, and topological Griffith cohesion $C \in [0, 1]$. |
| **`Luminescence`** | `(L_r, L_g, L_b, smoke_density)` | Chromatic reaction-diffusion exciton glow field $\mathbf{L}(\mathbf{x}, t)$ for dynamic 2.5D/3D lighting, magic auras, and condensing smoke. |

### 6.2 Real-Time Performance Contract ($\ge 24\text{ FPS}$ Target, Rust + Heavy GPU Compute)

**Hard Engineering Requirement:** The inference/unfoldment engine (`implicate unfold`) must achieve **real-time video generation at $\ge 24\text{ FPS}$ (targeting $60+\text{ FPS}$ at 1080p)** on the local NVIDIA RTX 4090 Laptop GPU. To guarantee this frame budget ($\le 41.6\text{ ms/frame}$ for $24\text{ FPS}$, $\le 16.6\text{ ms/frame}$ for $60\text{ FPS}$):
1. **Pure Rust + Native GPU Compute Architecture:** Zero Python in the engine runtime. All runtime simulation, holon kinematics, `GATr` verb conditioning, and `@rolpa` wavepacket/CliffordNet projection execute in **Rust (`Edition 2024`)** backed by **`wgpu` (Vulkan/SPIR-V) / CUDA compute shaders**.
2. **Zero-Readback GPU Spine:** The wave-matter lattice (`State`, `SpinTorque`, `Material`, `Luminescence`), articulated holon motor buffers (`StorageBuffer`), multi-plane atlas textures, and `@rolpa` output framebuffer stay resident in VRAM across the entire generation run. Only compact telemetry/audio ring-buffer frames ($< 4\text{ KB/frame}$) cross PCIe.
3. **Per-Frame Compute Budget ($\le 16.6\text{ ms}$ for $60\text{ FPS}$, $\le 41.6\text{ ms}$ for $24\text{ FPS}$):**
   * `GATr` Relational Verb Conditioning ($K \le 64$ holons): **$\le 0.4\text{ ms}$**
   * Staggered Yee FDTD Wave-Matter Substeps + $\mathcal{Q}_{\text{Cl}}$ + Holon Reduction (`wgpu` compute): **$\le 1.8\text{ ms}$**
   * `@rolpa` Multi-Plane Atlas + Oriented Gaussian-Gabor Wavepacket Splatting (`wgpu` compute/render): **$\le 2.5\text{ ms}$**
   * `@rolpa` 1-Pass `CliffordNet` Local Surface Refiner (fused WGSL/CUDA kernel): **$\le 4.5\text{ ms}$**
   * Total GPU Frame Time: **$\approx 9.2\text{ ms/frame}$ ($\sim 108\text{ FPS}$, $> 4\times$ faster than the $24\text{ FPS}$ floor)**.

### 6.3 Single-Pass Causal Step Loop (Per Frame $\Delta t$)
1. **Verb Beacon Injection ($1\text{–}10\text{ Hz}$ or on state transition):**
   `GATr` evaluates the active scene graph and stamps standing-wave carrier sources $s_{\text{verb}}(\mathbf{x}, t)$ and boundary impedances into the lattice.
2. **Yee-Staggered Wave-Matter Substeps (`dzog-wgpu` compute shaders):**
   * Updates face velocities $\mathbf{v}$ and evaluates $\mathcal{Q}_{\text{Cl}} = -(\nabla^2 \Psi)\Psi^{-1}$.
   * Updates cell pressure $p$ with the bounded rational de Broglie soliton law:
     $$f_{\text{nl}}(p) = \frac{\beta\,p}{1 + \alpha\,p^2} - \kappa\,p, \qquad K_{\text{eff}}(\mathbf{x}, p) = K(\mathbf{x})\max\!\left(0.35,\; 1 - \frac{\chi_{\text{lens}}\,p^2}{1 + \alpha\,p^2}\right)$$
   * Erodes cohesion $C$ where overpressure exceeds yield ($|p| > p_{\text{yield}}$) and diffuses chromatic luminescence $\mathbf{L}(\mathbf{x}, t)$.
3. **Articulated Holon Integration (`implicate-math`):**
   * Zero-readback GPU reduction (`reduce_holon.wgsl`) integrates net radiation pressure $-\nabla p$, Bohmian guidance force $-\nabla \langle \mathcal{Q}_{\text{Cl}} \rangle_0$, and bivector torque $\Omega_Q$ over each holon's bounding blade.
   * Advances root motors $M_{i,\text{root}}(t)$ and articulated child joint angles $\theta_{i,p}(t)$, coupling compliant sub-holons (capes, hair, foliage) directly to local lattice flow $\mathbf{v}(\mathbf{x}, t)$.
   * Resolves PGA boundary incidences ($P_i \wedge B_k = 0$) and executes discrete idempotent transitions $T_{s \to s'} = \epsilon_{s'} M \epsilon_s$.

---

## 7. Subsystem IV: The `@rolpa` Shadow-Manifold Projector (`implicate-rolpa`)

To render 1080p video at 60+ FPS with zero temporal flickering alongside synchronized stereo audio, `@rolpa` projects the implicate state in two coupled passes:

### 7.1 Visual Lens: Deterministic Wavepacket Compositing + 1-Pass `CliffordNet` Shader
1. **Multi-Plane & Wavepacket Splatting (Deterministic Base Frame $I_{\text{base}}$):**
   * Warps the depth-stratified background atlases $A_1 \dots A_L$ by the active camera motor $M_{\text{cam}}(t)$ with depth parallax $1/z_l$.
   * Evaluates each sub-holon's CFT/Gaussian wavepacket $\mathcal{W}_{i,p}(\mathbf{x}, t)$ (Eq. in §2.3) in depth order, applying PGA half-space occlusion clipping against opaque boundary blades.
   * Computes per-pixel 2.5D/3D normal lighting from the lattice's chromatic luminescence gradient $\nabla \mathbf{L}(\mathbf{x}, t)$ and pressure shockwave refraction $\nabla p(\mathbf{x}, t)$.
2. **1-Pass `CliffordNet` Surface Refiner (`gFFN-H`):**
   * On pure pixel-art benchmarks (`pacman`, `smb3`, `raptor`), $I_{\text{base}}$ is already pixel-exact and requires zero neural post-processing (`refiner_mode = Bypass`, running at $> 1,500\text{ Hz}$).
   * On continuous/cinematic benchmarks (*The Secret of NIMH*), articulated joint seams, extreme foreshortening, and fluid wakes benefit from local contour refinement. A 4-block isotropic **`CliffordNet-Lite` Refiner** ($2.6\text{M}$ params, shift set $\mathcal{S} = \{1, 2, 4, 8, 16\}$) takes the 12-channel input tensor:
     $$X_{\text{in}} = \operatorname{Cat}\!\left(I_{\text{base}},\; \nabla p,\; \nabla \langle \mathcal{Q}_{\text{Cl}} \rangle_0,\; C,\; \mathbf{L}\right) \in \mathbb{R}^{H \times W \times 12}$$
     and predicts a residual detail correction $\Delta I = \text{CliffordNet}(X_{\text{in}})$ in **one deterministic forward pass**.
   * **Why it never flickers (`VERIFIED` in Milestone R3.3):** Because `CliffordNet` receives no random Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ and its inputs $(I_{\text{base}}, p, \mathcal{Q}_{\text{Cl}}, C, \mathbf{L})$ evolve continuously via hyperbolic PDEs and $\text{Spin}(3,0,1)$ motors, the mapping is Lipschitz-continuous in time. Implemented as a fused $18 \times 18$ workgroup-tiled WGSL compute shader (`crates/implicate-rolpa/src/cliffordnet_rolpa.wgsl`), a full $1920 \times 1080$ 4-block pass executes in **`0.4595 ms/frame`** (`2,176.4 FPS`) on the RTX 4090 (`Vulkan`), achieving **`+5.37 dB`** mean PSNR gain, **`33.8%`** LPIPS reduction, and temporal warping error **$E_{\text{warp}} = 0.000011$** (`< 0.002` gate) across 3 test shots of *The Secret of NIMH*.

### 7.2 Acoustic Lens: Inseparable Stereo Modal & Harmonic Synthesis (`VERIFIED` in Milestone R4.2)
Virtual binaural pickups attached to the active camera pose $M_{\text{cam}}(t)$ sample port and starboard lattice pressures $(p_L, p_R)$, Doppler closing velocities, and holon collision impulses $\Delta \mathbf{P}_B$ at $60\text{ Hz}$, driving a lock-free $48\text{ kHz}$ audio stream (`crates/implicate-rolpa/src/acoustic.rs`):
* **Collision Modal Resonators:** Impacts excite 4-partial damped modal banks tuned to the struck holon's constitutive wave speed $f_0 = 440\sqrt{K/\rho}$ (`0.86%–1.25%` frequency error across soft wood, brass, and crystal amulet materials) at the exact sub-frame sample index of contact (**`0.0000 ms`** audio-visual synchronization error, `< 1.0 ms` gate, **`0` buffer underruns**).
* **Continuous Wake Sonification:** High-velocity movement and luminescence flares modulate continuous aerodynamic/hum harmonics proportional to local wave pressure and Poynting flux $\mathbf{I} = p\mathbf{v}$.

---

## 8. Binary Container Specification (`.implicate` v1 & `ENFD` v1 Payload)

Compiled worlds and lexicons are stored in a compact, zero-copy-friendly binary container (`.implicate`, implemented in `crates/implicate-unfold/src/{container.rs, payload.rs}`):

```text
+-----------------------------------------------------------------------+
| Magic: "IMPL" (4 bytes) | Version: u32 (1) | Flags: u32 (0x01 | 0x02) |
+-----------------------------------------------------------------------+
| Manifest JSON Length: u64 | UTF-8 JSON Manifest                       |
|   • ConceptLexicon (Noun/Verb/Emotion/Archetype Spinors ∈ (G_{2,0}^+)^16)
|   • VerbPotentialLexicon (11-mode Chebyshev-Fourier Q_Cl potentials)  |
|   • GatrVerbBinder Rules (Pin(3,0,1)-equivariant relational weights)  |
|   • EnfoldedClipSummary Index (clip_id, dimensions, shot table)       |
+-----------------------------------------------------------------------+
| Binary Payload Section ("ENFD" v1 Magic, 64-byte aligned):            |
|   1. Multi-Plane Atlas Layers (Depth-stratified RGB + bit-packed α)   |
|   2. Camera-Locked HUD Holon Layer (M_hud(t) = M_cam(t))              |
|   3. Luminescence Cel Sequence & Per-Frame Actor Bounding Boxes       |
|   4. Multi-Actor ArticulatedHolon Trees & 5-Mode CFT Kernels K̂_{p,m}  |
+-----------------------------------------------------------------------+
```

---

## 9. Cold Hard Truths: Engineering Risks, Failure Modes, & Uncertainty Ratings

To keep our engineering grounded, we explicitly separate what is **already proven** in `holonengine` / `dzoglang` from what carries genuine research risk when scaling to general video (*The Secret of NIMH*), along with calibrated **Uncertainty Ratings**:

### 1. Single-Shot 2D Orthographic Enfold/Unfold (`pacman`, `smb3`, `raptor`)
* **Uncertainty Rating:** **Low (5%)**
* **Why:** Already proven in `holonengine` at $1,500\text{–}3,000\text{ Hz}$. Porting these into `implicate-lifter` and `implicate-unfold` while adding `dzog-wgpu` field coupling and `Qwen-VL` semantic tagging is straightforward systems integration.

### 2. Multi-Shot Cut Detection & Multi-Plane Background Unwarping on *The Secret of NIMH*
* **Uncertainty Rating:** **Low–Medium (20%)**
* **Cold Hard Truth:** Traditional 2D animation (Don Bluth, 1982) uses physical multi-plane camera stands (typically 2 to 4 painted glass layers moving at constant depth ratios), which matches our `MultiPlane-2.5D` CFT layer model well. However, painted lighting overlays (animated glow cel layers, rain, fire) and camera zooms combined with pans can cause naive temporal-mode accumulation to ghost if foreground masks miss semi-transparent shadows.
* **Mitigation:** Separate additive/multiplicative luminescence cel layers ($L$) before computing the opaque RGB mode per parallax plane.

### 3. Unsupervised Articulated Sub-Holon Decomposition on Hand-Drawn Characters
* **Uncertainty Rating:** **Medium–High (45%)**
* **Cold Hard Truth:** This is the hardest computer vision problem in the whitepaper. In 2D pixel-art games (`smb3`), characters are rigid sprites with a 2-frame walk loop. In *The Secret of NIMH*, characters are hand-drawn on 1s and 2s (12–24 drawings/sec). When Mrs. Brisby turns her head $90^\circ$ or wraps her cape around her shoulders, topological contours appear and disappear non-isometrically—it is not a rigid 3D mesh projection.
* **Mitigation:** Do not force a single rigid 3D mesh onto hand-drawn 2D animation. Instead, index each sub-holon's CFT wavepacket bank by **discrete viewing/posture idempotents** $\epsilon_{s, \text{view}}$ (analogous to multi-aspect view cells), using `Spin(3,0,1)` motors for gross part translation/rotation/scale and the 1-pass `CliffordNet` `@rolpa` refiner to bridge non-rigid contour transitions.

### 4. Learning Generalizable Verb Potentials ($\mathcal{Q}_{\text{Cl}}$ + `GATr`) from Noisy `Qwen-VL` Labels
* **Uncertainty Rating:** **Medium (30%)**
* **Cold Hard Truth:** A local 7B VLM (`Qwen2.5-VL` via `llama.cpp`) is reliable at identifying *who* is in a frame and *coarse* action verbs (`"runs"`, `"flies"`, `"falls"`, `"glows"`), but struggles with fine sub-second temporal boundaries ($t_{\text{start}}, t_{\text{end}}$) and subtle physical causality.
* **Mitigation:** Never rely on `Qwen-VL` for temporal timestamps or trajectory math. Use **bottom-up empirical CFT/PGA kinematics** (velocity discontinuities $\Delta \mathbf{P}_B$, projective contacts $P \wedge B = 0$, and optical-flow kinetic energy peaks) to segment exact action boundaries first, and ask `Qwen-VL` only to label the pre-segmented event window.

### 5. Out-of-Distribution View Synthesis Without Multi-Step Diffusion
* **Uncertainty Rating:** **Medium (35%)**
* **Cold Hard Truth:** If a user prompts for a character or camera angle that was *never observed* in the enfolded video corpus (e.g., asking for a top-down $90^\circ$ overhead view of a character only ever seen from the side), pure CFT wavepacket projection + a 2.6M parameter 1-pass `CliffordNet` refiner will not hallucinate unseen 3D geometry the way a 30B diffusion model trained on 100M web videos does.
* **Mitigation:** Be explicit about the architectural boundary: `ImplicateModel` is an **Implicate World & Lexicon Engine** that learns the compositional laws, characters, materials, and verbs of an ingested video corpus (and can transfer verb laws across corpora via $\text{Pin}(3,0,1)$ equivariance), prioritizing 100% causal physics, zero temporal flicker, and $1000\times$ speed over unconstrained zero-shot texture hallucination.

---

## 10. Key Systems & Mathematical Insights from Empirical Implementation (`R1`–`R7`)

Across the 18 milestones and 36 verification gates spanning Tracks `R1`–`R7`, seven non-obvious systems and mathematical principles proved decisive:

### 10.1 Hann-Windowed Dirac-Spinor CFT & Algebraic HUD Holon Separation (`R2.1`–`R2.2`)
* **Failure Mode of Raw Luminance FFTs:** Standard phase correlation over raw luminance fails simultaneously on scrolling games with uniform sky bands (*SMB3*) and games with stationary HUD borders (*SMB3*, *Pac-Man*): zero-padded image boundaries create false stationary spectral peaks, while uniform sky strips yield degenerate zero-gradient phase.
* **Clifford Solution:** Lifting each viewport band into a Hann-windowed 2D Dirac-Spinor field $\Psi(x,y) = w_x(x)w_y(y)[0.35(L-\bar{L}) + \partial_x L + \partial_y L\,\mathbf{e}_{12}] \in \mathbb{G}_{2,0}^+$ eliminates boundary cliffs and DC sky bias. Furthermore, rather than cropping HUDs with brittle per-game heuristics, any border band whose CFT trajectory remains stationary in screen space while the interior world translates is algebraically promoted to a **Camera-Locked HUD Holon** ($M_{\text{hud}}(t) = M_{\text{cam}}(t)$, $1/z \to \infty$). During unfoldment, this same HUD holon locks with `0.0000 px` drift while its internal state idempotents (e.g., score counters) update dynamically from Bohmian work and collision events (`R6.1`).

### 10.2 Gram-Matrix Jacobi SVD for Open-Ended Online Concept Assimilation (`R1.3`, `R6.1`)
* **The Dynamic Vocabulary Problem:** When `Qwen2.5-VL` inspects a new film or game shot, it discovers previously unseen nouns, visual appearances, and action verbs, expanding the raw feature dimension $D_{\text{dyn}}$.
* **Clifford Solution:** By computing the thin symmetric Jacobi SVD on the $N \times N$ clause Gram matrix $G_{mn} = \hat{\mathbf{e}}_m^\top \hat{\mathbf{e}}_n$ rather than the $D_{\text{dyn}} \times D_{\text{dyn}}$ feature covariance, `ConceptLexicon::learn_from_shot` assimilates arbitrary new visual concepts online while projecting every concept into a fixed 16-channel wording-space spinor $\psi \in (\mathbb{G}_{2,0}^+)^{16}$. At unfold time, `VerbPotentialLexicon::resolve` matches open-vocabulary prompt verbs to learned $\mathcal{Q}_{\text{Cl},v}$ potentials via the coordinate-free bilinear scalar inner product $\langle \psi_{\text{query}} \widetilde{\psi}_{\text{entry}} \rangle_0$.

### 10.3 Projective-Join Orthonormal Triad in $\text{Pin}(3,0,1)$ `GATr` (`R3.2`)
* **Degeneracy of the Pure Geometric Product in PGA:** Because $e_0^2 = 0$ in $\mathbb{G}_{3,0,1}$, the geometric product $X_A X_B$ of two Euclidean point trivectors discards their translational offset vector.
* **Clifford Solution:** Constructing an explicit local orthonormal triad $[\mathbf{u}_{\text{rel}}, \mathbf{n}_{\text{up}}, \mathbf{u}_{\text{tang}}]$ from the normalized projective join $P_A \vee P_B = (X_A^* \wedge X_B^*)^*$ inside `GatrVerbBinder` guarantees that predicted relative target offsets and motor generators commute exactly with arbitrary 3D rigid motions $(R, \mathbf{t}) \in E(3)$, achieving `0.00%–0.02%` out-of-distribution trajectory error even from a single training demonstration ($N_{\text{train}} = 1$).

### 10.4 Symplectic Canonical-Relative Bohmian Guidance Decoupled from FDTD CFL Substeps (`R4.1`, `R4.3`)
* **Multi-Rate Stiffness:** The GPU wave-matter lattice requires Courant-Friedrichs-Lewy (CFL) substeps on a $256 \times 256$ staggered Yee grid, whereas learned verb potentials $\mathcal{Q}_{\text{Cl},v}(\mathbf{x}_{\text{rel}}, \tau)$ govern smooth macroscopic trajectories over normalized clip time $\tau \in [0, 1]$. Stepping holons in raw world pixels with first-order Euler integration causes cumulative phase drift when perturbed by wave pressure.
* **Clifford Solution:** Integrating each active verb binding in the target's canonical relative PGA frame (`CanonicalRelFrame`, normalized by $L_{AB} = \|P_A(t_0) \vee P_B(t_0)\|$) using symplectic **Velocity-Verlet** steps—while applying zero-readback GPU wave-lattice force/flow reductions (`GpuHolonReduction`) as bounded physical perturbations—preserves `< 0.7%` normalized Fréchet verb adherence alongside $100\%$ PGA boundary compliance (`P_i \wedge B_k = 0`) at `127–346+ FPS`.

### 10.5 Fused Workgroup-Tiled No-FFN `CliffordNet-Lite` WGSL Compute Kernel (`R3.3`)
* **Memory-Bandwidth Bottleneck:** Executing 4 blocks of $3 \times 3$ depthwise Laplacian convolutions and 5-shift geometric products ($D^{(s)} \oplus W^{(s)}$, $\mathcal{S} = \{1, 2, 4, 8, 16\}$) as separate GPU dispatches at $1920 \times 1080$ would require 24 full-frame VRAM read/write passes per frame.
* **Systems Solution:** Fusing all 4 `CliffordNet-Lite` blocks into a single $18 \times 18$ shared-memory (`var<workgroup>`) WGSL compute shader (`cliffordnet_rolpa.wgsl`, $16 \times 16$ interior + 1-pixel halo) keeps all intermediate multivector activations in fast on-chip SRAM. A full 1080p 4-block refinement pass executes in **`0.4595 ms/frame`** (`2,176.4 FPS`) on the RTX 4090 with **`+5.37 dB`** PSNR gain and zero temporal flicker ($E_{\text{warp}} = 0.000011$).

### 10.6 Sub-Sample PGA Incidence Timing for Zero-Drift Modal Audio (`R4.2`)
* **Frame-Rate Quantization Jitter:** Triggering collision audio only at integer video frame boundaries ($60\text{ Hz} \implies 16.67\text{ ms}$ quantization) produces audible synchronization jitter against visual contact.
* **Systems Solution:** When a PGA boundary incidence ($P_i \wedge B_k = 0$) produces a Bohmian momentum impulse $\Delta \mathbf{P}_B$, `AcousticRolpaProjector` computes the exact sub-frame sample index (`onset_delay_samples` within the 800-sample $48\text{ kHz}$ frame slice) and excites a 4-partial modal resonator bank tuned to the struck holon's constitutive wave speed $f_0 = 440\sqrt{K/\rho}$, achieving **`0.0000 ms`** audio-visual synchronization error and `0` buffer underruns.

### 10.7 Multi-Actor Pairwise PGA Assimilation & $\text{Spin}(2) \times S^1$ View-Idempotent Selection (`R7.1`)
* **Beyond Single-Actor Templates:** Extracting multiple persistent foreground tracks (`extract_character_observation_tracks`) and snapping verb windows to bottom-up Bohmian momentum discontinuities (`detect_cft_kinematic_impulses`) enables `assimilate_clip_into_manifest` to learn true two-body pairwise relative PGA potentials ($M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$, `enfolded_pair_<clip_id>`) directly from ingested video. During unfoldment, indexing each sub-holon's 5-mode CFT wavepacket bank by geodesic rotor and gait-phase coherence on $\text{Spin}(2) \times S^1$ (`select_view_idempotent_observation_index`) smoothly transitions across discrete posture idempotents $\epsilon_{s,\text{view}}$ without requiring a 3D mesh prior.
