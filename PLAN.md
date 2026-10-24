## PROPOSAL A: Track R9 — General-Purpose Semantic Generalization & Rigorous Clifford Abstraction

### Architectural Problem Statement
Video generation engines that rely on hardcoded string branching (`"mario"`, `"pacman"`, `"raptor"`, `"brisby"`), magic coordinate constants (`y=192.0`), or domain-specific heuristics are fundamentally brittle and fail the requirement of being a general-purpose implicate video engine. Track R9 purges all remaining content-specific code, replacing it with rigorous Clifford algebra, $\text{Spin}(16)$ semantic spinor inner products, physical constitutive classification, and resolution-invariant projective coordinates.

### Proposal Design Plan

```text
                  Prompt Text / DzogLang Spec
                              │
               ConceptLexicon::embed_text(...)
                              ▼
           Prompt Semantic Spinor ψ_prompt ∈ Spin(16)
           Constitutive Tuple (C, ρ, K, L)
           Normalized Coordinates x ∈ [0, 1]²
                              │
       ┌──────────────────────┴──────────────────────┐
       ▼                                             ▼
Spinor Inner Product                          Constitutive Type
⟨ψ_prompt, ψ_enfolded⟩                        Classification
       │                                             │
       ├─► Milieu Retrieval (clip selection)         ├─► C → 1.0 (Articulated Holon)
       └─► Entity Dispatch (character track)         ├─► C → 0.0, L > 0 (Field Soliton)
                                                     └─► M(t) = 1 (Camera-Locked HUD)
```

### Execution Tasks

- [ ] **R9.1 — $\text{Spin}(16)$ Semantic Inner-Product Dispatch (`implicate-unfold`)**
  - **Objective:** Attach `Spinor16` concepts to `ArticulatedHolon` during enfoldment. In `engine.rs`, replace string-based character matching in `match_holon_to_wavepacket` with maximal semantic spinor inner product $\langle \psi_{\text{prompt}}, \psi_{\text{asset}} \rangle$.
  - **Gate:** Dispatch character wavepackets purely through spinor projection without hardcoded character name checks.

- [ ] **R9.2 — Physical Constitutive Parameter Routing ($C, \rho, K, \mathbf{L}$) (`implicate-unfold`)**
  - **Objective:** Replace entity name string heuristics (`"drone"`, `"plasma"`, `"pellet"`) with constitutive classification: $C \ge 0.5$ routes to articulated matter wavepackets, $C < 0.5, L > 0$ routes to Yee wave-lattice luminescence.
  - **Gate:** Arbitrary novel entities with physical parameters route accurately to either field solitons or articulated bodies.

- [ ] **R9.3 — Resolution-Invariant Projective Canonical Coordinates $[0, 1]^2$ (`implicate-math`, `implicate-unfold`)**
  - **Objective:** Standardize scene definitions to continuous normalized coordinates $\mathbf{x} \in [0, 1]^2$ with projective scaling $S \in PGL(3)$ to viewport $(W, H)$.
  - **Gate:** Prompts define scenes in resolution-independent $[0, 1]^2$; arbitrary render dimensions $(W, H)$ scale correctly without magic pixel numbers.

- [ ] **R9.4 — Milieu Retrieval via Geodesic Distance on $\text{Spin}(16)$ (`implicate-unfold`)**
  - **Objective:** Remove `domain_sub` substring checks. Match scene prompts to enfolded clips using geodesic distance $\mathcal{D}(\psi_{\text{scene}}, \psi_{\text{clip}}) = \arccos |\langle \psi_{\text{scene}}, \psi_{\text{clip}} \rangle|$.
  - **Gate:** Automated clip selection by semantic spinor distance without scene ID parsing.

- [ ] **R9.5 — Zero-Heuristic Verification Gate Suite (`tests/r9_gates.rs`)**
  - **Objective:** Static code audit ensuring zero domain/character name strings exist in engine dispatch logic; novel scene unfoldment verifying generalized routing.
  - **Gate:** All tests pass with zero hardcoded character string checks in `crates/implicate-unfold/src/engine.rs`.

