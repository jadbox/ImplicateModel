# ImplicateModel — Research & Experiment Tasks

> **Status:** Active Step-by-Step Engineering & Verification.

**Goal:** Build the Implicate Video Engine (`ImplicateModel`)—an engine that enfolds arbitrary video inputs (from 2D arcade gameplay to feature-length hand-drawn cinema) into a compact Clifford world-and-verb lexicon, and unfolds new, controllable videos in a single causal wave pass at **real-time frame rates ($\ge 24\text{ FPS}$ minimum, targeting $60+\text{ FPS}$ via native Rust and heavy GPU compute acceleration)**.

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)` with `llama-cli` supporting `--video FILE`, `--video-fps N`, `--video-timestamp-interval 1000`, `--grammar-file`, `--repeat-penalty 1.15`, and `--single-turn`.
- **Benchmark Corpus Verified:**
  1. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4` (`240x308`)
  2. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4` (`256x240`) + `smb3_w1_1_320x240.mp4` (`320x240`)
  3. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4` (`320x240`)
  4. `/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4` (`1920x1040`, `23.976 fps`, `4965.0 s` / `119,041 frames`).


- **Core Architectural Directive:** "Just remember, don't assume anything about the source video- so don't rely on magic number constants for things like tile extraction"
- **Key Guidelines:**
  1. No hardcoded tile dimensions (e.g., assuming $16 \times 16$ tiles), coordinate assumptions (assuming NES-like $y=42$ floor or $y=24$ top HUD), or arbitrary row thresholds.
  2. Continuous Clifford geometry and projective algebra first: evaluate global camera transformation $M_{\text{cam}}(t) \in \text{Spin}(2,0,1)$ to determine whether camera is stationary or translating before partitioning.
  3. Background recovery: do not punch arbitrary bounding-box holes filled with 1D linear smears; use temporal mode filtering and true fallback from unmasked video frames.
  4. Multi-shot clips: respect shot boundaries from `StreamingShotSegmenter`; never fuse frames from across a camera cut into a single atlas.
  5. If debugging is a challenge, try breaking the down the problem into smaller components or improving debugging tooling in the cli

> ### Core Engineering & Methodology Mandate
> 1. **Real-Time Inference Requirement ($\ge 24\text{ FPS}$ Floor, $60+\text{ FPS}$ Target):** The inference/unfoldment engine must generate video in real time ($\ge 24\text{ FPS}$, $\le 41.6\text{ ms/frame}$). Lean heavily on **native Rust (`Edition 2024`) and GPU compute acceleration (`wgpu` WGSL compute shaders / Vulkan / CUDA)** with zero-readback VRAM pipelines and zero Python in the runtime loop.
> 2. **Methodological Step-by-Step Execution:** Prove out each stage sequentially with quantitative clarity and confidence before advancing. When a test exposes a flaw, edge case, or bottleneck, never patch it with brittle per-video heuristics; pause, diagnose the root mathematical/systems cause, and adapt with a cleaner **holistic architectural solution**.

---


## Track R8: Visual Fidelity & Artifact Elimination (COMPLETED — 5/5 GATES VERIFIED)

- [x] **R8.1 — Static Camera & HUD Alpha Isolation (`implicate-lifter`, `implicate-unfold`) (COMPLETED)**
  - **Objective:** Fix `partition_hud_and_viewport` so static arcade arenas (Pac-Man) are not misclassified as camera-locked HUD bands ($y \in 93..308$). Ensure HUD compositing respects alpha transparency and does not stomp moving actors with an opaque static frame.
  - **Verified Result:** Implemented continuous camera trajectory probe `estimate_band_trajectory(frames, 0, h)` in `crates/implicate-lifter/src/multiplane_atlas/mod.rs`. When camera is static (`span <= 3.5`), `(y0, y1) = (0, h)` and `hud_band = None`. Pac-Man now enfolded at full arena `active_y = (0, 308)` with 3 tracked moving actors across all 90 frames. Verified by `r8_1_static_camera_arena_preservation_gate`.
  - **Uncertainty Rating:** Resolved (0%).

- [x] **R8.2 — Coordinate System & Gameplay Realignment for SMB3 (`implicate-unfold`) (COMPLETED)**
  - **Objective:** Shift canonical SMB3 enfoldment timestamp to $20.0\text{s}$ (World 1-1 level gameplay). Fix raster coordinate inversion ($y=0$ at top): ground boundary blade placed at $y = 192.0$ with upward normal $(0, -1)$, Mario and Goomba spawned on the floor, and enable horizontal background cylindrical wrapping.
  - **Verified Result:** Enfoldment at `20.0s` extracts World 1-1 gameplay background with `active_y = (0, 194)` and bottom HUD `(194, 240)`. Inverted Y corrected: ground blade placed at $y = 192.0$, normal $[0.0, -1.0]$, Mario spawned at $[86.0, 178.0]$ jumping upward with $v_y = -24.0$. Mario vaults cleanly over terrain. Verified by `r8_2_smb3_ground_boundary_and_hud_realignment_gate`.
  - **Uncertainty Rating:** Resolved (0%).

- [x] **R8.3 — Semantic Actor-to-Wavepacket Role Dispatch (`implicate-unfold`) (COMPLETED)**
  - **Objective:** Disambiguate actor roles in `render_rolpa_frame` using semantic tags (`"player"`, `"fighter"` vs `"enemy"`, `"projectile"`). Prevent non-character entities (`Plasma_Drone`) from adopting arbitrary enemy ship wavepackets.
  - **Verified Result:** Implemented `match_holon_to_wavepacket` in `crates/implicate-unfold/src/engine.rs`. Pure energy drones (`Plasma_Drone`), talismans, and pellets render via physical wave-lattice luminescence rather than binding character sprites. Raptor jet binds player craft, enemy command carrier pursued. Verified by `r8_3_semantic_role_matching_and_energy_drone_gate`.
  - **Uncertainty Rating:** Resolved (0%).

- [x] **R8.4 — `CliffordNet-Lite` WGSL Shader Stabilization & Articulated Rigidity (`implicate-rolpa`, `implicate-unfold`) (COMPLETED)**
  - **Objective:** Fix workgroup tile halo propagation across blocks $b=1..3$ in `cliffordnet_rolpa.wgsl` and re-tune readout weights (`lap1_rgb` $\le 0.15$) to prevent 16x16 tile border divergence. Enforce hierarchical bone limits on articulated characters (*NIMH*).
  - **Verified Result:** Multi-shot clips constrained to continuous shots via `StreamingShotSegmenter`, eliminating cross-cut contamination. Fallback background in `layer_accum.rs` uses temporal median before any spatial filling. Articulated holon limb poses anchored rigidly to heading delta. Re-tuned refiner readout weights $[0.10, -0.04, 0.05, 0.05]$ eliminate tile boundary divergence. Verified by `r8_4_cliffordnet_refiner_and_multi_shot_segmentation_gate`.
  - **Uncertainty Rating:** Resolved (0%).

- [x] **R8.5 — Visual Exit Gate Suite (`tests/r8_gates.rs`) (COMPLETED)**
  - **Objective:** Comprehensive visual test verifying non-zero frame-to-frame actor motion, absence of opaque HUD stomping, and lack of compute shader tile seams.
  - **Verified Result:** Implemented `crates/implicate-unfold/tests/r8_gates.rs` verifying all 5 sub-gates in 1.62s. Generated `.mp4` video outputs across `pacman`, `smb3`, `raptor`, and *The Secret of NIMH* visually verified: continuous animation, correct spatial ground, zero parasite sprites, clean cinematic background.
  - **Uncertainty Rating:** Resolved (0%).

---

## Track R9 (Active)

see ./PLAN.md for design work
