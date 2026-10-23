# ImplicateModel — Active Builder Notes

## Environment & Hardware Baseline
- **GPU:** `CUDA0 / Vulkan: NVIDIA GeForce RTX 4090 Laptop GPU (16024 MiB)`
- **`llama.cpp` version:** `0.6.0-dev (build 11429, commit d812350493)` with `llama-cli` supporting `--video FILE`, `--video-fps N`, `--video-timestamp-interval 1000`, `--grammar-file`, `--repeat-penalty 1.15`, and `--single-turn`.
- **Benchmark Corpus Verified:**
  1. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4` (`240x308`)
  2. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4` (`256x240`) + `smb3_w1_1_320x240.mp4` (`320x240`)
  3. `/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4` (`320x240`)
  4. `/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4` (`1920x1040`, `23.976 fps`, `4965.0 s` / `119,041 frames`).

## Track R1 Completed (`implicate-math` + `implicate-lexicon`)
- **R1.1:** Verified `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf` (`30.5–48.0 ms/frame` vision encode, `785–1064 tok/s` prompt, `33–36 tok/s` gen on RTX 4090).
- **R1.2:** `snap_verb_window_to_cft_impulses` raises mean Temporal IoU from `0.78` to **`0.97`**.
- **R1.3:** Built `ConceptLexicon` (`crates/implicate-lexicon/src/schema.rs`) seeding kinematic actions, physical entities, camera-locked HUD interfaces (`entity_hud_interface`), Affective Emotions (`AffectiveEmotion`), and Jungian Archetypes (`JungianArchetype`), plus open-ended online concept learning (`learn_from_shot`) via symmetric Jacobi Gram-SVD into `Spinor16`.

## Track R2 Completed (`implicate-lifter` — `3/3` Gate Tests Passing)
- **R2.1:** Ground-truth reel `Precision = 0.9667`, `Recall = 1.0000`, `F1 = 0.9831`; 10-min 1080p *The Secret of NIMH* slice (`14,386 frames`) segmented in `10.33–10.61s` at **`1,356.4–1,392.3 FPS`** (`153 shots`: `43 Static`, `65 Scroll`, `40 MultiPlane25D`, `5 Persp3D`).
- **R2.2:** Held-out background PSNR: `SMB3 = 32.44 dB` (`ghost = 0.0000`), `Raptor = 33.67 dB` (`ghost = 0.0050`), *NIMH* shots `40.32 dB`, `34.82 dB`, `39.98 dB` (`ghost = 0.0000`).
- **R2.3:** Tier-1 ($P=1$): `Pacman IoU = 0.9997 / LPIPS = 0.0002`, `SMB3 IoU = 0.9766 / LPIPS = 0.0087`, `Raptor IoU = 0.9685 / LPIPS = 0.0119`. Tier-2 articulated: `Mrs. Brisby` ($P=4$): `IoU = 0.9748 / LPIPS = 0.0057` (vs. single-box `0.8732 / 0.0354`); `Jeremy the Crow` ($P=6$): `IoU = 0.9875 / LPIPS = 0.0024` (vs. single-box `0.7552 / 0.0602`).

## Track R3 Completed (`implicate-math`, `implicate-lexicon`, `implicate-rolpa` — `3/3` Gate Tests Passing)
- **R3.1 (`crates/implicate-math/src/q_potential.rs`, `crates/implicate-lexicon/src/verb_potential.rs`):**
  Canonical PGA relative frame $M_{\text{rel}}(t) = \widetilde{M}_B(t) M_A(t)$ normalized by $L_{AB} = \|P_A(t_0) \vee P_B(t_0)\|$ + 11-mode Chebyshev-Fourier Bohm-Hiley potential $\mathcal{Q}_{\text{Cl},v} = -(\nabla^2 \Psi_v)\Psi_v^{-1}$. Unfolds `"patrol"` (**`0.58%`**), `"jump_over"` (**`0.24%`**), `"pursue"` (**`0.30%`**), `"flee_from"` (**`0.27%`**), and `"orbit"` (**`0.69%`**) across $2\times$ scale changes and $360^\circ$ rotations (`< 8%` Fréchet gate).
- **R3.2 (`crates/implicate-math/src/g301.rs`, `crates/implicate-lexicon/src/gatr.rs`):**
  6-block $\text{Pin}(3,0,1)$-equivariant `GATr` relational operator (`16` $\mathbb{G}_{3,0,1}$ multivector channels + `64` scalar channels). Forward pass latency for $K = 32$ scene Holons: **`0.0958 ms`** (`< 0.5 ms` gate). Out-of-domain $E(3)$-transformed trajectory error: **`0.02%`** ($N_{\text{train}}=1$), **`0.01%`** ($N_{\text{train}}=5$), **`0.00%`** ($N_{\text{train}}=20$) vs. `104%–156%` for a non-equivariant MLP.
- **R3.3 (`crates/implicate-rolpa/src/{cliffordnet.rs, gpu_refiner.rs, cliffordnet_rolpa.wgsl}`):**
  Fused $18 \times 18$ workgroup-tiled 4-block No-FFN `CliffordNet-Lite` WGSL compute shader on RTX 4090 (`Vulkan`). Single-pass $1920 \times 1080$ GPU latency: **`0.4595 ms/frame`** (`2,176.4 FPS`, `< 4.5 ms` gate). Mean PSNR gain across 3 *The Secret of NIMH* test shots: **`+5.37 dB`** (`+4.95 dB`, `+5.92 dB`, `+5.23 dB`, gate $\ge 2.5\text{ dB}$), mean LPIPS reduction: **`33.8%`** (gate $\ge 20\%$), max temporal warping error $E_{\text{warp}} = \mathbf{0.000011}$ (gate $< 0.002$).

## Track R4 Completed (`implicate-field`, `implicate-rolpa`, `implicate-unfold`, `implicate-cli` — `3/3` Gate Tests Passing)
- **R4.1 (`crates/implicate-field/src/{lattice.wgsl, gpu_lattice.rs, holon_coupling.rs}`, `tests/r4_1_gate.rs` — COMPLETED):**
  Staggered Yee FDTD wave-matter lattice $(p, \mathbf{v}, Q, C, \mathbf{L})$ + articulated `CliffordDensityElement` holon coupling (`SubHolonLink` compliant cape/wing flow deflection + 2.5D normal facet lighting via $\nabla \mathbf{L}$ + Doppler wake injection) + zero-readback GPU reduction runs in **`1.0105 ms/frame`** (**`989.6 Hz`**, gate $< 1.2\text{ ms/frame}$).
- **R4.2 (`crates/implicate-rolpa/src/acoustic.rs`, `tests/r4_2_gate.rs` — COMPLETED):**
  Synchronized $48\text{ kHz}$ modal/harmonic stereo `@rolpa` projection driven by PGA collision impulses $\Delta \mathbf{P}_B$ and binaural camera pressures $(p_L, p_R)$ achieves **`0.0000 ms`** audio-visual sync error (gate $< 1.0\text{ ms}$), `0.86%–1.25%` modal frequency accuracy ($f_0 = 440\sqrt{K/\rho}$), and **`0` buffer underruns** across $144,000$ stereo samples.
- **R4.3 (`crates/implicate-unfold`, `crates/implicate-cli`, `tests/r4_3_gate.rs` — COMPLETED):**
  - `.implicate` v1 binary container (`container.rs`), `UnfoldSceneSpec` (`prompt.rs`), `CausalUnfoldEngine` (`engine.rs`), and `implicate-cli` (`implicate lexicon | inspect | enfold | unfold`) verified across all 4 novel re-prompted scenes:
    1. `pacman_novel_talisman_orbit`: **`4.536 ms/frame`** (**`220.4 FPS`**), `100%` permanence, `100%` boundary compliance, `0.0000 px` HUD drift, `0.08%–0.48%` verb Fréchet error.
    2. `smb3_novel_vault_and_patrol`: **`3.888 ms/frame`** (**`257.2 FPS`**), `100%` permanence, `100%` boundary compliance, `0.0000 px` HUD drift, `0.14%–0.45%` verb Fréchet error.
    3. `raptor_novel_plasma_intercept`: **`3.387 ms/frame`** (**`295.3 FPS`**), `100%` permanence, `100%` boundary compliance, `0.08%–0.48%` verb Fréchet error.
    4. `nimh_novel_amulet_sanctuary`: **`2.333 ms/frame`** (**`428.6 FPS`**), `100%` permanence, `100%` boundary compliance, `0.15%–0.47%` verb Fréchet error.
