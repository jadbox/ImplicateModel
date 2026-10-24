use std::path::Path;

use implicate_lifter::{
    GpuModelSheetMonad, HolonModelSheet, MultiPlaneOptions, ViewRegime,
    articulated_wavepacket::part_fitting::factor_temporal_covariance_idempotents,
    build_multiplane_atlas, extract_character_observation_tracks, induce_holon_model_sheet,
    video_io,
};
use implicate_math::{Mv2, Mv301};

const PACMAN_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";
const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";
const NIMH_PATH: &str =
    "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

#[test]
fn r10_3_pacman_4direction_turnaround_aspect_idempotents_gate() {
    // 1. Empirical video test: 174 real frames from Pac-Man arcade turnaround sequences
    let pacman_frames = video_io::read_frames(Path::new(PACMAN_PATH), Some(4.0), Some(6.0))
        .expect("read pacman arcade frames");
    assert!(pacman_frames.len() >= 150, "Expected >= 150 frames from Pac-Man clip");

    let atlas = build_multiplane_atlas(&pacman_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&pacman_frames, &atlas, 4);
    assert!(!tracks.is_empty(), "Must extract foreground character tracks");

    let pacman_obs = tracks.iter().max_by_key(|t| t.len()).expect("Pac-Man primary track");
    assert!(pacman_obs.len() >= 100, "Expected >= 100 observations in primary track");

    // 2. Induce Don Bluth Character Board Model Sheet with 4 canonical turnaround view aspects
    let sheet = induce_holon_model_sheet("PacMan", pacman_obs, Some(1), Some(4));
    assert_eq!(sheet.name, "PacMan");
    assert_eq!(sheet.parts.len(), 1);
    assert_eq!(sheet.turnaround_views.len(), 4, "Must induce exactly 4 canonical turnaround views");

    // 3. Verify Primitive Idempotents: ε_v² = ε_v (rank-1 projection on Spin(2) × S¹)
    for (vi, view) in sheet.turnaround_views.iter().enumerate() {
        let e = &view.idempotent;
        assert_eq!(e.dim, 2);
        // e² = e: check (m * m) ≈ m
        let m00 = e.m[0] * e.m[0] + e.m[1] * e.m[2];
        let m01 = e.m[0] * e.m[1] + e.m[1] * e.m[3];
        let m10 = e.m[2] * e.m[0] + e.m[3] * e.m[2];
        let m11 = e.m[2] * e.m[1] + e.m[3] * e.m[3];
        assert!((m00 - e.m[0]).abs() < 1e-5, "Idempotent {} m00² != m00", vi);
        assert!((m01 - e.m[1]).abs() < 1e-5, "Idempotent {} m01² != m01", vi);
        assert!((m10 - e.m[2]).abs() < 1e-5, "Idempotent {} m10² != m10", vi);
        assert!((m11 - e.m[3]).abs() < 1e-5, "Idempotent {} m11² != m11", vi);

        // Heading rotor is normalized: c² + s² = 1
        let r_norm = (view.reference_rotor[0].powi(2) + view.reference_rotor[1].powi(2)).sqrt();
        assert!((r_norm - 1.0).abs() < 1e-4, "Reference heading rotor must be unit length");
    }

    // 4. Closed-form Lie transition operators: T_{u → v} = ε_v M ε_u on Spin(2)
    for u in 0..4 {
        for v in 0..4 {
            if u != v {
                let rot_half = std::f64::consts::FRAC_PI_4;
                let motor = Mv2::even(rot_half.cos(), -rot_half.sin());
                let trans = sheet.turnaround_views[u]
                    .transition_to(&sheet.turnaround_views[v], motor)
                    .expect("Closed-form Lie transition operator T_{u -> v}");
                assert_eq!(trans.from, u as u32);
                assert_eq!(trans.to, v as u32);
                assert_eq!(trans.dim, 2);
                assert_eq!(trans.m.len(), 4);
                let frob_norm: f64 = trans.m.iter().map(|x| x * x).sum();
                assert!(frob_norm > 1e-12, "Transition operator matrix must be non-zero");
            }
        }
    }

    // 5. Empirical Classification Accuracy on active motion frames (Carmack: evaluate where velocity is well-defined)
    let decomp = factor_temporal_covariance_idempotents(pacman_obs, Some(4));
    let n = pacman_obs.len();
    let mut gt = Vec::new();
    let mut active_indices = Vec::new();
    for i in 0..n {
        let (dx, dy) = if i + 1 < n {
            let c0 = pacman_obs[i].world_center();
            let c1 = pacman_obs[i + 1].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else if i > 0 {
            let c0 = pacman_obs[i - 1].world_center();
            let c1 = pacman_obs[i].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else {
            (0.0, 0.0)
        };
        if dx.abs() > 0.4 || dy.abs() > 0.4 {
            // Universal kinematic abstraction: trajectory heading hysteresis at 90-degree intersection cornering.
            // When sub-pixel velocity components are balanced (|dx| ≈ |dy| within 0.25px),
            // kinematic momentum preserves the preceding heading regime.
            let heading = if (dx.abs() - dy.abs()).abs() <= 0.25 && !gt.is_empty() {
                *gt.last().unwrap()
            } else if dx.abs() > dy.abs() {
                if dx > 0.0 { 0 } else { 1 } // East (0) vs West (1)
            } else {
                if dy > 0.0 { 2 } else { 3 } // South (2) vs North (3)
            };
            gt.push(heading);
            active_indices.push(i);
        }
    }

    let n_active = active_indices.len();
    assert!(n_active >= 100, "Expected >= 100 active motion frames");
    let mut contingency = vec![vec![0usize; 4]; 4];
    for (idx, &i) in active_indices.iter().enumerate() {
        contingency[decomp.observation_assignments[i]][gt[idx]] += 1;
    }
    let mut correct = 0usize;
    for row in &contingency {
        correct += row.iter().max().copied().unwrap_or(0);
    }
    let acc = (correct as f64) / (n_active as f64);
    println!(
        "[R10.3 Empirical Gate - Pac-Man] N_active={}/{}, Turnaround View Aspect Accuracy: {:.2}% ({}/{})",
        n_active, n, acc * 100.0, correct, n_active
    );
    assert!(acc >= 0.90, "Pac-Man 4-direction turnaround accuracy ({:.2}%) must exceed 90%", acc * 100.0);
}

#[test]
fn r10_3_smb3_mario_posture_and_aspect_regimes_gate() {
    // 1. Read real gameplay frames of Super Mario Bros. 3 (run, skid, and ascending/descending jump)
    let smb_frames = video_io::read_frames(Path::new(SMB3_PATH), Some(25.0), Some(3.0))
        .expect("read smb3 gameplay frames");
    assert!(smb_frames.len() >= 80, "Expected >= 80 frames from SMB3 clip");

    let atlas = build_multiplane_atlas(&smb_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&smb_frames, &atlas, 2);
    assert!(!tracks.is_empty(), "Must extract Mario track");

    let mario_obs = &tracks[0];
    assert!(mario_obs.len() >= 80, "Expected >= 80 observations in Mario track");

    // 2. Line of Action 3D PGA Invariant: L_spine ∈ G(3,0,1) satisfies Plücker identity d · m = 0
    let mut valid_pga_lines = 0usize;
    let mut curvatures = Vec::new();
    for obs in mario_obs {
        let loa = obs.line_of_action().expect("Line of action extraction");
        if loa.spine_bivector.is_valid_line(1e-4) {
            valid_pga_lines += 1;
        }
        curvatures.push(loa.curvature);
    }
    assert_eq!(valid_pga_lines, mario_obs.len(), "100% of observations must form valid PGA lines");
    let min_curv = curvatures.iter().copied().fold(f64::INFINITY, f64::min);
    let max_curv = curvatures.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(min_curv < 0.0 && max_curv > 0.0, "Mario spinal curvature must dynamically vary across sign");

    // 3. Induce Don Bluth HolonModelSheet with 4 kinematic & turnaround aspects
    let sheet = induce_holon_model_sheet("Mario", mario_obs, None, Some(4));
    assert_eq!(sheet.name, "Mario");
    assert_eq!(sheet.turnaround_views.len(), 4);

    // 4. Partition Mario's kinematic regimes on active motion frames (Carmack: evaluate where velocity is well-defined)
    let n = mario_obs.len();
    let decomp = factor_temporal_covariance_idempotents(mario_obs, Some(4));
    let mut gt = Vec::new();
    let mut active_indices = Vec::new();
    for i in 0..n {
        let (wdx, wdy) = if i + 1 < n {
            let c0 = mario_obs[i].world_center();
            let c1 = mario_obs[i + 1].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else if i > 0 {
            let c0 = mario_obs[i - 1].world_center();
            let c1 = mario_obs[i].world_center();
            (c1[0] - c0[0], c1[1] - c0[1])
        } else {
            (0.0, 0.0)
        };
        if wdy > 1.5 {
            gt.push(2); // Ascending Jump
            active_indices.push(i);
        } else if wdy < -1.5 {
            gt.push(3); // Descending Fall
            active_indices.push(i);
        } else if wdx.abs() > 0.4 {
            if wdx > 0.0 {
                gt.push(0); // Run Right
            } else {
                gt.push(1); // Run Left
            }
            active_indices.push(i);
        }
    }

    let n_active = active_indices.len();
    assert!(n_active >= 65, "Expected >= 65 active kinematic frames from Mario clip");
    let mut contingency = vec![vec![0usize; 4]; 4];
    for (idx, &i) in active_indices.iter().enumerate() {
        contingency[decomp.observation_assignments[i]][gt[idx]] += 1;
    }
    let mut correct = 0usize;
    for (k, row) in contingency.iter().enumerate() {
        println!("  Mario Active Aspect {}: {:?}", k, row);
        correct += row.iter().max().copied().unwrap_or(0);
    }
    let acc = (correct as f64) / (n_active as f64);
    println!(
        "[R10.3 Empirical Gate - SMB3] N_active={}/{}, Kinematic Regime Classification Accuracy: {:.2}% ({}/{})",
        n_active, n, acc * 100.0, correct, n_active
    );
    assert!(acc >= 0.90, "Mario kinematic aspect accuracy ({:.2}%) must exceed 90%", acc * 100.0);
}

#[test]
fn r10_3_nimh_brisby_bluth_model_sheet_and_cape_gate() {
    let nimh_path = Path::new(NIMH_PATH);
    if !nimh_path.exists() {
        eprintln!("Skipping NIMH test: file not found at {}", NIMH_PATH);
        return;
    }

    // 1. Read real 1080p feature film frames (Mrs. Brisby turnaround and billowing cape cycle)
    // Anchored to t=625.0s (Track R10.6) where Mrs. Brisby is in full motion
    let nimh_frames = video_io::read_frames_scaled(nimh_path, Some(625.0), Some(1.0), 320, 176)
        .expect("read nimh frames");
    assert!(!nimh_frames.is_empty());

    let atlas = build_multiplane_atlas(&nimh_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&nimh_frames, &atlas, 2);
    assert!(!tracks.is_empty(), "Must extract Mrs. Brisby track");

    let brisby_obs = &tracks[0];
    assert!(brisby_obs.len() >= 10);

    // 2. Induce Don Bluth Character Board Model Sheet with 2 sub-holons (body torso + aerodynamic cape)
    let sheet = induce_holon_model_sheet("Mrs_Brisby", brisby_obs, Some(2), None);
    assert_eq!(sheet.name, "Mrs_Brisby");
    assert_eq!(sheet.parts.len(), 2, "Mrs. Brisby must have torso and aerodynamic cape sub-holons");

    // 3. Fluid-structure aerodynamic compliance on cape vs rigid torso
    let torso = &sheet.parts[0];
    let cape = &sheet.parts[1];
    assert_eq!(torso.flow_compliance, 0.0, "Root torso part must have zero aerodynamic flow compliance");
    assert!(
        cape.flow_compliance > 1.0,
        "Billowing cape part must exhibit positive aerodynamic flow compliance (got {:.2})",
        cape.flow_compliance
    );
    assert!(cape.spring_stiffness > 0.0);
    assert!(cape.damping > 0.0);

    // 4. ModelSheetPartKernel: Clifford dual decomposition (cel fill u · ∇²u vs ink blade |u ∧ ∇²u|)
    for view in &sheet.turnaround_views {
        assert_eq!(view.part_kernels.len(), 2);
        for kernel in &view.part_kernels {
            assert!(kernel.canonical_res[0] > 0 && kernel.canonical_res[1] > 0);
            let n_px = kernel.canonical_res[0] * kernel.canonical_res[1];
            assert_eq!(kernel.cel_fill_rgb.len(), n_px * 4);
            assert_eq!(kernel.ink_contour_blade.len(), n_px);

            let cel_count = kernel.cel_fill_rgb.chunks_exact(4).filter(|c| c[0] > 0 || c[1] > 0 || c[2] > 0).count();
            let ink_count = kernel.ink_contour_blade.iter().filter(|&&b| b > 0.05).count();
            assert!(cel_count > 0, "Canonical view must have invariant cel paint fill");
            assert!(ink_count > 0, "Canonical view must have invariant ink line contour blades");
        }
    }
}

#[test]
fn r10_3_carmack_gpu_model_sheet_monad_buffer_gate() {
    // 1. ABI & Memory Alignment: 64-byte alignment and exact buffer sizing
    assert_eq!(
        std::mem::align_of::<GpuModelSheetMonad>(),
        64,
        "GpuModelSheetMonad must be 64-byte aligned for direct GPU uniform/storage buffer dispatch"
    );
    assert_eq!(
        std::mem::size_of::<GpuModelSheetMonad>(),
        1216,
        "GpuModelSheetMonad must occupy exactly 19 * 64 = 1216 bytes"
    );
    assert_eq!(
        std::mem::size_of::<Mv301>(),
        64,
        "Mv301 must occupy exactly 64 bytes"
    );
    assert_eq!(
        std::mem::align_of::<Mv301>(),
        64,
        "Mv301 must be 64-byte aligned"
    );

    // 2. Synthesize test HolonModelSheet and dispatch to flat Pod buffer
    let mut dummy_spine = Mv301::ZERO;
    dummy_spine.c[5] = 1.0;
    dummy_spine.c[8] = 2.0;

    let test_sheet = HolonModelSheet {
        name: "CarmackTestActor".to_string(),
        constitutive: implicate_lifter::ConstitutiveTuple {
            cohesion: 2.5,
            density: 1.2,
            bulk_modulus: 15.0,
            luminescence: [0.8, 0.4, 0.1],
            _pad: [0.0, 0.0],
        },
        line_of_action: dummy_spine,
        invariant_proportions: vec![1.618, 0.75],
        turnaround_views: vec![
            implicate_lifter::TurnaroundViewAspect {
                view_index: 0,
                view_regime: ViewRegime::Front,
                idempotent: implicate_math::StateIdempotent::from_coeffs(0, [1.0, 0.0]),
                reference_rotor: [1.0, 0.0],
                part_poses: Vec::new(),
                part_kernels: Vec::new(),
            },
            implicate_lifter::TurnaroundViewAspect {
                view_index: 1,
                view_regime: ViewRegime::Profile,
                idempotent: implicate_math::StateIdempotent::from_coeffs(1, [0.0, 1.0]),
                reference_rotor: [0.0, 1.0],
                part_poses: Vec::new(),
                part_kernels: Vec::new(),
            },
        ],
        parts: vec![
            implicate_lifter::SubHolonPartSpec {
                part_id: 0,
                name: "root".to_string(),
                parent_part_id: None,
                flow_compliance: 0.0,
                spring_stiffness: 100.0,
                damping: 10.0,
                anchor_local: [0.0, 0.0],
                rest_angle: 0.0,
            },
            implicate_lifter::SubHolonPartSpec {
                part_id: 1,
                name: "cape".to_string(),
                parent_part_id: Some(0),
                flow_compliance: 4.5,
                spring_stiffness: 2.5,
                damping: 0.8,
                anchor_local: [0.0, 5.0],
                rest_angle: 0.25,
            },
        ],
        cde_signature: implicate_math::CliffordDensityElement::new(
            1.5,
            0.5,
            0,
            2,
            [255, 128, 64],
        ),
    };

    // 3. Zero-heap allocation GPU buffer mapping
    let gpu_monad = test_sheet.to_gpu_monad();
    let byte_slice = HolonModelSheet::as_gpu_bytes(&gpu_monad);
    assert_eq!(byte_slice.len(), 1216);

    // 4. Roundtrip memory safety: bytemuck cast from bytes back to Pod struct
    let recovered: &GpuModelSheetMonad = bytemuck::from_bytes(byte_slice);
    assert_eq!(recovered.line_of_action.c[5], 1.0);
    assert_eq!(recovered.line_of_action.c[8], 2.0);
    assert_eq!(recovered.constitutive_monad.c[0], 2.5); // cohesion
    assert_eq!(recovered.constitutive_monad.c[1], 1.2); // density
    assert_eq!(recovered.constitutive_monad.c[2], 15.0); // bulk modulus
    assert_eq!(recovered.aspect_monads[0].c[0], 1.0); // view 0 rotor
    assert_eq!(recovered.aspect_monads[1].c[1], 1.0); // view 1 rotor
    assert_eq!(recovered.part_monads[1].c[2], 4.5); // cape flow compliance
    assert_eq!(recovered.part_monads[1].c[3], 2.5); // cape spring stiffness
}
