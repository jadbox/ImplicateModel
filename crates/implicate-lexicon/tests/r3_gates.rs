use std::time::Instant;

use implicate_lexicon::{
    CORE_VERBS, ConceptLexicon, GatrHolonOutput, GatrHolonToken, GatrVerbBinder,
    NonEquivariantMlpBaseline, VerbPotentialLexicon, evaluate_verb_unfold_frechet,
    synthesize_corpus_verb_clip,
};
use implicate_math::{CliffordQuantumPotential, mat3_vec};

#[test]
fn r3_1_inverse_bohm_hiley_q_potential_frechet_gate() {
    let lexicon = ConceptLexicon::default();
    let num_train_clips = 16usize;
    let num_steps = 48usize;

    let mut ensembles = Vec::with_capacity(CORE_VERBS.len());
    for (v_idx, &verb) in CORE_VERBS.iter().enumerate() {
        let mut clips = Vec::with_capacity(num_train_clips);
        for c in 0..num_train_clips {
            let scale = 35.0 + 5.0 * (c as f64);
            let rot = 0.25 * (c as f64);
            let offset = [100.0 + 12.0 * (c as f64), 80.0 - 7.0 * (c as f64)];
            let seed = ((v_idx as u64) + 1) * 1000 + (c as u64);
            clips.push(synthesize_corpus_verb_clip(
                verb, num_steps, offset, scale, rot, 0.015, seed,
            ));
        }
        ensembles.push((verb, clips));
    }

    let verb_lex = VerbPotentialLexicon::from_labeled_ensembles(&lexicon, &ensembles);
    assert_eq!(verb_lex.entries.len(), 5);

    // 1. Verify numerical finite-difference Laplacian of \Psi_v agrees with analytic Q_Cl
    let orbit_pot = &verb_lex.get("orbit").expect("orbit entry").potential;
    let (mu_mid, _, _, _, _) = orbit_pot.canonical_kinematics(0.5);
    let test_pt = [mu_mid[0] + 0.03, mu_mid[1] - 0.02];
    let h = 1e-4;
    let psi_c = orbit_pot.evaluate_wavefield(test_pt, 0.5);
    let psi_px = orbit_pot.evaluate_wavefield([test_pt[0] + h, test_pt[1]], 0.5);
    let psi_mx = orbit_pot.evaluate_wavefield([test_pt[0] - h, test_pt[1]], 0.5);
    let psi_py = orbit_pot.evaluate_wavefield([test_pt[0], test_pt[1] + h], 0.5);
    let psi_my = orbit_pot.evaluate_wavefield([test_pt[0], test_pt[1] - h], 0.5);
    let lap_psi = (psi_px + psi_mx + psi_py + psi_my - psi_c.scale(4.0)).scale(1.0 / (h * h));
    let q_fd = CliffordQuantumPotential::from_field(psi_c, lap_psi).expect("valid even inverse");
    let (q_an, _, _) = orbit_pot.evaluate_q_potential_and_guidance(test_pt, 0.5);
    assert!(
        (q_fd.bivector - q_an.bivector).abs() < 0.05,
        "Expected finite-difference bivector torque {:.4} to match analytic {:.4}",
        q_fd.bivector,
        q_an.bivector
    );

    // 2. Evaluate unfolded trajectories across 2x scale changes (0.5x, 1.0x, 2.0x) and 360 deg rotations
    let test_scales = [30.0, 60.0, 120.0]; // 0.5x, 1.0x, 2.0x relative to 60 px baseline
    let num_angles = 8usize; // 0, 45, 90, 135, 180, 225, 270, 315 degrees

    for &verb in &CORE_VERBS {
        let entry = verb_lex.get(verb).expect("learned verb entry");
        let mut max_frechet = 0.0f64;
        let mut sum_frechet = 0.0f64;
        let mut count = 0usize;

        for &scale in &test_scales {
            for a_idx in 0..num_angles {
                let rot_rad =
                    (a_idx as f64) * (2.0 * std::f64::consts::PI / (num_angles as f64));
                let offset = [-240.0 + 55.0 * (a_idx as f64), 160.0 - 30.0 * (a_idx as f64)];
                let (gt_a, gt_b) = synthesize_corpus_verb_clip(
                    verb,
                    num_steps,
                    offset,
                    scale,
                    rot_rad,
                    0.0,
                    999,
                );
                let df = evaluate_verb_unfold_frechet(&entry.potential, &gt_a, &gt_b);
                max_frechet = max_frechet.max(df);
                sum_frechet += df;
                count += 1;
            }
        }
        let mean_frechet = sum_frechet / (count as f64);
        println!(
            "[R3.1 Q_Cl Verb '{}'] sigma={:.3} | mean Frechet={:.4} ({:.2}%), max Frechet={:.4} ({:.2}%) across 2x scales & 360 deg rotations",
            verb,
            entry.potential.sigma,
            mean_frechet,
            mean_frechet * 100.0,
            max_frechet,
            max_frechet * 100.0
        );
        assert!(
            max_frechet < 0.08,
            "Gate R3.1 failed for verb '{}': max normalized Frechet {:.4} >= 0.08",
            verb,
            max_frechet
        );
    }
}

#[test]
fn r3_2_pin301_equivariant_gatr_latency_and_e3_generalization() {
    let lexicon = ConceptLexicon::default();
    let verb_spinors = lexicon.embed_clauses(&CORE_VERBS);
    let noun_clauses = [
        "Mrs_Brisby hero mouse",
        "Jeremy_Crow trickster bird",
        "Dragon_Cat shadow beast",
        "Red_Amulet self talisman",
    ];
    let noun_spinors = lexicon.embed_clauses(&noun_clauses);

    // Helper to synthesize a canonical multi-body scene for verb v_idx
    let make_scene_example = |v_idx: usize, clip_idx: usize| -> (Vec<GatrHolonToken>, Vec<GatrHolonOutput>) {
        let tau = ((clip_idx % 5) as f32) * 0.2;
        let dist = 4.0 + 0.3 * (clip_idx as f32);
        let p_a = [0.0f32, 0.0, 0.0];
        let p_b = [dist, 0.5 * ((clip_idx % 3) as f32), 0.0];
        let n_up = [0.0f32, 1.0, 0.0];

        let rx = p_b[0] - p_a[0];
        let ry = p_b[1] - p_a[1];
        let rz = p_b[2] - p_a[2];
        let rn = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-3);
        let u_rel = [rx / rn, ry / rn, rz / rn];
        let dot_nr = n_up[0] * u_rel[0] + n_up[1] * u_rel[1] + n_up[2] * u_rel[2];
        let nx = n_up[0] - dot_nr * u_rel[0];
        let ny = n_up[1] - dot_nr * u_rel[1];
        let nz = n_up[2] - dot_nr * u_rel[2];
        let nn = (nx * nx + ny * ny + nz * nz).sqrt().max(1e-3);
        let n_orth = [nx / nn, ny / nn, nz / nn];
        let u_tang = [
            n_orth[1] * u_rel[2] - n_orth[2] * u_rel[1],
            n_orth[2] * u_rel[0] - n_orth[0] * u_rel[2],
            n_orth[0] * u_rel[1] - n_orth[1] * u_rel[0],
        ];

        // Ground-truth physical velocity impulse in the E(3)-equivariant frame of (A, B)
        let (w_rel, w_up, w_tang, beacon) = match v_idx {
            0 => (0.8f32, 0.0f32, 0.4f32, [2.0f32, 1.0, 1.2, 0.0]),   // patrol
            1 => (1.2f32, 1.5f32, 0.0f32, [1.5f32, 0.0, 2.0, 0.5]),   // jump_over
            2 => (2.0f32, 0.1f32, 0.0f32, [3.0f32, 0.0, 2.5, 0.0]),   // pursue
            3 => (-2.2f32, 0.3f32, 0.2f32, [3.0f32, 0.0, -2.5, 3.14]), // flee_from
            _ => (0.1f32, 0.0f32, 1.8f32, [2.5f32, 2.0, 1.8, 1.57]),  // orbit
        };

        let dv = [
            w_rel * u_rel[0] + w_up * n_orth[0] + w_tang * u_tang[0],
            w_rel * u_rel[1] + w_up * n_orth[1] + w_tang * u_tang[1],
            w_rel * u_rel[2] + w_up * n_orth[2] + w_tang * u_tang[2],
        ];
        let dw = [0.0f32, 0.5 * w_tang, 0.2 * w_up];

        let tok_a = GatrHolonToken {
            position: p_a,
            velocity: [0.2 * u_rel[0], 0.2 * u_rel[1], 0.2 * u_rel[2]],
            angular_velocity: [0.0, 0.1, 0.0],
            support_normal: n_up,
            support_dist: 0.0,
            noun_spinor: noun_spinors[v_idx % noun_spinors.len()],
            verb_spinor: verb_spinors[v_idx],
            target_idx: 1,
            phase_tau: tau,
        };
        let tok_b = GatrHolonToken {
            position: p_b,
            velocity: [0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
            support_normal: n_up,
            support_dist: 0.0,
            noun_spinor: noun_spinors[(v_idx + 1) % noun_spinors.len()],
            verb_spinor: verb_spinors[v_idx],
            target_idx: 0,
            phase_tau: tau,
        };
        let out_a = GatrHolonOutput {
            delta_v: dv,
            delta_omega: dw,
            q_beacon_params: beacon,
        };
        let out_b = GatrHolonOutput {
            delta_v: [-dv[0], -dv[1], -dv[2]],
            delta_omega: [-dw[0], -dw[1], -dw[2]],
            q_beacon_params: beacon,
        };
        (vec![tok_a, tok_b], vec![out_a, out_b])
    };

    // 1. Evaluate out-of-domain E(3)-transformed trajectory error for N_train in {1, 5, 20}
    let rot_ood: [[f32; 3]; 3] = [
        [0.0, -0.8, 0.6],
        [0.8, 0.36, 0.48],
        [-0.6, 0.48, 0.64],
    ];
    let trans_ood = [145.0f32, -82.0, 63.0];

    for &n_train in &[1usize, 5, 20] {
        let mut gatr = GatrVerbBinder::new();
        let mut mlp = NonEquivariantMlpBaseline::default();

        for v_idx in 0..CORE_VERBS.len() {
            let mut train_set = Vec::with_capacity(n_train);
            for c in 0..n_train {
                train_set.push(make_scene_example(v_idx, c));
            }
            gatr.fit_verb_from_examples(verb_spinors[v_idx], &train_set);
            mlp.fit_verb_from_examples(verb_spinors[v_idx], &train_set);
        }

        let mut gatr_err_sum = 0.0f32;
        let mut mlp_err_sum = 0.0f32;
        let mut count = 0usize;

        for v_idx in 0..CORE_VERBS.len() {
            for test_c in 20..26 {
                let (canon_toks, canon_gt) = make_scene_example(v_idx, test_c);
                let ood_toks: Vec<GatrHolonToken> = canon_toks
                    .iter()
                    .map(|t| t.transform_e3(rot_ood, trans_ood))
                    .collect();
                let gt_dv_ood = mat3_vec(rot_ood, canon_gt[0].delta_v);
                let gt_norm = (gt_dv_ood[0].powi(2) + gt_dv_ood[1].powi(2) + gt_dv_ood[2].powi(2))
                    .sqrt()
                    .max(1e-4);

                let gatr_pred = gatr.forward(&ood_toks)[0].delta_v;
                let mlp_pred = mlp.forward(&ood_toks)[0].delta_v;

                let g_err = ((gatr_pred[0] - gt_dv_ood[0]).powi(2)
                    + (gatr_pred[1] - gt_dv_ood[1]).powi(2)
                    + (gatr_pred[2] - gt_dv_ood[2]).powi(2))
                .sqrt()
                    / gt_norm;
                let m_err = ((mlp_pred[0] - gt_dv_ood[0]).powi(2)
                    + (mlp_pred[1] - gt_dv_ood[1]).powi(2)
                    + (mlp_pred[2] - gt_dv_ood[2]).powi(2))
                .sqrt()
                    / gt_norm;

                gatr_err_sum += g_err;
                mlp_err_sum += m_err;
                count += 1;
            }
        }

        let gatr_mean_err = gatr_err_sum / (count as f32);
        let mlp_mean_err = mlp_err_sum / (count as f32);
        println!(
            "[R3.2 E(3) OOD Generalization @ N_train={:2}] GATr error={:.4} ({:.2}%) vs Non-Equivariant MLP error={:.4} ({:.1}%)",
            n_train,
            gatr_mean_err,
            gatr_mean_err * 100.0,
            mlp_mean_err,
            mlp_mean_err * 100.0
        );
        assert!(
            gatr_mean_err < 0.05,
            "Gate R3.2 failed: GATr out-of-domain E(3) error {:.4} >= 0.05 at N_train={}",
            gatr_mean_err,
            n_train
        );
        assert!(
            mlp_mean_err > 0.50,
            "Expected non-equivariant MLP to fail on OOD E(3) transforms, got {:.4}",
            mlp_mean_err
        );
    }

    // 2. Benchmark 6-block GATr forward latency for K = 32 scene Holons (< 0.5 ms gate)
    let mut gatr = GatrVerbBinder::new();
    for v_idx in 0..CORE_VERBS.len() {
        let ex = vec![make_scene_example(v_idx, 0)];
        gatr.fit_verb_from_examples(verb_spinors[v_idx], &ex);
    }

    let k_holons = 32usize;
    let mut scene_32 = Vec::with_capacity(k_holons);
    for i in 0..k_holons {
        let (pair, _) = make_scene_example(i % CORE_VERBS.len(), i);
        let mut tok = pair[0].clone();
        tok.position = [(i as f32) * 1.5, ((i % 4) as f32) * 0.8, 0.0];
        tok.target_idx = (i + 1) % k_holons;
        scene_32.push(tok);
    }

    // Warmup
    for _ in 0..20 {
        let out = std::hint::black_box(gatr.forward(std::hint::black_box(&scene_32)));
        assert_eq!(out.len(), k_holons);
    }

    let iters = 500usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let out = std::hint::black_box(gatr.forward(std::hint::black_box(&scene_32)));
        std::hint::black_box(out);
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);
    println!(
        "[R3.2 GATr Latency (6 blocks, 16 MV + 64 Scalar channels, K=32 holons)] {:.4} ms/pass (gate < 0.5000 ms)",
        avg_ms
    );
    assert!(
        avg_ms < 0.5,
        "Gate R3.2 failed: GATr forward latency {:.4} ms >= 0.5 ms",
        avg_ms
    );
}
