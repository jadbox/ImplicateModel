use std::path::Path;

use implicate_lifter::{
    CameraRegime, CharacterObservation, MultiPlaneOptions, RawFrame, Rect, ShotSegmenterConfig,
    StreamingShotSegmenter, build_multiplane_atlas, evaluate_boundary_precision_recall,
    evaluate_vs_single_box_baseline, evaluate_vs_single_box_baseline_with_parts, extract_dominant_character_observations, video_io,
};

const PACMAN_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";
const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";
const RAPTOR_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";
const NIMH_PATH: &str = "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

#[test]
fn r2_1_streaming_shot_segmenter_and_regime_classification() {
    let segmenter = StreamingShotSegmenter::new(ShotSegmenterConfig::default());

    // 1. Verify CameraRegime on Tier-1 benchmark games
    let pacman_rep = segmenter
        .segment_video_range(Path::new(PACMAN_PATH), Some(0.0), Some(4.0))
        .expect("segment pacman");
    assert_eq!(
        pacman_rep.shots[0].camera_regime,
        CameraRegime::Ortho2DStatic,
        "Pacman should classify as Ortho2DStatic"
    );

    let smb3_rep = segmenter
        .segment_video_range(Path::new(SMB3_PATH), Some(1.0), Some(5.0))
        .expect("segment smb3");
    assert_eq!(
        smb3_rep.shots[0].camera_regime,
        CameraRegime::Ortho2DScroll,
        "SMB3 horizontal scroll should classify as Ortho2DScroll"
    );

    let raptor_rep = segmenter
        .segment_video_range(Path::new(RAPTOR_PATH), Some(1.0), Some(5.0))
        .expect("segment raptor");
    assert_eq!(
        raptor_rep.shots[0].camera_regime,
        CameraRegime::Ortho2DScroll,
        "Raptor vertical scroll should classify as Ortho2DScroll"
    );

    // 2. Ground-truth Precision / Recall benchmark across 26 hard cuts + 4 cross-dissolves + 3 lightning flashes
    // constructed from real frames of SMB3, Raptor, Pacman, and The Secret of NIMH.
    let f_pac = video_io::read_frames_scaled(Path::new(PACMAN_PATH), Some(0.5), Some(1.6), 128, 64)
        .expect("read pacman");
    let f_smb = video_io::read_frames_scaled(Path::new(SMB3_PATH), Some(1.5), Some(1.6), 128, 64)
        .expect("read smb3");
    let f_rap = video_io::read_frames_scaled(Path::new(RAPTOR_PATH), Some(1.5), Some(1.6), 128, 64)
        .expect("read raptor");
    let f_nimh_a =
        video_io::read_frames_scaled(Path::new(NIMH_PATH), Some(215.0), Some(1.6), 128, 64)
            .expect("read nimh a");
    let f_nimh_b =
        video_io::read_frames_scaled(Path::new(NIMH_PATH), Some(625.0), Some(1.6), 128, 64)
            .expect("read nimh b");

    let sources = [&f_pac, &f_smb, &f_rap, &f_nimh_a, &f_nimh_b];
    let mut reel: Vec<RawFrame> = Vec::new();
    let mut gt_boundaries: Vec<usize> = Vec::new();

    for seg_idx in 0..30 {
        let src = sources[seg_idx % sources.len()];
        let seg_len = 28usize;
        if seg_idx > 0 {
            if seg_idx % 7 == 0 {
                // 8-frame cross-dissolve transition
                let prev_src = sources[(seg_idx - 1) % sources.len()];
                let d_len = 8usize;
                let dissolve_mid = reel.len() + d_len / 2;
                gt_boundaries.push(dissolve_mid);
                for d in 0..d_len {
                    let alpha = (d + 1) as f32 / (d_len + 1) as f32;
                    let a = &prev_src[(seg_len - d_len + d).min(prev_src.len() - 1)];
                    let b = &src[d.min(src.len() - 1)];
                    let mut blended = RawFrame::blank(128, 64);
                    for i in 0..blended.rgb.len() {
                        blended.rgb[i] = ((1.0 - alpha) * a.rgb[i] as f32
                            + alpha * b.rgb[i] as f32)
                            .round() as u8;
                    }
                    reel.push(blended);
                }
            } else {
                // Hard cut
                gt_boundaries.push(reel.len());
            }
        }
        for k in 0..seg_len {
            let mut frame = src[k.min(src.len() - 1)].clone();
            // Inject 1-frame lightning flash inside 3 segments (must NOT trigger false cut)
            if (seg_idx == 3 || seg_idx == 11 || seg_idx == 19) && k == 9 {
                for px in &mut frame.rgb {
                    *px = px.saturating_add(210);
                }
            }
            reel.push(frame);
        }
    }

    let gt_rep = segmenter.segment_frames(&reel, 24.0, 0.0);
    let pred_boundaries: Vec<usize> = gt_rep.shots.iter().skip(1).map(|s| s.frame_start).collect();
    let (precision, recall, f1) =
        evaluate_boundary_precision_recall(&pred_boundaries, &gt_boundaries, 3);
    println!(
        "[R2.1 Ground-Truth Reel] transitions={}, detected={}, Precision={:.4}, Recall={:.4}, F1={:.4}",
        gt_boundaries.len(),
        gt_rep.shots.len().saturating_sub(1),
        precision,
        recall,
        f1
    );
    assert!(
        precision > 0.96,
        "Expected cut/dissolve precision > 0.96, got {precision:.4}"
    );
    assert!(
        recall > 0.94,
        "Expected cut/dissolve recall > 0.94, got {recall:.4}"
    );

    // 3. Full 10-minute (600s) streaming slice of 1080p The Secret of NIMH
    let nimh_10m = segmenter
        .segment_video_range(Path::new(NIMH_PATH), Some(120.0), Some(600.0))
        .expect("segment 10m slice of The Secret of NIMH");
    let count_static = nimh_10m
        .shots
        .iter()
        .filter(|s| s.camera_regime == CameraRegime::Ortho2DStatic)
        .count();
    let count_scroll = nimh_10m
        .shots
        .iter()
        .filter(|s| s.camera_regime == CameraRegime::Ortho2DScroll)
        .count();
    let count_multiplane = nimh_10m
        .shots
        .iter()
        .filter(|s| s.camera_regime == CameraRegime::MultiPlane25D)
        .count();
    let count_persp = nimh_10m
        .shots
        .iter()
        .filter(|s| s.camera_regime == CameraRegime::Perspective3D)
        .count();

    println!(
        "[R2.1 NIMH 10-min Slice] frames={}, elapsed={:.2}s, throughput={:.1} FPS, shots={} (Static={}, Scroll={}, MultiPlane25D={}, Persp3D={})",
        nimh_10m.total_frames,
        nimh_10m.elapsed_sec,
        nimh_10m.throughput_fps,
        nimh_10m.shots.len(),
        count_static,
        count_scroll,
        count_multiplane,
        count_persp
    );
    assert!(
        nimh_10m.throughput_fps > 180.0,
        "Expected streaming throughput > 180 FPS, got {:.1} FPS",
        nimh_10m.throughput_fps
    );
    assert!(nimh_10m.shots.len() >= 40, "Expected >= 40 shots in 10m slice");
    assert!(count_static > 0 && count_scroll > 0 && count_multiplane > 0);
}

#[test]
fn r2_2_multiplane_atlas_psnr_and_zero_ghosting() {
    // 1. Single-layer regression on SMB3 and Raptor
    let smb_frames =
        video_io::read_frames(Path::new(SMB3_PATH), Some(1.0), Some(2.0)).expect("smb3 frames");
    let smb_opts = MultiPlaneOptions {
        held_out_frames: vec![10, 20, 30],
        ..Default::default()
    };
    let smb_atlas = build_multiplane_atlas(&smb_frames, &smb_opts);
    let smb_psnr = [10usize, 20, 30]
        .iter()
        .map(|&fi| smb_atlas.evaluate_background_psnr(fi, &smb_frames[fi]))
        .sum::<f64>()
        / 3.0;
    let smb_ghost = smb_atlas.measure_ghost_contamination_ratio(&smb_frames);

    let rap_frames =
        video_io::read_frames(Path::new(RAPTOR_PATH), Some(1.0), Some(2.0)).expect("raptor frames");
    let rap_opts = MultiPlaneOptions {
        held_out_frames: vec![10, 20, 30],
        ..Default::default()
    };
    let rap_atlas = build_multiplane_atlas(&rap_frames, &rap_opts);
    let rap_psnr = [10usize, 20, 30]
        .iter()
        .map(|&fi| rap_atlas.evaluate_background_psnr(fi, &rap_frames[fi]))
        .sum::<f64>()
        / 3.0;
    let rap_ghost = rap_atlas.measure_ghost_contamination_ratio(&rap_frames);

    println!(
        "[R2.2 Tier-1 Regression] SMB3: layers={}, PSNR={:.2} dB, ghost={:.4} | Raptor: layers={}, PSNR={:.2} dB, ghost={:.4}",
        smb_atlas.num_layers(),
        smb_psnr,
        smb_ghost,
        rap_atlas.num_layers(),
        rap_psnr,
        rap_ghost
    );
    assert_eq!(smb_atlas.num_layers(), 1);
    assert!(smb_psnr > 32.0);
    assert!(smb_ghost < 0.01);
    assert!(rap_psnr > 32.0);
    assert!(rap_ghost < 0.01);

    // 2. Three contiguous shots from The Secret of NIMH (scaled to 320x176 for fast multi-band CFT)
    let nimh_shot_starts = [194.0, 226.0, 350.0];
    for (idx, &t0) in nimh_shot_starts.iter().enumerate() {
        let frames =
            video_io::read_frames_scaled(Path::new(NIMH_PATH), Some(t0), Some(1.5), 320, 176)
                .expect("read nimh shot");
        let held_out = vec![6, 14, 22];
        let opts = MultiPlaneOptions {
            max_layers: 4,
            held_out_frames: held_out.clone(),
            ..Default::default()
        };
        let atlas = build_multiplane_atlas(&frames, &opts);
        let mean_psnr = held_out
            .iter()
            .filter(|&&fi| fi < frames.len())
            .map(|&fi| atlas.evaluate_background_psnr(fi, &frames[fi]))
            .sum::<f64>()
            / held_out.len() as f64;
        let ghost = atlas.measure_ghost_contamination_ratio(&frames);
        println!(
            "[R2.2 NIMH Shot {} @ {:.1}s] layers={}, held-out PSNR={:.2} dB, ghost={:.4}",
            idx + 1,
            t0,
            atlas.num_layers(),
            mean_psnr,
            ghost
        );
        assert!(
            mean_psnr > 32.0,
            "Expected held-out background PSNR > 32 dB on NIMH shot {}, got {:.2} dB",
            idx + 1,
            mean_psnr
        );
        assert!(
            ghost < 0.01,
            "Expected zero ghost silhouettes (< 1%) on NIMH shot {}, got {:.4}",
            idx + 1,
            ghost
        );
    }
}

#[test]
fn r2_3_articulated_sub_holon_wavepackets_vs_single_box() {
    // 1. Tier-1 rigid sprites (Pacman, SMB3, Raptor) must degenerate to P=1 with IoU > 0.88 & LPIPS < 0.12
    let mut tier1_failures: Vec<String> = Vec::new();
    for (name, path) in [("Pacman", PACMAN_PATH), ("SMB3", SMB3_PATH), ("Raptor", RAPTOR_PATH)] {
        let frames = video_io::read_frames(Path::new(path), Some(1.0), Some(1.5)).expect("frames");
        let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
        let obs = extract_dominant_character_observations(&frames, &atlas);
        let rep = evaluate_vs_single_box_baseline(&obs);
        println!(
            "[R2.3 Tier-1 {name}] obs={}, P={}, views={}, IoU={:.4}, LPIPS={:.4}",
            obs.len(),
            rep.num_parts,
            rep.num_view_cells,
            rep.articulated_wavepacket_iou,
            rep.articulated_wavepacket_lpips
        );
        if rep.num_parts != 1 {
            tier1_failures.push(format!("{name} should degenerate to P=1 rigid holon (P={})", rep.num_parts));
        }
        if rep.articulated_wavepacket_iou <= 0.88 || rep.articulated_wavepacket_lpips >= 0.12 {
            tier1_failures.push(format!(
                "{name} reconstruction IoU={:.4} LPIPS={:.4}",
                rep.articulated_wavepacket_iou, rep.articulated_wavepacket_lpips
            ));
        }
    }

    // 2. Tier-2 articulated characters (Mrs. Brisby walking/turning with red cape & Jeremy the Crow flapping wings)
    let brisby_obs = build_brisby_cape_turn_sequence();
    let brisby_rep = evaluate_vs_single_box_baseline(&brisby_obs);
    println!(
        "[R2.3 Tier-2 Mrs. Brisby] P={}, views={} | Baseline Single-Box: IoU={:.4}, LPIPS={:.4} | Articulated Wavepacket: IoU={:.4}, LPIPS={:.4}",
        brisby_rep.num_parts,
        brisby_rep.num_view_cells,
        brisby_rep.baseline_single_box_iou,
        brisby_rep.baseline_single_box_lpips,
        brisby_rep.articulated_wavepacket_iou,
        brisby_rep.articulated_wavepacket_lpips
    );
    // P is chosen by posture-recurrence model selection; any choice must meet the quality gate.
    assert!(brisby_rep.num_parts >= 1 && brisby_rep.num_parts <= 8);
    assert!(brisby_rep.articulated_wavepacket_iou > 0.88);
    assert!(brisby_rep.articulated_wavepacket_lpips < 0.12);
    assert!(brisby_rep.articulated_wavepacket_iou > brisby_rep.baseline_single_box_iou + 0.08);
    assert!(brisby_rep.articulated_wavepacket_lpips < brisby_rep.baseline_single_box_lpips * 0.5);

    let jeremy_obs = build_jeremy_crow_wing_flap_sequence();
    let jeremy_rep = evaluate_vs_single_box_baseline(&jeremy_obs);
    println!(
        "[R2.3 Tier-2 Jeremy the Crow] P={}, views={} | Baseline Single-Box: IoU={:.4}, LPIPS={:.4} | Articulated Wavepacket: IoU={:.4}, LPIPS={:.4}",
        jeremy_rep.num_parts,
        jeremy_rep.num_view_cells,
        jeremy_rep.baseline_single_box_iou,
        jeremy_rep.baseline_single_box_lpips,
        jeremy_rep.articulated_wavepacket_iou,
        jeremy_rep.articulated_wavepacket_lpips
    );
    // P is chosen by posture-recurrence model selection; any choice must meet the quality gate.
    assert!(jeremy_rep.num_parts >= 1 && jeremy_rep.num_parts <= 8);
    assert!(jeremy_rep.articulated_wavepacket_iou > 0.88);
    assert!(jeremy_rep.articulated_wavepacket_lpips < 0.12);
    assert!(jeremy_rep.articulated_wavepacket_iou > jeremy_rep.baseline_single_box_iou + 0.08);
    assert!(jeremy_rep.articulated_wavepacket_lpips < jeremy_rep.baseline_single_box_lpips * 0.5);

    // 3. Explicit articulated decomposition must also reconstruct Tier-2 characters within the gate.
    for (name, seq) in [("Mrs. Brisby", &brisby_obs), ("Jeremy the Crow", &jeremy_obs)] {
        for p in [3usize, 4, 6] {
            let rep = evaluate_vs_single_box_baseline_with_parts(seq, Some(p));
            println!(
                "[R2.3 Tier-2 {name} forced P={}] IoU={:.4}, LPIPS={:.4}",
                rep.num_parts, rep.articulated_wavepacket_iou, rep.articulated_wavepacket_lpips
            );
            assert_eq!(rep.num_parts, p);
            assert!(rep.articulated_wavepacket_iou > 0.88);
            assert!(rep.articulated_wavepacket_lpips < 0.12);
            assert!(rep.articulated_wavepacket_iou > rep.baseline_single_box_iou);
        }
    }
    assert!(tier1_failures.is_empty(), "Tier-1 failures: {tier1_failures:?}");
}

fn build_brisby_cape_turn_sequence() -> Vec<CharacterObservation> {
    let mut seq = Vec::new();
    let (w, h) = (84usize, 76usize);
    for t in 0..16 {
        let phase = t as f64 * std::f64::consts::TAU / 16.0;
        let cape_swing = 0.45 * phase.sin();
        let stride = 5.0 * phase.cos();
        let mut mask = vec![false; w * h];
        let mut rgb = vec![0u8; w * h * 3];

        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let idx = y * w + x;

                // Head/ears (warm brown rodent)
                let dh = ((fx - 42.0) / 11.0).powi(2) + ((fy - 20.0) / 10.0).powi(2);
                // Red flowing cape (rotates with cape_swing)
                let dx_c = fx - (42.0 + 8.0 * cape_swing);
                let dy_c = fy - 44.0;
                let (s, c) = cape_swing.sin_cos();
                let rx_c = c * dx_c + s * dy_c;
                let ry_c = -s * dx_c + c * dy_c;
                let dc = (rx_c / 18.0).powi(2) + (ry_c / 15.0).powi(2);
                // Lower paws/skirt stride
                let dl = ((fx - (42.0 + stride)) / 12.0).powi(2) + ((fy - 60.0) / 9.0).powi(2);

                if dh <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[158, 118, 88]);
                } else if dc <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[212, 42, 48]);
                } else if dl <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[138, 102, 76]);
                }
            }
        }
        seq.push(CharacterObservation {
            frame_idx: t,
            bbox: Rect::new(
                100 + t as i32 * 2,
                80,
                100 + t as i32 * 2 + w as i32 - 1,
                80 + h as i32 - 1,
            ),
            mask,
            rgb,
            camera_offset: [0.0, 0.0],
            truncated: false,
        });
    }
    seq
}

fn build_jeremy_crow_wing_flap_sequence() -> Vec<CharacterObservation> {
    let mut seq = Vec::new();
    let (w, h) = (104usize, 80usize);
    for t in 0..16 {
        let phase = t as f64 * std::f64::consts::TAU / 16.0;
        let wing_angle = 0.65 * phase.sin();
        let mut mask = vec![false; w * h];
        let mut rgb = vec![0u8; w * h * 3];

        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let idx = y * w + x;

                // Crow torso (slate blue-charcoal)
                let db = ((fx - 52.0) / 13.0).powi(2) + ((fy - 42.0) / 16.0).powi(2);
                // Yellow beak / head
                let dk = ((fx - 64.0) / 9.0).powi(2) + ((fy - 26.0) / 7.0).powi(2);
                // Articulated flapping wing
                let wx = fx - 44.0;
                let wy = fy - (36.0 - 12.0 * wing_angle);
                let (s, c) = wing_angle.sin_cos();
                let rwx = c * wx + s * wy;
                let rwy = -s * wx + c * wy;
                let dw = (rwx / 24.0).powi(2) + (rwy / 10.0).powi(2);

                if dk <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[228, 186, 54]);
                } else if dw <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[64, 72, 98]);
                } else if db <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&[44, 48, 66]);
                }
            }
        }
        seq.push(CharacterObservation {
            frame_idx: t,
            bbox: Rect::new(
                140,
                60 + t as i32,
                140 + w as i32 - 1,
                60 + t as i32 + h as i32 - 1,
            ),
            mask,
            rgb,
            camera_offset: [0.0, 0.0],
            truncated: false,
        });
    }
    seq
}
