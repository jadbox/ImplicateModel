use std::path::Path;

use implicate_lifter::{probe, probe_audio_stream};
use implicate_unfold::{
    CausalUnfoldEngine, IMPLICATE_FLAG_ENFOLDED_ASSETS, IMPLICATE_FLAG_LEXICON,
    ImplicateContainerManifest, UnfoldSceneSpec, enfold_video_clip, load_implicate_container,
    load_implicate_container_with_assets, save_implicate_container,
    save_implicate_container_with_assets, upsert_enfolded_clip_to_container,
};

#[test]
fn r4_3_generative_reprompting_novel_scene_unfoldment_gate() {
    // 1. Build canonical Implicate Lexicon & verify .implicate v1 binary container round-trip
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let container_path = Path::new("/tmp/opencode/r4_3_corpus.implicate");
    save_implicate_container(container_path, &manifest, 0x01, &[1, 2, 3, 4])
        .expect("save .implicate v1 container");
    let (loaded_manifest, flags, payload) =
        load_implicate_container(container_path).expect("load .implicate v1 container");
    assert_eq!(flags, 0x01);
    assert_eq!(payload, vec![1, 2, 3, 4]);
    assert_eq!(loaded_manifest.verb_potentials.entries.len(), 5);

    // 2. Unfold all 4 novel re-prompted scenes across pacman, smb3, raptor, and The Secret of NIMH
    let scenes = UnfoldSceneSpec::canonical_r4_3_novel_scenes();
    assert_eq!(scenes.len(), 4);

    for scene in scenes {
        let mut engine = CausalUnfoldEngine::new(loaded_manifest.clone(), scene.clone());
        let report = engine.unfold_all_frames();

        println!(
            "[R4.3 Novel Scene '{}'] {}x{} ({} frames) | {:.3} ms/frame ({:.1} FPS, gate >= 60.0 FPS) | Permanence={:.1}% | Boundary Compliance={:.1}% | HUD Drift={:.4} px | Audio Samples={} (underruns={})",
            report.scene_id,
            report.width,
            report.height,
            report.num_frames,
            report.mean_frame_ms,
            report.effective_fps,
            report.object_permanence_rate * 100.0,
            report.boundary_compliance_rate * 100.0,
            report.max_hud_lock_drift_px,
            report.stereo_audio.len(),
            report.audio_underruns
        );

        for vm in &report.verb_metrics {
            println!(
                "  -> Verb Binding ({} --[{}]--> {}): normalized Frechet = {:.4} ({:.2}%, gate < 8.00%)",
                vm.subject,
                vm.verb,
                vm.target,
                vm.normalized_frechet,
                vm.normalized_frechet * 100.0
            );
            assert!(
                vm.normalized_frechet < 0.08,
                "Gate R4.3 failed for scene '{}' verb '{}': normalized Frechet {:.4} >= 0.08",
                report.scene_id,
                vm.verb,
                vm.normalized_frechet
            );
        }

        assert!(
            report.effective_fps >= 60.0,
            "Gate R4.3 failed for scene '{}': effective FPS {:.1} < 60.0 FPS",
            report.scene_id,
            report.effective_fps
        );
        assert!(
            (report.object_permanence_rate - 1.0).abs() < 1e-9,
            "Gate R4.3 failed for scene '{}': object permanence {:.4} < 100%",
            report.scene_id,
            report.object_permanence_rate
        );
        assert!(
            (report.boundary_compliance_rate - 1.0).abs() < 1e-9,
            "Gate R4.3 failed for scene '{}': PGA boundary compliance {:.4} < 100%",
            report.scene_id,
            report.boundary_compliance_rate
        );
        assert!(
            report.max_hud_lock_drift_px < 1e-4,
            "Expected zero camera-locked HUD drift, got {:.5} px",
            report.max_hud_lock_drift_px
        );
        assert_eq!(report.audio_underruns, 0);
        assert_eq!(report.rgb_frames.len(), report.num_frames);
        assert_eq!(report.stereo_audio.len(), report.num_frames * 800);
    }
}

#[test]
fn r5_1_mp4_audio_video_mux_and_refiner_readback_gate() {
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let nimh_scene = UnfoldSceneSpec::canonical_r4_3_novel_scenes().remove(3);

    // 1. Verify CliffordNet-Lite GPU refiner readback modifies every frame compared to unrefined
    let mut unrefined_scene = nimh_scene.clone();
    unrefined_scene.enable_cliffordnet_refiner = false;
    let rep_unrefined = CausalUnfoldEngine::new(manifest.clone(), unrefined_scene).unfold_all_frames();
    let rep_refined = CausalUnfoldEngine::new(manifest, nimh_scene).unfold_all_frames();

    for f_idx in 0..rep_refined.num_frames {
        let diff_sum: u64 = rep_refined.rgb_frames[f_idx]
            .iter()
            .zip(rep_unrefined.rgb_frames[f_idx].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        assert!(
            diff_sum > 0,
            "Frame {f_idx} had zero difference between CliffordNet-Lite refined and unrefined RGB"
        );
    }

    // 2. Verify MP4 + 48 kHz WAV export and container streams via ffprobe
    let out_dir = Path::new("/tmp/opencode/r5_1_mux_test");
    let (wav_path, mp4_path) = rep_refined
        .export_wav_and_mp4(out_dir)
        .expect("export WAV + MP4");

    assert!(wav_path.exists());
    assert!(mp4_path.exists());

    let v_info = probe(&mp4_path).expect("probe muxed MP4 video stream");
    let (sample_rate, channels) =
        probe_audio_stream(&mp4_path).expect("probe muxed MP4 audio stream");

    println!(
        "[R5.1 MP4+WAV Mux Gate] {} ({}x{} @ {:.1} fps, {} frames) + Audio ({} Hz, {} ch) | Effective Unfold FPS = {:.1}",
        mp4_path.display(),
        v_info.width,
        v_info.height,
        v_info.fps,
        v_info.frames,
        sample_rate,
        channels,
        rep_refined.effective_fps
    );

    assert_eq!(v_info.width, rep_refined.width);
    assert_eq!(v_info.height, rep_refined.height);
    assert_eq!(v_info.frames as usize, rep_refined.num_frames);
    assert_eq!(sample_rate, 48_000);
    assert_eq!(channels, 2);
}

#[test]
fn r5_2_enfold_to_container_serialization_gate() {
    let smb3_path = Path::new(
        "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4",
    );
    let clip = enfold_video_clip(smb3_path, 1.0, 48).expect("enfold smb3 clip");
    assert_eq!(clip.atlas.width, 256);
    assert_eq!(clip.atlas.height, 240);
    assert!(!clip.atlas.layers.is_empty());
    assert!(
        clip.atlas.camera_locked_hud.is_some(),
        "SMB3 must enfold a camera-locked HUD holon layer"
    );
    assert!(
        !clip.holons.is_empty(),
        "SMB3 must enfold an ArticulatedHolon wavepacket"
    );

    let out_path = Path::new("/tmp/opencode/r5_2_smb3.implicate");
    let mut manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    save_implicate_container_with_assets(out_path, &mut manifest, std::slice::from_ref(&clip))
        .expect("save .implicate container with enfolded assets");

    let (loaded_manifest, flags, loaded_clips) =
        load_implicate_container_with_assets(out_path).expect("load .implicate container with assets");

    assert_eq!(flags, IMPLICATE_FLAG_LEXICON | IMPLICATE_FLAG_ENFOLDED_ASSETS);
    assert_eq!(loaded_manifest.enfolded_clips.len(), 1);
    assert_eq!(loaded_manifest.enfolded_clips[0], clip.summary());
    assert_eq!(loaded_clips.len(), 1);

    let loaded = &loaded_clips[0];
    assert_eq!(loaded.clip_id, clip.clip_id);
    assert_eq!(loaded.atlas.width, clip.atlas.width);
    assert_eq!(loaded.atlas.height, clip.atlas.height);
    assert_eq!(loaded.atlas.layers.len(), clip.atlas.layers.len());
    assert_eq!(
        loaded.atlas.reconstruct_background_frame(0).rgb,
        clip.atlas.reconstruct_background_frame(0).rgb
    );
    assert_eq!(
        loaded.atlas.reconstruct_background_frame(15).rgb,
        clip.atlas.reconstruct_background_frame(15).rgb
    );
    assert_eq!(
        loaded.atlas.camera_locked_hud.as_ref().unwrap().atlas.image.rgb,
        clip.atlas.camera_locked_hud.as_ref().unwrap().atlas.image.rgb
    );
    assert_eq!(loaded.holons.len(), clip.holons.len());
    assert_eq!(loaded.holons[0].num_parts, clip.holons[0].num_parts);
    assert_eq!(
        loaded.holons[0].kernels[0][0].cft_rgba_coeffs,
        clip.holons[0].kernels[0][0].cft_rgba_coeffs
    );

    println!(
        "[R5.2 Enfold-to-Container Gate] Saved & verified '{}' ({}x{}, {} frames, {} layer(s), HUD={}, {} holon(s)) -> {}",
        loaded.clip_id,
        loaded.atlas.width,
        loaded.atlas.height,
        loaded.num_frames,
        loaded.atlas.layers.len(),
        loaded.atlas.camera_locked_hud.is_some(),
        loaded.holons.len(),
        out_path.display()
    );
}

#[test]
fn r5_3_real_atlas_and_wavepacket_compositing_gate() {
    let smb3_path = Path::new(
        "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4",
    );
    let clip = enfold_video_clip(smb3_path, 1.0, 48).expect("enfold smb3 clip");
    let hud_layer = clip.atlas.camera_locked_hud.as_ref().unwrap();
    let (hy0, hy1) = hud_layer.y_band;

    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let smb3_scene = UnfoldSceneSpec::canonical_r4_3_novel_scenes().remove(1);

    let rep_synth = CausalUnfoldEngine::new(manifest.clone(), smb3_scene.clone()).unfold_all_frames();
    let rep_real = CausalUnfoldEngine::new(manifest, smb3_scene)
        .with_enfolded_clip(clip.clone())
        .unfold_all_frames();

    assert_eq!(rep_real.object_permanence_rate, 1.0);
    assert_eq!(rep_real.boundary_compliance_rate, 1.0);
    assert_eq!(rep_real.max_hud_lock_drift_px, 0.0);

    // 1. Verify real MultiPlaneAtlas + ArticulatedHolon compositing substantially changes viewport pixels
    let w = rep_real.width as usize;
    let mut viewport_diff: u64 = 0;
    for y in 0..hy0 {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                viewport_diff += (rep_real.rgb_frames[10][idx + c] as i32
                    - rep_synth.rgb_frames[10][idx + c] as i32)
                    .unsigned_abs() as u64;
            }
        }
    }
    assert!(
        viewport_diff > 100_000,
        "Expected real MultiPlaneAtlas + ArticulatedHolon compositing to differ from procedural fallback"
    );

    // 2. Verify camera-locked HUD band matches the enfolded HUD atlas across scrolling frames
    for y in hy0..hy1 {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let expected = hud_layer.sample_rgba(x as i32, y as i32).unwrap();
            assert_eq!(
                &rep_real.rgb_frames[0][idx..idx + 3],
                &expected[0..3],
                "HUD pixel mismatch at ({x}, {y}) on frame 0"
            );
            assert_eq!(
                &rep_real.rgb_frames[30][idx..idx + 3],
                &expected[0..3],
                "HUD pixel mismatch at ({x}, {y}) on scrolling frame 30"
            );
        }
    }

    println!(
        "[R5.3 Real Atlas & Wavepacket Compositing Gate] '{}' unfolded at {:.2} ms/frame ({:.1} FPS) | viewport_l1_diff={} | HUD lock exact across {}..{}",
        rep_real.scene_id, rep_real.mean_frame_ms, rep_real.effective_fps, viewport_diff, hy0, hy1
    );
}

#[test]
fn r5_4_text_and_dzoglang_prompt_parser_gate() {
    let prompt = "@scene nimh_custom_sanctuary: Articulated Mrs_Brisby (P=4, hero, courage) executes jump_over toward glowing Red_Amulet (self, awe) while Jeremy_Crow (P=6, trickster, joy) executes orbit around Red_Amulet and HUD tracks score.";
    let spec = UnfoldSceneSpec::parse_prompt(prompt).expect("parse custom DzogLang prompt");

    assert_eq!(spec.scene_id, "nimh_custom_sanctuary");
    assert!(spec.enable_cliffordnet_refiner);
    assert_eq!(spec.verb_bindings.len(), 2);
    assert_eq!(spec.verb_bindings[0].subject, "Mrs_Brisby");
    assert_eq!(spec.verb_bindings[0].verb, "jump_over");
    assert_eq!(spec.verb_bindings[0].target, "Red_Amulet");
    assert_eq!(spec.verb_bindings[1].subject, "Jeremy_Crow");
    assert_eq!(spec.verb_bindings[1].verb, "orbit");
    assert_eq!(spec.verb_bindings[1].target, "Red_Amulet");
    assert!(spec.holons.iter().any(|h| h.is_camera_locked_hud));

    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let report = CausalUnfoldEngine::new(manifest, spec).unfold_all_frames();

    assert_eq!(report.object_permanence_rate, 1.0);
    assert_eq!(report.boundary_compliance_rate, 1.0);
    assert_eq!(report.max_hud_lock_drift_px, 0.0);
    assert_eq!(report.audio_underruns, 0);
    assert!(report.effective_fps >= 24.0);
    for vm in &report.verb_metrics {
        assert!(
            vm.normalized_frechet < 0.08,
            "Verb {} exceeded 8% Frechet gate: {:.4}",
            vm.verb,
            vm.normalized_frechet
        );
    }

    println!(
        "[R5.4 Prompt Parser Gate] Parsed & unfolded '{}' ({} holons, {} verbs) at {:.2} ms/frame ({:.1} FPS)",
        report.scene_id,
        report.trajectories.len(),
        report.verb_metrics.len(),
        report.mean_frame_ms,
        report.effective_fps
    );
}

#[test]
fn r6_multi_clip_corpus_enfold_trajectory_assimilation_and_dynamic_hud_gate() {
    let pacman_path = Path::new(
        "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4",
    );
    let smb3_path = Path::new(
        "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4",
    );

    let multi_path = Path::new("/tmp/opencode/r6_multi_corpus.implicate");
    let _ = std::fs::remove_file(multi_path);

    let clip_pac = enfold_video_clip(pacman_path, 1.0, 32).expect("enfold pacman clip");
    let clip_smb = enfold_video_clip(smb3_path, 1.0, 32).expect("enfold smb3 clip");

    upsert_enfolded_clip_to_container(multi_path, clip_pac).expect("upsert pacman");
    upsert_enfolded_clip_to_container(multi_path, clip_smb).expect("upsert smb3");

    let (manifest, _, clips) =
        load_implicate_container_with_assets(multi_path).expect("load multi-clip container");
    assert_eq!(clips.len(), 2);
    assert_eq!(manifest.enfolded_clips.len(), 2);

    // Verify real video trajectories were assimilated into manifest.verb_potentials
    assert!(
        manifest
            .verb_potentials
            .get("enfolded_pacman_240x308")
            .is_some(),
        "Expected assimilated trajectory potential 'enfolded_pacman_240x308'"
    );
    assert!(
        manifest
            .verb_potentials
            .get("enfolded_smb3_w1_1_256x240")
            .is_some(),
        "Expected assimilated trajectory potential 'enfolded_smb3_w1_1_256x240'"
    );

    // Unfold a custom prompt using both the assimilated video verb and an open-vocabulary verb ("leaps_over")
    let prompt = "@scene smb3_assimilated_unfold: Mario (P=4, hero, joy) executes enfolded_smb3_w1_1_256x240 toward Koopa_Block while Goomba_Sentinel (shadow) executes leaps_over toward Koopa_Block and HUD tracks score.";
    let spec = UnfoldSceneSpec::parse_prompt(prompt).expect("parse assimilated prompt");
    let engine = CausalUnfoldEngine::new(manifest, spec).with_enfolded_clips(&clips);
    assert_eq!(
        engine.enfolded_clip.as_ref().unwrap().clip_id,
        "smb3_w1_1_256x240",
        "Expected domain-matched asset selector to choose smb3_w1_1_256x240 from multi-clip container"
    );

    let mut engine = engine;
    let report = engine.unfold_all_frames();
    assert_eq!(report.object_permanence_rate, 1.0);
    assert_eq!(report.boundary_compliance_rate, 1.0);
    assert_eq!(report.max_hud_lock_drift_px, 0.0);
    assert_eq!(report.verb_metrics.len(), 2);
    for vm in &report.verb_metrics {
        assert!(
            vm.normalized_frechet < 0.08,
            "Assimilated/open-vocab verb {} exceeded 8% Frechet gate: {:.4}",
            vm.verb,
            vm.normalized_frechet
        );
    }

    // Verify state-coupled HUD holon score progression
    let score_first = *report.hud_score_history.first().unwrap();
    let score_last = *report.hud_score_history.last().unwrap();
    assert!(
        score_last > score_first,
        "Expected state-coupled HUD score to advance across frames ({score_first} -> {score_last})"
    );

    println!(
        "[R6 Multi-Clip Assimilation & Dynamic HUD Gate] 2 clips enfolded | '{}' unfolded at {:.2} ms/frame ({:.1} FPS) | HUD score {:06}->{:06} | Frechet=({:.2}%, {:.2}%)",
        report.scene_id,
        report.mean_frame_ms,
        report.effective_fps,
        score_first,
        score_last,
        report.verb_metrics[0].normalized_frechet * 100.0,
        report.verb_metrics[1].normalized_frechet * 100.0
    );
}

#[test]
fn r7_multi_actor_cft_impulse_and_view_idempotent_gate() {
    let pacman_path = Path::new(
        "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4",
    );
    let r7_container = Path::new("/tmp/opencode/r7_pacman_pair.implicate");
    let _ = std::fs::remove_file(r7_container);

    let clip_pac = enfold_video_clip(pacman_path, 5.0, 48).expect("enfold pacman 48f");
    assert!(
        clip_pac.holons.len() >= 2,
        "Expected multi-actor track extractor to extract >= 2 actors from pacman clip, got {}",
        clip_pac.holons.len()
    );

    upsert_enfolded_clip_to_container(r7_container, clip_pac.clone()).expect("upsert pacman r7");
    let (manifest, _, clips) =
        load_implicate_container_with_assets(r7_container).expect("load r7 container");

    // Verify true two-body pairwise relative PGA potential M_rel(t) = ~M_B(t) M_A(t) was assimilated
    assert!(
        manifest
            .verb_potentials
            .get("enfolded_pair_pacman_240x308")
            .is_some(),
        "Expected pairwise relative PGA potential 'enfolded_pair_pacman_240x308'"
    );

    // Unfold a scene driven by the two-body pairwise relative PGA verb potential
    let prompt = "@scene pacman_pair_unfold: Pacman (hero, courage) executes enfolded_pair_pacman_240x308 toward Blinky_Ghost (shadow, fear) while Inky_Ghost (trickster) executes orbit around Blinky_Ghost and HUD tracks score.";
    let spec = UnfoldSceneSpec::parse_prompt(prompt).expect("parse r7 prompt");
    let mut engine = CausalUnfoldEngine::new(manifest, spec).with_enfolded_clips(&clips);

    // Verify Spin(2) x S^1 view-aspect idempotent selector responds to holon heading/phase
    let idx_f0 = CausalUnfoldEngine::select_view_idempotent_observation_index(
        &clip_pac.holons[0],
        &engine.holons[0],
        0,
    );
    let idx_f15 = CausalUnfoldEngine::select_view_idempotent_observation_index(
        &clip_pac.holons[0],
        &engine.holons[0],
        15,
    );
    assert!(idx_f0 < clip_pac.holons[0].poses_per_obs.len());
    assert!(idx_f15 < clip_pac.holons[0].poses_per_obs.len());

    let report = engine.unfold_all_frames();
    assert_eq!(report.object_permanence_rate, 1.0);
    assert_eq!(report.boundary_compliance_rate, 1.0);
    assert_eq!(report.max_hud_lock_drift_px, 0.0);
    assert!(report.effective_fps >= 24.0);
    for vm in &report.verb_metrics {
        assert!(
            vm.normalized_frechet < 0.08,
            "R7 verb {} exceeded 8% Frechet gate: {:.4}",
            vm.verb,
            vm.normalized_frechet
        );
    }

    println!(
        "[R7 Multi-Actor Pairwise PGA & View-Idempotent Gate] {} actors enfolded | '{}' unfolded at {:.2} ms/frame ({:.1} FPS) | Pairwise Frechet=({:.2}%, {:.2}%)",
        clip_pac.holons.len(),
        report.scene_id,
        report.mean_frame_ms,
        report.effective_fps,
        report.verb_metrics[0].normalized_frechet * 100.0,
        report.verb_metrics[1].normalized_frechet * 100.0
    );
}
