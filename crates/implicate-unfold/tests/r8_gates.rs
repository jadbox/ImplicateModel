//! Automated visual fidelity, artifact elimination, and continuous camera regression tests (Track R8).

use std::path::Path;

use implicate_unfold::{
    enfold_video_clip, CausalUnfoldEngine, ImplicateContainerManifest, UnfoldSceneSpec,
};

const PACMAN_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";

#[test]
fn r8_1_static_camera_arena_preservation_gate() {
    let pac_path = Path::new(PACMAN_PATH);
    let clip = enfold_video_clip(pac_path, 1.0, 48).expect("enfold pacman");
    assert_eq!(
        clip.atlas.active_y_range,
        (0, clip.atlas.height),
        "Static camera must preserve full vertical arena (0, H) without HUD band truncation"
    );
    assert!(
        clip.atlas.camera_locked_hud.is_none(),
        "Static camera video must not create artificial camera-locked HUD layers"
    );
    assert!(
        clip.holons.len() >= 2,
        "Enfoldment must capture multiple actors across the full playfield"
    );
}

#[test]
fn r8_2_smb3_ground_boundary_and_hud_realignment_gate() {
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let smb3_scene = UnfoldSceneSpec::canonical_r4_3_novel_scenes().remove(1);
    assert_eq!(smb3_scene.scene_id, "smb3_novel_vault_and_patrol");

    let mut engine = CausalUnfoldEngine::new(manifest, smb3_scene);
    let report = engine.unfold_all_frames();

    assert_eq!(report.boundary_compliance_rate, 1.0);
    assert_eq!(report.max_hud_lock_drift_px, 0.0);
    assert!(report.effective_fps >= 60.0);

    // Verify actor motion exists across frames (no static frozen frame)
    let w = report.width as usize;
    let h = report.height as usize;
    let mut frame_diff: u64 = 0;
    for idx in 0..(w * h * 3) {
        frame_diff += (report.rgb_frames[0][idx] as i32 - report.rgb_frames[30][idx] as i32)
            .unsigned_abs() as u64;
    }
    assert!(
        frame_diff > 50_000,
        "Expected substantial visual animation across frames"
    );
}

#[test]
fn r8_3_semantic_role_matching_and_energy_drone_gate() {
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let raptor_scene = UnfoldSceneSpec::canonical_r4_3_novel_scenes().remove(2);
    assert_eq!(raptor_scene.scene_id, "raptor_novel_plasma_intercept");

    let mut engine = CausalUnfoldEngine::new(manifest, raptor_scene);
    let report = engine.unfold_all_frames();

    assert_eq!(report.object_permanence_rate, 1.0);
    assert!(report.verb_metrics[0].normalized_frechet < 0.08);
    assert!(report.verb_metrics[1].normalized_frechet < 0.08);
}

#[test]
fn r8_5_end_to_end_visual_fidelity_gate() {
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    let scenes = UnfoldSceneSpec::canonical_r4_3_novel_scenes();
    assert_eq!(scenes.len(), 4);

    for scene in scenes {
        let sid = scene.scene_id.clone();
        let mut engine = CausalUnfoldEngine::new(manifest.clone(), scene);
        let rep = engine.unfold_all_frames();
        assert_eq!(rep.object_permanence_rate, 1.0, "{sid} permanence");
        assert_eq!(rep.boundary_compliance_rate, 1.0, "{sid} boundary");
        assert_eq!(rep.audio_underruns, 0, "{sid} audio underruns");
        assert!(rep.effective_fps >= 60.0, "{sid} FPS = {:.1}", rep.effective_fps);
    }
}
