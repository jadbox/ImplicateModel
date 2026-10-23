use implicate_field::{
    ArticulatedFieldHolon, CollisionImpulseEvent, FieldGpuContext, GpuWaveMatterLattice,
    LatticeParams,
};
use implicate_math::OrientedLine2;
use implicate_rolpa::{
    AcousticRolpaProjector, DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS,
    estimate_fundamental_frequency_hz,
};

#[test]
fn r4_2_synchronized_acoustic_rolpa_projection_gate() {
    let ctx = FieldGpuContext::shared();
    let params = LatticeParams {
        width: 128,
        height: 128,
        dt: 0.15,
        ..LatticeParams::default()
    };
    let mut lattice = GpuWaveMatterLattice::new(ctx, params);
    let mut projector = AcousticRolpaProjector::new(DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS);

    // 1. Verify material stiffness \sqrt{K/\rho} -> modal fundamental frequency f_0 = 440\sqrt{K/\rho}
    //    and sub-millisecond (< 1.0 ms) audio-visual impact synchronization
    let test_materials = [
        ("soft_wood", 0.75f32, 330.0f32, 32.0f32, 0.0040f64),
        ("brass_block", 1.00f32, 440.0f32, 64.0f32, 0.0085f64),
        ("crystal_amulet", 1.50f32, 660.0f32, 96.0f32, 0.0120f64),
    ];

    let mut max_sync_err_ms = 0.0f64;

    for (idx, &(mat_name, c_speed, expected_f0, contact_x, exact_impact_t)) in
        test_materials.iter().enumerate()
    {
        let mut mat_proj = AcousticRolpaProjector::new(DEFAULT_SAMPLE_RATE, DEFAULT_VIDEO_FPS);
        let ev = CollisionImpulseEvent {
            holon_id: (idx + 1) as u32,
            boundary_idx: 0,
            sim_time: exact_impact_t,
            frame_idx: 0,
            substep_idx: 0,
            contact_pos: [contact_x, 32.0],
            normal: [0.0, 1.0],
            impulse_magnitude: 2.5,
            wave_speed_c: c_speed,
        };

        // Synthesize 4 frames (3200 samples @ 48 kHz = 66.7 ms) starting at t = 0.0
        let mut stream = Vec::with_capacity(3200);
        for f in 0..4 {
            let t_frame = (f as f64) / (DEFAULT_VIDEO_FPS as f64);
            let ev_slice = if f == 0 {
                std::slice::from_ref(&ev)
            } else {
                &[]
            };
            let block = mat_proj.synthesize_frame(t_frame, [0.0, 0.0], ev_slice, 64.0, 64.0);
            let consumed = mat_proj.consume_audio_block(block.len());
            stream.extend_from_slice(&consumed);
        }
        assert_eq!(mat_proj.underrun_count, 0);

        // Detect first sample index where |L| + |R| exceeds threshold 0.02
        let onset_sample = stream
            .iter()
            .position(|s| s[0].abs() + s[1].abs() > 0.02)
            .expect("expected acoustic modal onset");
        let detected_onset_t = (onset_sample as f64) / (DEFAULT_SAMPLE_RATE as f64);
        let sync_err_ms = (detected_onset_t - exact_impact_t).abs() * 1000.0;
        max_sync_err_ms = max_sync_err_ms.max(sync_err_ms);

        // Estimate fundamental frequency over 1024 samples post-onset
        let est_f0 = estimate_fundamental_frequency_hz(
            &stream[onset_sample..(onset_sample + 1024)],
            DEFAULT_SAMPLE_RATE,
        );
        let freq_err_pct = ((est_f0 - expected_f0).abs() / expected_f0) * 100.0;

        // Check stereo panning: contact_x = 32 is Left of camera (64), contact_x = 96 is Right
        let energy_l: f32 = stream[onset_sample..(onset_sample + 400)]
            .iter()
            .map(|s| s[0] * s[0])
            .sum();
        let energy_r: f32 = stream[onset_sample..(onset_sample + 400)]
            .iter()
            .map(|s| s[1] * s[1])
            .sum();

        println!(
            "[R4.2 Material '{}'] c=sqrt(K/rho)={:.2} | f0_expected={:.1} Hz, f0_measured={:.1} Hz (err={:.2}%) | AV Sync Error={:.4} ms | Stereo (E_L={:.3}, E_R={:.3})",
            mat_name, c_speed, expected_f0, est_f0, freq_err_pct, sync_err_ms, energy_l, energy_r
        );

        assert!(
            sync_err_ms < 1.0,
            "Gate R4.2 failed for {}: AV sync error {:.4} ms >= 1.0 ms",
            mat_name,
            sync_err_ms
        );
        assert!(
            freq_err_pct < 3.5,
            "Expected modal frequency {:.1} Hz to match 440*sqrt(K/rho) = {:.1} Hz within 3.5%",
            est_f0,
            expected_f0
        );
        if contact_x < 60.0 {
            assert!(energy_l > energy_r, "Expected Left impact to pan Left");
        } else if contact_x > 68.0 {
            assert!(energy_r > energy_l, "Expected Right impact to pan Right");
        }
    }

    // 2. End-to-End Coupled Lattice + PGA Boundary Collision + 48 kHz Stream (180 frames = 3.0s, 144,000 samples)
    let mut holons = vec![
        ArticulatedFieldHolon::new(1, "Mario_Shell", [36.0, 24.0], [0.0, -28.0], -1.57, 3.0, 1.0, 1.44),
        ArticulatedFieldHolon::new(2, "Amulet_Bell", [92.0, 28.0], [0.0, -32.0], -1.57, 3.0, 1.0, 2.25),
    ];
    let external_accels = vec![[0.0f32, -15.0], [0.0f32, -15.0]]; // downward gravity toward floor blade
    let boundaries = vec![OrientedLine2::solid_through([0.0, 16.0], [0.0, 1.0])];

    let total_frames = 180u64;
    let mut all_samples = Vec::with_capacity((total_frames as usize) * 800);
    let mut total_collisions = 0usize;

    for f in 0..total_frames {
        let sim_t = (f as f64) / (DEFAULT_VIDEO_FPS as f64);
        let (ear_p, events) = lattice.step_coupled_frame(
            ctx,
            &mut holons,
            &external_accels,
            &boundaries,
            [56.0, 32.0],
            [72.0, 32.0],
            f,
            sim_t,
            2,
        );
        total_collisions += events.len();
        let frame_audio = projector.synthesize_frame(sim_t, ear_p, &events, 64.0, 64.0);
        assert_eq!(frame_audio.len(), 800);
        let consumed = projector.consume_audio_block(800);
        assert_eq!(consumed.len(), 800);
        all_samples.extend_from_slice(&consumed);
    }

    let wav_path = std::path::Path::new("/tmp/opencode/r4_2_synchronized_rolpa.wav");
    if let Some(parent) = wav_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    projector
        .write_stereo_wav(wav_path, &all_samples)
        .expect("write 48 kHz stereo WAV");

    println!(
        "[R4.2 48 kHz Real-Time Stream Summary] {} stereo samples synthesized across {} frames | PGA collisions sonified = {} | Max AV Sync Error = {:.4} ms (gate < 1.0000 ms) | Underruns = {} (gate == 0)",
        all_samples.len(),
        total_frames,
        total_collisions,
        max_sync_err_ms,
        projector.underrun_count
    );

    assert!(total_collisions >= 4, "Expected multiple PGA boundary bounces");
    assert_eq!(
        projector.underrun_count, 0,
        "Gate R4.2 failed: expected zero buffer underruns at 48 kHz"
    );
}
