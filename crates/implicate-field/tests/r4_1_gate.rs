use std::time::Instant;

use implicate_field::{
    ArticulatedFieldHolon, FieldGpuContext, GpuWaveMatterLattice, LatticeParams, SubHolonLink,
};
use implicate_math::OrientedLine2;

#[test]
fn r4_1_unified_wave_matter_lattice_and_articulated_holon_gate() {
    let ctx = FieldGpuContext::shared();
    let adapter_info = ctx.adapter.get_info();
    println!(
        "[R4.1 GPU Adapter] {} ({:?})",
        adapter_info.name, adapter_info.backend
    );

    let params = LatticeParams {
        width: 256,
        height: 256,
        dt: 0.15,
        ..LatticeParams::default()
    };
    let mut lattice = GpuWaveMatterLattice::new(ctx, params);

    // 1. Construct articulated Mrs. Brisby (P=4: torso, head, compliant cape, feet)
    let mut brisby = ArticulatedFieldHolon::new(
        1,
        "Mrs_Brisby",
        [96.0, 128.0],
        [14.0, 2.5],
        0.15,
        4.5,
        1.1,
        1.4,
    );
    brisby.parts = vec![
        SubHolonLink::new_compliant("head", [1.5, 1.8], 0.0, 0.4, [0.8, 0.2, 0.5]),
        SubHolonLink::new_compliant("cape", [-2.2, 0.5], -0.1, 4.5, [-0.7, 0.3, 0.6]),
        SubHolonLink::new_compliant("left_arm", [0.8, 0.4], 0.2, 1.2, [0.6, 0.4, 0.7]),
        SubHolonLink::new_compliant("feet", [0.0, -2.0], 0.0, 0.8, [0.3, -0.8, 0.5]),
    ];

    // 2. Construct articulated Jeremy the Crow (P=6)
    let mut jeremy = ArticulatedFieldHolon::new(
        2,
        "Jeremy_Crow",
        [140.0, 152.0],
        [-10.0, 6.0],
        2.6,
        5.5,
        0.85,
        1.1,
    );
    jeremy.parts = vec![
        SubHolonLink::new_compliant("beak_head", [2.5, 1.5], 0.0, 0.5, [0.9, 0.1, 0.4]),
        SubHolonLink::new_compliant("left_wing", [0.0, 2.8], 0.35, 3.8, [0.0, 0.8, 0.6]),
        SubHolonLink::new_compliant("right_wing", [0.0, -2.8], -0.35, 3.8, [0.0, -0.8, 0.6]),
        SubHolonLink::new_compliant("tail_plumage", [-3.0, 0.0], 0.0, 4.2, [-0.8, 0.0, 0.6]),
        SubHolonLink::new_compliant("left_talon", [-0.5, -2.2], -0.1, 1.0, [0.2, -0.9, 0.4]),
        SubHolonLink::new_compliant("right_talon", [0.5, -2.2], 0.1, 1.0, [0.2, -0.9, 0.4]),
    ];

    // 3. Construct the glowing Red Amulet (strong chromatic luminescence source)
    let mut amulet = ArticulatedFieldHolon::new(
        3,
        "Red_Amulet",
        [104.0, 132.0],
        [0.0, 0.0],
        0.0,
        3.5,
        2.4,
        5.8,
    );
    amulet.luminescence_emission = [4.5, 1.2, 0.6];

    // 4. Construct Raptor Fighter & Plasma Bolt holons
    let raptor = ArticulatedFieldHolon::new(
        4,
        "Raptor_Fighter",
        [64.0, 64.0],
        [18.0, 0.0],
        0.0,
        4.0,
        1.8,
        3.6,
    );
    let mut plasma = ArticulatedFieldHolon::new(
        5,
        "Plasma_Bolt",
        [76.0, 64.0],
        [28.0, 0.0],
        0.0,
        2.5,
        0.4,
        2.2,
    );
    plasma.luminescence_emission = [1.0, 3.2, 4.8];

    let mut holons = vec![brisby, jeremy, amulet, raptor, plasma];
    let external_accels = vec![[0.0f32, 0.0]; holons.len()];
    let boundaries = vec![
        OrientedLine2::solid_through([16.0, 16.0], [0.0, 1.0]),
        OrientedLine2::solid_through([240.0, 16.0], [-1.0, 0.0]),
    ];

    // Step 12 frames so Doppler wakes, luminescence diffusion, and cape compliance develop
    for frame in 0..12 {
        let sim_t = (frame as f64) / 60.0;
        let _ = lattice.step_coupled_frame(
            ctx,
            &mut holons,
            &external_accels,
            &boundaries,
            [120.0, 128.0],
            [136.0, 128.0],
            frame,
            sim_t,
            2,
        );
    }

    // Verify 1: Compliant cape on Mrs. Brisby and wings on Jeremy deformed under lattice flow v(x,t)
    let cape_deflection = (holons[0].parts[1].theta_rel - holons[0].parts[1].rest_angle).abs();
    let wing_deflection = (holons[1].parts[1].theta_rel - holons[1].parts[1].rest_angle).abs();
    println!(
        "[R4.1 Compliant Sub-Holon Deformation] Brisby cape |dtheta| = {:.5} rad, Jeremy wing |dtheta| = {:.5} rad",
        cape_deflection, wing_deflection
    );
    assert!(
        cape_deflection > 1e-4 && wing_deflection > 1e-4,
        "Expected compliant sub-holon parts to deform with local lattice flow v(x,t)"
    );

    // Verify 2: Glowing Red Amulet illuminates Mrs. Brisby's 2.5D normal facets via \nabla L
    let max_facet_irr = holons[0]
        .parts
        .iter()
        .map(|p| p.facet_irradiance[0])
        .fold(0.0f32, f32::max);
    println!(
        "[R4.1 Chromatic Normal Facet Lighting] Mrs. Brisby peak facet irradiance from Red Amulet \\nabla L = {:.5}",
        max_facet_irr
    );
    assert!(
        max_facet_irr > 1e-3,
        "Expected glowing Red Amulet to illuminate nearby 2.5D normal facets via \\nabla L"
    );

    // Verify 3: Moving holon (Raptor / Brisby) sheds a physical Doppler compression/rarefaction wake in pressure p
    let state_grid = lattice.readback_state_grid(ctx);
    let macro_dt = params.dt * 2.0;
    let src_x = holons[3].position[0] - holons[3].velocity[0] * macro_dt;
    let rx = (src_x.round() as usize).clamp(6, 248);
    let ry = (holons[3].position[1].round() as usize).clamp(6, 248);
    let p_ahead = state_grid[ry * 256 + (rx + 3)][0];
    let p_behind = state_grid[ry * 256 + (rx - 3)][0];
    println!(
        "[R4.1 Doppler Wake Verification] Raptor p_ahead = {:+.5}, p_behind = {:+.5} (dipole delta = {:+.5})",
        p_ahead,
        p_behind,
        p_ahead - p_behind
    );
    assert!(
        p_ahead > p_behind,
        "Expected positive Doppler pressure dipole ahead of moving holon: p_ahead={} <= p_behind={}",
        p_ahead,
        p_behind
    );

    // Verify 4: Benchmark full lattice + articulated holon step + zero-readback GPU reduction (< 1.2 ms/frame gate)
    let bench_frames = 200u64;
    let t0 = Instant::now();
    for f in 0..bench_frames {
        let sim_t = (f as f64) / 60.0;
        let (ears, _events) = lattice.step_coupled_frame(
            ctx,
            &mut holons,
            &external_accels,
            &boundaries,
            [120.0, 128.0],
            [136.0, 128.0],
            f,
            sim_t,
            2,
        );
        std::hint::black_box(ears);
    }
    let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / (bench_frames as f64);
    let hz = 1000.0 / avg_ms;
    println!(
        "[R4.1 Coupled Wave-Matter + Articulated Holon + Zero-Readback GPU Latency] {:.4} ms/frame ({:.1} Hz, gate < 1.2000 ms / > 800 Hz)",
        avg_ms, hz
    );
    assert!(
        avg_ms < 1.2,
        "Gate R4.1 failed: coupled lattice + holon step latency {:.4} ms >= 1.2 ms",
        avg_ms
    );
}
