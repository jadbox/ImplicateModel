use std::path::Path;

use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, decompose_observation_dual_product,
    extract_character_observation_tracks, video_io,
};

const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";

#[test]
fn r10_2_clifford_dual_decomposition_empirical_gate() {
    // 1. Empirical test on real Super Mario Bros. 3 gameplay frames
    let smb_frames = video_io::read_frames(Path::new(SMB3_PATH), Some(25.0), Some(27.0))
        .expect("read smb3 gameplay frames");
    assert!(smb_frames.len() >= 30, "Expected >= 30 frames from SMB3");

    let atlas = build_multiplane_atlas(&smb_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&smb_frames, &atlas, 1);
    assert!(!tracks.is_empty(), "Expected Mario track to be extracted");

    let mario_obs = &tracks[0];
    assert!(mario_obs.len() >= 20, "Expected Mario track to have >= 20 observations, got {}", mario_obs.len());

    let mut total_ink_blades = 0usize;
    let mut total_cel_fills = 0usize;

    for (i, obs) in mario_obs.iter().enumerate() {
        let decomp = decompose_observation_dual_product(obs);

        // Properties of Clifford Dual Product Decomposition:
        // 1. Invariant dimensions
        assert_eq!(decomp.width, obs.width());
        assert_eq!(decomp.height, obs.height());

        // 2. Statistical noise floor derived without magic constants
        assert!(decomp.ink_blade_threshold > 0.05,
            "Ink blade noise floor ({:.4}) must exceed minimum floor", decomp.ink_blade_threshold);

        // 3. Ink contour blades (u ∧ ∇²u) and cel paint fills (u · ∇²u) must have zero overlap
        let mut obs_ink = 0usize;
        let mut obs_cel = 0usize;
        for idx in 0..obs.width() * obs.height() {
            assert!(!(decomp.ink_blade_mask[idx] && decomp.cel_fill_mask[idx]),
                "Ink blade and cel fill masks must be mutually exclusive at pixel {}", idx);
            if decomp.ink_blade_mask[idx] {
                obs_ink += 1;
            }
            if decomp.cel_fill_mask[idx] {
                obs_cel += 1;
            }
        }

        assert!(obs_ink > 0, "Observation {} must have non-zero ink contour blade pixels", i);
        assert!(obs_cel > 0, "Observation {} must have non-zero cel paint fill pixels", i);

        total_ink_blades += obs_ink;
        total_cel_fills += obs_cel;
    }

    println!(
        "[R10.2 Empirical Dual Product Gate - SMB3] Mario obs={}, mean ink blade px/obs={:.1}, mean cel fill px/obs={:.1}",
        mario_obs.len(),
        total_ink_blades as f64 / mario_obs.len() as f64,
        total_cel_fills as f64 / mario_obs.len() as f64
    );
    assert!(total_ink_blades > 0);
    assert!(total_cel_fills > total_ink_blades, "Cel fill volume must exceed perimeter ink blade area");
}

#[test]
fn r10_2_line_of_action_bivector_and_spinal_curvature_gate() {
    let smb_frames = video_io::read_frames(Path::new(SMB3_PATH), Some(25.0), Some(27.0))
        .expect("read smb3 gameplay frames");
    let atlas = build_multiplane_atlas(&smb_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&smb_frames, &atlas, 1);
    assert!(!tracks.is_empty());

    let mario_obs = &tracks[0];
    let mut valid_pga_lines = 0usize;
    let mut curvature_variance = Vec::new();

    for obs in mario_obs {
        let loa = obs.line_of_action().expect("LineOfAction extraction");

        // 1. Plücker Line condition in 3D PGA: d · m = 0
        if loa.spine_bivector.is_valid_line(1e-4) {
            valid_pga_lines += 1;
        }

        // 2. Physical inertia moments
        assert!(loa.spread_parallel > 0.0);
        assert!(loa.spread_perpendicular > 0.0);
        assert!(loa.total_mass > 0.0);

        curvature_variance.push(loa.curvature);
    }

    assert_eq!(
        valid_pga_lines,
        mario_obs.len(),
        "100% of observations must produce exact, valid 3D PGA Line of Action bivectors L_spine"
    );

    // Compute dynamic curvature range
    let min_k = curvature_variance.iter().copied().fold(f64::INFINITY, f64::min);
    let max_k = curvature_variance.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "[R10.2 Line of Action Gate - SMB3] Valid PGA lines: {}/{}, Curvature range: [{:.5}, {:.5}]",
        valid_pga_lines,
        mario_obs.len(),
        min_k,
        max_k
    );
}

#[test]
fn r10_2_visual_contact_sheet_generation() {
    let smb_frames = video_io::read_frames(Path::new(SMB3_PATH), Some(25.0), Some(26.0))
        .expect("read smb3 frames");
    let atlas = build_multiplane_atlas(&smb_frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&smb_frames, &atlas, 1);
    assert!(!tracks.is_empty());

    let obs = &tracks[0][tracks[0].len() / 2];
    let decomp = decompose_observation_dual_product(obs);
    let loa = obs.line_of_action().expect("LineOfAction");

    let w = obs.width();
    let h = obs.height();

    // Create a 4-panel contact sheet:
    // Panel 1: Original RGB crop
    // Panel 2: Cel Paint Fill coherence (u · ∇²u)
    // Panel 3: Animated film ink contour blade (u ∧ ∇²u)
    // Panel 4: Line of Action spine overlay
    let padding = 4usize;
    let total_w = (w * 4) + (padding * 5);
    let total_h = h + (padding * 2);

    let mut sheet_rgb = vec![30u8; total_w * total_h * 3]; // dark gray background

    let blit_panel = |sheet: &mut [u8], panel_x: usize, panel_rgb: &[u8], pw: usize, ph: usize| {
        for y in 0..ph {
            for x in 0..pw {
                let sx = panel_x + x;
                let sy = padding + y;
                let s_idx = (sy * total_w + sx) * 3;
                let p_idx = (y * pw + x) * 3;
                sheet[s_idx..s_idx + 3].copy_from_slice(&panel_rgb[p_idx..p_idx + 3]);
            }
        }
    };

    // Panel 1: Original RGB
    let mut p1_rgb = obs.rgb.clone();
    for i in 0..w * h {
        if !obs.mask[i] {
            p1_rgb[i * 3] = 15;
            p1_rgb[i * 3 + 1] = 15;
            p1_rgb[i * 3 + 2] = 20;
        }
    }
    blit_panel(&mut sheet_rgb, padding, &p1_rgb, w, h);

    // Panel 2: Cel Paint Fill
    let mut p2_rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        if decomp.cel_fill_mask[i] {
            let v = (decomp.cel_fill[i] * 255.0).clamp(0.0, 255.0) as u8;
            p2_rgb[i * 3] = v;
            p2_rgb[i * 3 + 1] = (v as f32 * 0.7) as u8;
            p2_rgb[i * 3 + 2] = (v as f32 * 0.3) as u8;
        } else {
            p2_rgb[i * 3] = 15;
            p2_rgb[i * 3 + 1] = 15;
            p2_rgb[i * 3 + 2] = 20;
        }
    }
    blit_panel(&mut sheet_rgb, padding * 2 + w, &p2_rgb, w, h);

    // Panel 3: Ink Contour Blade
    let mut p3_rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        if decomp.ink_blade_mask[i] {
            // Bright cyan/white ink blade
            p3_rgb[i * 3] = 60;
            p3_rgb[i * 3 + 1] = 240;
            p3_rgb[i * 3 + 2] = 255;
        } else {
            let mag = (decomp.ink_contour[i] * 255.0).clamp(0.0, 255.0) as u8;
            p3_rgb[i * 3] = mag / 4;
            p3_rgb[i * 3 + 1] = mag / 4;
            p3_rgb[i * 3 + 2] = mag / 3;
        }
    }
    blit_panel(&mut sheet_rgb, padding * 3 + w * 2, &p3_rgb, w, h);

    // Panel 4: Line of Action spine line overlay
    let mut p4_rgb = p1_rgb.clone();
    let cx = loa.center_of_mass[0];
    let cy = loa.center_of_mass[1];
    let dx = loa.direction[0];
    let dy = loa.direction[1];

    // Draw spine line across character extent
    for step in -30..=30 {
        let lx = (cx + step as f64 * 0.5 * dx).round() as i32;
        let ly = (cy + step as f64 * 0.5 * dy).round() as i32;
        if lx >= 0 && ly >= 0 && lx < w as i32 && ly < h as i32 {
            let idx = (ly as usize * w + lx as usize) * 3;
            // Yellow spine axis
            p4_rgb[idx] = 255;
            p4_rgb[idx + 1] = 230;
            p4_rgb[idx + 2] = 30;
        }
    }
    // Draw center of mass (magenta 3x3 dot)
    let icx = cx.round() as i32;
    let icy = cy.round() as i32;
    for dy_dot in -1..=1 {
        for dx_dot in -1..=1 {
            let px = icx + dx_dot;
            let py = icy + dy_dot;
            if px >= 0 && py >= 0 && px < w as i32 && py < h as i32 {
                let idx = (py as usize * w + px as usize) * 3;
                p4_rgb[idx] = 255;
                p4_rgb[idx + 1] = 40;
                p4_rgb[idx + 2] = 220;
            }
        }
    }
    blit_panel(&mut sheet_rgb, padding * 4 + w * 3, &p4_rgb, w, h);

    let out_path = Path::new("/tmp/implicate_r10_2_contact_sheet.png");
    video_io::write_png(out_path, total_w as u32, total_h as u32, &sheet_rgb)
        .expect("save contact sheet");

    println!(
        "[R10.2 Visual Contact Sheet Generated] Path: {:?}, Dimensions: {}x{}, Mario bbox={:?}, LOA: angle={:.2}rad, dir=({:.2},{:.2}), curv={:.4}",
        out_path, total_w, total_h, obs.bbox, loa.angle_rad, loa.direction[0], loa.direction[1], loa.curvature
    );
    assert!(out_path.exists());
}
