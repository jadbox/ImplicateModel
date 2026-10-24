use implicate_lifter::{
    CharacterObservation,
    articulated_wavepacket::{fit_articulated_holon, render_articulated_holon_observation, mask_iou, choose_part_and_view_counts},
};

fn test_obs(name: &str, obs: &[CharacterObservation]) {
    let (p, v) = choose_part_and_view_counts(obs);
    let m1 = fit_articulated_holon(obs, Some(1));
    let iou1: f64 = obs.iter().enumerate().map(|(i, o)| mask_iou(&o.mask, &render_articulated_holon_observation(&m1, o, i).0)).sum::<f64>() / obs.len() as f64;
    println!("{name:<20}: P_chosen={} views={} | P=1 IoU={:.4}", p, v, iou1);
}

fn build_brisby() -> Vec<CharacterObservation> {
    let mut seq = Vec::new();
    let (w, h) = (84usize, 76usize);
    for t in 0..16 {
        let phase = t as f64 * std::f64::consts::TAU / 16.0;
        let cape_swing = 0.45 * phase.sin();
        let _stride = 5.0 * phase.cos();
        let mut mask = vec![false; w * h];
        let mut rgb = vec![0u8; w * h * 3];

        for y in 0..h {
            for x in 0..w {
                let fx = x as f64;
                let fy = y as f64;
                let idx = y * w + x;

                let dx_h = (fx - 42.0) / 10.0;
                let dy_h = (fy - 20.0) / 10.0;
                if dx_h * dx_h + dy_h * dy_h <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3] = 168;
                    rgb[idx * 3 + 1] = 112;
                    rgb[idx * 3 + 2] = 72;
                }
                let dx_t = (fx - 42.0) / 11.0;
                let dy_t = (fy - 37.0) / 12.0;
                if dx_t * dx_t + dy_t * dy_t <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3] = 138;
                    rgb[idx * 3 + 1] = 88;
                    rgb[idx * 3 + 2] = 52;
                }
                let cape_cx = 40.0 - 14.0 * cape_swing.cos();
                let cape_cy = 38.0 + 16.0;
                let dx_c = (fx - cape_cx) / (14.0 * (1.0 + 0.3 * cape_swing.abs()));
                let dy_c = (fy - cape_cy) / 16.0;
                if dx_c * dx_c + dy_c * dy_c <= 1.0 {
                    mask[idx] = true;
                    rgb[idx * 3] = 208;
                    rgb[idx * 3 + 1] = 28;
                    rgb[idx * 3 + 2] = 36;
                }
            }
        }
        seq.push(CharacterObservation {
            frame_idx: t,
            bbox: implicate_lifter::Rect::new(20, 10, 20 + w as i32 - 1, 10 + h as i32 - 1),
            mask,
            rgb,
            camera_offset: [0.0, 0.0],
            truncated: false,
            line_of_action: None,
        });
    }
    seq
}

fn main() {
    let smb3_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";
    let pacman_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";
    let raptor_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";

    for (name, path) in [("Pacman", pacman_path), ("SMB3", smb3_path), ("Raptor", raptor_path)] {
        let frames = implicate_lifter::video_io::read_frames(std::path::Path::new(path), Some(1.0), Some(1.5)).expect("frames");
        let atlas = implicate_lifter::build_multiplane_atlas(&frames, &implicate_lifter::MultiPlaneOptions::default());
        let obs = implicate_lifter::extract_dominant_character_observations(&frames, &atlas);
        test_obs(name, &obs);
    }
    let brisby = build_brisby();
    test_obs("Mrs. Brisby (cape)", &brisby);
}
