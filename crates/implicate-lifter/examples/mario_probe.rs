use std::path::Path;
use implicate_lifter::{
    video_io, build_multiplane_atlas, MultiPlaneOptions, extract_dominant_character_observations,
};

const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";

fn print_obs_ascii(label: &str, obs: &implicate_lifter::CharacterObservation) {
    println!("=== {} ({}x{}, bbox={:?}, fi={}) ===", label, obs.width(), obs.height(), obs.bbox, obs.frame_idx);
    for y in 0..obs.height() {
        print!("  ");
        for x in 0..obs.width() {
            let i = y * obs.width() + x;
            if obs.mask[i] {
                let (r, g, b) = (obs.rgb[i * 3], obs.rgb[i * 3 + 1], obs.rgb[i * 3 + 2]);
                if r < 40 && g < 40 && b < 40 {
                    print!("K");
                } else if r > 220 && g > 180 && b > 140 {
                    print!("W");
                } else if r > 180 && g > 100 && b < 80 {
                    print!("O");
                } else if r > 140 && g < 90 {
                    print!("R");
                } else {
                    print!("B");
                }
            } else {
                print!(".");
            }
        }
        println!();
    }
}

fn main() {
    let smb_path = Path::new(SMB3_PATH);
    let w1_frames = video_io::read_frames(smb_path, Some(22.0), Some(2.5)).unwrap();
    let w1_atlas = build_multiplane_atlas(&w1_frames, &MultiPlaneOptions::default());

    // Let's inspect all tracks extracted from t = 22.0..24.5s
    let tracks = implicate_lifter::extract_character_observation_tracks(&w1_frames, &w1_atlas, 12);
    for (ti, tr) in tracks.iter().enumerate() {
        let f0 = tr.first().unwrap();
        let fl = tr.last().unwrap();
        let w0 = f0.world_center();
        let wl = fl.world_center();
        // Measure vertical bounce (min wy vs max wy) and horizontal drift
        let min_wy = tr.iter().map(|o| o.bbox.y0 as f64 + o.camera_offset[1]).fold(f64::INFINITY, f64::min);
        let max_wy = tr.iter().map(|o| o.bbox.y0 as f64 + o.camera_offset[1]).fold(f64::NEG_INFINITY, f64::max);
        let min_wx = tr.iter().map(|o| o.bbox.x0 as f64 + o.camera_offset[0]).fold(f64::INFINITY, f64::min);
        let max_wx = tr.iter().map(|o| o.bbox.x0 as f64 + o.camera_offset[0]).fold(f64::NEG_INFINITY, f64::max);
        println!(
            "Track {:02}: len={:02}, fi={:02}..{:02}, b0={:?} ({}x{}), world_start=({:.1},{:.1}), world_end=({:.1},{:.1}), wx_span={:.1}, wy_span={:.1}",
            ti, tr.len(), f0.frame_idx, fl.frame_idx, f0.bbox, f0.width(), f0.height(),
            w0[0], w0[1], wl[0], wl[1], max_wx - min_wx, max_wy - min_wy
        );
    }
}
