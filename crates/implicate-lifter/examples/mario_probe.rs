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
    let frames = video_io::read_frames(smb_path, Some(1.0), Some(1.5)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let mario_obs = implicate_lifter::extract_dominant_character_observations(&frames, &atlas);
    println!("Total mario obs: {}", mario_obs.len());
    for (i, obs) in mario_obs.iter().take(5).enumerate() {
        println!(
            "obs {:02}: frame_idx={}, bbox={:?}, w={}, h={}, fg={}",
            i, obs.frame_idx, obs.bbox, obs.width(), obs.height(), obs.mask.iter().filter(|&&m| m).count()
        );
    }
    if let Some(obs) = mario_obs.first() {
        print_obs_ascii("mario_obs[0]", obs);
    }
}
