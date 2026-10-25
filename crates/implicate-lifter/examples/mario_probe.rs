use std::path::Path;
use implicate_lifter::{
    video_io, build_multiplane_atlas, MultiPlaneOptions, extract_dominant_character_observations,
};

const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";

fn main() {
    let smb_path = Path::new(SMB3_PATH);
    let frames = video_io::read_frames(smb_path, Some(1.0), Some(1.5)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let mario_obs = extract_dominant_character_observations(&frames, &atlas);
    println!("Total mario obs: {}", mario_obs.len());
    for (i, obs) in mario_obs.iter().enumerate() {
        println!("obs {:02}: frame_idx={}, bbox={:?}, w={}, h={}", i, obs.frame_idx, obs.bbox, obs.width(), obs.height());
    }
}
