use std::path::Path;
use implicate_lifter::{
    video_io, build_multiplane_atlas, MultiPlaneOptions,
};

const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";

fn main() {
    let smb_path = Path::new(SMB3_PATH);
    let frames = video_io::read_frames(smb_path, Some(1.0), Some(1.5)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let mario_obs = implicate_lifter::articulated_wavepacket::extract_dominant_character_observations(&frames, &atlas);
    println!("Extracted {} Mario observations", mario_obs.len());
}
