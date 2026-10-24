use implicate_lifter::{
    video_io, MultiPlaneOptions, build_multiplane_atlas, extract_dominant_character_observations,
};
use std::path::Path;

const RAPTOR_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";

fn main() {
    let frames = video_io::read_frames(Path::new(RAPTOR_PATH), Some(1.0), Some(2.0)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let raptor_obs = extract_dominant_character_observations(&frames, &atlas);
    for &oi in &[17, 27] {
        let obs = &raptor_obs[oi];
        let name = format!("output_tmp/obs_{}.png", oi);
        video_io::write_png(Path::new(&name), obs.width() as u32, obs.height() as u32, &obs.rgb).unwrap();
    }
}
