use std::path::Path;
use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, probe, video_io,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let video_path = Path::new(&args[1]);
    let start_sec: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let n_frames: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(90);

    let info = probe(video_path).unwrap();
    let dur = n_frames as f64 / info.fps;
    let mut frames = if info.width > 512 {
        video_io::read_frames_scaled(video_path, Some(start_sec), Some(dur), 320, 180).unwrap()
    } else {
        video_io::read_frames(video_path, Some(start_sec), Some(dur)).unwrap()
    };
    frames.truncate(n_frames);

    let _atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());

    // We want to simulate the exact loop of extract_character_observation_tracks around frame 49-54
    // but print every rejection condition for the Mario candidate!
    println!("See why Cand 0 was rejected in frames 49..55");
}
