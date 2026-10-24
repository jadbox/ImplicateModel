use std::path::Path;
use implicate_lifter::{video_io, MultiPlaneOptions, build_multiplane_atlas};

fn main() {
    let nimh_path = Path::new("/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4");
    let frames = video_io::read_frames_scaled(nimh_path, Some(364.0), Some(1.0), 320, 176).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let bg = atlas.reconstruct_background_frame(4);
    println!("Pixels at x=30, y=70..105 in Frame 4:");
    for y in 70..=105 {
        let p = frames[4].pixel(30, y);
        let b = bg.pixel(30, y as usize);
        let d = (p[0] as i32 - b[0] as i32).abs().max((p[1] as i32 - b[1] as i32).abs()).max((p[2] as i32 - b[2] as i32).abs());
        println!("y={}: frame={:?}, bg={:?}, d={}", y, p, b, d);
    }
}
