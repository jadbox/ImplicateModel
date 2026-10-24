use std::path::Path;
use implicate_lifter::video_io;

const NIMH_PATH: &str =
    "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

fn main() {
    let nimh_path = Path::new(NIMH_PATH);
    for i in 0..10 {
        let t = 350.0 + i as f64 * 1.5;
        let frames = video_io::read_frames_scaled(nimh_path, Some(t), Some(0.05), 320, 176).unwrap();
        if !frames.is_empty() {
            let name = format!("output_tmp/scan_sub_{}.png", i);
            video_io::write_png(Path::new(&name), 320, 176, &frames[0].rgb).unwrap();
            println!("Saved {} at t={:.1}", name, t);
        }
    }
}
