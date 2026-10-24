use std::path::Path;
use implicate_lifter::video_io;

fn main() {
    let nimh_path = Path::new("/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4");
    let frames = video_io::read_frames_scaled(nimh_path, Some(625.0), Some(1.0), 320, 176).unwrap();
    let f = &frames[4];
    let (x0, y0, x1, y1) = (50, 10, 220, 170);
    let cw = x1 - x0;
    let ch = y1 - y0;
    let mut crop = vec![0u8; cw * ch * 3];
    for y in 0..ch {
        for x in 0..cw {
            let p = f.pixel((x0 + x) as u32, (y0 + y) as u32);
            let idx = (y * cw + x) * 3;
            crop[idx..idx+3].copy_from_slice(&p);
        }
    }
    video_io::write_png(Path::new("nimh_crop_625.png"), cw as u32, ch as u32, &crop).unwrap();
    println!("Saved nimh_crop_625.png");
}
