//! R10.10 debug: dump the selected character track (raw bbox crop | masked cutout) per frame,
//! upscaled, for visual inspection of filament/sky contamination.
//! `cargo run --release -p implicate-lifter --example r10_10_dump -- [nimh|smb3|pacman|raptor] [out_dir]`
use std::path::Path;

use image::{Rgb, RgbImage};
use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, extract_character_observation_tracks,
    select_character_track, video_io,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args.get(1).map(String::as_str).unwrap_or("nimh");
    let out = args.get(2).cloned().unwrap_or_else(|| "/tmp/r10_10".into());
    std::fs::create_dir_all(&out).unwrap();
    let frames = match which {
        "nimh" => video_io::read_frames_scaled(
            Path::new("/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4"),
            Some(480.0), Some(1.0), 320, 176).unwrap(),
        "smb3" => video_io::read_frames(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4"), Some(1.0), Some(1.5)).unwrap(),
        "pacman" => video_io::read_frames(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4"), Some(1.0), Some(2.0)).unwrap(),
        _ => video_io::read_frames(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4"), Some(1.0), Some(2.0)).unwrap(),
    };
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let bg = atlas.reconstruct_background_frame(0);
    RgbImage::from_raw(bg.width as u32, bg.height as u32, bg.rgb.clone()).unwrap().save(format!("{out}/{which}_bg0.png")).ok();
    // Full-frame residual overlay (red tint = atlas residual; green = actor box outline).
    for fi in [0usize, 8, 16] {
        let Some(f) = frames.get(fi) else { continue };
        let bgf = atlas.reconstruct_background_frame(fi);
        let (w, h) = (f.width, f.height);
        let mut img = RgbImage::new(w * 2, h);
        for y in 0..h {
            for x in 0..w {
                let c = f.pixel(x, y);
                let b = bgf.pixel(x as usize, y as usize);
                let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap();
                img.put_pixel(x, y, Rgb(c));
                let v = (d * 4).min(255) as u8;
                img.put_pixel(x + w, y, Rgb([v, v, v]));
            }
        }
        for r in &atlas.actor_boxes_per_frame[fi] {
            for x in r.x0.max(0)..=r.x1.min(w as i32 - 1) {
                for y in [r.y0.max(0), r.y1.min(h as i32 - 1)] {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0]));
                }
            }
            for y in r.y0.max(0)..=r.y1.min(h as i32 - 1) {
                for x in [r.x0.max(0), r.x1.min(w as i32 - 1)] {
                    img.put_pixel(x as u32, y as u32, Rgb([0, 255, 0]));
                }
            }
        }
        img.save(format!("{out}/{which}_full{fi:02}.png")).unwrap();
    }
    let tracks = extract_character_observation_tracks(&frames, &atlas, 2);
    let track = select_character_track(&tracks).expect("track");
    let s = if frames[0].width < 300 { 6 } else { 4 };
    for obs in track.iter() {
        let (w, h) = (obs.width(), obs.height());
        let mut img = RgbImage::from_pixel((w * 2 + 2) as u32 * s, h as u32 * s, Rgb([30, 30, 40]));
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let c = [obs.rgb[i * 3], obs.rgb[i * 3 + 1], obs.rgb[i * 3 + 2]];
                let m = if obs.mask[i] { c } else { [0, 0, 0] };
                for sy in 0..s {
                    for sx in 0..s {
                        img.put_pixel(x as u32 * s + sx, y as u32 * s + sy, Rgb(c));
                        img.put_pixel((x + w + 2) as u32 * s + sx, y as u32 * s + sy, Rgb(m));
                    }
                }
            }
        }
        img.save(format!("{out}/{which}_f{:02}.png", obs.frame_idx)).unwrap();
        let fg = obs.mask.iter().filter(|&&m| m).count();
        println!("frame {:02} bbox {:?} fg {fg} trunc {}", obs.frame_idx, obs.bbox, obs.truncated);
    }
}
