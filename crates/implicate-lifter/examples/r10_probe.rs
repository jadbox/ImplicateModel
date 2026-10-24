//! R10 diagnostic probe: decode a real clip, build the multi-plane atlas, extract holon tracks, and
//! emit (1) per-track frame coverage, (2) residual |ΔΨ| robust statistics, and (3) a contact sheet of
//! every frame with each track's silhouette box drawn in its label colour.
//!
//! Usage: cargo run --release -p implicate-lifter --example r10_probe -- <video> <start_sec> <frames> <out_dir>

use std::path::Path;

use image::{Rgb, RgbImage};
use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, extract_character_observation_tracks, probe, video_io,
};

fn label_color(i: usize) -> [u8; 3] {
    let h = (i as f32 * 137.508).rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h.rem_euclid(2.0) - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let video = Path::new(&a[1]);
    let start: f64 = a[2].parse().unwrap();
    let n: usize = a[3].parse().unwrap();
    let out = Path::new(&a[4]);
    std::fs::create_dir_all(out).unwrap();
    let info = probe(video).unwrap();
    let dur = n as f64 / info.fps;
    let mut frames = if info.width > 512 {
        video_io::read_frames_scaled(video, Some(start), Some(dur), 320, 180).unwrap()
    } else {
        video_io::read_frames(video, Some(start), Some(dur)).unwrap()
    };
    frames.truncate(n);
    let t0 = std::time::Instant::now();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let t_atlas = t0.elapsed();
    let t1 = std::time::Instant::now();
    let tracks = extract_character_observation_tracks(&frames, &atlas, 8);
    let t_tracks = t1.elapsed();
    println!("frames={} atlas={:?} tracks={:?}", frames.len(), t_atlas, t_tracks);

    // Residual statistics |ΔΨ| = max-channel |frame - background| over the active viewport.
    let mut all: Vec<u16> = Vec::new();
    for (fi, f) in frames.iter().enumerate().step_by((frames.len() / 8).max(1)) {
        let bg = atlas.reconstruct_background_frame(fi);
        for y in atlas.active_y_range.0..atlas.active_y_range.1 {
            for x in 0..f.width as usize {
                let c = f.pixel(x as u32, y as u32);
                let b = bg.pixel(x, y);
                all.push((0..3).map(|k| (c[k] as i32 - b[k] as i32).unsigned_abs() as u16).max().unwrap());
            }
        }
    }
    all.sort_unstable();
    let med = all[all.len() / 2] as f64;
    let mut dev: Vec<f64> = all.iter().map(|&v| (v as f64 - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = dev[dev.len() / 2];
    let q = |p: f64| all[((all.len() - 1) as f64 * p) as usize];
    println!(
        "residual: median={med} MAD={mad} tau(3σ)={:.1} p90={} p95={} p99={}",
        med + 3.0 * 1.4826 * mad,
        q(0.90),
        q(0.95),
        q(0.99)
    );

    for (ti, tr) in tracks.iter().enumerate() {
        let fset: Vec<usize> = tr.iter().map(|o| o.frame_idx).collect();
        let missing: Vec<usize> = (fset[0]..=*fset.last().unwrap()).filter(|f| !fset.contains(f)).collect();
        let area: usize = tr.iter().map(|o| o.mask.iter().filter(|&&m| m).count()).sum::<usize>() / tr.len();
        println!(
            "track {ti}: n={} span={}..{} missing_in_span={:?} mean_area={} bbox0={:?}",
            tr.len(),
            fset[0],
            fset.last().unwrap(),
            missing,
            area,
            tr[0].bbox
        );
    }

    // Contact sheet: every `stride`-th frame, tracks overlaid.
    let (w, h) = (frames[0].width, frames[0].height);
    let stride = (frames.len() / 24).max(1);
    let picks: Vec<usize> = (0..frames.len()).step_by(stride).collect();
    let cols = 6u32;
    let rows = (picks.len() as u32).div_ceil(cols);
    let mut sheet = RgbImage::new(cols * w, rows * h);
    for (k, &fi) in picks.iter().enumerate() {
        let (ox, oy) = ((k as u32 % cols) * w, (k as u32 / cols) * h);
        let f = &frames[fi];
        for y in 0..h {
            for x in 0..w {
                let p = f.pixel(x, y);
                sheet.put_pixel(ox + x, oy + y, Rgb([p[0] / 2, p[1] / 2, p[2] / 2]));
            }
        }
        for (ti, tr) in tracks.iter().enumerate() {
            let Some(o) = tr.iter().find(|o| o.frame_idx == fi) else { continue };
            let c = label_color(ti);
            for ly in 0..o.height() {
                for lx in 0..o.width() {
                    if o.mask[ly * o.width() + lx] {
                        let i = (ly * o.width() + lx) * 3;
                        let (x, y) = (o.bbox.x0 as u32 + lx as u32, o.bbox.y0 as u32 + ly as u32);
                        sheet.put_pixel(ox + x, oy + y, Rgb([o.rgb[i], o.rgb[i + 1], o.rgb[i + 2]]));
                    }
                }
            }
            let b = o.bbox;
            for x in b.x0..=b.x1 {
                for y in [b.y0, b.y1] {
                    sheet.put_pixel(ox + x as u32, oy + y as u32, Rgb(c));
                }
            }
            for y in b.y0..=b.y1 {
                for x in [b.x0, b.x1] {
                    sheet.put_pixel(ox + x as u32, oy + y as u32, Rgb(c));
                }
            }
        }
    }
    let p = out.join("tracks_overlay.png");
    sheet.save(&p).unwrap();
    println!("sheet -> {}", p.display());
}
