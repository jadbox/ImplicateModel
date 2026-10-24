use implicate_lifter::{
    video_io, MultiPlaneOptions, build_multiplane_atlas,
    dominant_silhouette,
};
use std::path::Path;

const RAPTOR_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";
const SMB3_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";
const PACMAN_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";
const NIMH_PATH: &str =
    "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

fn test_extraction(name: &str, path: &str, start_s: f64, dur_s: f64, scale: Option<(u32, u32)>, fi: usize, box_idx: usize) {
    let frames = if let Some((w, h)) = scale {
        video_io::read_frames_scaled(Path::new(path), Some(start_s), Some(dur_s), w, h).unwrap()
    } else {
        video_io::read_frames(Path::new(path), Some(start_s), Some(dur_s)).unwrap()
    };
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let frame = &frames[fi];
    let recon_bg = atlas.reconstruct_background_frame(fi);
    let bbox = atlas.actor_boxes_per_frame[fi][box_idx];
    let (x0, y0, x1, y1) = (bbox.x0, bbox.y0, bbox.x1, bbox.y1);
    let (bw, bh) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);

    // Compute empirical Q98 pointwise outside boxes
    let mut bg_diffs = Vec::new();
    let (ay0, ay1) = atlas.active_y_range;
    let n = frames.len();
    let stride = (n / 8).max(1);
    for f_idx in (0..n).step_by(stride) {
        let fr = &frames[f_idx];
        let bg_f = atlas.reconstruct_background_frame(f_idx);
        let boxes = atlas.actor_boxes_per_frame.get(f_idx);
        for y in (ay0..ay1).step_by(2) {
            for x in (0..atlas.width).step_by(2) {
                let in_actor = boxes.map(|bs| bs.iter().any(|b| {
                    (b.x0 - 5..=b.x1 + 5).contains(&(x as i32)) && (b.y0 - 5..=b.y1 + 5).contains(&(y as i32))
                })).unwrap_or(false);
                if !in_actor {
                    let c = fr.pixel(x as u32, y as u32);
                    let b = bg_f.pixel(x, y);
                    let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                    if d > 0 {
                        bg_diffs.push(d);
                    }
                }
            }
        }
    }
    bg_diffs.sort_unstable();
    let q98 = if bg_diffs.is_empty() { 18.0 } else {
        bg_diffs[((bg_diffs.len() - 1) as f64 * 0.98).round() as usize] as f64
    };
    let median = if bg_diffs.is_empty() { 0.0 } else { bg_diffs[bg_diffs.len() / 2] as f64 };
    let mut devs: Vec<f64> = bg_diffs.iter().map(|&x| (x as f64 - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if devs.is_empty() { 1.0 } else { devs[devs.len() / 2] };
    let tau_mad = median + 3.0 * 1.4826 * mad;
    let tau = tau_mad.max(q98).clamp(6.0, 36.0);

    println!("{}: median={:.1}, mad={:.1}, tau_mad={:.2}, Q98={:.2} => final tau={:.2}", name, median, mad, tau_mad, q98, tau);

    // 1. Pointwise with Q98
    let mut res_q98 = vec![false; bw * bh];
    for ly in 0..bh {
        for lx in 0..bw {
            let fx = x0 as usize + lx;
            let fy = y0 as usize + ly;
            let c = frame.pixel(fx as u32, fy as u32);
            let b = recon_bg.pixel(fx, fy);
            let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
            res_q98[ly * bw + lx] = (d as f64) > tau;
        }
    }
    let sil_q98 = dominant_silhouette(&res_q98, bw, bh);
    let count_q98 = sil_q98.iter().filter(|&&m| m).count();

    // 2. 3x3 lattice tolerance with tau=12 or tau_mad
    let mut res_lattice = vec![false; bw * bh];
    for ly in 0..bh {
        for lx in 0..bw {
            let fx = x0 as usize + lx;
            let fy = y0 as usize + ly;
            let c = frame.pixel(fx as u32, fy as u32);
            let mut best_d = i32::MAX;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let bx = fx as i32 + dx;
                    let by = fy as i32 + dy;
                    if bx >= 0 && by >= 0 && bx < atlas.width as i32 && by < atlas.height as i32 {
                        let b = recon_bg.pixel(bx as usize, by as usize);
                        let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                        best_d = best_d.min(d);
                    }
                }
            }
            res_lattice[ly * bw + lx] = best_d > 12; // 3x3 lattice at tau=12
        }
    }
    let sil_lattice = dominant_silhouette(&res_lattice, bw, bh);
    let count_lattice = sil_lattice.iter().filter(|&&m| m).count();

    println!("{}: fg count pointwise Q98 (T={}) = {}, lattice (T=12) = {}", name, q98, count_q98, count_lattice);

    // Save Q98 mask visual
    let mut rgb = vec![0u8; bw * bh * 3];
    for ly in 0..bh {
        for lx in 0..bw {
            let idx = ly * bw + lx;
            if sil_q98[idx] {
                let fx = (x0 as usize + lx) as u32;
                let fy = (y0 as usize + ly) as u32;
                let c = frame.pixel(fx, fy);
                rgb[idx * 3..idx * 3 + 3].copy_from_slice(&c);
            }
        }
    }
    let out_path = format!("output_tmp/{}_q98.png", name);
    video_io::write_png(Path::new(&out_path), bw as u32, bh as u32, &rgb).unwrap();
}

fn main() {
    test_extraction("Pacman", PACMAN_PATH, 5.0, 1.0, None, 0, 1);
    test_extraction("SMB3", SMB3_PATH, 25.0, 1.0, None, 6, 2);
    test_extraction("NIMH", NIMH_PATH, 356.0, 1.0, Some((320, 176)), 0, 0);
    test_extraction("Raptor", RAPTOR_PATH, 1.0, 1.0, None, 6, 1);
}
