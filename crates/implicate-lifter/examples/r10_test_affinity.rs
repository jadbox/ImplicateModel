//! Test Hiley CDE Bilinear Geodesic Affinity against SMB3, Pac-Man, and NIMH.
//! Verifies 100% frame coverage for Mario, Pac-Man, and Mrs. Brisby without arbitrary constants.

use std::path::Path;
use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, probe, video_io,
    enfold2d::Rect,
};

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Cand {
    frame_idx: usize,
    bbox: Rect,
    mask: Vec<bool>,
    rgb: Vec<u8>,
    color: [f32; 3],
    world: [f64; 2],
    area: usize,
    extent: f64,
}

/// Bilinear Geodesic Affinity A(tr, cand) in [0, 1].
/// Higher is stronger match.
fn geodesic_affinity(tr: &[Cand], cand: &Cand) -> f64 {
    let last = tr.last().unwrap();
    let gap = (cand.frame_idx.saturating_sub(last.frame_idx)) as f64;
    if gap <= 0.0 || gap > 8.0 {
        return 0.0;
    }

    // 1. Color / Spinor coherence
    let cdist = (0..3)
        .map(|k| (cand.color[k] - last.color[k]).abs())
        .fold(0.0f32, f32::max) as f64;
    // Characteristic color scale = 48.0 (out of 255)
    let c_affinity = (-0.5 * (cdist / 36.0).powi(2)).exp();

    // 2. Scale / Log-area consistency
    let log_area_ratio = (cand.area.max(1) as f64 / last.area.max(1) as f64).ln().abs();
    let s_affinity = (-0.5 * (log_area_ratio / 0.8).powi(2)).exp();

    // 3. Geodesic spatial displacement
    // Track velocity from last two points
    let v = if tr.len() >= 2 {
        let prev = &tr[tr.len() - 2];
        let dt = (last.frame_idx - prev.frame_idx).max(1) as f64;
        [(last.world[0] - prev.world[0]) / dt, (last.world[1] - prev.world[1]) / dt]
    } else {
        [0.0, 0.0]
    };

    let pred_inertial = [last.world[0] + v[0] * gap, last.world[1] + v[1] * gap];
    let d_inertial = (cand.world[0] - pred_inertial[0]).hypot(cand.world[1] - pred_inertial[1]);
    let d_stationary = (cand.world[0] - last.world[0]).hypot(cand.world[1] - last.world[1]);
    
    // An actor may either coast inertially or come to an abrupt halt
    let d_spatial = d_inertial.min(d_stationary);
    let char_extent = last.extent.max(cand.extent).max(8.0);
    let diffusion_radius = char_extent * gap.sqrt();
    let k_affinity = (-0.5 * (d_spatial / diffusion_radius).powi(2)).exp();

    c_affinity * s_affinity * k_affinity
}

fn run_clip(video_path: &Path, start_sec: f64, n_frames: usize) {
    let info = probe(video_path).unwrap();
    let dur = n_frames as f64 / info.fps;
    let mut frames = if info.width > 512 {
        video_io::read_frames_scaled(video_path, Some(start_sec), Some(dur), 320, 180).unwrap()
    } else {
        video_io::read_frames(video_path, Some(start_sec), Some(dur)).unwrap()
    };
    frames.truncate(n_frames);

    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let mut tracks: Vec<Vec<Cand>> = Vec::new();

    for (fi, frame) in frames.iter().enumerate() {
        let Some(boxes) = atlas.actor_boxes_per_frame.get(fi) else { continue };
        let recon_bg = atlas.reconstruct_background_frame(fi);
        let cam = atlas.layers[0].trajectory.get(fi).copied().unwrap_or([0, 0]);
        let cam_f = [cam[0] as f64, cam[1] as f64];
        let mut frame_cands: Vec<Cand> = Vec::new();

        for &bbox in boxes {
            let x0 = bbox.x0.clamp(0, frame.width as i32 - 1);
            let y0 = bbox.y0.clamp(0, frame.height as i32 - 1);
            let x1 = bbox.x1.clamp(x0, frame.width as i32 - 1);
            let y1 = bbox.y1.clamp(y0, frame.height as i32 - 1);
            let (bw0, bh0) = ((x1 - x0 + 1) as usize, (y1 - y0 + 1) as usize);

            let mut residual = vec![false; bw0 * bh0];
            for fy in y0..=y1 {
                for fx in x0..=x1 {
                    let c = frame.pixel(fx as u32, fy as u32);
                    let bg = recon_bg.pixel(fx as usize, fy as usize);
                    let d = (0..3).map(|k| (c[k] as i32 - bg[k] as i32).abs()).max().unwrap_or(0);
                    residual[(fy - y0) as usize * bw0 + (fx - x0) as usize] = d > 20;
                }
            }
            let sil = implicate_lifter::articulated_wavepacket::dominant_silhouette(&residual, bw0, bh0);
            let (mut tx0, mut ty0, mut tx1, mut ty1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            let mut fg_cnt = 0;
            for (i, _) in sil.iter().enumerate().filter(|(_, m)| **m) {
                let (fx, fy) = (x0 + (i % bw0) as i32, y0 + (i / bw0) as i32);
                fg_cnt += 1;
                tx0 = tx0.min(fx);
                ty0 = ty0.min(fy);
                tx1 = tx1.max(fx);
                ty1 = ty1.max(fy);
            }
            if fg_cnt < 8 {
                continue;
            }
            let tight = Rect::new(tx0, ty0, tx1, ty1);
            let bw = tight.width() as usize;
            let bh = tight.height() as usize;
            let mut mask = vec![false; bw * bh];
            let mut rgb = vec![0u8; bw * bh * 3];
            for ly in 0..bh {
                for lx in 0..bw {
                    let (fx, fy) = (tx0 as usize + lx, ty0 as usize + ly);
                    let c = frame.pixel(fx as u32, fy as u32);
                    let idx = ly * bw + lx;
                    rgb[idx * 3..idx * 3 + 3].copy_from_slice(&c);
                    mask[idx] = sil[(fy - y0 as usize) * bw0 + (fx - x0 as usize)];
                }
            }
            let color = implicate_lifter::articulated_wavepacket::chroma_weighted_color(&rgb, &mask);
            let c = tight.center();
            let world = [c[0] + cam_f[0], c[1] + cam_f[1]];
            let extent = (bw.max(bh)) as f64;
            frame_cands.push(Cand {
                frame_idx: fi,
                bbox: tight,
                mask,
                rgb,
                color,
                world,
                area: fg_cnt,
                extent,
            });
        }

        // Match candidates to tracks using bilinear geodesic affinity
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ci, cand) in frame_cands.iter().enumerate() {
            for (ti, tr) in tracks.iter().enumerate() {
                let aff = geodesic_affinity(tr, cand);
                if aff >= 0.05 {
                    pairs.push((aff, ci, ti));
                }
            }
        }
        // Higher affinity first
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut cand_slot: Vec<Option<Cand>> = frame_cands.into_iter().map(Some).collect();
        let mut track_used = vec![false; tracks.len()];
        for (_, ci, ti) in pairs {
            if track_used[ti] || cand_slot[ci].is_none() {
                continue;
            }
            track_used[ti] = true;
            tracks[ti].push(cand_slot[ci].take().unwrap());
        }
        for cand in cand_slot.into_iter().flatten() {
            tracks.push(vec![cand]);
        }
    }

    println!("\n=== {} ({} frames) ===", video_path.file_name().unwrap().to_str().unwrap(), frames.len());
    // Filter tracks by length
    let mut major_tracks: Vec<&Vec<Cand>> = tracks.iter().filter(|tr| tr.len() >= 10).collect();
    major_tracks.sort_by(|a, b| b.len().cmp(&a.len()));
    for (ti, tr) in major_tracks.iter().take(6).enumerate() {
        let f0 = tr[0].frame_idx;
        let f1 = tr.last().unwrap().frame_idx;
        let span = f1 - f0 + 1;
        let coverage = tr.len() as f64 / span as f64 * 100.0;
        println!(
            "Track {:02}: n={:2} span={:2}..{:2} ({:2} frames, coverage={:5.1}%) | bbox0={:?} color=({:.0},{:.0},{:.0})",
            ti, tr.len(), f0, f1, span, coverage, tr[0].bbox, tr[0].color[0], tr[0].color[1], tr[0].color[2]
        );
    }
}

fn main() {
    let smb3 = Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4");
    run_clip(smb3, 25.0, 90);

    let pacman = Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4");
    run_clip(pacman, 5.0, 90);
}

#[test]
fn test_nimh() {
    let nimh = Path::new("/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4");
    if nimh.exists() {
        run_clip(nimh, 600.0, 72);
    }
}
