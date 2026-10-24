use std::path::Path;
use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, probe, video_io,
};

/// Compute the statistical noise floor tau = median(I_nz) + 3 * 1.4826 * MAD(I_nz)
/// over the non-zero lattice-tolerant residual multivector |ΔΨ|.
pub fn compute_shot_noise_floor(
    frames: &[crate::video_io::RawFrame],
    atlas: &implicate_lifter::MultiPlaneAtlas,
) -> f64 {
    let n = frames.len();
    if n == 0 {
        return 18.0;
    }
    let stride = (n / 8).max(1);
    let mut sample_res = Vec::new();
    let (ay0, ay1) = atlas.active_y_range;
    let w = atlas.width;
    let h = atlas.height;

    for fi in (0..n).step_by(stride) {
        let f = &frames[fi];
        let bg = atlas.reconstruct_background_frame(fi);
        for y in (ay0..ay1).step_by(2) {
            for x in (0..w).step_by(2) {
                let c = f.pixel(x as u32, y as u32);
                let mut best_d = i32::MAX;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (bx, by) = (x as i32 + dx, y as i32 + dy);
                        if bx >= 0 && by >= 0 && bx < w as i32 && by < h as i32 {
                            let b = bg.pixel(bx as usize, by as usize);
                            let d = (0..3).map(|k| (c[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                            best_d = best_d.min(d);
                        }
                    }
                }
                if best_d > 0 && best_d != i32::MAX {
                    sample_res.push(best_d as f64);
                }
            }
        }
    }

    if sample_res.is_empty() {
        return 18.0;
    }

    sample_res.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sample_res[sample_res.len() / 2];
    let mut devs: Vec<f64> = sample_res.iter().map(|&x| (x - median).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = devs[devs.len() / 2];

    // tau = median + 3.0 * 1.4826 * MAD
    let tau = median + 3.0 * 1.4826 * mad;
    tau.clamp(6.0, 36.0)
}

fn test_video(path: &Path, start: f64, n: usize, name: &str) {
    if !path.exists() {
        println!("{name}: not found at {}", path.display());
        return;
    }
    let info = probe(path).unwrap();
    let dur = n as f64 / info.fps;
    let mut frames = if info.width > 512 {
        video_io::read_frames_scaled(path, Some(start), Some(dur), 320, 180).unwrap()
    } else {
        video_io::read_frames(path, Some(start), Some(dur)).unwrap()
    };
    frames.truncate(n);
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let tau = compute_shot_noise_floor(&frames, &atlas);
    println!("{name:<20}: auto-tau = {:.2} (dim {}x{}, {} frames)", tau, frames[0].width, frames[0].height, frames.len());
}

fn main() {
    test_video(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4"), 25.0, 90, "SMB3");
    test_video(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4"), 5.0, 90, "Pac-Man");
    test_video(Path::new("/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4"), 20.0, 90, "Raptor");
    test_video(Path::new("/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4"), 600.0, 72, "The Secret of NIMH");
}
