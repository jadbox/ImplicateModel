//! R10.1 hypothesis test: which parameter-free noise floor separates holons from background residual
//! |ΔΨ| = ‖frame − atlas‖₂ (RGB grade-1 difference norm) on real footage?
//!
//! Candidates: (a) median + 3·1.4826·MAD, (b) same over the non-zero residual population,
//! (c) Otsu two-idempotent split, (d) Rosin/Zack triangle unimodal threshold, vs. the legacy 24/26.
//! Writes `tau_masks.png`: rows = sample frames, cols = [frame, legacy, b, c, d].
//!
//! Usage: cargo run --release -p implicate-lifter --example r10_tau -- <video> <start_sec> <frames> <out_dir>

use std::path::Path;

use image::{Rgb, RgbImage};
use implicate_lifter::{MultiPlaneOptions, build_multiplane_atlas, probe, video_io};

fn mad_tau(v: &[u32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let med = s[s.len() / 2] as f64;
    let mut d: Vec<f64> = s.iter().map(|&x| (x as f64 - med).abs()).collect();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    med + 3.0 * 1.4826 * d[d.len() / 2]
}

fn otsu(hist: &[u64]) -> usize {
    let total: u64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();
    let (mut w0, mut s0, mut best, mut t) = (0u64, 0.0f64, 0.0f64, 0usize);
    for (i, &c) in hist.iter().enumerate() {
        w0 += c;
        if w0 == 0 || w0 == total {
            continue;
        }
        s0 += i as f64 * c as f64;
        let w1 = total - w0;
        let m0 = s0 / w0 as f64;
        let m1 = (sum_all - s0) / w1 as f64;
        let between = w0 as f64 * w1 as f64 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            t = i;
        }
    }
    t
}

/// Zack triangle: line from histogram peak to the last non-empty bin; threshold at max distance.
fn triangle(hist: &[u64]) -> usize {
    let lh: Vec<f64> = hist.iter().map(|&c| ((c + 1) as f64).ln()).collect();
    let peak = (0..lh.len()).max_by(|&a, &b| lh[a].partial_cmp(&lh[b]).unwrap()).unwrap();
    let last = (0..hist.len()).rev().find(|&i| hist[i] > 0).unwrap_or(peak);
    if last <= peak {
        return peak;
    }
    let (x0, y0, x1, y1) = (peak as f64, lh[peak], last as f64, lh[last]);
    let (dx, dy) = (x1 - x0, y1 - y0);
    let norm = dx.hypot(dy);
    (peak..=last)
        .max_by(|&a, &b| {
            let da = (dy * a as f64 - dx * lh[a] + x1 * y0 - y1 * x0).abs() / norm;
            let db = (dy * b as f64 - dx * lh[b] + x1 * y0 - y1 * x0).abs() / norm;
            da.partial_cmp(&db).unwrap()
        })
        .unwrap()
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
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let (w, h) = (frames[0].width as usize, frames[0].height as usize);
    let (ay0, ay1) = atlas.active_y_range;

    let mut res: Vec<Vec<u32>> = Vec::new();
    let mut hist = vec![0u64; 443];
    let mut all = Vec::new();
    for (fi, f) in frames.iter().enumerate() {
        let bg = atlas.reconstruct_background_frame(fi);
        let mut r = vec![0u32; w * h];
        for y in ay0..ay1 {
            for x in 0..w {
                let c = f.pixel(x as u32, y as u32);
                // Lattice-tolerant residual: min over the 1-px registration neighbourhood.
                let tol = std::env::var_os("R10_TOL").is_some();
                let mut d2 = i32::MAX;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if !tol && (dx != 0 || dy != 0) {
                            continue;
                        }
                        let (bx, by) = (x as i32 + dx, y as i32 + dy);
                        if bx < 0 || by < 0 || bx >= w as i32 || by >= h as i32 {
                            continue;
                        }
                        let b = bg.pixel(bx as usize, by as usize);
                        d2 = d2.min((0..3).map(|k| (c[k] as i32 - b[k] as i32).pow(2)).sum());
                    }
                }
                let d = (d2 as f64).sqrt().round() as u32;
                r[y * w + x] = d;
                hist[d as usize] += 1;
                all.push(d);
            }
        }
        res.push(r);
    }
    let nz: Vec<u32> = all.iter().copied().filter(|&v| v > 0).collect();
    let t_a = mad_tau(&all);
    let t_b = mad_tau(&nz);
    let t_c = otsu(&hist) as f64;
    let t_d = triangle(&hist) as f64;
    let zero_frac = 1.0 - nz.len() as f64 / all.len() as f64;
    println!(
        "zero_frac={zero_frac:.3} tau: a(MAD)={t_a:.1} b(MAD nz)={t_b:.1} c(Otsu)={t_c} d(triangle)={t_d} legacy=24/26(max-ch)"
    );
    let frac = |t: f64| all.iter().filter(|&&v| v as f64 > t).count() as f64 / all.len() as f64;
    println!(
        "fg fraction: a={:.4} b={:.4} c={:.4} d={:.4} legacy(L2>~40)={:.4}",
        frac(t_a),
        frac(t_b),
        frac(t_c),
        frac(t_d),
        frac(40.0)
    );

    let picks: Vec<usize> = (0..frames.len()).step_by((frames.len() / 4).max(1)).take(4).collect();
    let taus = [None, Some(40.0), Some(t_b), Some(t_c), Some(t_d)];
    let mut sheet = RgbImage::new((w * taus.len()) as u32, (h * picks.len()) as u32);
    for (r, &fi) in picks.iter().enumerate() {
        for (c, t) in taus.iter().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    let p = frames[fi].pixel(x as u32, y as u32);
                    let px = match t {
                        None => p,
                        Some(t) if res[fi][y * w + x] as f64 > *t => p,
                        Some(_) => [p[0] / 5, p[1] / 5, p[2] / 5 + 30],
                    };
                    sheet.put_pixel((c * w + x) as u32, (r * h + y) as u32, Rgb(px));
                }
            }
        }
    }
    let p = out.join("tau_masks.png");
    sheet.save(&p).unwrap();
    println!("sheet -> {}", p.display());
}
