//! 2D CFT phase-correlation scroll estimator and HUD/border partitioning (`enfold2d::fft_scroll`).

use std::io;
use std::path::Path;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use super::color_dist;
use crate::video_io::{self, RawFrame, VideoInfo};

pub struct Fft2 {
    pub w: usize,
    pub h: usize,
    rows_fwd: Arc<dyn Fft<f32>>,
    rows_inv: Arc<dyn Fft<f32>>,
    cols_fwd: Arc<dyn Fft<f32>>,
    cols_inv: Arc<dyn Fft<f32>>,
}

impl Fft2 {
    pub fn new(w: usize, h: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            w,
            h,
            rows_fwd: planner.plan_fft_forward(w),
            rows_inv: planner.plan_fft_inverse(w),
            cols_fwd: planner.plan_fft_forward(h),
            cols_inv: planner.plan_fft_inverse(h),
        }
    }

    pub fn transform(&self, buf: &mut [Complex<f32>], inverse: bool) {
        let (w, h) = (self.w, self.h);
        let (row_fft, col_fft) = if inverse {
            (self.rows_inv.clone(), self.cols_inv.clone())
        } else {
            (self.rows_fwd.clone(), self.cols_fwd.clone())
        };
        let mut row = vec![Complex::new(0.0, 0.0); w];
        for y in 0..h {
            row.copy_from_slice(&buf[y * w..(y + 1) * w]);
            row_fft.process(&mut row);
            if inverse {
                for c in row.iter_mut() {
                    *c /= w as f32;
                }
            }
            buf[y * w..(y + 1) * w].copy_from_slice(&row);
        }
        let mut col = vec![Complex::new(0.0, 0.0); h];
        for x in 0..w {
            for y in 0..h {
                col[y] = buf[y * w + x];
            }
            col_fft.process(&mut col);
            if inverse {
                for c in col.iter_mut() {
                    *c /= h as f32;
                }
            }
            for y in 0..h {
                buf[y * w + x] = col[y];
            }
        }
    }
}

/// Phase-correlation shift estimator over a fixed viewport band.
pub struct ScrollEstimator {
    fft: Fft2,
    w: usize,
    h: usize,
    band_h: usize,
    win_x: Vec<f32>,
    win_y: Vec<f32>,
    prev: Option<Vec<Complex<f32>>>,
}

impl ScrollEstimator {
    pub fn new(width: usize, band_h: usize) -> Self {
        let w = width.next_power_of_two();
        let h = band_h.next_power_of_two();
        let cols = width.min(w).max(1);
        let rows = band_h.min(h).max(1);
        let mut win_x = vec![0.0f32; w];
        let mut win_y = vec![0.0f32; h];
        for x in 0..cols {
            let s = (std::f32::consts::PI * (x as f32 + 0.5) / cols as f32).sin();
            win_x[x] = 0.25 + 0.75 * s * s;
        }
        for y in 0..rows {
            let s = (std::f32::consts::PI * (y as f32 + 0.5) / rows as f32).sin();
            win_y[y] = 0.25 + 0.75 * s * s;
        }
        Self {
            fft: Fft2::new(w, h),
            w,
            h,
            band_h: rows,
            win_x,
            win_y,
            prev: None,
        }
    }

    /// Lift the viewport band into a 2D Clifford Spinor field $\Psi(x,y) \in \mathbb{G}_{2,0}^+$
    /// combining zero-mean luminance with the Dirac gradient spinor $\mathbf{e}_1 \nabla L = \partial_x L + \partial_y L\,\mathbf{e}_{12}$.
    fn clifford_spinor_spectrum(&self, frame: &RawFrame) -> Vec<Complex<f32>> {
        let w = self.w;
        let rows = self.band_h.min(frame.height as usize);
        let cols = self.w.min(frame.width as usize);
        let mut luma = vec![0.0f32; rows * cols];
        let mut sum = 0.0f32;
        for y in 0..rows {
            for x in 0..cols {
                let i = (y * frame.width as usize + x) * 3;
                let g = 0.299 * frame.rgb[i] as f32
                    + 0.587 * frame.rgb[i + 1] as f32
                    + 0.114 * frame.rgb[i + 2] as f32;
                luma[y * cols + x] = g;
                sum += g;
            }
        }
        let mean = if rows * cols > 0 {
            sum / (rows * cols) as f32
        } else {
            0.0
        };

        let mut buf = vec![Complex::new(0.0f32, 0.0f32); self.w * self.h];
        for y in 0..rows {
            let wy = self.win_y[y];
            let y_up = y.saturating_sub(1);
            let y_dn = (y + 1).min(rows.saturating_sub(1));
            for x in 0..cols {
                let wx = self.win_x[x];
                let w_xy = wx * wy;
                let x_lt = x.saturating_sub(1);
                let x_rt = (x + 1).min(cols.saturating_sub(1));
                let gx = 0.5 * (luma[y * cols + x_rt] - luma[y * cols + x_lt]);
                let gy = 0.5 * (luma[y_dn * cols + x] - luma[y_up * cols + x]);
                let dc = 0.35 * (luma[y * cols + x] - mean);
                buf[y * w + x] = Complex::new((dc + gx) * w_xy, gy * w_xy);
            }
        }
        self.fft.transform(&mut buf, false);
        buf
    }

    fn plane_peak(&self, plane: &[Complex<f32>], max_shift: i32) -> ([i32; 2], f32) {
        let mut best = (0i32, 0i32, f32::MIN);
        let mut sum = 0.0f32;
        let mut count = 0.0f32;
        for dy in -max_shift..=max_shift {
            for dx in -max_shift..=max_shift {
                let yy = dy.rem_euclid(self.h as i32) as usize;
                let xx = dx.rem_euclid(self.w as i32) as usize;
                let mag = plane[yy * self.w + xx].norm();
                sum += mag;
                count += 1.0;
                if mag > best.2 {
                    best = (dx, dy, mag);
                }
            }
        }
        let mean = if count > 0.0 { sum / count } else { 1.0 };
        let confidence = if mean > 1e-9 { best.2 / mean } else { 1.0 };
        ([best.0, best.1], confidence)
    }

    pub fn push(&mut self, frame: &RawFrame) -> ([i32; 2], f32) {
        let cur = self.clifford_spinor_spectrum(frame);
        let Some(prev) = self.prev.as_ref() else {
            self.prev = Some(cur);
            return ([0, 0], 1.0);
        };
        let n = self.w * self.h;
        let mut cross = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            cross[i] = prev[i] * cur[i].conj();
        }
        self.fft.transform(&mut cross, true);
        let (peak, confidence) = self.plane_peak(&cross, 8);
        self.prev = Some(cur);
        (peak, confidence)
    }
}

/// Frame-by-frame cumulative integer scroll offsets over the video band.
pub fn estimate_2d_scroll(frames: &[RawFrame], hud_top: usize) -> Vec<[i32; 2]> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut est = ScrollEstimator::new(frames[0].width as usize, hud_top);
    let mut cum = [0i32, 0];
    let mut out = Vec::with_capacity(frames.len());
    for f in frames {
        let (d, _) = est.push(f);
        cum[0] += d[0];
        cum[1] += d[1];
        out.push(cum);
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct HudPartition {
    pub top: usize,
    pub row_deviation: Vec<f32>,
}

pub fn partition_hud_and_viewport(frames: &[RawFrame]) -> HudPartition {
    assert!(!frames.is_empty(), "need at least one frame");
    let (w, h) = (frames[0].width as usize, frames[0].height as usize);
    let n_f = frames.len() as f32;
    let mut row_deviation = Vec::with_capacity(h);
    for y in 0..h {
        let mut devs = Vec::with_capacity(w);
        for x in 0..w {
            let mut max_ch_std = 0.0f32;
            for ch in 0..3 {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for f in frames {
                    let v = f.rgb[(y * w + x) * 3 + ch] as f32;
                    sum += v;
                    sum_sq += v * v;
                }
                let mean = sum / n_f;
                let var = (sum_sq / n_f - mean * mean).max(0.0);
                max_ch_std = max_ch_std.max(var.sqrt());
            }
            devs.push(max_ch_std);
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p94 = ((devs.len() * 94) / 100).min(devs.len().saturating_sub(1));
        row_deviation.push(devs[p94]);
    }
    let mut top = h;
    for y in (0..h).rev() {
        if row_deviation[y] < 6.0 {
            top = y;
        } else {
            break;
        }
    }
    HudPartition { top, row_deviation }
}

pub fn detect_camera_is_static(
    path: &Path,
    start: f64,
    w: usize,
    h: usize,
) -> io::Result<bool> {
    let mut reader = video_io::FrameReader::open_range(path, Some(start), Some(4.0))?;
    let playfield_h = (h * 7) / 10;
    let mut est = ScrollEstimator::new(w, playfield_h);
    let mut cum_x = 0i32;
    let mut cum_y = 0i32;
    let mut max_disp = 0i32;
    let mut count = 0usize;
    while let Some(frame) = reader.next_frame()? {
        let (d, _) = est.push(&frame);
        cum_x += d[0];
        cum_y += d[1];
        max_disp = max_disp.max(cum_x.abs()).max(cum_y.abs());
        count += 1;
        if count >= 120 {
            break;
        }
    }
    Ok(max_disp <= 4)
}

pub fn detect_static_borders(
    path: &Path,
    start: f64,
    w: usize,
    h: usize,
) -> io::Result<(usize, usize)> {
    let mut reader = video_io::FrameReader::open_range(path, Some(start), Some(1.0))?;
    let mut frames = Vec::new();
    reader.for_each(|_, f| frames.push(f.clone()))?;
    if frames.is_empty() {
        return Ok((0, h));
    }
    let bg = [frames[0].rgb[0], frames[0].rgb[1], frames[0].rgb[2]];
    let mut wall_counts = vec![0usize; h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let c = [frames[0].rgb[i], frames[0].rgb[i + 1], frames[0].rgb[i + 2]];
            if color_dist(c, bg) > 30 {
                wall_counts[y] += 1;
            }
        }
    }
    let threshold = (w * 6) / 10;
    let mut first_border = None;
    let mut last_border = None;
    for y in 0..h {
        if wall_counts[y] >= threshold {
            if first_border.is_none() {
                first_border = Some(y);
            }
            last_border = Some(y);
        }
    }
    match (first_border, last_border) {
        (Some(fb), Some(lb)) if lb > fb + 40 => {
            let top = (fb / 8) * 8;
            let bot = ((lb + 8) / 8) * 8;
            Ok((top, bot.min(h)))
        }
        _ => Ok((0, h)),
    }
}

pub fn detect_playfield_segment(path: &Path) -> io::Result<(f64, f64, VideoInfo)> {
    let info = video_io::probe(path)?;
    let mut reader = video_io::FrameReader::open(path)?;
    let fps = info.fps.max(1.0);
    let mut run_start: Option<usize> = None;
    let mut curr_bg: Option<[u8; 3]> = None;
    let mut last_run: Option<(usize, usize)> = None;
    let mut idx = 0usize;
    let mut best: Option<(usize, usize)> = None;
    while let Some(frame) = reader.next_frame()? {
        let bg = frame.pixel(5, 5);
        let mut nonblack = 0usize;
        let mut samples = 0usize;
        let mut y = 0;
        while y < frame.height as usize {
            let mut x = 0;
            while x < frame.width as usize {
                let i = (y * frame.width as usize + x) * 3;
                if frame.rgb[i] as u32 + frame.rgb[i + 1] as u32 + frame.rgb[i + 2] as u32 > 30 {
                    nonblack += 1;
                }
                samples += 1;
                x += 16;
            }
            y += 16;
        }
        let bright = nonblack as f32 / samples.max(1) as f32;
        let active = bright > 0.04;

        if active {
            match curr_bg {
                None => {
                    curr_bg = Some(bg);
                    run_start = Some(idx);
                }
                Some(prev_bg) => {
                    if color_dist(bg, prev_bg) > 25 {
                        if let Some(s) = run_start {
                            if idx - s >= 60 {
                                last_run = Some((s, idx));
                                if best.map_or(true, |(bs, be)| idx - s > be - bs) {
                                    best = Some((s, idx));
                                }
                            }
                        }
                        curr_bg = Some(bg);
                        run_start = Some(idx);
                    }
                }
            }
        } else {
            if let Some(s) = run_start {
                if idx - s >= 60 {
                    last_run = Some((s, idx));
                    if best.map_or(true, |(bs, be)| idx - s > be - bs) {
                        best = Some((s, idx));
                    }
                }
            }
            curr_bg = None;
            run_start = None;
        }
        idx += 1;
    }
    if let Some(s) = run_start {
        if idx - s >= 60 {
            last_run = Some((s, idx));
            if best.map_or(true, |(bs, be)| idx - s > be - bs) {
                best = Some((s, idx));
            }
        }
    }
    let (s, e) = best.or(last_run).ok_or_else(|| {
        io::Error::other("no playfield segment found (no stable scene run of >= 60 frames)")
    })?;
    let raw_start = s as f64 / fps;
    let raw_dur = (e - s) as f64 / fps;
    let (start_sec, dur_sec) = if s > 0 && raw_dur > 2.0 {
        let trim = 0.6;
        (raw_start + trim, (raw_dur - trim).clamp(1.0, 21.0))
    } else {
        (raw_start, raw_dur.clamp(1.0, 25.0))
    };
    Ok((start_sec, dur_sec, info))
}
