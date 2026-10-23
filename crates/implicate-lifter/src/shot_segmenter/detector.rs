//! Streaming CFT + Histogram Shot Segmenter engine (`shot_segmenter::detector`).

use std::io;
use std::path::Path;
use std::time::Instant;

use rustfft::num_complex::Complex;

use super::{
    ANALYSIS_H, ANALYSIS_W, CameraRegime, FrameTransitionMetrics, HIST_BINS, PARALLAX_STRIPS,
    ShotSegment, ShotSegmentationReport, ShotSegmenterConfig, TransitionKind,
    downsample_to_analysis, find_primary_and_secondary_cft_peaks, grid_l1_distance,
    isolate_hud_and_bezels, linear_dissolve_fit, symmetric_kl,
};
use crate::enfold2d::Fft2;
use crate::video_io::{self, FrameReader, RawFrame};

#[derive(Clone)]
struct CompactFrameState {
    hist: [f32; HIST_BINS],
    grid_rgb: [f32; 16 * 8 * 3],
    global_spec: Vec<Complex<f32>>,
    strip_specs: [Vec<Complex<f32>>; PARALLAX_STRIPS],
    has_signal: bool,
    mean_luma: f32,
}

/// Streaming CFT + Histogram Shot Segmenter engine.
pub struct StreamingShotSegmenter {
    pub config: ShotSegmenterConfig,
    fft_global: Fft2,
    fft_strip: Fft2,
}

impl StreamingShotSegmenter {
    pub fn new(config: ShotSegmenterConfig) -> Self {
        Self {
            config,
            fft_global: Fft2::new(ANALYSIS_W, ANALYSIS_H),
            fft_strip: Fft2::new(ANALYSIS_W, ANALYSIS_H / PARALLAX_STRIPS),
        }
    }

    pub fn segment_video_range(
        &self,
        path: &Path,
        start_sec: Option<f64>,
        duration_sec: Option<f64>,
    ) -> io::Result<ShotSegmentationReport> {
        let info = video_io::probe(path)?;
        let t0 = Instant::now();
        let mut reader = FrameReader::open_range_scaled(
            path,
            start_sec,
            duration_sec,
            ANALYSIS_W as u32,
            ANALYSIS_H as u32,
        )?;
        let mut states: Vec<CompactFrameState> = Vec::new();
        let mut var_sum = vec![0.0f32; ANALYSIS_W * ANALYSIS_H];
        let mut var_sq_sum = vec![0.0f32; ANALYSIS_W * ANALYSIS_H];
        let mut raw = RawFrame::blank(ANALYSIS_W as u32, ANALYSIS_H as u32);

        while reader.next_frame_into(&mut raw)? {
            let st = self.extract_frame_state(&raw, &mut var_sum, &mut var_sq_sum);
            states.push(st);
        }

        let elapsed_sec = t0.elapsed().as_secs_f64().max(1e-6);
        Ok(self.finalize_states(
            &states,
            &var_sum,
            &var_sq_sum,
            info.width,
            info.height,
            info.fps,
            start_sec.unwrap_or(0.0),
            elapsed_sec,
        ))
    }

    pub fn segment_frames(
        &self,
        frames: &[RawFrame],
        fps: f64,
        start_sec: f64,
    ) -> ShotSegmentationReport {
        let t0 = Instant::now();
        let (native_w, native_h) = frames
            .first()
            .map(|f| (f.width, f.height))
            .unwrap_or((ANALYSIS_W as u32, ANALYSIS_H as u32));
        let mut states = Vec::with_capacity(frames.len());
        let mut var_sum = vec![0.0f32; ANALYSIS_W * ANALYSIS_H];
        let mut var_sq_sum = vec![0.0f32; ANALYSIS_W * ANALYSIS_H];
        let mut scaled = RawFrame::blank(ANALYSIS_W as u32, ANALYSIS_H as u32);

        for f in frames {
            downsample_to_analysis(f, &mut scaled);
            let st = self.extract_frame_state(&scaled, &mut var_sum, &mut var_sq_sum);
            states.push(st);
        }

        let elapsed_sec = t0.elapsed().as_secs_f64().max(1e-6);
        self.finalize_states(
            &states,
            &var_sum,
            &var_sq_sum,
            native_w,
            native_h,
            fps,
            start_sec,
            elapsed_sec,
        )
    }

    fn extract_frame_state(
        &self,
        frame: &RawFrame,
        var_sum: &mut [f32],
        var_sq_sum: &mut [f32],
    ) -> CompactFrameState {
        let mut hist = [1e-4f32; HIST_BINS];
        let mut grid_rgb = [0.0f32; 16 * 8 * 3];
        let mut luma = [0.0f32; ANALYSIS_W * ANALYSIS_H];
        let mut luma_sum = 0.0f32;

        let mut y_min = 0usize;
        let mut y_max = ANALYSIS_H;
        for y in 0..8 {
            let row_sum: u32 = (0..ANALYSIS_W)
                .map(|x| {
                    let i = (y * ANALYSIS_W + x) * 3;
                    frame.rgb[i] as u32 + frame.rgb[i + 1] as u32 + frame.rgb[i + 2] as u32
                })
                .sum();
            if row_sum < (ANALYSIS_W as u32 * 18) {
                y_min = y + 1;
            } else {
                break;
            }
        }
        for y in (ANALYSIS_H - 8..ANALYSIS_H).rev() {
            let row_sum: u32 = (0..ANALYSIS_W)
                .map(|x| {
                    let i = (y * ANALYSIS_W + x) * 3;
                    frame.rgb[i] as u32 + frame.rgb[i + 1] as u32 + frame.rgb[i + 2] as u32
                })
                .sum();
            if row_sum < (ANALYSIS_W as u32 * 18) {
                y_max = y;
            } else {
                break;
            }
        }
        if y_max <= y_min + 16 {
            y_min = 0;
            y_max = ANALYSIS_H;
        }

        let active_px = ((y_max - y_min) * ANALYSIS_W).max(1) as f32;
        for y in 0..ANALYSIS_H {
            let gy = y / 8;
            for x in 0..ANALYSIS_W {
                let gx = x / 8;
                let idx = y * ANALYSIS_W + x;
                let p = idx * 3;
                let (r, g, b) = (
                    frame.rgb[p] as f32,
                    frame.rgb[p + 1] as f32,
                    frame.rgb[p + 2] as f32,
                );
                let y_val = 0.299 * r + 0.587 * g + 0.114 * b;
                luma[idx] = y_val;
                var_sum[idx] += y_val;
                var_sq_sum[idx] += y_val * y_val;

                let g_idx = (gy * 16 + gx) * 3;
                grid_rgb[g_idx] += r / (64.0 * 255.0);
                grid_rgb[g_idx + 1] += g / (64.0 * 255.0);
                grid_rgb[g_idx + 2] += b / (64.0 * 255.0);

                if y >= y_min && y < y_max {
                    luma_sum += y_val;
                    let y_bin = ((y_val as usize) >> 3).min(31);
                    let cr = ((r - y_val + 128.0).clamp(0.0, 255.0) as usize >> 4).min(15);
                    let cb = ((b - y_val + 128.0).clamp(0.0, 255.0) as usize >> 4).min(15);
                    hist[y_bin] += 1.0;
                    hist[32 + cr] += 0.5;
                    hist[48 + cb] += 0.5;
                }
            }
        }

        let hist_sum: f32 = hist.iter().sum();
        for h in &mut hist {
            *h /= hist_sum;
        }

        let mean_luma = luma_sum / active_px;
        let mut global_buf = vec![Complex::new(0.0f32, 0.0f32); ANALYSIS_W * ANALYSIS_H];
        let mut energy = 0.0f32;
        for y in y_min..y_max {
            let wy = 0.5
                * (1.0
                    - (2.0 * std::f32::consts::PI * (y - y_min) as f32
                        / (y_max - y_min).max(1) as f32)
                        .cos());
            for x in 0..ANALYSIS_W {
                let wx = 0.5
                    * (1.0 - (2.0 * std::f32::consts::PI * x as f32 / ANALYSIS_W as f32).cos());
                let w = (0.25 + 0.75 * wx * wy).sqrt();
                let v = (luma[y * ANALYSIS_W + x] - mean_luma) * w;
                global_buf[y * ANALYSIS_W + x] = Complex::new(v, 0.0);
                energy += v * v;
            }
        }
        let has_signal = energy > 250.0;
        if has_signal {
            let inv_norm = 1.0 / energy.sqrt();
            for c in &mut global_buf {
                *c *= inv_norm;
            }
            self.fft_global.transform(&mut global_buf, false);
        }

        let strip_h = ANALYSIS_H / PARALLAX_STRIPS;
        let strip_len = ANALYSIS_W * strip_h;
        let mut strip_specs: [Vec<Complex<f32>>; PARALLAX_STRIPS] =
            std::array::from_fn(|_| vec![Complex::new(0.0, 0.0); strip_len]);
        for s in 0..PARALLAX_STRIPS {
            let sy0 = s * strip_h;
            let sy1 = sy0 + strip_h;
            let mut s_sum = 0.0f32;
            for y in sy0..sy1 {
                for x in 0..ANALYSIS_W {
                    s_sum += luma[y * ANALYSIS_W + x];
                }
            }
            let s_mean = s_sum / strip_len as f32;
            let mut s_energy = 0.0f32;
            for y in sy0..sy1 {
                for x in 0..ANALYSIS_W {
                    let v = luma[y * ANALYSIS_W + x] - s_mean;
                    strip_specs[s][(y - sy0) * ANALYSIS_W + x] = Complex::new(v, 0.0);
                    s_energy += v * v;
                }
            }
            if s_energy > 80.0 {
                let inv = 1.0 / s_energy.sqrt();
                for c in &mut strip_specs[s] {
                    *c *= inv;
                }
                self.fft_strip.transform(&mut strip_specs[s], false);
            } else {
                strip_specs[s].fill(Complex::new(0.0, 0.0));
            }
        }

        CompactFrameState {
            hist,
            grid_rgb,
            global_spec: global_buf,
            strip_specs,
            has_signal,
            mean_luma,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finalize_states(
        &self,
        states: &[CompactFrameState],
        var_sum: &[f32],
        var_sq_sum: &[f32],
        native_width: u32,
        native_height: u32,
        fps: f64,
        start_sec: f64,
        elapsed_sec: f64,
    ) -> ShotSegmentationReport {
        let n = states.len();
        let fps = fps.max(1.0);
        let throughput_fps = n as f64 / elapsed_sec.max(1e-6);
        let (bezel_and_hud_rects, active_y_bounds) =
            isolate_hud_and_bezels(var_sum, var_sq_sum, n, native_width, native_height);

        if n == 0 {
            return ShotSegmentationReport {
                native_width,
                native_height,
                fps,
                total_frames: 0,
                elapsed_sec,
                throughput_fps,
                bezel_and_hud_rects,
                active_y_bounds,
                shots: Vec::new(),
                frame_metrics: Vec::new(),
            };
        }

        let mut frame_metrics = Vec::with_capacity(n);
        frame_metrics.push(FrameTransitionMetrics {
            frame_idx: 0,
            cross_corr_peak: 1.0,
            secondary_peak_ratio: 0.0,
            kl_divergence: 0.0,
            grid_l1_jump: 0.0,
            global_flow: [0.0, 0.0],
            strip_flows: [[0.0, 0.0]; PARALLAX_STRIPS],
            radial_divergence: 0.0,
        });

        let scale_x = 320.0 / ANALYSIS_W as f32;
        let scale_y = 180.0 / ANALYSIS_H as f32;
        let mut cross_buf = vec![Complex::new(0.0f32, 0.0f32); ANALYSIS_W * ANALYSIS_H];
        let strip_h = ANALYSIS_H / PARALLAX_STRIPS;
        let mut strip_cross = vec![Complex::new(0.0f32, 0.0f32); ANALYSIS_W * strip_h];

        for t in 1..n {
            let (prev, curr) = (&states[t - 1], &states[t]);
            let kl = symmetric_kl(&prev.hist, &curr.hist);
            let grid_l1 = grid_l1_distance(&prev.grid_rgb, &curr.grid_rgb);

            let (corr_peak, sec_ratio, g_shift) = if prev.has_signal && curr.has_signal {
                for i in 0..cross_buf.len() {
                    cross_buf[i] = prev.global_spec[i] * curr.global_spec[i].conj();
                }
                self.fft_global.transform(&mut cross_buf, true);
                find_primary_and_secondary_cft_peaks(&cross_buf, ANALYSIS_W, ANALYSIS_H, 16)
            } else {
                let luma_diff = (prev.mean_luma - curr.mean_luma).abs() / 255.0;
                ((1.0 - luma_diff * 3.0).clamp(0.0, 1.0), 0.0, [0.0, 0.0])
            };

            let global_flow = [g_shift[0] * scale_x, g_shift[1] * scale_y];
            let mut strip_flows = [[0.0f32; 2]; PARALLAX_STRIPS];
            for s in 0..PARALLAX_STRIPS {
                let mut non_zero = false;
                for i in 0..strip_cross.len() {
                    let z = prev.strip_specs[s][i] * curr.strip_specs[s][i].conj();
                    if z.norm_sqr() > 1e-12 {
                        non_zero = true;
                    }
                    strip_cross[i] = z;
                }
                if non_zero {
                    self.fft_strip.transform(&mut strip_cross, true);
                    let (pk, _, sh) = find_primary_and_secondary_cft_peaks(
                        &strip_cross,
                        ANALYSIS_W,
                        strip_h,
                        14,
                    );
                    if pk > 0.15 {
                        strip_flows[s] = [sh[0] * scale_x, sh[1] * scale_y];
                    }
                }
            }

            let radial_divergence = (strip_flows[PARALLAX_STRIPS - 1][1] - strip_flows[0][1]) * 0.5;
            frame_metrics.push(FrameTransitionMetrics {
                frame_idx: t,
                cross_corr_peak: corr_peak,
                secondary_peak_ratio: sec_ratio,
                kl_divergence: kl,
                grid_l1_jump: grid_l1,
                global_flow,
                strip_flows,
                radial_divergence,
            });
        }

        let min_gap = ((self.config.min_shot_duration_sec * fps).round() as usize).max(5);
        let mut boundaries: Vec<(usize, TransitionKind)> = vec![(0, TransitionKind::StreamStart)];
        let mut cut_scores = vec![0.0f32; n];
        for t in 1..n {
            let m = &frame_metrics[t];
            cut_scores[t] =
                m.kl_divergence + 1.6 * (1.0 - m.cross_corr_peak).max(0.0) * m.grid_l1_jump;
        }

        let mut last_boundary = 0usize;
        let mut t = 1usize;
        while t < n {
            if t - last_boundary < min_gap {
                t += 1;
                continue;
            }
            let m = &frame_metrics[t];
            let raw_cut_trigger =
                (m.kl_divergence >= self.config.tau_hist && m.grid_l1_jump >= 0.075)
                    || (m.cross_corr_peak < self.config.tau_cut && m.grid_l1_jump >= 0.11)
                    || (cut_scores[t] >= 0.42 && m.grid_l1_jump >= 0.08);

            if raw_cut_trigger {
                let prev_s = if t > 1 { cut_scores[t - 1] } else { 0.0 };
                let next_s = if t + 1 < n { cut_scores[t + 1] } else { 0.0 };
                if cut_scores[t] >= prev_s && cut_scores[t] >= next_s {
                    let pre_idx = t.saturating_sub(2).max(last_boundary);
                    let post_idx = (t + 2).min(n - 1);
                    let sustained_kl = symmetric_kl(&states[pre_idx].hist, &states[post_idx].hist);
                    let sustained_l1 =
                        grid_l1_distance(&states[pre_idx].grid_rgb, &states[post_idx].grid_rgb);
                    let post_internal_l1 = if t + 1 < n {
                        grid_l1_distance(&states[t].grid_rgb, &states[t + 1].grid_rgb)
                    } else {
                        0.0
                    };
                    if (sustained_kl >= self.config.tau_hist * 0.55 || sustained_l1 >= 0.075)
                        && post_internal_l1 <= m.grid_l1_jump * 0.85 + 0.02
                    {
                        boundaries.push((t, TransitionKind::HardCut));
                        last_boundary = t;
                        t += min_gap;
                        continue;
                    }
                }
            }

            let w = 5usize;
            if t >= w && t + w < n && t - last_boundary >= min_gap {
                let span_kl = symmetric_kl(&states[t - w].hist, &states[t + w].hist);
                let span_l1 = grid_l1_distance(&states[t - w].grid_rgb, &states[t + w].grid_rgb);
                let max_step_l1 = (t - w + 1..=t + w)
                    .map(|k| frame_metrics[k].grid_l1_jump)
                    .fold(0.0f32, f32::max);
                if max_step_l1 < span_l1 * 0.72
                    && (span_kl >= self.config.tau_dissolve * 0.70 || span_l1 >= 0.11)
                    && span_l1 >= 0.07
                {
                    let (alpha, blend_err) = linear_dissolve_fit(
                        &states[t - w].grid_rgb,
                        &states[t].grid_rgb,
                        &states[t + w].grid_rgb,
                    );
                    let span_kl_prev = if t > w {
                        symmetric_kl(&states[t - 1 - w].hist, &states[t - 1 + w].hist)
                    } else {
                        0.0
                    };
                    let span_kl_next = if t + 1 + w < n {
                        symmetric_kl(&states[t + 1 - w].hist, &states[t + 1 + w].hist)
                    } else {
                        0.0
                    };
                    if (0.15..=0.85).contains(&alpha)
                        && blend_err < 0.52 * span_l1
                        && span_kl >= span_kl_prev
                        && span_kl >= span_kl_next
                    {
                        boundaries.push((
                            t,
                            TransitionKind::CrossDissolve {
                                duration_frames: 2 * w,
                            },
                        ));
                        last_boundary = t;
                        t += min_gap;
                        continue;
                    }
                }
            }
            t += 1;
        }

        let mut shots = Vec::with_capacity(boundaries.len());
        for (shot_id, &(b_frame, trans)) in boundaries.iter().enumerate() {
            let f_end = boundaries
                .get(shot_id + 1)
                .map(|&(nf, _)| nf.saturating_sub(1))
                .unwrap_or(n - 1)
                .max(b_frame);
            shots.push(self.build_shot_segment(
                shot_id,
                b_frame,
                f_end,
                trans,
                &frame_metrics,
                fps,
                start_sec,
            ));
        }

        ShotSegmentationReport {
            native_width,
            native_height,
            fps,
            total_frames: n,
            elapsed_sec,
            throughput_fps,
            bezel_and_hud_rects,
            active_y_bounds,
            shots,
            frame_metrics,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_shot_segment(
        &self,
        shot_id: usize,
        frame_start: usize,
        frame_end: usize,
        transition_in: TransitionKind,
        metrics: &[FrameTransitionMetrics],
        fps: f64,
        base_start_sec: f64,
    ) -> ShotSegment {
        let entry_kl = metrics
            .get(frame_start)
            .map(|m| m.kl_divergence)
            .unwrap_or(0.0);
        let motion_start = (frame_start + 1).min(frame_end);
        let (mut cum_x, mut cum_y, mut max_disp) = (0.0f32, 0.0f32, 0.0f32);
        let (mut sum_vx, mut sum_vy, mut sum_corr, mut sum_div) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut strip_cum = [[0.0f32; 2]; PARALLAX_STRIPS];
        let mut strip_active = [0usize; PARALLAX_STRIPS];
        let mut sec_peak_hits = 0usize;
        let mut count = 0usize;

        for m in &metrics[motion_start..=frame_end] {
            if m.frame_idx == 0 {
                continue;
            }
            cum_x += m.global_flow[0];
            cum_y += m.global_flow[1];
            max_disp = max_disp.max(cum_x.hypot(cum_y));
            sum_vx += m.global_flow[0];
            sum_vy += m.global_flow[1];
            sum_corr += m.cross_corr_peak;
            sum_div += m.radial_divergence;
            if m.secondary_peak_ratio >= 0.34
                && m.global_flow[0].hypot(m.global_flow[1]) >= 0.85
            {
                sec_peak_hits += 1;
            }
            for s in 0..PARALLAX_STRIPS {
                let sf = m.strip_flows[s][0].hypot(m.strip_flows[s][1]);
                if sf > 0.25 {
                    strip_cum[s][0] += m.strip_flows[s][0];
                    strip_cum[s][1] += m.strip_flows[s][1];
                    strip_active[s] += 1;
                }
            }
            count += 1;
        }

        let denom = count.max(1) as f32;
        let mean_vx = sum_vx / denom;
        let mean_vy = sum_vy / denom;
        let mean_corr = if count > 0 { sum_corr / denom } else { 1.0 };
        let mean_div = (sum_div / denom).abs();

        // Compare per-active-frame velocity across strips that have sustained signal (>= 50% of frames).
        let min_active = (count / 2).max(3);
        let mut strip_speeds: Vec<f32> = (0..PARALLAX_STRIPS)
            .filter(|&s| strip_active[s] >= min_active)
            .map(|s| {
                let disp = strip_cum[s][0].hypot(strip_cum[s][1]);
                if disp > 3.0 {
                    disp / strip_active[s] as f32
                } else {
                    0.0
                }
            })
            .filter(|&v| v > 0.4)
            .collect();
        strip_speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let strip_shear = if strip_speeds.len() >= 2 {
            let slowest = strip_speeds[0];
            let fastest = *strip_speeds.last().unwrap();
            if slowest >= 0.5 && fastest >= 1.2 {
                (fastest - slowest) / fastest
            } else {
                0.0
            }
        } else {
            0.0
        };

        let parallax_shear_ratio = strip_shear.max((sec_peak_hits as f32 / denom) * 0.65);
        let camera_regime = if max_disp <= self.config.static_max_disp_px && mean_div < 0.18 {
            CameraRegime::Ortho2DStatic
        } else if mean_div >= 0.55 && mean_div > mean_vx.hypot(mean_vy) * 0.65 {
            CameraRegime::Perspective3D
        } else if parallax_shear_ratio >= self.config.multiplane_shear_threshold
            || (mean_div >= 0.16 && max_disp > self.config.static_max_disp_px)
        {
            CameraRegime::MultiPlane25D
        } else {
            CameraRegime::Ortho2DScroll
        };

        ShotSegment {
            shot_id,
            frame_start,
            frame_end,
            start_sec: base_start_sec + frame_start as f64 / fps,
            end_sec: base_start_sec + (frame_end + 1) as f64 / fps,
            transition_in,
            camera_regime,
            mean_camera_velocity: [mean_vx, mean_vy],
            max_camera_displacement: max_disp,
            parallax_shear_ratio,
            divergence_score: mean_div,
            mean_cross_corr: mean_corr,
            entry_kl_divergence: entry_kl,
        }
    }
}
