//! CFT Kinematic Timestamp Grounding & Snapping (`implicate-lexicon`).

use implicate_math::HolonPose2D;

/// An empirical CFT/PGA kinematic impulse boundary ($\|\Delta \mathbf{P}_B\| > \tau_{\text{imp}}$)
/// observed in the video stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicImpulseBoundary {
    /// Exact onset time in seconds ($t_{\text{start}} = f_{\text{start}} / \text{fps}$).
    pub t_start: f64,
    /// Exact offset time in seconds ($t_{\text{end}} = f_{\text{end}} / \text{fps}$).
    pub t_end: f64,
    /// Peak Bohmian momentum discontinuity $\|\Delta \mathbf{P}_B\|$.
    pub delta_momentum: f64,
}

/// Parse a timestamp string `HH:MM:SS` or `HH:MM:SS.mmm` into seconds.
pub fn parse_timestamp_sec(ts: &str) -> Option<f64> {
    let trimmed = ts.trim();
    let parts: Vec<&str> = trimmed.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let hours: f64 = parts[0].parse().ok()?;
    let mins: f64 = parts[1].parse().ok()?;
    let secs: f64 = parts[2].parse().ok()?;
    Some(hours * 3600.0 + mins * 60.0 + secs)
}

/// Parse a bracketed timestamp range `"[HH:MM:SS - HH:MM:SS]"` into `(t_start, t_end)` in seconds.
pub fn parse_timestamp_range(range_str: &str) -> Option<(f64, f64)> {
    let s = range_str.trim().trim_start_matches('[').trim_end_matches(']');
    let (left, right) = s.split_once('-')?;
    let t0 = parse_timestamp_sec(left)?;
    let t1 = parse_timestamp_sec(right)?;
    Some((t0.min(t1), t0.max(t1)))
}

/// Format `(t_start, t_end)` in seconds as `"[HH:MM:SS.mmm - HH:MM:SS.mmm]"`.
pub fn format_timestamp_range(t0: f64, t1: f64) -> String {
    fn fmt_one(t: f64) -> String {
        let total_ms = (t.max(0.0) * 1000.0).round() as u64;
        let ms = total_ms % 1000;
        let total_s = total_ms / 1000;
        let s = total_s % 60;
        let m = (total_s / 60) % 60;
        let h = total_s / 3600;
        format!("{h:02}:{m:02}:{s:02}.{ms:03}")
    }
    format!("[{} - {}]", fmt_one(t0), fmt_one(t1))
}

/// Compute 1D Temporal Intersection-over-Union (tIoU) between two time intervals `[a0, a1]` and `[b0, b1]`.
pub fn temporal_iou(a: (f64, f64), b: (f64, f64)) -> f64 {
    let inter_start = a.0.max(b.0);
    let inter_end = a.1.min(b.1);
    let inter = (inter_end - inter_start).max(0.0);
    let len_a = (a.1 - a.0).max(0.0);
    let len_b = (b.1 - b.0).max(0.0);
    let union = len_a + len_b - inter;
    if union <= 1e-9 {
        if (a.0 - b.0).abs() < 1e-6 { 1.0 } else { 0.0 }
    } else {
        inter / union
    }
}

/// Snap a coarse VLM timestamp interval `(vlm_t0, vlm_t1)` to the best-matching empirical
/// CFT kinematic impulse window (`KinematicImpulseBoundary`) within `max_snap_radius_sec`.
pub fn snap_verb_window_to_cft_impulses(
    vlm_window: (f64, f64),
    cft_impulses: &[KinematicImpulseBoundary],
    max_snap_radius_sec: f64,
) -> (f64, f64) {
    let vlm_mid = 0.5 * (vlm_window.0 + vlm_window.1);
    let mut best: Option<(f64, KinematicImpulseBoundary)> = None;

    for &imp in cft_impulses {
        let imp_mid = 0.5 * (imp.t_start + imp.t_end);
        let start_dist = (vlm_window.0 - imp.t_start).abs();
        let end_dist = (vlm_window.1 - imp.t_end).abs();
        let mid_dist = (vlm_mid - imp_mid).abs();

        if start_dist <= max_snap_radius_sec
            || end_dist <= max_snap_radius_sec
            || mid_dist <= max_snap_radius_sec
        {
            let tiou = temporal_iou(vlm_window, (imp.t_start, imp.t_end));
            let proximity = 1.0 / (1.0 + mid_dist + 0.5 * (start_dist + end_dist));
            let score =
                (0.65 * proximity + 0.35 * tiou) * (1.0 + 0.1 * imp.delta_momentum.min(10.0));
            if best.is_none_or(|(b_score, _)| score > b_score) {
                best = Some((score, imp));
            }
        }
    }

    if let Some((_, imp)) = best {
        (imp.t_start, imp.t_end)
    } else {
        vlm_window
    }
}

/// Detect bottom-up CFT kinematic impulse windows (`KinematicImpulseBoundary`) from a tracked
/// holon pose trajectory by locating Bohmian momentum discontinuities $\|\Delta \mathbf{P}_B(t)\|$.
pub fn detect_cft_kinematic_impulses(
    trajectory: &[HolonPose2D],
    fps: f64,
) -> Vec<KinematicImpulseBoundary> {
    let fps = fps.max(1.0);
    if trajectory.len() < 3 {
        let dur = (trajectory.len().max(1) as f64) / fps;
        return vec![KinematicImpulseBoundary {
            t_start: 0.0,
            t_end: dur,
            delta_momentum: 1.0,
        }];
    }

    let n = trajectory.len();
    let mut vels = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dx = trajectory[i + 1].x - trajectory[i].x;
        let dy = trajectory[i + 1].y - trajectory[i].y;
        vels.push([dx * fps, dy * fps]);
    }

    let mut impulses = Vec::with_capacity(vels.len().saturating_sub(1));
    let mut sum_dp = 0.0;
    for i in 0..(vels.len() - 1) {
        let dvx = vels[i + 1][0] - vels[i][0];
        let dvy = vels[i + 1][1] - vels[i][1];
        let dp = (dvx * dvx + dvy * dvy).sqrt();
        impulses.push(dp);
        sum_dp += dp;
    }
    let mean_dp = sum_dp / (impulses.len().max(1) as f64);
    let threshold = (mean_dp * 1.15).max(1e-3);

    // Partition trajectory at significant momentum discontinuities
    let mut cut_frames = vec![0usize];
    let mut peak_between = Vec::new();
    let mut cur_peak = mean_dp.max(0.5);
    for (i, &dp) in impulses.iter().enumerate() {
        cur_peak = cur_peak.max(dp);
        let frame_idx = i + 1;
        let prev_cut = *cut_frames.last().unwrap();
        if dp >= threshold && (frame_idx - prev_cut) >= 4 && (n - frame_idx) >= 4 {
            cut_frames.push(frame_idx);
            peak_between.push(cur_peak);
            cur_peak = dp;
        }
    }
    cut_frames.push(n - 1);
    peak_between.push(cur_peak.max(1.0));

    let mut out = Vec::with_capacity(peak_between.len());
    for w_idx in 0..peak_between.len() {
        let f0 = cut_frames[w_idx];
        let f1 = cut_frames[w_idx + 1].max(f0 + 1);
        out.push(KinematicImpulseBoundary {
            t_start: (f0 as f64) / fps,
            t_end: (f1 as f64) / fps,
            delta_momentum: peak_between[w_idx],
        });
    }
    out
}
