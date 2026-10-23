//! Spinor-coherent multi-frame actor tracker (`enfold2d::tracker2d`).

use serde::{Deserialize, Serialize};
use super::{Rect, color_dist};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorKind {
    Player,
    Autonomous,
    Unknown,
}

impl ActorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ActorKind::Player => "player",
            ActorKind::Autonomous => "autonomous",
            ActorKind::Unknown => "unknown",
        }
    }
}

/// Canonical RGBA sprite kernel with DC mode ($\omega_0$) and first temporal CFT harmonic ($\omega_1$).
#[derive(Clone, Debug, PartialEq)]
pub struct Sprite2D {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub harmonic_rgba: Vec<u8>,
}

impl Sprite2D {
    pub fn alpha_at(&self, x: usize, y: usize) -> u8 {
        self.rgba[(y * self.width + x) * 4 + 3]
    }
}

#[derive(Clone, Debug)]
pub struct ActorTrack {
    pub kind: ActorKind,
    pub boxes: Vec<(usize, Rect)>,
    pub color: [u8; 3],
    pub sprite: Option<Sprite2D>,
}

impl ActorTrack {
    pub fn median_size(&self) -> (i32, i32) {
        if self.boxes.is_empty() {
            return (0, 0);
        }
        let mut ws: Vec<i32> = self.boxes.iter().map(|(_, b)| b.width()).collect();
        let mut hs: Vec<i32> = self.boxes.iter().map(|(_, b)| b.height()).collect();
        ws.sort_unstable();
        hs.sort_unstable();
        (ws[ws.len() / 2], hs[hs.len() / 2])
    }

    pub fn first_world_center(&self) -> Option<[f64; 2]> {
        self.boxes.first().map(|(_, b)| b.center())
    }
}

pub fn track_actors(boxes_per_frame: &[Vec<Rect>], max_gap: usize) -> Vec<ActorTrack> {
    let colored: Vec<Vec<(Rect, [u8; 3])>> = boxes_per_frame
        .iter()
        .map(|v| v.iter().map(|b| (*b, [128, 128, 128])).collect())
        .collect();
    let scroll = vec![[0, 0]; boxes_per_frame.len()];
    track_actors_coherent(&colored, &scroll, 256, false, max_gap).0
}

pub fn track_actors_coherent(
    boxes_per_frame: &[Vec<(Rect, [u8; 3])>],
    scroll: &[[i32; 2]],
    viewport_w: usize,
    is_static: bool,
    max_gap: usize,
) -> (Vec<ActorTrack>, Vec<ActorTrack>) {
    struct Open {
        boxes: Vec<(usize, Rect)>,
        last: usize,
        last_box: Rect,
        color: [f64; 3],
    }
    let mut open: Vec<Open> = Vec::new();
    let mut closed: Vec<ActorTrack> = Vec::new();
    let max_dist = if is_static { 24.0 } else { 32.0 };
    let vw = viewport_w.max(1) as f64;

    for (fi, boxes) in boxes_per_frame.iter().enumerate() {
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, t) in open.iter().enumerate() {
            if fi - t.last > max_gap {
                continue;
            }
            let tc = t.last_box.center();
            let tw = t.last_box.width();
            let th = t.last_box.height();
            for (bi, (b, bcol)) in boxes.iter().enumerate() {
                let c = b.center();
                let dx = if is_static {
                    let raw = (c[0] - tc[0]).abs();
                    raw.min(vw - raw)
                } else {
                    (c[0] - tc[0]).abs()
                };
                let dy = (c[1] - tc[1]).abs();
                let d = (dx * dx + dy * dy).sqrt();
                let cd = (bcol[0] as f64 - t.color[0])
                    .abs()
                    .max((bcol[1] as f64 - t.color[1]).abs())
                    .max((bcol[2] as f64 - t.color[2]).abs());
                let size_ok = (b.width() - tw).abs() <= 12 && (b.height() - th).abs() <= 12;
                if d <= max_dist && cd <= 55.0 && size_ok {
                    pairs.push((d + 0.4 * cd, ti, bi));
                }
            }
        }
        let orig_open_len = open.len();
        let mut used_t = vec![false; orig_open_len];
        let mut used_b = vec![false; boxes.len()];

        for (bi, (b, _)) in boxes.iter().enumerate() {
            let has_direct_size_match = open.iter().take(orig_open_len).any(|t| {
                fi - t.last <= max_gap
                    && (b.width() - t.last_box.width()).abs() <= 6
                    && (b.height() - t.last_box.height()).abs() <= 6
            });
            if has_direct_size_match {
                continue;
            }
            let mut nearby_open: Vec<(usize, f64)> = open
                .iter()
                .enumerate()
                .take(orig_open_len)
                .filter(|(ti, t)| {
                    !used_t[*ti]
                        && fi - t.last <= max_gap
                        && t.boxes.len() >= 3
                        && t.last_box.width() >= 10
                        && t.last_box.height() >= 10
                })
                .filter_map(|(ti, t)| {
                    let tc = t.last_box.center();
                    let c = b.center();
                    let dx = if is_static {
                        let raw = (c[0] - tc[0]).abs();
                        raw.min(vw - raw)
                    } else {
                        (c[0] - tc[0]).abs()
                    };
                    let dy = (c[1] - tc[1]).abs();
                    let d = (dx * dx + dy * dy).sqrt();
                    let tw = t.last_box.width();
                    let th = t.last_box.height();
                    if d <= max_dist * 1.25
                        && b.area() >= (tw * th * 5) / 4
                        && b.area() <= tw * th * 3
                    {
                        Some((ti, d))
                    } else {
                        None
                    }
                })
                .collect();
            nearby_open.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if nearby_open.len() >= 2 {
                for &(ti, _) in nearby_open.iter().take(2) {
                    used_t[ti] = true;
                    let t = &mut open[ti];
                    let tw = t.last_box.width();
                    let th = t.last_box.height();
                    let tc = t.last_box.center();
                    let sx0 =
                        (tc[0].round() as i32 - tw / 2).clamp(b.x0, (b.x1 - tw + 1).max(b.x0));
                    let sy0 =
                        (tc[1].round() as i32 - th / 2).clamp(b.y0, (b.y1 - th + 1).max(b.y0));
                    let sub = Rect {
                        x0: sx0,
                        y0: sy0,
                        x1: (sx0 + tw - 1).min(b.x1),
                        y1: (sy0 + th - 1).min(b.y1),
                    };
                    t.boxes.push((fi, sub));
                    t.last = fi;
                    t.last_box = sub;
                }
                used_b[bi] = true;
            }
        }

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, ti, bi) in pairs {
            if used_t[ti] || used_b[bi] {
                continue;
            }
            used_t[ti] = true;
            used_b[bi] = true;
            let (b, bcol) = boxes[bi];
            let t = &mut open[ti];
            t.boxes.push((fi, b));
            t.last = fi;
            t.last_box = b;
            for k in 0..3 {
                t.color[k] = 0.85 * t.color[k] + 0.15 * bcol[k] as f64;
            }
        }
        for (bi, (b, bcol)) in boxes.iter().enumerate() {
            if !used_b[bi] {
                open.push(Open {
                    boxes: vec![(fi, *b)],
                    last: fi,
                    last_box: *b,
                    color: [bcol[0] as f64, bcol[1] as f64, bcol[2] as f64],
                });
            }
        }
        open.retain(|t| {
            if fi - t.last > max_gap {
                closed.push(ActorTrack {
                    kind: ActorKind::Unknown,
                    boxes: t.boxes.clone(),
                    color: [
                        t.color[0].round() as u8,
                        t.color[1].round() as u8,
                        t.color[2].round() as u8,
                    ],
                    sprite: None,
                });
                false
            } else {
                true
            }
        });
    }
    for t in open {
        closed.push(ActorTrack {
            kind: ActorKind::Unknown,
            boxes: t.boxes,
            color: [
                t.color[0].round() as u8,
                t.color[1].round() as u8,
                t.color[2].round() as u8,
            ],
            sprite: None,
        });
    }

    let scroll_is_trivial = scroll.iter().all(|s| s[0] == 0 && s[1] == 0);
    let world_span = |t: &ActorTrack| -> (f64, f64) {
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        for &(fi, b) in &t.boxes {
            let s = scroll.get(fi).copied().unwrap_or([0, 0]);
            let wx = b.x0 as f64 + s[0] as f64;
            let wy = b.y0 as f64 + s[1] as f64;
            min_x = min_x.min(wx);
            max_x = max_x.max(wx);
            min_y = min_y.min(wy);
            max_y = max_y.max(wy);
        }
        ((max_x - min_x).max(0.0), (max_y - min_y).max(0.0))
    };

    closed.retain(|t| {
        if t.boxes.len() < 6 || t.median_size() == (0, 0) {
            return false;
        }
        if is_static {
            let (dx, dy) = world_span(t);
            dx.max(dy) >= 12.0
        } else {
            t.boxes.len() >= 8
        }
    });

    closed.sort_by_key(|t| std::cmp::Reverse(t.boxes.len()));
    let all_moving = closed.clone();
    let mut merged: Vec<ActorTrack> = Vec::new();
    for t in closed {
        let (tw, th) = t.median_size();
        let mut placed = false;
        for m in &mut merged {
            if color_dist(t.color, m.color) > 35 {
                continue;
            }
            let (mw, mh) = m.median_size();
            if (tw - mw).abs() > 8 || (th - mh).abs() > 8 {
                continue;
            }
            let overlaps = t
                .boxes
                .iter()
                .any(|(fi, _)| m.boxes.iter().any(|(mfi, _)| mfi == fi));
            if overlaps && !is_static {
                continue;
            }
            let can_merge = if is_static {
                true
            } else {
                let (tfi, tb) = t.boxes.first().unwrap();
                let ts = scroll.get(*tfi).copied().unwrap_or([0, 0]);
                let tx = tb.center()[0] + ts[0] as f64;
                let ty = tb.center()[1] + ts[1] as f64;
                let (mfi, mb) = m.boxes.last().unwrap();
                let ms = scroll.get(*mfi).copied().unwrap_or([0, 0]);
                let mx = mb.center()[0] + ms[0] as f64;
                let my = mb.center()[1] + ms[1] as f64;
                (tx - mx).abs() <= 96.0 && (ty - my).abs() <= 28.0
            };
            if can_merge {
                for &(fi, rect) in &t.boxes {
                    if !m.boxes.iter().any(|(mfi, _)| *mfi == fi) {
                        m.boxes.push((fi, rect));
                    }
                }
                m.boxes.sort_by_key(|(fi, _)| *fi);
                placed = true;
                break;
            }
        }
        if !placed {
            merged.push(t);
        }
    }

    merged.retain(|t| {
        if t.boxes.len() < 8 {
            return false;
        }
        let (w, h) = t.median_size();
        if !is_static && (w < 12 || h < 13) {
            return false;
        }
        if !is_static && scroll_is_trivial {
            return true;
        }
        let (dx, dy) = world_span(t);
        if is_static {
            dx.max(dy) >= 12.0
        } else {
            dx >= 28.0 || dy >= 24.0
        }
    });
    merged.sort_by_key(|t| std::cmp::Reverse(t.boxes.len()));
    if is_static && !merged.is_empty() {
        let (pw, ph) = merged[0].median_size();
        let primary_area = (pw * ph).max(1);
        let primary_len = merged[0].boxes.len();
        merged.retain(|t| {
            let (w, h) = t.median_size();
            !(w * h > primary_area * 3 / 2 && t.boxes.len() * 3 < primary_len)
        });
    }
    merged.truncate(12);
    classify_tracks(&mut merged);
    if !is_static && !scroll_is_trivial {
        for t in &mut merged {
            let (dx, dy) = world_span(t);
            let mut vx_sum = 0.0f64;
            let mut v_cnt = 0usize;
            for pair in t.boxes.windows(2) {
                let dt = pair[1].0.saturating_sub(pair[0].0).max(1) as f64;
                if dt <= 5.0 {
                    let s0 = scroll.get(pair[0].0).copied().unwrap_or([0, 0]);
                    let s1 = scroll.get(pair[1].0).copied().unwrap_or([0, 0]);
                    let wx0 = pair[0].1.x0 as f64 + s0[0] as f64;
                    let wx1 = pair[1].1.x0 as f64 + s1[0] as f64;
                    let vx = (wx1 - wx0) / dt;
                    if vx.abs() <= 16.0 {
                        vx_sum += vx;
                        v_cnt += 1;
                    }
                }
            }
            let mean_vx = if v_cnt > 0 { vx_sum / v_cnt as f64 } else { 0.0 };
            if (dx >= 150.0 && mean_vx >= 1.2) || (dx >= 60.0 && dy >= 48.0) {
                t.kind = ActorKind::Player;
            }
        }
    }
    (merged, all_moving)
}

fn classify_tracks(tracks: &mut [ActorTrack]) {
    let mut best: Option<(usize, usize)> = None;
    for (i, t) in tracks.iter().enumerate() {
        let (w, h) = t.median_size();
        if !(8..=44).contains(&w) || !(10..=44).contains(&h) {
            continue;
        }
        let count = t.boxes.len();
        if best.map_or(true, |(_, max_c)| count > max_c) {
            best = Some((i, count));
        }
    }
    if let Some((i, _)) = best {
        tracks[i].kind = ActorKind::Player;
    }
    for t in tracks.iter_mut() {
        if t.kind == ActorKind::Unknown {
            t.kind = ActorKind::Autonomous;
        }
    }
}
