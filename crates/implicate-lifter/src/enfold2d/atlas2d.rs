//! Quantized temporal-mode world atlas accumulation and residual detection (`enfold2d::atlas2d`).

use std::collections::HashMap;

use super::{Rect, RgbImage8, color_dist, connected_boxes};
use crate::video_io::RawFrame;

const MAX_ENTRIES: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct BucketEntry {
    pub key: u32,
    pub count: u32,
    pub sr: u32,
    pub sg: u32,
    pub sb: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Bucket {
    pub entries: Vec<BucketEntry>,
}

impl Bucket {
    pub fn add(&mut self, c: [u8; 3]) {
        let key = ((c[0] as u32 >> 3) << 10) | ((c[1] as u32 >> 3) << 5) | (c[2] as u32 >> 3);
        for e in self.entries.iter_mut() {
            if e.key == key {
                e.count += 1;
                e.sr += c[0] as u32;
                e.sg += c[1] as u32;
                e.sb += c[2] as u32;
                return;
            }
        }
        if self.entries.len() < MAX_ENTRIES {
            self.entries.push(BucketEntry {
                key,
                count: 1,
                sr: c[0] as u32,
                sg: c[1] as u32,
                sb: c[2] as u32,
            });
        } else {
            let idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.count)
                .map(|(i, _)| i)
                .unwrap();
            self.entries[idx] = BucketEntry {
                key,
                count: 1,
                sr: c[0] as u32,
                sg: c[1] as u32,
                sb: c[2] as u32,
            };
        }
    }

    pub fn mode(&self) -> Option<([u8; 3], u32)> {
        let e = self.entries.iter().max_by_key(|e| e.count)?;
        let n = e.count.max(1);
        Some((
            [
                ((e.sr + n / 2) / n) as u8,
                ((e.sg + n / 2) / n) as u8,
                ((e.sb + n / 2) / n) as u8,
            ],
            e.count,
        ))
    }

    pub fn total(&self) -> u32 {
        self.entries.iter().map(|e| e.count).sum()
    }
}

/// Streaming world atlas accumulator.
pub struct AtlasAccumulator {
    pub viewport_h: usize,
    buckets: HashMap<(i32, i32), Bucket>,
    min: [i32; 2],
    max: [i32; 2],
    frames: usize,
}

impl AtlasAccumulator {
    pub fn new(viewport_h: usize) -> Self {
        Self {
            viewport_h,
            buckets: HashMap::new(),
            min: [i32::MAX, i32::MAX],
            max: [i32::MIN, i32::MIN],
            frames: 0,
        }
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn add_frame(&mut self, frame: &RawFrame, off: [i32; 2], mask: &[Rect]) {
        self.frames += 1;
        let fw = frame.width as usize;
        let rows = self.viewport_h.min(frame.height as usize);
        let x_start = if off[0] > 0 && fw >= 128 { 8 } else { 0 };
        for y in 0..rows {
            let wy = y as i32 + off[1];
            for x in x_start..fw {
                let wx = x as i32 + off[0];
                if wx < self.min[0] {
                    self.min[0] = wx;
                }
                if wy < self.min[1] {
                    self.min[1] = wy;
                }
                if wx > self.max[0] {
                    self.max[0] = wx;
                }
                if wy > self.max[1] {
                    self.max[1] = wy;
                }
                if !mask.is_empty() && mask.iter().any(|r| r.contains(x as i32, y as i32)) {
                    continue;
                }
                let i = (y * fw + x) * 3;
                let c = [frame.rgb[i], frame.rgb[i + 1], frame.rgb[i + 2]];
                self.buckets.entry((wx, wy)).or_default().add(c);
            }
        }
    }

    pub fn to_atlas(&self) -> WorldAtlas {
        if self.buckets.is_empty() {
            return WorldAtlas {
                image: RgbImage8::new(0, 0),
                origin: [0, 0],
            };
        }
        let origin = self.min;
        let w = (self.max[0] - self.min[0] + 1).max(0) as usize;
        let h = (self.max[1] - self.min[1] + 1).max(0) as usize;
        let mut image = RgbImage8::new(w, h);
        for (&(wx, wy), bucket) in &self.buckets {
            if let Some((c, _)) = bucket.mode() {
                let x = (wx - origin[0]) as usize;
                let y = (wy - origin[1]) as usize;
                if x < w && y < h {
                    image.set_pixel(x, y, c);
                }
            }
        }
        WorldAtlas { image, origin }
    }

    pub fn to_atlas_with_fallback(&self, fallback: &WorldAtlas) -> WorldAtlas {
        let mut atlas = self.to_atlas();
        let w = atlas.image.width;
        let h = atlas.image.height;
        for y in 0..h {
            let wy = atlas.origin[1] + y as i32;
            for x in 0..w {
                let wx = atlas.origin[0] + x as i32;
                let has = self.buckets.get(&(wx, wy)).and_then(|b| b.mode()).is_some();
                if !has {
                    if let Some(c) = fallback.pixel(wx, wy) {
                        atlas.image.set_pixel(x, y, c);
                    }
                }
            }
        }
        atlas
    }

    pub fn to_atlas_inpainted(
        &self,
        fallback: &WorldAtlas,
        has_viewport_hud: bool,
        bg: [u8; 3],
    ) -> WorldAtlas {
        if !has_viewport_hud {
            return self.to_atlas_with_fallback(fallback);
        }
        let mut atlas = self.to_atlas();
        let w = atlas.image.width;
        let h = atlas.image.height;
        if w == 0 || h == 0 {
            return atlas;
        }
        let mut valid = vec![false; w * h];
        for y in 0..h {
            let wy = atlas.origin[1] + y as i32;
            for x in 0..w {
                let wx = atlas.origin[0] + x as i32;
                if let Some(b) = self.buckets.get(&(wx, wy)) {
                    if let Some((c, _)) = b.mode() {
                        let tot = b.total();
                        if y >= 10 || tot >= 12 || color_dist(c, bg) <= 32 {
                            valid[y * w + x] = true;
                        }
                    }
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                if valid[y * w + x] {
                    continue;
                }
                let mut best_c = bg;
                let mut best_d = usize::MAX;
                for sx in (0..x).rev() {
                    if valid[y * w + sx] {
                        let d = x - sx;
                        if d < best_d {
                            best_d = d;
                            best_c = atlas.image.pixel(sx, y);
                        }
                        break;
                    }
                }
                for sx in (x + 1)..w {
                    if valid[y * w + sx] {
                        let d = sx - x;
                        if d < best_d {
                            best_d = d;
                            best_c = atlas.image.pixel(sx, y);
                        }
                        break;
                    }
                }
                for sy in (y + 1)..h {
                    if valid[sy * w + x] {
                        let d = sy - y;
                        if d < best_d {
                            best_d = d;
                            best_c = atlas.image.pixel(x, sy);
                        }
                        break;
                    }
                }
                for sy in (0..y).rev() {
                    if valid[sy * w + x] {
                        let d = y - sy;
                        if d < best_d {
                            best_c = atlas.image.pixel(x, sy);
                        }
                        break;
                    }
                }
                atlas.image.set_pixel(x, y, best_c);
            }
        }
        atlas
    }
}

/// Dense world-space atlas with its world origin.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldAtlas {
    pub image: RgbImage8,
    pub origin: [i32; 2],
}

impl WorldAtlas {
    #[inline]
    pub fn pixel(&self, wx: i32, wy: i32) -> Option<[u8; 3]> {
        let x = wx - self.origin[0];
        let y = wy - self.origin[1];
        if x < 0 || y < 0 || x >= self.image.width as i32 || y >= self.image.height as i32 {
            None
        } else {
            Some(self.image.pixel(x as usize, y as usize))
        }
    }
}

pub fn build_2d_world_atlas(
    frames: &[RawFrame],
    scroll: &[[i32; 2]],
    viewport_h: usize,
) -> WorldAtlas {
    let mut acc = AtlasAccumulator::new(viewport_h);
    for (f, off) in frames.iter().zip(scroll) {
        acc.add_frame(f, *off, &[]);
    }
    acc.to_atlas()
}

pub fn residual_boxes(
    atlas: &WorldAtlas,
    frame: &RawFrame,
    off: [i32; 2],
    hud_top: usize,
    threshold: i32,
    min_pixels: usize,
) -> Vec<Rect> {
    residual_boxes_range(atlas, frame, off, 0, hud_top, threshold, min_pixels)
}

pub fn residual_boxes_range(
    atlas: &WorldAtlas,
    frame: &RawFrame,
    off: [i32; 2],
    y_start: usize,
    y_end: usize,
    threshold: i32,
    min_pixels: usize,
) -> Vec<Rect> {
    let w = frame.width as usize;
    let h = y_end.min(frame.height as usize);
    let mut mask = vec![false; w * h];
    let x_start = if off[0] > 0 && w >= 128 { 8 } else { 0 };
    for y in y_start..h {
        for x in x_start..w {
            let wx = x as i32 + off[0];
            let wy = y as i32 + off[1];
            if let Some(a) = atlas.pixel(wx, wy) {
                let i = (y * w + x) * 3;
                let c = [frame.rgb[i], frame.rgb[i + 1], frame.rgb[i + 2]];
                let d = (c[0] as i32 - a[0] as i32)
                    .abs()
                    .max((c[1] as i32 - a[1] as i32).abs())
                    .max((c[2] as i32 - a[2] as i32).abs());
                if d > threshold {
                    mask[y * w + x] = true;
                }
            }
        }
    }
    connected_boxes(&mask, w, h, min_pixels)
}

pub fn residual_boxes_with_color(
    atlas: &WorldAtlas,
    frame: &RawFrame,
    off: [i32; 2],
    y_start: usize,
    y_end: usize,
    threshold: i32,
    min_pixels: usize,
) -> Vec<(Rect, [u8; 3])> {
    let w = frame.width as usize;
    let h = y_end.min(frame.height as usize);
    let mut mask = vec![false; w * h];
    let x_start = if off[0] > 0 && w >= 128 { 8 } else { 0 };
    for y in y_start..h {
        for x in x_start..w {
            let wx = x as i32 + off[0];
            let wy = y as i32 + off[1];
            if let Some(a) = atlas.pixel(wx, wy) {
                let i = (y * w + x) * 3;
                let c = [frame.rgb[i], frame.rgb[i + 1], frame.rgb[i + 2]];
                let d = (c[0] as i32 - a[0] as i32)
                    .abs()
                    .max((c[1] as i32 - a[1] as i32).abs())
                    .max((c[2] as i32 - a[2] as i32).abs());
                if d > threshold {
                    mask[y * w + x] = true;
                }
            }
        }
    }
    let rects = connected_boxes(&mask, w, h, min_pixels);
    rects
        .into_iter()
        .map(|r| {
            let mut sr = 0u64;
            let mut sg = 0u64;
            let mut sb = 0u64;
            let mut n = 0u64;
            for y in r.y0.max(0)..=r.y1.min(h as i32 - 1) {
                for x in r.x0.max(0)..=r.x1.min(w as i32 - 1) {
                    let idx = y as usize * w + x as usize;
                    if mask[idx] {
                        let p = idx * 3;
                        sr += frame.rgb[p] as u64;
                        sg += frame.rgb[p + 1] as u64;
                        sb += frame.rgb[p + 2] as u64;
                        n += 1;
                    }
                }
            }
            let col = if n > 0 {
                [(sr / n) as u8, (sg / n) as u8, (sb / n) as u8]
            } else {
                [128, 128, 128]
            };
            (r, col)
        })
        .collect()
}

pub fn refine_offset(
    atlas: &WorldAtlas,
    frame: &RawFrame,
    off: [i32; 2],
    hud_top: usize,
) -> [i32; 2] {
    let w = frame.width as usize;
    let h = hud_top.min(frame.height as usize);
    let mut best = off;
    let mut best_err = u64::MAX;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cand = [off[0] + dx, off[1] + dy];
            let mut err = 0u64;
            let mut n = 0u64;
            for y in (0..h).step_by(2) {
                for x in (0..w).step_by(2) {
                    let Some(a) = atlas.pixel(x as i32 + cand[0], y as i32 + cand[1]) else {
                        continue;
                    };
                    let i = (y * w + x) * 3;
                    err += (frame.rgb[i] as i32 - a[0] as i32).unsigned_abs() as u64
                        + (frame.rgb[i + 1] as i32 - a[1] as i32).unsigned_abs() as u64
                        + (frame.rgb[i + 2] as i32 - a[2] as i32).unsigned_abs() as u64;
                    n += 1;
                }
            }
            if n > 0 {
                let e = err.saturating_mul(1000) / n;
                if e < best_err {
                    best_err = e;
                    best = cand;
                }
            }
        }
    }
    best
}
