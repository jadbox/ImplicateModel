//! 2D orthographic & scrolling CFT primitives (`enfold2d`).
//!
//! Organized into focused submodules:
//! - `fft_scroll`: 2D FFT cross-correlation (`Fft2`, `ScrollEstimator`) and HUD/border partitioning.
//! - `atlas2d`: Quantized RGB mode world atlas (`Bucket`, `AtlasAccumulator`, `WorldAtlas`) and residual detection.
//! - `tracker2d`: Spinor-coherent multi-frame actor tracker (`ActorTrack`, `Sprite2D`).
//! - `tiles2d`: Game-agnostic terrain and platform tile classifier (`TileExtraction`, `TileOptions`).

pub mod atlas2d;
pub mod fft_scroll;
pub mod tiles2d;
pub mod tracker2d;

use serde::{Deserialize, Serialize};

pub use atlas2d::{
    AtlasAccumulator, Bucket, BucketEntry, WorldAtlas, build_2d_world_atlas, refine_offset,
    residual_boxes, residual_boxes_range, residual_boxes_with_color,
};
pub use fft_scroll::{
    Fft2, HudPartition, ScrollEstimator, detect_camera_is_static, detect_playfield_segment,
    detect_static_borders, estimate_2d_scroll, partition_hud_and_viewport,
};
pub use tiles2d::{
    TileExtraction, TileOptions, extract_2d_tiles, extract_2d_tiles_with, learn_background,
};
pub use tracker2d::{ActorKind, ActorTrack, Sprite2D, track_actors, track_actors_coherent};

/// Tile edge in pixels.
pub const TILE: usize = 16;
/// World y of the first tile row's top edge.
pub const TILE_ORIGIN_Y: i32 = 1;
/// Sprite-scale prior for 2D tile games: actors are at most two tiles across.
pub const ACTOR_MAX_PX: i32 = 48;
/// Minimum residual area for a 2D tile-game actor candidate.
pub const ACTOR_MIN_AREA: i32 = 120;

/// Collision class of a 16×16 atlas tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TileKind {
    Empty = 0,
    Solid = 1,
    OneWay = 2,
    Background = 3,
}

impl TileKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => TileKind::Solid,
            2 => TileKind::OneWay,
            3 => TileKind::Background,
            _ => TileKind::Empty,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Inclusive pixel rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    pub fn width(&self) -> i32 {
        (self.x1 - self.x0 + 1).max(0)
    }

    pub fn height(&self) -> i32 {
        (self.y1 - self.y0 + 1).max(0)
    }

    pub fn center(&self) -> [f64; 2] {
        [
            (self.x0 + self.x1) as f64 / 2.0,
            (self.y0 + self.y1) as f64 / 2.0,
        ]
    }

    pub fn area(&self) -> i32 {
        self.width() * self.height()
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    pub fn expand(&self, m: i32) -> Self {
        Self::new(self.x0 - m, self.y0 - m, self.x1 + m, self.y1 + m)
    }

    pub fn shifted(&self, dx: i32, dy: i32) -> Self {
        Self::new(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x0 <= other.x1 && other.x0 <= self.x1 && self.y0 <= other.y1 && other.y0 <= self.y1
    }

    pub fn iou(&self, other: &Rect) -> f64 {
        let ix0 = self.x0.max(other.x0);
        let iy0 = self.y0.max(other.y0);
        let ix1 = self.x1.min(other.x1);
        let iy1 = self.y1.min(other.y1);
        if ix1 < ix0 || iy1 < iy0 {
            return 0.0;
        }
        let inter = ((ix1 - ix0 + 1) * (iy1 - iy0 + 1)) as f64;
        let union = (self.area() + other.area()) as f64 - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
}

/// Dense RGB image with direct pixel access.
#[derive(Clone, Debug, PartialEq)]
pub struct RgbImage8 {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

impl RgbImage8 {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgb: vec![0; width * height * 3],
        }
    }

    pub fn from_raw(width: usize, height: usize, rgb: Vec<u8>) -> Self {
        debug_assert_eq!(rgb.len(), width * height * 3);
        Self { width, height, rgb }
    }

    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let i = (y * self.width + x) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    #[inline]
    pub fn set_pixel(&mut self, x: usize, y: usize, c: [u8; 3]) {
        let i = (y * self.width + x) * 3;
        self.rgb[i] = c[0];
        self.rgb[i + 1] = c[1];
        self.rgb[i + 2] = c[2];
    }

    /// Blit `src` at `(dx, dy)`, clipping at the edges.
    pub fn blit(&mut self, src: &RgbImage8, dx: i32, dy: i32) {
        for sy in 0..src.height {
            let ty = dy + sy as i32;
            if ty < 0 || ty >= self.height as i32 {
                continue;
            }
            for sx in 0..src.width {
                let tx = dx + sx as i32;
                if tx < 0 || tx >= self.width as i32 {
                    continue;
                }
                let s = (sy * src.width + sx) * 3;
                let d = (ty as usize * self.width + tx as usize) * 3;
                self.rgb[d..d + 3].copy_from_slice(&src.rgb[s..s + 3]);
            }
        }
    }
}

#[inline]
pub fn color_dist(a: [u8; 3], b: [u8; 3]) -> i32 {
    let d0 = a[0] as i32 - b[0] as i32;
    let d1 = a[1] as i32 - b[1] as i32;
    let d2 = a[2] as i32 - b[2] as i32;
    ((d0 * d0 + d1 * d1 + d2 * d2) as f64).sqrt() as i32
}

/// 4-connected components → bounding boxes (dilated by 1 px).
pub fn connected_boxes(mask: &[bool], w: usize, h: usize, min_pixels: usize) -> Vec<Rect> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if mask[start] && !seen[start] {
            let mut min_x = w as i32;
            let mut min_y = h as i32;
            let mut max_x = -1i32;
            let mut max_y = -1i32;
            let mut count = 0usize;
            stack.clear();
            stack.push(start);
            seen[start] = true;
            while let Some(p) = stack.pop() {
                let (x, y) = (p % w, p / w);
                count += 1;
                min_x = min_x.min(x as i32);
                max_x = max_x.max(x as i32);
                min_y = min_y.min(y as i32);
                max_y = max_y.max(y as i32);
                for (nx, ny) in [
                    (x as i32 - 1, y as i32),
                    (x as i32 + 1, y as i32),
                    (x as i32, y as i32 - 1),
                    (x as i32, y as i32 + 1),
                ] {
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let np = ny as usize * w + nx as usize;
                    if mask[np] && !seen[np] {
                        seen[np] = true;
                        stack.push(np);
                    }
                }
            }
            if count >= min_pixels {
                out.push(Rect::new(min_x, min_y, max_x, max_y).expand(1));
            }
        }
    }
    out.sort_by_key(|r| (r.y0, r.x0));
    out
}

/// Connected components over a W×H mask without dilation.
pub fn connected_boxes_xy(mask: &[bool], w: usize, h: usize, min_pixels: usize) -> Vec<Rect> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || seen[start] {
            continue;
        }
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (w as i32, h as i32, -1i32, -1i32);
        let mut count = 0usize;
        stack.clear();
        stack.push(start);
        seen[start] = true;
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            count += 1;
            min_x = min_x.min(x as i32);
            max_x = max_x.max(x as i32);
            min_y = min_y.min(y as i32);
            max_y = max_y.max(y as i32);
            for (nx, ny) in [
                (x as i32 - 1, y as i32),
                (x as i32 + 1, y as i32),
                (x as i32, y as i32 - 1),
                (x as i32, y as i32 + 1),
            ] {
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let np = ny as usize * w + nx as usize;
                if mask[np] && !seen[np] {
                    seen[np] = true;
                    stack.push(np);
                }
            }
        }
        if count >= min_pixels {
            out.push(Rect::new(min_x, min_y, max_x, max_y));
        }
    }
    out.sort_by_key(|r| (r.y0, r.x0));
    out
}
