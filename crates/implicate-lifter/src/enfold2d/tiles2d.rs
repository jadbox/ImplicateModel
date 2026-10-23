//! Game-agnostic terrain and platform tile classifier (`enfold2d::tiles2d`).

use super::{Rect, RgbImage8, TILE, TileKind, WorldAtlas, color_dist, connected_boxes_xy};

#[derive(Clone, Copy, Debug)]
pub struct TileOptions {
    pub bg_tol: i32,
    pub min_run: usize,
    pub min_area: usize,
    pub min_w: i32,
    pub min_h: i32,
    pub min_fill: f64,
    pub min_flat: f64,
    pub tile_size: usize,
    pub is_static: bool,
    pub hud_bottom: usize,
    pub hud_top: usize,
}

impl Default for TileOptions {
    fn default() -> Self {
        Self {
            bg_tol: 18,
            min_run: 3,
            min_area: 120,
            min_w: 12,
            min_h: 10,
            min_fill: 0.6,
            min_flat: 0.8,
            tile_size: 16,
            is_static: false,
            hud_bottom: 0,
            hud_top: usize::MAX,
        }
    }
}

pub fn learn_background(img: &RgbImage8) -> [u8; 3] {
    if img.width == 0 || img.height == 0 {
        return [0, 0, 0];
    }
    const BUCKETS: usize = 32 * 32 * 32;
    let band = img.height.min(3);
    let mut counts = vec![0u32; BUCKETS];
    let mut sums = vec![[0u64; 3]; BUCKETS];
    for y in 0..band {
        for x in 0..img.width {
            let c = img.pixel(x, y);
            let idx = (((c[0] as usize) >> 3) << 10)
                | (((c[1] as usize) >> 3) << 5)
                | ((c[2] as usize) >> 3);
            counts[idx] += 1;
            for (k, s) in sums[idx].iter_mut().enumerate() {
                *s += c[k] as u64;
            }
        }
    }
    let mut best = 0usize;
    for (i, c) in counts.iter().enumerate() {
        if *c > counts[best] {
            best = i;
        }
    }
    let n = counts[best] as u64;
    if n == 0 {
        return [0, 0, 0];
    }
    [
        (sums[best][0] / n) as u8,
        (sums[best][1] / n) as u8,
        (sums[best][2] / n) as u8,
    ]
}

#[derive(Clone, Debug)]
pub struct TileExtraction {
    pub cols: usize,
    pub rows: usize,
    pub tile_size: usize,
    pub kinds: Vec<TileKind>,
    pub platforms: Vec<Rect>,
    pub solids: Vec<Rect>,
    pub boxes: Vec<Rect>,
    pub collectibles: Vec<Rect>,
    pub ground_top: i32,
    pub origin_y: i32,
    pub background: [u8; 3],
    pub surface: Vec<i32>,
}

impl TileExtraction {
    pub fn kind(&self, tx: usize, ty: usize) -> TileKind {
        if tx < self.cols && ty < self.rows {
            self.kinds[ty * self.cols + tx]
        } else {
            TileKind::Empty
        }
    }
}

pub fn extract_2d_tiles(atlas: &WorldAtlas) -> TileExtraction {
    extract_2d_tiles_with(atlas, &TileOptions::default())
}

pub fn extract_2d_tiles_with(atlas: &WorldAtlas, opts: &TileOptions) -> TileExtraction {
    let img = &atlas.image;
    let (w, h) = (img.width, img.height);
    let ts = opts.tile_size.max(4);
    if w == 0 || h == 0 {
        return TileExtraction {
            cols: 0,
            rows: 0,
            tile_size: ts,
            kinds: Vec::new(),
            platforms: Vec::new(),
            solids: Vec::new(),
            boxes: Vec::new(),
            collectibles: Vec::new(),
            ground_top: 0,
            origin_y: 0,
            background: [0, 0, 0],
            surface: Vec::new(),
        };
    }
    let background = learn_background(img);
    let mut nonbg = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            nonbg[y * w + x] = color_dist(img.pixel(x, y), background) > opts.bg_tol;
        }
    }

    if opts.is_static {
        let cols = w / ts;
        let rows = h / ts;
        let mut kinds = vec![TileKind::Empty; cols * rows];
        let mut solids = Vec::new();
        let mut collectibles = Vec::new();
        let y_min = opts.hud_bottom.min(h);
        let y_max = opts.hud_top.min(h);
        let mut visited = vec![false; w * h];
        let mut wall_mask = vec![false; w * h];
        let mut queue = Vec::new();
        for y in y_min..y_max {
            for x in 0..w {
                let start_idx = y * w + x;
                if !nonbg[start_idx] || visited[start_idx] {
                    continue;
                }
                visited[start_idx] = true;
                queue.clear();
                queue.push((x, y));
                let mut head = 0usize;
                let mut min_x = x as i32;
                let mut max_x = x as i32;
                let mut min_y = y as i32;
                let mut max_y = y as i32;
                while head < queue.len() {
                    let (cx, cy) = queue[head];
                    head += 1;
                    min_x = min_x.min(cx as i32);
                    max_x = max_x.max(cx as i32);
                    min_y = min_y.min(cy as i32);
                    max_y = max_y.max(cy as i32);
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = cx as i32 + dx;
                            let ny = cy as i32 + dy;
                            if nx >= 0
                                && (nx as usize) < w
                                && ny >= y_min as i32
                                && (ny as usize) < y_max
                            {
                                let nidx = ny as usize * w + nx as usize;
                                if nonbg[nidx] && !visited[nidx] {
                                    visited[nidx] = true;
                                    queue.push((nx as usize, ny as usize));
                                }
                            }
                        }
                    }
                }
                let bw = max_x - min_x + 1;
                let bh = max_y - min_y + 1;
                let span = bw.max(bh);
                if span >= (2 * ts) as i32 {
                    for &(px, py) in &queue {
                        wall_mask[py * w + px] = true;
                    }
                } else if span <= (ts + 4) as i32 {
                    collectibles.push(Rect::new(min_x, min_y, max_x, max_y));
                }
            }
        }

        for ty in 0..rows {
            let y0 = ty * ts;
            let y1 = (y0 + ts).min(h);
            if y1 <= opts.hud_bottom || y0 >= opts.hud_top {
                for tx in 0..cols {
                    kinds[ty * cols + tx] = TileKind::Background;
                }
                continue;
            }
            for tx in 0..cols {
                let x0 = tx * ts;
                let x1 = (x0 + ts).min(w);
                let mut wall_pixels = 0usize;
                for y in y0..y1 {
                    for x in x0..x1 {
                        if wall_mask[y * w + x] {
                            wall_pixels += 1;
                        }
                    }
                }
                let total_px = (y1 - y0) * (x1 - x0);
                if wall_pixels >= (total_px / 10).max(4) {
                    kinds[ty * cols + tx] = TileKind::Solid;
                    solids.push(Rect::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32));
                } else {
                    kinds[ty * cols + tx] = TileKind::Empty;
                }
            }
        }
        return TileExtraction {
            cols,
            rows,
            tile_size: ts,
            kinds,
            platforms: Vec::new(),
            solids,
            boxes: Vec::new(),
            collectibles,
            ground_top: h as i32,
            origin_y: 0,
            background,
            surface: vec![h as i32; cols],
        };
    }

    let mut dense_start = vec![h; w];
    for x in 0..w {
        let mut last = None;
        for y in (0..h).rev() {
            if nonbg[y * w + x] {
                last = Some(y);
                break;
            }
        }
        let Some(mut top) = last else { continue };
        while top > 0 && nonbg[(top - 1) * w + x] {
            top -= 1;
        }
        dense_start[x] = top;
    }

    let need = (w / 12).max(8);
    let mut hist = vec![0usize; h];
    for &y in &dense_start {
        if y < h {
            hist[y] += 1;
        }
    }
    let mut ground_px = h;
    for (y, &n) in hist.iter().enumerate() {
        if n >= need {
            ground_px = y;
        }
    }
    let has_ground = ground_px < h;
    let origin_y = if has_ground {
        (ground_px % TILE) as i32
    } else {
        0
    };
    let rows = (((h as i32 - origin_y).max(0) as usize).div_ceil(TILE)).max(1);
    let cols = w.div_ceil(TILE);
    let ground_row = if has_ground {
        (((ground_px as i32 - origin_y) / TILE as i32).clamp(0, rows as i32 - 1)) as usize
    } else {
        rows - 1
    };

    let mut surface_row = vec![ground_row; cols];
    for tx in 0..cols {
        let x0 = tx * TILE;
        let x1 = (x0 + TILE).min(w);
        if x1 <= x0 {
            continue;
        }
        let mut counts = vec![0usize; rows];
        let mut best = usize::MAX;
        let mut best_n = 0usize;
        for x in x0..x1 {
            let d = dense_start[x];
            if d >= h {
                continue;
            }
            let row = (((d as f64 - origin_y as f64) / TILE as f64).round()).max(0.0) as usize;
            let row = row.min(rows - 1);
            counts[row] += 1;
            if counts[row] > best_n {
                best_n = counts[row];
                best = row;
            }
        }
        if best != usize::MAX && best + 1 == ground_row {
            let shelf_y = origin_y + best as i32 * TILE as i32;
            let flat_count = (x0..x1)
                .filter(|&x| (dense_start[x] as i32 - shelf_y).abs() <= 1)
                .count();
            if flat_count >= (x1 - x0) / 2 {
                surface_row[tx] = best;
            }
        }
    }

    let surface_y = |tx: usize| -> i32 {
        let r = surface_row.get(tx).copied().unwrap_or(ground_row);
        origin_y + r as i32 * TILE as i32
    };

    let mut kinds = vec![TileKind::Empty; cols * rows];
    if has_ground {
        for (tx, &sr) in surface_row.iter().enumerate() {
            for ty in sr..rows {
                kinds[ty * cols + tx] = TileKind::Solid;
            }
        }
    }

    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            mask[y * w + x] = nonbg[y * w + x] && (y as i32) < surface_y(x / TILE);
        }
    }
    let mut boxes = Vec::new();
    for c in connected_boxes_xy(&mask, w, h, opts.min_area) {
        if c.width() < opts.min_w || c.height() < opts.min_h {
            continue;
        }
        let (cx0, cx1) = (c.x0 as usize, (c.x1 as usize).min(w - 1));
        let (cy0, cy1) = (c.y0 as usize, (c.y1 as usize).min(h - 1));
        let mut n = 0usize;
        for y in cy0..=cy1 {
            for x in cx0..=cx1 {
                if mask[y * w + x] {
                    n += 1;
                }
            }
        }
        let area = (c.width() * c.height()).max(1) as f64;
        if (n as f64 / area) < opts.min_fill {
            continue;
        }
        let mut tops: Vec<i32> = Vec::new();
        for x in cx0..=cx1 {
            let mut y_bot = None;
            for y in (cy0..=cy1).rev() {
                if mask[y * w + x] {
                    y_bot = Some(y);
                    break;
                }
            }
            let Some(mut top) = y_bot else { continue };
            while top > cy0 && mask[(top - 1) * w + x] {
                top -= 1;
            }
            tops.push(top as i32);
        }
        if tops.is_empty() {
            continue;
        }
        tops.sort_unstable();
        let med = tops[tops.len() / 2];
        let flat =
            tops.iter().filter(|t| (**t - med).abs() <= 1).count() as f64 / tops.len() as f64;
        if flat < opts.min_flat {
            continue;
        }
        boxes.push(c);
        let ty = ((med - origin_y).max(0) as usize) / TILE;
        if ty < rows {
            for tx in 0..cols {
                let tx0 = (tx * TILE) as i32;
                let tx1 = tx0 + TILE as i32 - 1;
                if tx1.min(c.x1) - tx0.max(c.x0) + 1 >= 4 {
                    kinds[ty * cols + tx] = TileKind::OneWay;
                }
            }
        }
    }

    let mut solids = Vec::new();
    let mut platforms = Vec::new();
    for ty in 0..rows {
        let mut tx = 0;
        while tx < cols {
            let k = kinds[ty * cols + tx];
            if k != TileKind::Solid && k != TileKind::OneWay {
                tx += 1;
                continue;
            }
            let start = tx;
            while tx + 1 < cols && kinds[ty * cols + tx + 1] == k {
                tx += 1;
            }
            let y0 = origin_y + ty as i32 * TILE as i32;
            let x0 = (start * TILE) as i32;
            let x1 = ((tx * TILE + TILE - 1) as i32).min(w as i32 - 1);
            let r = Rect::new(x0, y0, x1, y0 + TILE as i32 - 1);
            if k == TileKind::Solid {
                solids.push(r);
            } else {
                platforms.push(r);
            }
            tx += 1;
        }
    }

    TileExtraction {
        cols,
        rows,
        tile_size: opts.tile_size.max(8),
        kinds,
        platforms,
        solids,
        boxes,
        collectibles: Vec::new(),
        ground_top: if has_ground {
            origin_y + ground_row as i32 * TILE as i32
        } else {
            h as i32
        },
        origin_y,
        background,
        surface: (0..cols).map(surface_y).collect(),
    }
}
