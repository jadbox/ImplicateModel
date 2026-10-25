//! Track R10.10 — binary morphology primitives for the Dimension Filter.
//!
//! A 1-blade (filament: string, wire, tail tip, ink stroke) and a 2-blade (holon body) differ
//! in their inscribed radius. The chessboard distance transform `D(x)` measures, for every
//! foreground pixel, the radius of the largest square structuring element centred on `x` that
//! still fits inside the mask. Erosion by radius `r` is therefore exactly `D > r`, and
//! the opening `(M ⊖ B_r) ⊕ B_r` keeps precisely the pixels covered by an inscribed square of
//! radius `> r`. All operations are O(N) and parameter-free apart from the radius itself,
//! which callers derive from the mask's own thickness statistics.

/// Chessboard (L∞) distance from each foreground pixel to the nearest background pixel,
/// with the region outside the raster counted as background. Background pixels get `0`;
/// an isolated foreground pixel gets `1`.
pub fn chessboard_distance(mask: &[bool], w: usize, h: usize) -> Vec<u32> {
    let mut d = vec![0u32; w * h];
    if w == 0 || h == 0 {
        return d;
    }
    let get = |d: &[u32], x: isize, y: isize| -> u32 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize { 0 } else { d[y as usize * w + x as usize] }
    };
    // Forward pass (top-left causal neighbours).
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            let m = get(&d, xi - 1, yi)
                .min(get(&d, xi - 1, yi - 1))
                .min(get(&d, xi, yi - 1))
                .min(get(&d, xi + 1, yi - 1));
            d[i] = m + 1;
        }
    }
    // Backward pass (bottom-right causal neighbours).
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let (xi, yi) = (x as isize, y as isize);
            let m = get(&d, xi + 1, yi)
                .min(get(&d, xi + 1, yi + 1))
                .min(get(&d, xi, yi + 1))
                .min(get(&d, xi - 1, yi + 1));
            d[i] = d[i].min(m + 1);
        }
    }
    d
}

/// Geodesic dilation of `seed` inside `mask` (8-connected). `steps = None` iterates to
/// convergence (morphological reconstruction by dilation); `Some(k)` stops after `k` rings,
/// i.e. keeps the mask pixels whose geodesic distance to the seed is at most `k`.
pub fn geodesic_dilate(seed: &[bool], mask: &[bool], w: usize, h: usize, steps: Option<u32>) -> Vec<bool> {
    let mut out: Vec<bool> = seed.iter().zip(mask).map(|(&s, &m)| s && m).collect();
    let mut frontier: Vec<usize> = (0..w * h).filter(|&i| out[i]).collect();
    let mut ring = 0u32;
    while !frontier.is_empty() && steps.is_none_or(|k| ring < k) {
        let mut next = Vec::new();
        for &i in &frontier {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && !out[j] {
                        out[j] = true;
                        next.push(j);
                    }
                }
            }
        }
        frontier = next;
        ring += 1;
    }
    out
}

/// Dilate `mask` by a square structuring element of radius `r` (separable max filter).
pub fn dilate(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    if r == 0 {
        return mask.to_vec();
    }
    let mut tmp = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r).min(w - 1));
            tmp[y * w + x] = (a..=b).any(|xx| mask[y * w + xx]);
        }
    }
    let mut out = vec![false; w * h];
    for y in 0..h {
        let (a, b) = (y.saturating_sub(r), (y + r).min(h - 1));
        for x in 0..w {
            out[y * w + x] = (a..=b).any(|yy| tmp[yy * w + x]);
        }
    }
    out
}

/// 8-connected component labelling. Returns `(labels, sizes)` with label `0` = background and
/// `sizes[0] = 0`.
pub fn label_components(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, Vec<usize>) {
    let mut label = vec![0u32; w * h];
    let mut sizes: Vec<usize> = vec![0];
    let mut stack = Vec::new();
    for start in 0..w * h {
        if !mask[start] || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        sizes.push(0);
        label[start] = id;
        stack.push(start);
        while let Some(i) = stack.pop() {
            sizes[id as usize] += 1;
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if mask[j] && label[j] == 0 {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
    }
    (label, sizes)
}

/// Topological Cavity Isolation (Track S1.2):
/// Flood-fill from raster borders to isolate all interior cavities {C_k} enclosed
/// by foreground silhouette boundaries.
/// Returns a list of cavities, where each cavity is a Vec<usize> of raster pixel indices.
pub fn detect_enclosed_cavities(mask: &[bool], w: usize, h: usize) -> Vec<Vec<usize>> {
    if w == 0 || h == 0 || mask.len() < w * h {
        return Vec::new();
    }
    let mut outside = vec![false; w * h];
    let mut stack = Vec::new();

    // 1. Seed boundary background pixels (x=0, x=w-1, y=0, y=h-1 where !mask)
    for x in 0..w {
        let top = x;
        if !mask[top] && !outside[top] {
            outside[top] = true;
            stack.push(top);
        }
        let btm = (h - 1) * w + x;
        if !mask[btm] && !outside[btm] {
            outside[btm] = true;
            stack.push(btm);
        }
    }
    for y in 0..h {
        let left = y * w;
        if !mask[left] && !outside[left] {
            outside[left] = true;
            stack.push(left);
        }
        let right = y * w + (w - 1);
        if !mask[right] && !outside[right] {
            outside[right] = true;
            stack.push(right);
        }
    }

    // 2. 4-connected flood-fill of exterior ambient background
    // (4-connectivity ensures thin 8-connected diagonal foreground boundaries don't leak)
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        if x > 0 {
            let j = i - 1;
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
        if x + 1 < w {
            let j = i + 1;
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
        if y > 0 {
            let j = i - w;
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
        if y + 1 < h {
            let j = i + w;
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
    }

    // 3. Any pixel with !mask[i] && !outside[i] is an enclosed interior cavity.
    // Label 8-connected components of interior cavities.
    let mut cavity_mask = vec![false; w * h];
    for i in 0..w * h {
        if !mask[i] && !outside[i] {
            cavity_mask[i] = true;
        }
    }

    let (labels, sizes) = label_components(&cavity_mask, w, h);
    if sizes.len() <= 1 {
        return Vec::new();
    }

    let mut cavities = vec![Vec::new(); sizes.len() - 1];
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl > 0 {
            cavities[(lbl - 1) as usize].push(i);
        }
    }

    cavities
}

/// Medial-ridge thickness histogram: `hist[d]` counts ridge pixels (local maxima of the
/// chessboard distance in their 8-neighbourhood) with inscribed radius `d`. Each stroke of
/// half-width `d` contributes ridge pixels proportional to its length, so a long filament
/// forms a mode at its own radius.
pub fn ridge_radius_histogram(dist: &[u32], w: usize, h: usize) -> Vec<usize> {
    let max_d = dist.iter().copied().max().unwrap_or(0) as usize;
    let mut hist = vec![0usize; max_d + 1];
    for y in 0..h {
        for x in 0..w {
            let d = dist[y * w + x];
            if d == 0 {
                continue;
            }
            let mut ridge = true;
            'n: for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let (nx, ny) = (x as isize + dx, y as isize + dy);
                    if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                        continue;
                    }
                    if dist[ny as usize * w + nx as usize] > d {
                        ridge = false;
                        break 'n;
                    }
                }
            }
            if ridge {
                hist[d as usize] += 1;
            }
        }
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_and_opening_separate_filament_from_body() {
        // 20x20 square body with a 2px-wide, 15px-long filament attached on the right.
        let (w, h) = (40usize, 24usize);
        let mut m = vec![false; w * h];
        for y in 2..22 {
            for x in 2..22 {
                m[y * w + x] = true;
            }
        }
        for y in 11..13 {
            for x in 22..37 {
                m[y * w + x] = true;
            }
        }
        let d = chessboard_distance(&m, w, h);
        assert_eq!(d[12 * w + 30], 1);
        assert_eq!(d[11 * w + 11], 10);
        let core: Vec<bool> = d.iter().map(|&v| v > 1).collect();
        assert!(!core[12 * w + 30]);
        let opened = dilate(&core, w, h, 1);
        let rec = geodesic_dilate(&opened, &m, w, h, Some(1));
        assert!(rec[11 * w + 11]);
        assert!(!rec[12 * w + 30], "filament beyond r_B must not regrow");
        let full = geodesic_dilate(&opened, &m, w, h, None);
        assert!(full[12 * w + 30], "unbounded reconstruction regrows connected filaments");
        let hist = ridge_radius_histogram(&d, w, h);
        let mode = (1..hist.len()).max_by_key(|&r| hist[r]).unwrap();
        assert_eq!(mode, 1);
    }

    #[test]
    fn test_detect_enclosed_cavities() {
        // 20x20 box with a 4x4 hole in the center, and a 4x4 open bay at the border
        let (w, h) = (30usize, 30usize);
        let mut m = vec![false; w * h];
        for y in 5..25 {
            for x in 5..25 {
                m[y * w + x] = true;
            }
        }
        // Hollow out center 4x4: (13..17, 13..17)
        for y in 13..17 {
            for x in 13..17 {
                m[y * w + x] = false;
            }
        }
        // Create an open inlet / bay connecting to outside at the left edge
        for y in 10..15 {
            for x in 5..10 {
                m[y * w + x] = false;
            }
        }

        let cavities = detect_enclosed_cavities(&m, w, h);
        assert_eq!(cavities.len(), 1, "Only the fully enclosed center cavity should be detected");
        assert_eq!(cavities[0].len(), 16, "Center cavity must be 4x4 = 16 pixels");
        for y in 13..17 {
            for x in 13..17 {
                let idx = y * w + x;
                assert!(cavities[0].contains(&idx));
            }
        }
    }
}
