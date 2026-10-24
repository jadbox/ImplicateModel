//! Visual diagnostics: atlas/track contact sheets and unfold dispatch reports.

use std::path::{Path, PathBuf};

use image::{Rgb, RgbImage};
use implicate_lifter::ArticulatedHolon;
use implicate_unfold::{CausalUnfoldEngine, EnfoldedClipAssets};

/// Distinct, saturated label colors (golden-angle hue walk).
pub fn label_color(i: usize) -> [u8; 3] {
    let h = (i as f32 * 137.508).rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h.rem_euclid(2.0) - 1.0).abs();
    let (r, g, b) = match h as u32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

const DIGITS_3X5: [u16; 10] = [
    0b111_101_101_101_111,
    0b010_110_010_010_111,
    0b111_001_111_100_111,
    0b111_001_111_001_111,
    0b101_101_111_001_001,
    0b111_100_111_001_111,
    0b111_100_111_101_111,
    0b111_001_001_001_001,
    0b111_101_111_101_111,
    0b111_101_111_001_111,
];

fn put(img: &mut RgbImage, x: i64, y: i64, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, Rgb(c));
    }
}

pub fn draw_number(img: &mut RgbImage, n: usize, x0: i64, y0: i64, scale: i64, c: [u8; 3]) {
    let s = n.to_string();
    for (k, ch) in s.chars().enumerate() {
        let d = ch.to_digit(10).unwrap_or(0) as usize;
        let glyph = DIGITS_3X5[d];
        for ry in 0..5i64 {
            for rx in 0..3i64 {
                if (glyph >> (14 - (ry * 3 + rx))) & 1 == 1 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put(img, x0 + (k as i64 * 4 + rx) * scale + sx, y0 + ry * scale + sy, c);
                        }
                    }
                }
            }
        }
    }
}

fn draw_line(img: &mut RgbImage, a: [f64; 2], b: [f64; 2], c: [u8; 3]) {
    let n = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).ceil() as usize).max(1);
    for k in 0..=n {
        let t = k as f64 / n as f64;
        put(img, (a[0] + t * (b[0] - a[0])).round() as i64, (a[1] + t * (b[1] - a[1])).round() as i64, c);
    }
}

pub fn draw_rect(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3]) {
    for x in x0..=x1 {
        put(img, x, y0, c);
        put(img, x, y1, c);
    }
    for y in y0..=y1 {
        put(img, x0, y, c);
        put(img, x1, y, c);
    }
}

fn track_root_path(h: &ArticulatedHolon) -> Vec<([f64; 2], [f64; 2])> {
    h.poses_per_obs
        .iter()
        .filter_map(|p| p.first().map(|r| (r.center, r.radii)))
        .collect()
}

/// Render every atlas layer (plus HUD) with each track's root trajectory overlaid.
fn render_atlas_sheet(clip: &EnfoldedClipAssets) -> RgbImage {
    let mut layers: Vec<&implicate_lifter::ParallaxLayer> = clip.atlas.layers.iter().collect();
    if let Some(hud) = &clip.atlas.camera_locked_hud {
        layers.push(hud);
    }
    let gap = 4u32;
    let w = layers.iter().map(|l| l.atlas.image.width as u32).max().unwrap_or(1).max(clip.atlas.width as u32);
    let h: u32 = layers.iter().map(|l| l.atlas.image.height as u32 + gap).sum::<u32>().max(1);
    let mut img = RgbImage::from_pixel(w, h, Rgb([255, 0, 255]));
    let mut y_off = 0u32;
    for (li, layer) in layers.iter().enumerate() {
        let lw = layer.atlas.image.width;
        let lh = layer.atlas.image.height;
        for y in 0..lh {
            for x in 0..lw {
                let a = layer.alpha[y * lw + x];
                let c = layer.atlas.image.pixel(x, y);
                // Transparent texels shown as checkerboard.
                let chk = if ((x / 4) + (y / 4)) % 2 == 0 { 60u8 } else { 90u8 };
                let px = if a == 0 { [chk, chk, chk] } else { c };
                img.put_pixel(x as u32, y_off + y as u32, Rgb(px));
            }
        }
        // Tracks are drawn on the primary layer in screen/frame coordinates.
        if li == 0 {
            let ox = -layer.atlas.origin[0] as f64;
            let oy = y_off as f64 - layer.atlas.origin[1] as f64;
            let first_traj = layer.trajectory.first().copied().unwrap_or([0, 0]);
            for (ti, tr) in clip.holons.iter().enumerate() {
                let col = label_color(ti);
                let path = track_root_path(tr);
                // Frame coordinates -> world coordinates via the layer camera trajectory is
                // not stored per observation, so paths are drawn relative to the first frame.
                let to_img = |p: [f64; 2]| [p[0] + ox + first_traj[0] as f64, p[1] + oy + first_traj[1] as f64];
                for w2 in path.windows(2) {
                    draw_line(&mut img, to_img(w2[0].0), to_img(w2[1].0), col);
                }
                if let Some((c0, r0)) = path.first() {
                    let p = to_img(*c0);
                    draw_rect(
                        &mut img,
                        (p[0] - r0[0]) as i64,
                        (p[1] - r0[1]) as i64,
                        (p[0] + r0[0]) as i64,
                        (p[1] + r0[1]) as i64,
                        col,
                    );
                    draw_number(&mut img, ti, (p[0] + r0[0]) as i64 + 2, (p[1] - r0[1]) as i64, 1, col);
                }
            }
        }
        y_off += lh as u32 + gap;
    }
    img
}

/// Render a holon at explicit part poses onto a tight canvas: `(w, h, mask, rgb)`.
fn render_posed(h: &implicate_lifter::ArticulatedHolon, poses: &[implicate_lifter::SubHolonPose]) -> (usize, usize, Vec<bool>, Vec<u8>) {
    use implicate_lifter::articulated_wavepacket::render_with_poses;
    use implicate_lifter::{CharacterObservation, Rect};
    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in poses {
        let r = p.radii[0].max(p.radii[1]);
        x0 = x0.min(p.center[0] - r);
        y0 = y0.min(p.center[1] - r);
        x1 = x1.max(p.center[0] + r);
        y1 = y1.max(p.center[1] + r);
    }
    let bbox = Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32);
    let (w, hh) = (bbox.width().max(1) as usize, bbox.height().max(1) as usize);
    let canvas = CharacterObservation { frame_idx: 0, bbox, mask: vec![false; w * hh], rgb: vec![0; w * hh * 3], camera_offset: [0.0, 0.0], truncated: false, line_of_action: None };
    let (m, c) = render_with_poses(h, poses, &canvas);
    (w, hh, m, c)
}

/// Strip of every stored observation of track `ti`, rendered through its model (labelled with
/// observation index and view cell) to separate model defects from unfold selection defects.
pub fn write_recon_strip(clip: &EnfoldedClipAssets, ti: usize, path: &Path) -> Result<(), String> {
    let h = clip.holons.get(ti).ok_or_else(|| format!("no track {ti}"))?;
    let renders: Vec<_> = h.poses_per_obs.iter().map(|p| render_posed(h, p)).collect();
    let cell = renders.iter().map(|(w, hh, _, _)| (*w).max(*hh)).max().unwrap_or(16);
    let cols = 8usize;
    let rows = renders.len().div_ceil(cols).max(1);
    let lab = 8usize;
    let mut img = RgbImage::from_pixel((cols * (cell + 2)) as u32, (rows * (cell + lab + 2)) as u32, Rgb([255, 255, 255]));
    for (k, (w, hh, m, c)) in renders.iter().enumerate() {
        let (cx, cy) = ((k % cols) * (cell + 2), (k / cols) * (cell + lab + 2));
        draw_number(&mut img, k, cx as i64, cy as i64 + 1, 1, [0, 0, 0]);
        let vc = h.poses_per_obs[k].first().map_or(0, |p| p.view_idempotent_idx);
        draw_number(&mut img, vc, cx as i64 + 16, cy as i64 + 1, 1, [200, 0, 0]);
        for y in 0..cell {
            for x in 0..cell {
                let chk = if ((x / 4) + (y / 4)) % 2 == 0 { 200u8 } else { 160u8 };
                let px = if x < *w && y < *hh && m[y * w + x] { let i = (y * w + x) * 3; [c[i], c[i + 1], c[i + 2]] } else { [chk, chk, chk] };
                img.put_pixel((cx + x) as u32, (cy + lab + y) as u32, Rgb(px));
            }
        }
    }
    img.save(path).map_err(|e| e.to_string())
}

/// Render each track as the unfold engine would draw it: one column per view-idempotent cell,
/// posed at the first observation assigned to that cell (all parts, harmonic phase included).
fn render_track_sheet(clip: &EnfoldedClipAssets, scale: u32) -> RgbImage {
    let mut renders: Vec<Vec<(usize, usize, Vec<bool>, Vec<u8>)>> = Vec::new();
    for h in &clip.holons {
        let row = (0..h.kernels.len())
            .filter_map(|vc| h.poses_per_obs.iter().find(|p| p.first().is_some_and(|r| r.view_idempotent_idx == vc)))
            .map(|poses| render_posed(h, poses))
            .collect();
        renders.push(row);
    }
    let cell = renders.iter().flatten().map(|(w, h, _, _)| (*w).max(*h)).max().unwrap_or(16) as u32;
    let label_w = 16u32;
    let cols = renders.iter().map(Vec::len).max().unwrap_or(1).max(1) as u32;
    let cw = cell * scale + 2;
    let mut img = RgbImage::from_pixel(label_w + cols * cw, renders.len().max(1) as u32 * cw, Rgb([255, 255, 255]));
    for (ti, row) in renders.iter().enumerate() {
        let y0 = ti as u32 * cw;
        draw_number(&mut img, ti, 2, y0 as i64 + 2, 2, label_color(ti));
        for (ci, (w, h, m, c)) in row.iter().enumerate() {
            let x0 = label_w + ci as u32 * cw;
            for y in 0..cell * scale {
                for x in 0..cell * scale {
                    let chk = if ((x / 6) + (y / 6)) % 2 == 0 { 200u8 } else { 160u8 };
                    let (sx, sy) = ((x / scale) as usize, (y / scale) as usize);
                    let px = if sx < *w && sy < *h && m[sy * w + sx] {
                        let i = (sy * w + sx) * 3;
                        [c[i], c[i + 1], c[i + 2]]
                    } else {
                        [chk, chk, chk]
                    };
                    img.put_pixel(x0 + x, y0 + y, Rgb(px));
                }
            }
        }
    }
    img
}

/// Write `<clip>_atlas.png` and `<clip>_tracks.png` under `dir`.
pub fn write_clip_sheets(clip: &EnfoldedClipAssets, dir: &Path) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let a = dir.join(format!("{}_atlas.png", clip.clip_id));
    let t = dir.join(format!("{}_tracks.png", clip.clip_id));
    render_atlas_sheet(clip).save(&a).map_err(|e| e.to_string())?;
    render_track_sheet(clip, 3).save(&t).map_err(|e| e.to_string())?;
    Ok(vec![a, t])
}

/// Print the prompt-holon -> enfolded-track dispatch table and initial kinematics.
pub fn print_dispatch_report(engine: &CausalUnfoldEngine) {
    let clip_id = engine.enfolded_clip.as_ref().map(|c| c.clip_id.as_str()).unwrap_or("<none>");
    println!("[debug] scene '{}' {}x{} -> clip '{}'", engine.scene.scene_id, engine.scene.width, engine.scene.height, clip_id);
    for (i, spec) in engine.scene.holons.iter().enumerate() {
        let m = engine.wavepacket_matches.get(i).copied().flatten();
        let h = &engine.holons[i];
        println!(
            "  holon {i} '{}' P={} C={:.2} rho={:.2} K={:.2} hud={} color={:?} pos=({:.1},{:.1}) vel=({:.1},{:.1}) r={:.1} -> track {}",
            spec.name,
            spec.num_parts,
            spec.cohesion_c,
            spec.rho,
            spec.bulk_modulus_k,
            spec.is_camera_locked_hud,
            spec.base_color,
            h.position[0],
            h.position[1],
            h.velocity[0],
            h.velocity[1],
            h.radius,
            m.map(|t| t.to_string()).unwrap_or_else(|| "-".to_string())
        );
    }
    for vb in &engine.active_verbs {
        println!(
            "  verb {} --[{}]--> {} (span L_AB={:.1})",
            engine.scene.holons[vb.subject_idx].name,
            vb.verb_name,
            engine.scene.holons[vb.target_idx].name,
            vb.canonical_frame.span_l_ab
        );
    }
}

/// Overlay holon trajectories, labels, and the current pose box onto rendered frames.
pub fn overlay_frames(frames: &mut [Vec<u8>], w: usize, h: usize, trajectories: &[Vec<implicate_math::q_potential::HolonPose2D>], radii: &[f32], cam_offsets: &[[f32; 2]]) {
    for (fi, frame) in frames.iter_mut().enumerate() {
        let mut img = RgbImage::from_raw(w as u32, h as u32, std::mem::take(frame)).expect("frame size");
        let off = cam_offsets.get(fi).copied().unwrap_or([0.0, 0.0]);
        for (i, tr) in trajectories.iter().enumerate() {
            let col = label_color(i);
            let Some(p) = tr.get(fi) else { continue };
            let x = p.x - off[0] as f64;
            let y = p.y - off[1] as f64;
            let r = radii.get(i).copied().unwrap_or(4.0) as f64;
            draw_rect(&mut img, (x - r) as i64, (y - r) as i64, (x + r) as i64, (y + r) as i64, col);
            draw_number(&mut img, i, (x + r) as i64 + 2, (y - r) as i64, 1, col);
        }
        *frame = img.into_raw();
    }
}

/// Tile `cols x rows` evenly sampled frames (with holon overlays) into one PNG.
pub fn write_unfold_sheet(
    report: &implicate_unfold::UnfoldRunReport,
    engine: &CausalUnfoldEngine,
    path: &Path,
) -> Result<(), String> {
    let (w, h) = (report.width as usize, report.height as usize);
    let n = report.rgb_frames.len();
    let (cols, rows) = (5usize, 2usize);
    let picks: Vec<usize> = (0..cols * rows).map(|k| k * (n.saturating_sub(1)) / (cols * rows - 1).max(1)).collect();
    let mut frames: Vec<Vec<u8>> = picks.iter().map(|&i| report.rgb_frames[i].clone()).collect();
    let trajs: Vec<Vec<implicate_math::q_potential::HolonPose2D>> = report
        .trajectories
        .iter()
        .map(|t| picks.iter().filter_map(|&i| t.get(i).copied()).collect())
        .collect();
    let radii: Vec<f32> = engine.holons.iter().map(|h| h.radius).collect();
    let offs: Vec<[f32; 2]> = picks.iter().map(|&i| report.camera_offsets.get(i).copied().unwrap_or([0.0, 0.0])).collect();
    overlay_frames(&mut frames, w, h, &trajs, &radii, &offs);
    let gap = 2usize;
    let mut sheet = RgbImage::from_pixel((cols * (w + gap)) as u32, (rows * (h + gap)) as u32, Rgb([255, 255, 255]));
    for (k, f) in frames.iter().enumerate() {
        let (cx, cy) = ((k % cols) * (w + gap), (k / cols) * (h + gap));
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                sheet.put_pixel((cx + x) as u32, (cy + y) as u32, Rgb([f[i], f[i + 1], f[i + 2]]));
            }
        }
        draw_number(&mut sheet, picks[k], cx as i64 + 1, cy as i64 + h as i64 - 6, 1, [255, 255, 255]);
    }
    sheet.save(path).map_err(|e| e.to_string())
}

/// Print each track's dominant kernel texel colours (alpha-weighted, 4-bit/channel bins).
pub fn print_track_palettes(clip: &EnfoldedClipAssets) {
    for (ti, h) in clip.holons.iter().enumerate() {
        let mut bins: std::collections::HashMap<[u8; 3], (f32, [f32; 3])> = std::collections::HashMap::new();
        let mut total = 0.0f32;
        for k in h.kernels.iter().flatten() {
            for (t, px) in k.dc_rgba.chunks_exact(4).enumerate() {
                let a = px[3] as f32 / 255.0 * k.gaussian_envelope.get(t).copied().unwrap_or(1.0);
                if a <= 0.0 {
                    continue;
                }
                let e = bins.entry([px[0] >> 4, px[1] >> 4, px[2] >> 4]).or_insert((0.0, [0.0; 3]));
                e.0 += a;
                for c in 0..3 {
                    e.1[c] += a * px[c] as f32;
                }
                total += a;
            }
        }
        let mut v: Vec<(f32, [f32; 3])> = bins.into_values().collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let raw: Vec<String> = v
            .iter()
            .take(6)
            .map(|(w, c)| format!("({:.0},{:.0},{:.0})={:.2}", c[0] / w, c[1] / w, c[2] / w, w / total.max(1e-6)))
            .collect();
        println!("    Track {ti} (n_obs={}): {}", h.poses_per_obs.len(), raw.join(" "));
    }
}
