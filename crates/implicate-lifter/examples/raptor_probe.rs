use std::path::Path;
use implicate_lifter::{
    video_io, build_multiplane_atlas, MultiPlaneOptions,
    articulated_wavepacket::{extract_dominant_character_observations, induce_holon_model_sheet, fit_articulated_holon},
};
use implicate_rolpa::{RolpaGpuContext, GpuModelSheetUnfolder, UnfoldUniforms, GpuPartPose};

const RAPTOR_PATH: &str =
    "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";

fn main() {
    let raptor_path = Path::new(RAPTOR_PATH);
    let frames = video_io::read_frames(raptor_path, Some(3.0), Some(2.0)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let bg0 = atlas.reconstruct_background_frame(0);
    video_io::write_png(Path::new("/tmp/opencode/raptor_f0.png"), frames[0].width, frames[0].height, &frames[0].rgb).unwrap();
    video_io::write_png(Path::new("/tmp/opencode/raptor_bg0.png"), bg0.width as u32, bg0.height as u32, &bg0.rgb).unwrap();
    let obs = extract_dominant_character_observations(&frames, &atlas);
    for (oi, o) in obs.iter().enumerate() {
        let yellow_px = (0..o.width() * o.height())
            .filter(|&i| o.mask[i] && o.rgb[i * 3] > 180 && o.rgb[i * 3 + 1] > 140 && o.rgb[i * 3 + 2] < 80)
            .count();
        if yellow_px > 0 || o.width() > 20 || o.height() > 36 {
            println!("obs[{:02}] f={:02} bbox={:?} ({}x{}) yellow_px={}", oi, o.frame_idx, o.bbox, o.width(), o.height(), yellow_px);
            if oi == 47 || oi == 49 {
                for ly in 18..32 {
                    print!("  rgb y={:02}: ", ly);
                    for lx in 0..7 {
                        let i = ly * o.width() + lx;
                        if o.mask[i] {
                            print!("[{:3},{:3},{:3}] ", o.rgb[i*3], o.rgb[i*3+1], o.rgb[i*3+2]);
                        } else {
                            print!("[  .   .   .] ");
                        }
                    }
                    println!();
                }
                for ly in 0..o.height() {
                    let mut line = format!("  y={:02}: ", ly);
                    for lx in 0..o.width() {
                        let i = ly * o.width() + lx;
                        if o.mask[i] {
                            let (r, g, b) = (o.rgb[i * 3], o.rgb[i * 3 + 1], o.rgb[i * 3 + 2]);
                            if r > 180 && g > 140 && b < 80 {
                                line.push('Y');
                            } else {
                                line.push('#');
                            }
                        } else {
                            line.push('.');
                        }
                    }
                    println!("{}", line);
                }
            }
        }
    }
    let sheet = induce_holon_model_sheet("Raptor_Jet", &obs, Some(1), None);
    let gpu_monad = sheet.to_gpu_monad();
    let ctx = RolpaGpuContext::shared();
    let art_holon = fit_articulated_holon(&obs, Some(1));

    let (mut all_iou, mut clean_iou) = (0.0, 0.0);
    let (mut all_n, mut clean_n) = (0, 0);

    for oi in 0..obs.len() {
        let obsi = &obs[oi];
        let (w, h) = (obsi.width(), obsi.height());
        let unfolder = GpuModelSheetUnfolder::new(ctx, w as u32, h as u32);

        let poses = &art_holon.poses_per_obs[oi];
        let vc = poses[0].view_idempotent_idx.min(art_holon.kernels.len().saturating_sub(1));
        let kernels = &art_holon.kernels[vc];
        let (kernel_data, part_kernel_res) =
            GpuModelSheetUnfolder::pack_wavepacket_kernels_for_phase(kernels, poses[0].phase_rad);

        let mut part_poses = [GpuPartPose::default(); 8];
        part_poses[0] = GpuPartPose {
            center: [
                (poses[0].center[0] - obsi.bbox.x0 as f64) as f32,
                (poses[0].center[1] - obsi.bbox.y0 as f64) as f32,
            ],
            radii: [poses[0].radii[0] as f32, poses[0].radii[1] as f32],
            angle_rad: poses[0].angle_rad as f32,
            phase_rad: poses[0].phase_rad as f32,
            view_idx: vc as u32,
            part_idx: 0,
        };
        let uniforms = UnfoldUniforms {
            canvas_width: w as u32,
            canvas_height: h as u32,
            num_parts: 1,
            active_view: vc as u32,
            holon_center: [(w as f32) * 0.5, (h as f32) * 0.5],
            scale: [1.0, 1.0],
            phase_rad: poses[0].phase_rad as f32,
            ink_threshold: 0.12,
            ink_width: 0.08,
            aspect_ratio: (w as f32) / (h as f32).max(1.0),
            volume_scale: [1.0, 1.0],
            flags: 0,
            _pad0: 0,
            part_poses,
            part_kernel_res,
        };
        let mut pred_rgb = vec![0u8; w * h * 3];
        let mut pred_mask = vec![false; w * h];
        unfolder.unfold_and_readback_u8(ctx, &gpu_monad, &uniforms, Some(&kernel_data), &mut pred_rgb, &mut pred_mask);

        let inter = (0..w * h).filter(|&i| obsi.mask[i] && pred_mask[i]).count();
        let union = (0..w * h).filter(|&i| obsi.mask[i] || pred_mask[i]).count();
        let iou = inter as f64 / union.max(1) as f64;
        all_iou += iou;
        all_n += 1;

        let is_firing_merge = obsi.height() > 53;
        if !is_firing_merge {
            clean_iou += iou;
            clean_n += 1;
        }
        let pred_yellow = (0..w * h)
            .filter(|&i| pred_mask[i] && pred_rgb[i * 3] > 180 && pred_rgb[i * 3 + 1] > 140 && pred_rgb[i * 3 + 2] < 80)
            .count();
        println!("Frame {:02}: w={} h={}, IoU={:.4}, pred_yellow={}", oi, w, h, iou, pred_yellow);
    }
    println!("All {} frames: Mean IoU = {:.4}", all_n, all_iou / all_n as f64);
    println!("Clean {} frames: Mean IoU = {:.4}", clean_n, clean_iou / clean_n as f64);
}
