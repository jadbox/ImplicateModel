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
    let frames = video_io::read_frames(raptor_path, Some(1.0), Some(2.0)).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let obs = extract_dominant_character_observations(&frames, &atlas);
    let sheet = induce_holon_model_sheet("Raptor_Jet", &obs, Some(1), None);
    let gpu_monad = sheet.to_gpu_monad();
    let ctx = RolpaGpuContext::shared();
    let art_holon = fit_articulated_holon(&obs, Some(1));

    let (mut all_iou, mut clean_iou) = (0.0, 0.0);
    let (mut all_n, mut clean_n) = (0, 0);

    for oi in 0..obs.len().min(30) {
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
        println!("Frame {:02}: h={}, IoU={:.4} (clean={})", oi, h, iou, !is_firing_merge);
    }
    println!("All {} frames: Mean IoU = {:.4}", all_n, all_iou / all_n as f64);
    println!("Clean {} frames: Mean IoU = {:.4}", clean_n, clean_iou / clean_n as f64);
}
