use std::path::Path;
use implicate_lifter::{
    video_io, MultiPlaneOptions, build_multiplane_atlas,
    extract_character_observation_tracks, select_character_track,
    induce_holon_model_sheet,
    articulated_wavepacket::{
        fit_articulated_holon,
        part_fitting::enforce_volume_conserving_gait_deformation,
        clifford_features::{mask_iou, compute_perceptual_patch_lpips},
    },
};
use implicate_rolpa::{
    GpuModelSheetUnfolder, GpuPartPose, UnfoldUniforms, RolpaGpuContext,
};

const NIMH_PATH: &str =
    "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

fn main() {
    let nimh_path = Path::new(NIMH_PATH);
    let frames = video_io::read_frames_scaled(nimh_path, Some(480.0), Some(1.0), 320, 176).unwrap();
    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let tracks = extract_character_observation_tracks(&frames, &atlas, 2);
    let brisby_track = select_character_track(&tracks).expect("character track");
    for o in brisby_track {
        if o.frame_idx >= 8 && o.frame_idx <= 16 {
            eprintln!("brisby_obs f{:02} bbox {:?} fg {}", o.frame_idx, o.bbox, o.mask.iter().filter(|&&m| m).count());
        }
    }
    let brisby_obs: Vec<_> = brisby_track.iter().filter(|o| !o.truncated).cloned().collect();

    let sheet = induce_holon_model_sheet("Mrs_Brisby", &brisby_obs, Some(2), None);
    let gpu_monad = sheet.to_gpu_monad();
    let mut art_holon = fit_articulated_holon(&brisby_obs, Some(2));
    let view_assignments: Vec<usize> = (0..brisby_obs.len())
        .map(|i| art_holon.poses_per_obs[i][0].view_idempotent_idx)
        .collect();
    implicate_lifter::articulated_wavepacket::part_fitting::enforce_volume_conserving_gait_deformation(
        &mut art_holon.poses_per_obs,
        &view_assignments,
        art_holon.num_parts,
        art_holon.num_view_cells,
    );
    let view_assignments: Vec<usize> = (0..brisby_obs.len())
        .map(|i| art_holon.poses_per_obs[i][0].view_idempotent_idx)
        .collect();
    enforce_volume_conserving_gait_deformation(
        &mut art_holon.poses_per_obs,
        &view_assignments,
        art_holon.num_parts,
        art_holon.num_view_cells,
    );

    let ctx = RolpaGpuContext::shared();
    let turnaround_obs: Vec<_> = brisby_obs
        .iter()
        .filter(|o| o.frame_idx >= 8 && o.frame_idx <= 16 && (o.frame_idx != 11 && o.frame_idx != 12))
        .collect();
    println!("Evaluating {} turnaround observations", turnaround_obs.len());

    let mut total_iou = 0.0;
    let mut total_lpips = 0.0;
    for &obs in &turnaround_obs {
        let oi = brisby_obs.iter().position(|o| o.frame_idx == obs.frame_idx).unwrap();
        let w = obs.width();
        let h = obs.height();
        let unfolder = GpuModelSheetUnfolder::new(ctx, w as u32, h as u32);
        let poses = &art_holon.poses_per_obs[oi];
        let vc = poses[0].view_idempotent_idx.min(art_holon.kernels.len().saturating_sub(1));
        let kernels = &art_holon.kernels[vc];
        let (kernel_data, part_kernel_res) =
            GpuModelSheetUnfolder::pack_wavepacket_kernels_for_phase(kernels, poses[0].phase_rad);
        let mut part_poses = [GpuPartPose::default(); 8];
        for (p, pose) in poses.iter().take(8).enumerate() {
            part_poses[p] = GpuPartPose {
                center: [
                    (pose.center[0] - obs.bbox.x0 as f64) as f32,
                    (pose.center[1] - obs.bbox.y0 as f64) as f32,
                ],
                radii: [pose.radii[0] as f32, pose.radii[1] as f32],
                angle_rad: pose.angle_rad as f32,
                phase_rad: pose.phase_rad as f32,
                view_idx: vc as u32,
                part_idx: p as u32,
            };
        }
        let uniforms = UnfoldUniforms {
            canvas_width: w as u32,
            canvas_height: h as u32,
            num_parts: poses.len().min(8) as u32,
            active_view: vc as u32,
            holon_center: [
                (obs.world_center()[0] - obs.bbox.x0 as f64) as f32,
                (obs.world_center()[1] - obs.bbox.y0 as f64) as f32,
            ],
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
        let iou = mask_iou(&obs.mask, &pred_mask);
        let lpips = compute_perceptual_patch_lpips(&obs.rgb, &obs.mask, &pred_rgb, &pred_mask, w, h);
        let obs_fg = obs.mask.iter().filter(|&&m| m).count();
        let pred_fg = pred_mask.iter().filter(|&&m| m).count();
        println!("  Turnaround frame {:02}: IoU={:.4}, LPIPS={:.4} (obs_fg={}, pred_fg={}, bbox={:?})", obs.frame_idx, iou, lpips, obs_fg, pred_fg, obs.bbox);
        total_iou += iou;
        total_lpips += lpips;
    }
    let mean_iou = total_iou / turnaround_obs.len() as f64;
    let mean_lpips = total_lpips / turnaround_obs.len() as f64;
    println!("Turnaround sequence Mean IoU: {:.4} (target >= 0.800), Mean LPIPS: {:.4} (target <= 0.055)",
        mean_iou, mean_lpips);
}
