use std::path::Path;

use implicate_lifter::video_io::read_frames_scaled;
use implicate_rolpa::{
    CliffordNetLiteRefiner, GpuCliffordNetRefiner, RolpaGpuContext, RolpaInputGrid, RolpaUniforms,
    compute_lpips_rgb, compute_psnr_rgb, compute_temporal_warping_error,
    simulate_raw_wavepacket_splat,
};

const NIMH_PATH: &str = "/mnt/storage/DDownloads/The Secret Of NIMH (1982) [BluRay] [1080p] [YTS.AM]/The.Secret.Of.NIMH.1982.1080p.BluRay.x264-[YTS.AM].mp4";

#[test]
fn r3_3_cliffordnet_rolpa_gpu_refiner_benchmark() {
    let ctx = RolpaGpuContext::shared();
    let adapter_info = ctx.adapter.get_info();
    println!(
        "[R3.3 GPU Adapter] {} ({:?})",
        adapter_info.name, adapter_info.backend
    );

    // 1. Benchmark 1920x1080 single forward-pass GPU latency (< 4.5 ms gate)
    let uniforms_1080p = RolpaUniforms {
        width: 1920,
        height: 1080,
        ..RolpaUniforms::default()
    };
    let gpu_1080p = GpuCliffordNetRefiner::new(ctx, uniforms_1080p);
    let latency_1080p_ms = gpu_1080p.benchmark_forward_latency_ms(ctx, 100);
    println!(
        "[R3.3 1920x1080 1-Pass CliffordNet-Lite GPU Latency] {:.4} ms/frame ({:.1} FPS, gate < 4.5000 ms)",
        latency_1080p_ms,
        1000.0 / latency_1080p_ms
    );
    assert!(
        latency_1080p_ms < 4.5,
        "Gate R3.3 failed: 1920x1080 GPU latency {:.4} ms >= 4.5 ms",
        latency_1080p_ms
    );

    // 2. Evaluate PSNR gain (>= 2.5 dB), LPIPS reduction (>= 20%), and temporal warping error (E_warp < 0.002)
    // across 3 test shots of The Secret of NIMH using hardware GPU execution.
    let eval_w = 640usize;
    let eval_h = 352usize;
    let shot_times = [194.0, 226.0, 350.0];

    let mut total_psnr_gain = 0.0f64;
    let mut total_lpips_red = 0.0f64;
    let mut max_ewarp = 0.0f64;

    for (shot_idx, &t0) in shot_times.iter().enumerate() {
        let frames_u8 = read_frames_scaled(
            Path::new(NIMH_PATH),
            Some(t0),
            Some(0.5),
            eval_w as u32,
            eval_h as u32,
        )
        .expect("decode NIMH test shot for R3.3");
        assert!(frames_u8.len() >= 6);

        let gt_frames: Vec<Vec<[f32; 3]>> = frames_u8
            .iter()
            .take(6)
            .map(|f| {
                f.rgb
                    .chunks_exact(3)
                    .map(|p| {
                        [
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                        ]
                    })
                    .collect()
            })
            .collect();

        let base_frames: Vec<Vec<[f32; 3]>> = gt_frames
            .iter()
            .map(|gt| simulate_raw_wavepacket_splat(eval_w, eval_h, gt, 1))
            .collect();
        let grids: Vec<RolpaInputGrid> = base_frames
            .iter()
            .map(|base| RolpaInputGrid::from_base_rgb(eval_w, eval_h, base))
            .collect();

        // Calibrate the 4 readout weights on frame 0 of the shot
        let mut cpu_refiner = CliffordNetLiteRefiner::new(eval_w as u32, eval_h as u32);
        cpu_refiner.calibrate_from_pair(&grids[0], &gt_frames[0]);

        let gpu_eval = GpuCliffordNetRefiner::new(ctx, cpu_refiner.uniforms);

        let mut shot_psnr_base = 0.0f64;
        let mut shot_psnr_ref = 0.0f64;
        let mut shot_lpips_base = 0.0f64;
        let mut shot_lpips_ref = 0.0f64;
        let mut refined_frames = Vec::with_capacity(grids.len());

        for k in 0..grids.len() {
            let ref_rgb = gpu_eval.refine_and_readback(ctx, &grids[k]);
            if k >= 1 {
                shot_psnr_base += compute_psnr_rgb(&base_frames[k], &gt_frames[k]);
                shot_psnr_ref += compute_psnr_rgb(&ref_rgb, &gt_frames[k]);
                shot_lpips_base +=
                    compute_lpips_rgb(&base_frames[k], &gt_frames[k], eval_w, eval_h);
                shot_lpips_ref += compute_lpips_rgb(&ref_rgb, &gt_frames[k], eval_w, eval_h);
            }
            refined_frames.push(ref_rgb);
        }

        let held_out = (grids.len() - 1) as f64;
        shot_psnr_base /= held_out;
        shot_psnr_ref /= held_out;
        shot_lpips_base /= held_out;
        shot_lpips_ref /= held_out;

        let psnr_gain = shot_psnr_ref - shot_psnr_base;
        let lpips_reduction = (shot_lpips_base - shot_lpips_ref) / shot_lpips_base.max(1e-9);

        let mut shot_ewarp = 0.0f64;
        for k in 0..(refined_frames.len() - 1) {
            shot_ewarp += compute_temporal_warping_error(
                &refined_frames[k],
                &refined_frames[k + 1],
                &gt_frames[k],
                &gt_frames[k + 1],
            );
        }
        shot_ewarp /= (refined_frames.len() - 1) as f64;

        println!(
            "[R3.3 NIMH Shot {} @ {:.1}s] Base PSNR={:.2} dB -> Refined PSNR={:.2} dB (+{:.2} dB) | LPIPS {:.4} -> {:.4} (-{:.1}%) | E_warp={:.6}",
            shot_idx + 1,
            t0,
            shot_psnr_base,
            shot_psnr_ref,
            psnr_gain,
            shot_lpips_base,
            shot_lpips_ref,
            lpips_reduction * 100.0,
            shot_ewarp
        );

        total_psnr_gain += psnr_gain;
        total_lpips_red += lpips_reduction;
        max_ewarp = max_ewarp.max(shot_ewarp);
    }

    let mean_psnr_gain = total_psnr_gain / (shot_times.len() as f64);
    let mean_lpips_red = total_lpips_red / (shot_times.len() as f64);
    println!(
        "[R3.3 Summary] Mean PSNR Gain = +{:.2} dB (gate >= +2.50 dB) | Mean LPIPS Reduction = {:.1}% (gate >= 20.0%) | Max E_warp = {:.6} (gate < 0.002000)",
        mean_psnr_gain,
        mean_lpips_red * 100.0,
        max_ewarp
    );

    assert!(
        mean_psnr_gain >= 2.5,
        "Gate R3.3 failed: mean PSNR gain +{:.2} dB < 2.5 dB",
        mean_psnr_gain
    );
    assert!(
        mean_lpips_red >= 0.20,
        "Gate R3.3 failed: mean LPIPS reduction {:.2}% < 20%",
        mean_lpips_red * 100.0
    );
    assert!(
        max_ewarp < 0.002,
        "Gate R3.3 failed: temporal warping error {:.6} >= 0.002",
        max_ewarp
    );
}
