//! Unified CLI (`implicate enfold | lexicon | unfold | inspect`) for `ImplicateModel`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod debug;

use implicate_lifter::probe;
use implicate_unfold::{
    CausalUnfoldEngine, ImplicateContainerManifest, UnfoldSceneSpec, enfold_video_clip,
    load_implicate_container, load_implicate_container_with_assets, save_implicate_container,
    upsert_enfolded_clip_to_container,
};

fn print_usage() {
    eprintln!(
        "Usage: implicate <command> [options]\n\n\
         Commands:\n  \
           lexicon  [--output <path.implicate>]                 Compile canonical Implicate Lexicon container\n  \
           inspect  --input <path.implicate> [--tracks] [--dump-tracks] [--dump-png] [--sheet <dir>] [--palette] [--recon-track <N>] Inspect an .implicate v1 binary container\n  \
           enfold   --video <video.mp4> [--output <path.implicate>] [--max-frames <N>] [--start-sec <S>]\n  \
           unfold   [--scene <id|all>] [--prompt \"...\"] [--dzog <file.dzog>] [--lexicon <path.implicate>] [--out-dir <dir>] [--debug]"
    );
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn cmd_lexicon(args: &[String]) -> Result<(), String> {
    let out_path = get_arg(args, "--output")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/opencode/corpus.implicate"));
    let manifest = ImplicateContainerManifest::canonical_corpus_manifest();
    save_implicate_container(&out_path, &manifest, 0x01, &[])
        .map_err(|e| format!("Failed to save container to {}: {e}", out_path.display()))?;
    println!(
        "[implicate lexicon] Saved '{}' ({} concept clusters, {} verb potentials, {} GATr rules) -> {}",
        manifest.title,
        manifest.concept_lexicon.clusters.len(),
        manifest.verb_potentials.entries.len(),
        manifest.gatr_binder.rules.len(),
        out_path.display()
    );
    Ok(())
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let in_path = get_arg(args, "--input")
        .map(PathBuf::from)
        .ok_or_else(|| "Missing required --input <path.implicate>".to_string())?;
    let (manifest, flags, payload) = load_implicate_container(&in_path)
        .map_err(|e| format!("Failed to load {}: {e}", in_path.display()))?;
    let (_, _, clips) = load_implicate_container_with_assets(&in_path)
        .map_err(|e| format!("Failed to decode enfolded payload {}: {e}", in_path.display()))?;

    println!("=== .implicate v1 Container ===");
    println!("Title:          {}", manifest.title);
    println!("Flags:          0x{:08x}", flags);
    println!("Corpus Sources: {:?}", manifest.corpus_sources);
    println!("Concepts:       {}", manifest.concept_lexicon.clusters.len());
    println!("Verb Potentials ({}):", manifest.verb_potentials.entries.len());
    for vp in &manifest.verb_potentials.entries {
        println!(
            "  - {:<12} (sigma = {:.4})",
            vp.verb, vp.potential.sigma
        );
    }
    println!("GATr Rules:     {}", manifest.gatr_binder.rules.len());
    println!("Binary Payload: {} bytes ({} enfolded clip(s))", payload.len(), clips.len());
    let do_dump = args.iter().any(|a| a == "--dump-png");
    let show_tracks = args.iter().any(|a| a == "--tracks" || a == "--dump-tracks");
    let dump_tracks = args.iter().any(|a| a == "--dump-tracks");
    for clip in &clips {
        let hud_info = clip.atlas.camera_locked_hud.as_ref().map(|h| format!("{:?}", h.y_band)).unwrap_or_else(|| "none".to_string());
        println!(
            "  - Clip '{}' ({}x{}, {} frames): {} layer(s) (active_y={:?}), HUD={}, hud_band={}, {} articulated holon(s)",
            clip.clip_id,
            clip.atlas.width,
            clip.atlas.height,
            clip.num_frames,
            clip.atlas.layers.len(),
            clip.atlas.active_y_range,
            clip.atlas.camera_locked_hud.is_some(),
            hud_info,
            clip.holons.len()
        );
        for (hi, h) in clip.holons.iter().enumerate() {
            let root_color = h.part_colors.first().copied().unwrap_or([0, 0, 0]);
            let (min_x, max_x, min_y, max_y, max_disp) = if let Some(first_poses) = h.poses_per_obs.first().and_then(|p| p.first()) {
                let mut min_x = first_poses.center[0];
                let mut max_x = first_poses.center[0];
                let mut min_y = first_poses.center[1];
                let mut max_y = first_poses.center[1];
                let x0 = first_poses.center[0];
                let y0 = first_poses.center[1];
                let mut max_disp = 0.0f64;
                for f in &h.poses_per_obs {
                    if let Some(r) = f.first() {
                        min_x = min_x.min(r.center[0]);
                        max_x = max_x.max(r.center[0]);
                        min_y = min_y.min(r.center[1]);
                        max_y = max_y.max(r.center[1]);
                        let d = (r.center[0] - x0).hypot(r.center[1] - y0);
                        if d > max_disp {
                            max_disp = d;
                        }
                    }
                }
                (min_x, max_x, min_y, max_y, max_disp)
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };
            if show_tracks {
                println!(
                    "    Track {}: P={}, views={}, obs_len={}, root_rgb={:?}, x=[{:.1}, {:.1}], y=[{:.1}, {:.1}], disp={:.1}px",
                    hi, h.num_parts, h.num_view_cells, h.poses_per_obs.len(), root_color, min_x, max_x, min_y, max_y, max_disp
                );
            } else {
                println!("    Holon {}: P={}, views={}, obs_len={}", hi, h.num_parts, h.num_view_cells, h.poses_per_obs.len());
            }
            if dump_tracks {
                if let Some(view0) = h.kernels.first().and_then(|k| k.first()) {
                    let dump_ppm = format!("/tmp/opencode/dump_track_{}_{}.ppm", clip.clip_id, hi);
                    let mut header = format!("P6\n{} {}\n255\n", view0.res_w, view0.res_h).into_bytes();
                    let mut rgb_bytes = Vec::with_capacity(view0.res_w * view0.res_h * 3);
                    for chunk in view0.dc_rgba.chunks_exact(4) {
                        let a = chunk[3] as f32 / 255.0;
                        rgb_bytes.push((chunk[0] as f32 * a) as u8);
                        rgb_bytes.push((chunk[1] as f32 * a) as u8);
                        rgb_bytes.push((chunk[2] as f32 * a) as u8);
                    }
                    header.extend_from_slice(&rgb_bytes);
                    let _ = fs::write(&dump_ppm, header);
                    println!("      Saved track {hi} root kernel -> {dump_ppm}");
                }
            }
        }
        if let Some(ti) = get_arg(args, "--recon-track").and_then(|v| v.parse::<usize>().ok()) {
            let dir = get_arg(args, "--sheet").unwrap_or_else(|| "/tmp/opencode".to_string());
            let _ = fs::create_dir_all(&dir);
            let path = Path::new(&dir).join(format!("{}_track{ti}_recon.png", clip.clip_id));
            if debug::write_recon_strip(clip, ti, &path).is_ok() {
                println!("    Saved recon strip -> {}", path.display());
            }
        }
        if args.iter().any(|a| a == "--palette") {
            debug::print_track_palettes(clip);
        }
        if let Some(dir) = get_arg(args, "--sheet") {
            for p in debug::write_clip_sheets(clip, Path::new(&dir))? {
                println!("    Saved sheet -> {}", p.display());
            }
        }
        if do_dump && !clip.atlas.layers.is_empty() {
            let img = &clip.atlas.layers[0].atlas.image;
            let dump_ppm = format!("/tmp/opencode/dump_{}.ppm", clip.clip_id);
            let mut header = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
            header.extend_from_slice(&img.rgb);
            let _ = fs::write(&dump_ppm, header);
            println!("    Saved atlas layer 0 -> {dump_ppm}");
        }
    }
    Ok(())
}

fn cmd_enfold(args: &[String]) -> Result<(), String> {
    let video_str = get_arg(args, "--video")
        .ok_or_else(|| "Missing required --video <video.mp4>".to_string())?;
    let video_path = Path::new(&video_str);
    let max_frames: usize = get_arg(args, "--max-frames")
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let start_sec: f64 = get_arg(args, "--start-sec")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    let info = probe(video_path).map_err(|e| format!("probe failed: {e}"))?;
    let clip = enfold_video_clip(video_path, start_sec, max_frames)
        .map_err(|e| format!("enfold_video_clip failed: {e}"))?;

    let out_path = get_arg(args, "--output").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(format!("/tmp/opencode/{}.implicate", clip.clip_id))
    });

    upsert_enfolded_clip_to_container(&out_path, clip.clone())
        .map_err(|e| format!("upsert_enfolded_clip_to_container failed: {e}"))?;

    println!(
        "[implicate enfold] Video: {} ({}x{} @ {:.2} fps, {} frames enfolded -> {}x{})",
        video_path.display(),
        info.width,
        info.height,
        info.fps,
        clip.num_frames,
        clip.atlas.width,
        clip.atlas.height,
    );
    println!("  Shots Detected: {}", clip.shots.len());
    for (idx, shot) in clip.shots.iter().enumerate() {
        println!(
            "    Shot {:02}: frames {}..{} ({:.2}s..{:.2}s) | regime={:?} | mean_scroll=({:+.2}, {:+.2}) px/frame",
            idx + 1,
            shot.frame_start,
            shot.frame_end,
            shot.start_sec,
            shot.end_sec,
            shot.camera_regime,
            shot.mean_camera_velocity[0],
            shot.mean_camera_velocity[1]
        );
    }
    let holon_desc: Vec<String> = clip
        .holons
        .iter()
        .map(|h| format!("P={}, views={}", h.num_parts, h.num_view_cells))
        .collect();
    println!(
        "  Multi-Plane Atlas: {} depth layer(s) | Camera-Locked HUD Holon: {} | Articulated Holons: {} {:?} | Saved -> {}",
        clip.atlas.layers.len(),
        clip.atlas.camera_locked_hud.is_some(),
        clip.holons.len(),
        holon_desc,
        out_path.display()
    );
    Ok(())
}

fn cmd_unfold(args: &[String]) -> Result<(), String> {
    let scene_filter = get_arg(args, "--scene").unwrap_or_else(|| "all".to_string());
    let out_dir = get_arg(args, "--out-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/opencode/unfold_out"));
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("Failed to create output dir {}: {e}", out_dir.display()))?;

    let (manifest, enfolded_clips) = if let Some(lex_path) =
        get_arg(args, "--lexicon").or_else(|| get_arg(args, "--container"))
    {
        let (m, _, clips) = load_implicate_container_with_assets(Path::new(&lex_path))
            .map_err(|e| format!("Failed to load container {lex_path}: {e}"))?;
        (m, clips)
    } else {
        (
            ImplicateContainerManifest::canonical_corpus_manifest(),
            Vec::new(),
        )
    };

    let mut selected: Vec<UnfoldSceneSpec> = if let Some(prompt_str) = get_arg(args, "--prompt") {
        vec![UnfoldSceneSpec::parse_prompt(&prompt_str)?]
    } else if let Some(dzog_path) = get_arg(args, "--dzog") {
        let script = fs::read_to_string(&dzog_path)
            .map_err(|e| format!("Failed to read {dzog_path}: {e}"))?;
        vec![UnfoldSceneSpec::parse_prompt(&script)?]
    } else {
        let all_scenes = UnfoldSceneSpec::canonical_r4_3_novel_scenes();
        if scene_filter == "all" {
            all_scenes
        } else {
            all_scenes
                .into_iter()
                .filter(|s| s.scene_id == scene_filter)
                .collect()
        }
    };

    if selected.is_empty() {
        return Err(format!("No scene matched --scene '{scene_filter}'"));
    }

    if let Some(nf) = get_arg(args, "--frames").and_then(|s| s.parse::<usize>().ok()) {
        for s in &mut selected {
            s.num_frames = nf;
        }
    }
    if let Some(name) = get_arg(args, "--name") {
        for s in &mut selected {
            s.scene_id = name.clone();
        }
    }
    if let Some(cv_str) = get_arg(args, "--cam-vel") {
        let parts: Vec<f32> = cv_str
            .trim_matches(|c| c == '(' || c == ')')
            .split(',')
            .filter_map(|p| p.trim().parse::<f32>().ok())
            .collect();
        if parts.len() == 2 {
            for s in &mut selected {
                s.camera_scroll_vel = [parts[0], parts[1]];
            }
        }
    }

    let debug_mode = args.iter().any(|a| a == "--debug");
    for scene in selected {
        let mut engine = CausalUnfoldEngine::new(manifest.clone(), scene.clone())
            .with_enfolded_clips(&enfolded_clips);
        if debug_mode {
            debug::print_dispatch_report(&engine);
        }
        let report = engine.unfold_all_frames();
        if debug_mode {
            let sheet = out_dir.join(format!("{}_debug.png", report.scene_id));
            debug::write_unfold_sheet(&report, &engine, &sheet)?;
            println!("[debug] overlay contact sheet -> {}", sheet.display());
        }

        let (wav_path, mp4_path) = report
            .export_wav_and_mp4(&out_dir)
            .map_err(|e| format!("Failed to export media for '{}': {e}", report.scene_id))?;

        let hud_start = report.hud_score_history.first().copied().unwrap_or(0);
        let hud_end = report.hud_score_history.last().copied().unwrap_or(0);
        println!(
            "[implicate unfold] Scene '{}' ({}x{}, {} frames): {:.2} ms/frame ({:.1} FPS) | Permanence={:.1}% | Boundary={:.1}% | HUD Score={:06}->{:06} | MP4 -> {} | WAV -> {}",
            report.scene_id,
            report.width,
            report.height,
            report.num_frames,
            report.mean_frame_ms,
            report.effective_fps,
            report.object_permanence_rate * 100.0,
            report.boundary_compliance_rate * 100.0,
            hud_start,
            hud_end,
            mp4_path.display(),
            wav_path.display()
        );
        for vm in &report.verb_metrics {
            println!(
                "  -> Verb ({} --[{}]--> {}): normalized Frechet = {:.2}%",
                vm.subject,
                vm.verb,
                vm.target,
                vm.normalized_frechet * 100.0
            );
        }
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let res = match args[1].as_str() {
        "lexicon" => cmd_lexicon(&args[2..]),
        "inspect" => cmd_inspect(&args[2..]),
        "enfold" => cmd_enfold(&args[2..]),
        "unfold" => cmd_unfold(&args[2..]),
        _ => {
            print_usage();
            Err(format!("Unknown command '{}'", args[1]))
        }
    };

    if let Err(err) = res {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
