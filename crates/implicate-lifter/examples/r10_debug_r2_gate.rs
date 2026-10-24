use implicate_lifter::{
    MultiPlaneOptions, build_multiplane_atlas, video_io,
    extract_dominant_character_observations,
};

fn main() {
    let smb3_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d/smb3_w1_1_256x240.mp4";
    let pacman_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_pacman/pacman_240x308.mp4";
    let raptor_path = "/home/jdunlap/github/jadbox/implicate_order/holonengine/example_input2d_raptor/raptor_320x240.mp4";

    for (name, path) in [("Pacman", pacman_path), ("SMB3", smb3_path), ("Raptor", raptor_path)] {
        let frames = video_io::read_frames(std::path::Path::new(path), Some(1.0), Some(1.5)).expect("frames");
        let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
        let obs = extract_dominant_character_observations(&frames, &atlas);
        println!("\n=== {name} ===");
        println!("obs count: {}", obs.len());
        for (i, o) in obs.iter().take(5).enumerate() {
            println!("  obs {i}: frame={} bbox={:?} area={}", o.frame_idx, o.bbox, o.mask.iter().filter(|&&m| m).count());
        }
        // Let's inspect the track score and other tracks
        let all_tracks = implicate_lifter::extract_character_observation_tracks(&frames, &atlas, 5);
        println!("All tracks count: {}", all_tracks.len());
        for (ti, tr) in all_tracks.iter().enumerate() {
            println!("  Track {ti}: n={} frame_span={}..{} bbox0={:?}", tr.len(), tr[0].frame_idx, tr.last().unwrap().frame_idx, tr[0].bbox);
        }
    }
}
