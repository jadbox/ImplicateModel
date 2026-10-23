//! Binary payload encoder/decoder for enfolded `MultiPlaneAtlas` layers,
//! camera-locked HUD holons, and `ArticulatedHolon` CFT wavepackets inside `.implicate` v1 containers.

use std::io::{self, ErrorKind};
use std::path::Path;

use implicate_lexicon::{
    ShotLexicon, VlmHolonRecord, VlmVerbRecord, detect_cft_kinematic_impulses,
    format_timestamp_range, snap_verb_window_to_cft_impulses,
};
use implicate_lifter::{
    ArticulatedHolon, LuminescenceFrame, MultiPlaneAtlas, MultiPlaneOptions, ParallaxLayer, Rect,
    RgbImage8, ShotSegment, ShotSegmenterConfig, StreamingShotSegmenter, SubHolonPose,
    WavepacketPartKernel, WorldAtlas, build_multiplane_atlas,
    extract_character_observation_tracks, fit_articulated_holon, probe, video_io,
};
use implicate_math::{HolonPose2D, LearnedVerbPotential};
use serde::{Deserialize, Serialize};

use crate::container::{
    IMPLICATE_FLAG_ENFOLDED_ASSETS, IMPLICATE_FLAG_LEXICON, ImplicateContainerManifest,
    load_implicate_container, save_implicate_container,
};

pub const ENFOLDED_PAYLOAD_MAGIC: &[u8; 4] = b"ENFD";
pub const ENFOLDED_PAYLOAD_VERSION: u32 = 1;

/// Lightweight manifest summary for an enfolded video clip stored in the `.implicate` binary payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnfoldedClipSummary {
    pub clip_id: String,
    pub source_video: String,
    pub width: usize,
    pub height: usize,
    pub num_frames: usize,
    pub num_shots: usize,
    pub num_layers: usize,
    pub has_camera_locked_hud: bool,
    pub num_articulated_holons: usize,
    pub holon_part_counts: Vec<usize>,
    pub holon_view_counts: Vec<usize>,
}

/// Complete enfolded assets (`MultiPlaneAtlas` background/HUD layers + `ArticulatedHolon` wavepackets)
/// for a video clip stored inside a `.implicate` v1 container.
#[derive(Clone, Debug)]
pub struct EnfoldedClipAssets {
    pub clip_id: String,
    pub source_video: String,
    pub num_frames: usize,
    pub shots: Vec<ShotSegment>,
    pub atlas: MultiPlaneAtlas,
    pub holons: Vec<ArticulatedHolon>,
}

impl EnfoldedClipAssets {
    pub fn summary(&self) -> EnfoldedClipSummary {
        EnfoldedClipSummary {
            clip_id: self.clip_id.clone(),
            source_video: self.source_video.clone(),
            width: self.atlas.width,
            height: self.atlas.height,
            num_frames: self.num_frames,
            num_shots: self.shots.len(),
            num_layers: self.atlas.layers.len(),
            has_camera_locked_hud: self.atlas.camera_locked_hud.is_some(),
            num_articulated_holons: self.holons.len(),
            holon_part_counts: self.holons.iter().map(|h| h.num_parts).collect(),
            holon_view_counts: self.holons.iter().map(|h| h.num_view_cells).collect(),
        }
    }
}

/// Enfold a video clip `[start_sec, start_sec + duration]` into a `MultiPlaneAtlas` (including
/// camera-locked HUD holon) and `ArticulatedHolon` CFT wavepacket.
pub fn enfold_video_clip(
    video_path: &Path,
    start_sec: f64,
    max_frames: usize,
) -> io::Result<EnfoldedClipAssets> {
    let info = probe(video_path)?;
    let duration_sec = (max_frames.max(1) as f64) / info.fps.max(1.0);

    // For high-resolution sources (> 512px wide), decode at 320x180 for fast CFT multi-plane & wavepacket lifting.
    let frames = if info.width > 512 {
        let mut f =
            video_io::read_frames_scaled(video_path, Some(start_sec), Some(duration_sec), 320, 180)?;
        f.truncate(max_frames);
        f
    } else {
        let mut f = video_io::read_frames(video_path, Some(start_sec), Some(duration_sec))?;
        f.truncate(max_frames);
        f
    };

    if frames.is_empty() {
        return Err(io::Error::new(
            ErrorKind::UnexpectedEof,
            "No frames decoded from video clip",
        ));
    }

    let segmenter = StreamingShotSegmenter::new(ShotSegmenterConfig::default());
    let seg_report =
        segmenter.segment_video_range(video_path, Some(start_sec), Some(duration_sec))?;

    let atlas = build_multiplane_atlas(&frames, &MultiPlaneOptions::default());
    let obs_tracks = extract_character_observation_tracks(&frames, &atlas, 3);
    let mut holons = Vec::with_capacity(obs_tracks.len());
    for obs in &obs_tracks {
        if !obs.is_empty() {
            holons.push(fit_articulated_holon(obs, None));
        }
    }

    let clip_id = video_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip")
        .to_string();
    let source_video = video_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("video.mp4")
        .to_string();

    Ok(EnfoldedClipAssets {
        clip_id,
        source_video,
        num_frames: frames.len(),
        shots: seg_report.shots,
        atlas,
        holons,
    })
}

/// Assimilate an enfolded clip's semantic concepts, CFT kinematic impulse intervals, and real tracked
/// `ArticulatedHolon` trajectories ($\mathcal{Q}_{\text{Cl},v}$ potentials) into an `ImplicateContainerManifest`.
pub fn assimilate_clip_into_manifest(
    manifest: &mut ImplicateContainerManifest,
    clip: &EnfoldedClipAssets,
) {
    let verb_name = format!("enfolded_{}", clip.clip_id);
    let pair_verb_name = format!("enfolded_pair_{}", clip.clip_id);

    let extract_root_traj = |h: &ArticulatedHolon| -> Vec<HolonPose2D> {
        let mut traj = Vec::with_capacity(h.poses_per_obs.len());
        for poses in &h.poses_per_obs {
            if let Some(root) = poses.first() {
                traj.push(HolonPose2D::new(
                    root.center[0],
                    root.center[1],
                    root.angle_rad,
                ));
            }
        }
        traj
    };

    let traj_a = clip
        .holons
        .first()
        .map(extract_root_traj)
        .unwrap_or_default();

    let fps = 24.0;
    let cft_impulses = detect_cft_kinematic_impulses(&traj_a, fps);
    let raw_dur = (clip.num_frames.max(1) as f64) / fps;
    let snapped = snap_verb_window_to_cft_impulses((0.0, raw_dur), &cft_impulses, raw_dur.max(2.0));
    let time_range = format_timestamp_range(snapped.0, snapped.1);

    let mut vlm_holons = Vec::new();
    for (idx, _) in clip.holons.iter().enumerate() {
        vlm_holons.push(VlmHolonRecord {
            id: (idx + 1) as u32,
            noun: format!("{}_actor_{}", clip.clip_id, idx + 1),
            appearance: "articulated_wavepacket_actor".to_string(),
            location: "viewport".to_string(),
        });
    }
    if vlm_holons.is_empty() {
        vlm_holons.push(VlmHolonRecord {
            id: 1,
            noun: format!("{}_actor", clip.clip_id),
            appearance: "articulated_wavepacket_actor".to_string(),
            location: "viewport".to_string(),
        });
    }
    let anchor_id = (vlm_holons.len() + 1) as u32;
    vlm_holons.push(VlmHolonRecord {
        id: anchor_id,
        noun: format!("{}_anchor", clip.clip_id),
        appearance: "multiplane_atlas_anchor".to_string(),
        location: "center".to_string(),
    });
    if clip.atlas.camera_locked_hud.is_some() {
        vlm_holons.push(VlmHolonRecord {
            id: anchor_id + 1,
            noun: format!("{}_hud_bar", clip.clip_id),
            appearance: "camera_locked_hud_overlay".to_string(),
            location: "hud_border".to_string(),
        });
    }

    let mut verbs = vec![VlmVerbRecord {
        time_range: time_range.clone(),
        subject_id: 1,
        verb: verb_name.clone(),
        target: format!("{}_anchor", clip.clip_id),
    }];
    if clip.holons.len() >= 2 {
        verbs.push(VlmVerbRecord {
            time_range,
            subject_id: 1,
            verb: pair_verb_name.clone(),
            target: format!("{}_actor_2", clip.clip_id),
        });
    }

    let shot_lexicon = ShotLexicon {
        milieu: format!("{}_milieu", clip.clip_id),
        holons: vlm_holons,
        verbs,
    };
    manifest.concept_lexicon.learn_from_shot(&shot_lexicon);

    if traj_a.len() >= 4 {
        let a0 = traj_a[0];
        let anchor_pos = [
            (clip.atlas.width as f64) * 0.5,
            (clip.atlas.height as f64) * 0.5,
        ];
        let dx = anchor_pos[0] - a0.x;
        let dy = anchor_pos[1] - a0.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let (bx, by) = if dist < 20.0 {
            (a0.x + 36.0, a0.y)
        } else {
            (anchor_pos[0], anchor_pos[1])
        };
        let traj_b = vec![HolonPose2D::new(bx, by, 0.0); traj_a.len()];
        let potential =
            LearnedVerbPotential::fit_from_trajectories(&verb_name, &[(traj_a.clone(), traj_b)]);
        if let Some(spinor) = manifest
            .concept_lexicon
            .embed_clauses(&[verb_name.as_str()])
            .into_iter()
            .next()
        {
            manifest.verb_potentials.upsert(verb_name, spinor, potential);
        }

        // If a second tracked actor is present, also fit the true two-body relative PGA potential M_rel(t) = ~M_B(t) M_A(t)
        if let Some(holon_b) = clip.holons.get(1) {
            let traj_b_actor = extract_root_traj(holon_b);
            let n_pair = traj_a.len().min(traj_b_actor.len());
            if n_pair >= 4 {
                let mut pair_a = traj_a[..n_pair].to_vec();
                let pair_b = traj_b_actor[..n_pair].to_vec();
                let d0 = (pair_a[0].x - pair_b[0].x).hypot(pair_a[0].y - pair_b[0].y);
                if d0 < 12.0 {
                    pair_a[0].x += 16.0;
                }
                let pair_pot = LearnedVerbPotential::fit_from_trajectories(
                    &pair_verb_name,
                    &[(pair_a, pair_b)],
                );
                if let Some(spinor) = manifest
                    .concept_lexicon
                    .embed_clauses(&[pair_verb_name.as_str()])
                    .into_iter()
                    .next()
                {
                    manifest
                        .verb_potentials
                        .upsert(pair_verb_name, spinor, pair_pot);
                }
            }
        }
    }
}

/// Save a `.implicate` v1 container with both the `ImplicateContainerManifest` and binary-encoded
/// `EnfoldedClipAssets` (`MultiPlaneAtlas` + `ArticulatedHolon` CFT wavepackets).
pub fn save_implicate_container_with_assets(
    path: &Path,
    manifest: &mut ImplicateContainerManifest,
    clips: &[EnfoldedClipAssets],
) -> io::Result<()> {
    for clip in clips {
        assimilate_clip_into_manifest(manifest, clip);
        if !manifest.corpus_sources.contains(&clip.source_video) {
            manifest.corpus_sources.push(clip.source_video.clone());
        }
    }
    manifest.enfolded_clips = clips.iter().map(EnfoldedClipAssets::summary).collect();
    let payload = encode_enfolded_payload(clips)?;
    let flags = IMPLICATE_FLAG_LEXICON | IMPLICATE_FLAG_ENFOLDED_ASSETS;
    save_implicate_container(path, manifest, flags, &payload)
}

/// Load an existing `.implicate` v1 container (or create a canonical one if `path` does not exist),
/// upsert `clip` by `clip_id`, assimilate its trajectory potential and concepts, and save back to `path`.
pub fn upsert_enfolded_clip_to_container(path: &Path, clip: EnfoldedClipAssets) -> io::Result<()> {
    let (mut manifest, mut clips) = if path.exists() {
        let (m, _, c) = load_implicate_container_with_assets(path)?;
        (m, c)
    } else {
        (
            ImplicateContainerManifest::canonical_corpus_manifest(),
            Vec::new(),
        )
    };
    if let Some(existing) = clips.iter_mut().find(|c| c.clip_id == clip.clip_id) {
        *existing = clip;
    } else {
        clips.push(clip);
    }
    save_implicate_container_with_assets(path, &mut manifest, &clips)
}

/// Load a `.implicate` v1 container and decode its `EnfoldedClipAssets` from the binary payload if present.
pub fn load_implicate_container_with_assets(
    path: &Path,
) -> io::Result<(ImplicateContainerManifest, u32, Vec<EnfoldedClipAssets>)> {
    let (manifest, flags, payload) = load_implicate_container(path)?;
    let clips = if (flags & IMPLICATE_FLAG_ENFOLDED_ASSETS) != 0 && !payload.is_empty() {
        decode_enfolded_payload(&payload)?
    } else {
        Vec::new()
    };
    Ok((manifest, flags, clips))
}

/// Encode `EnfoldedClipAssets` into a binary tensor payload for `.implicate` v1.
pub fn encode_enfolded_payload(clips: &[EnfoldedClipAssets]) -> io::Result<Vec<u8>> {
    let mut w = PayloadWriter::new();
    w.write_raw(ENFOLDED_PAYLOAD_MAGIC);
    w.write_u32(ENFOLDED_PAYLOAD_VERSION);
    w.write_u32(clips.len() as u32);

    for clip in clips {
        w.write_str(&clip.clip_id);
        w.write_str(&clip.source_video);
        w.write_u32(clip.num_frames as u32);

        let shots_json = serde_json::to_vec(&clip.shots)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;
        w.write_bytes(&shots_json);

        // MultiPlaneAtlas
        w.write_u32(clip.atlas.width as u32);
        w.write_u32(clip.atlas.height as u32);
        w.write_u32(clip.atlas.active_y_range.0 as u32);
        w.write_u32(clip.atlas.active_y_range.1 as u32);

        w.write_u32(clip.atlas.layers.len() as u32);
        for layer in &clip.atlas.layers {
            w.write_parallax_layer(layer);
        }

        match &clip.atlas.camera_locked_hud {
            Some(hud) => {
                w.write_u8(1);
                w.write_parallax_layer(hud);
            }
            None => w.write_u8(0),
        }

        w.write_u32(clip.atlas.actor_boxes_per_frame.len() as u32);
        for boxes in &clip.atlas.actor_boxes_per_frame {
            w.write_u32(boxes.len() as u32);
            for r in boxes {
                w.write_i32(r.x0);
                w.write_i32(r.y0);
                w.write_i32(r.x1);
                w.write_i32(r.y1);
            }
        }

        w.write_u32(clip.atlas.actor_masks_per_frame.len() as u32);
        for mask in &clip.atlas.actor_masks_per_frame {
            w.write_bitmask(mask);
        }

        let lum_json = serde_json::to_vec(&clip.atlas.luminescence_per_frame)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;
        w.write_bytes(&lum_json);

        // ArticulatedHolons
        w.write_u32(clip.holons.len() as u32);
        for holon in &clip.holons {
            w.write_articulated_holon(holon)?;
        }
    }

    Ok(w.buf)
}

/// Decode `EnfoldedClipAssets` from a `.implicate` v1 binary tensor payload.
pub fn decode_enfolded_payload(bytes: &[u8]) -> io::Result<Vec<EnfoldedClipAssets>> {
    let mut r = PayloadReader::new(bytes);
    let magic = r.read_raw(4)?;
    if magic != ENFOLDED_PAYLOAD_MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "Invalid enfolded payload magic (expected 'ENFD')",
        ));
    }
    let version = r.read_u32()?;
    if version != ENFOLDED_PAYLOAD_VERSION {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("Unsupported enfolded payload version {version}"),
        ));
    }
    let num_clips = r.read_u32()? as usize;
    let mut clips = Vec::with_capacity(num_clips);

    for _ in 0..num_clips {
        let clip_id = r.read_str()?;
        let source_video = r.read_str()?;
        let num_frames = r.read_u32()? as usize;

        let shots_bytes = r.read_bytes()?;
        let shots: Vec<ShotSegment> = serde_json::from_slice(shots_bytes)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;

        let width = r.read_u32()? as usize;
        let height = r.read_u32()? as usize;
        let y0 = r.read_u32()? as usize;
        let y1 = r.read_u32()? as usize;

        let num_layers = r.read_u32()? as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(r.read_parallax_layer()?);
        }

        let has_hud = r.read_u8()? != 0;
        let camera_locked_hud = if has_hud {
            Some(r.read_parallax_layer()?)
        } else {
            None
        };

        let n_box_frames = r.read_u32()? as usize;
        let mut actor_boxes_per_frame = Vec::with_capacity(n_box_frames);
        for _ in 0..n_box_frames {
            let nb = r.read_u32()? as usize;
            let mut boxes = Vec::with_capacity(nb);
            for _ in 0..nb {
                boxes.push(Rect::new(
                    r.read_i32()?,
                    r.read_i32()?,
                    r.read_i32()?,
                    r.read_i32()?,
                ));
            }
            actor_boxes_per_frame.push(boxes);
        }

        let n_mask_frames = r.read_u32()? as usize;
        let mut actor_masks_per_frame = Vec::with_capacity(n_mask_frames);
        for _ in 0..n_mask_frames {
            actor_masks_per_frame.push(r.read_bitmask()?);
        }

        let lum_bytes = r.read_bytes()?;
        let luminescence_per_frame: Vec<LuminescenceFrame> = serde_json::from_slice(lum_bytes)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;

        let atlas = MultiPlaneAtlas {
            width,
            height,
            active_y_range: (y0, y1),
            layers,
            camera_locked_hud,
            actor_boxes_per_frame,
            actor_masks_per_frame,
            luminescence_per_frame,
        };

        let num_holons = r.read_u32()? as usize;
        let mut holons = Vec::with_capacity(num_holons);
        for _ in 0..num_holons {
            holons.push(r.read_articulated_holon()?);
        }

        clips.push(EnfoldedClipAssets {
            clip_id,
            source_video,
            num_frames,
            shots,
            atlas,
            holons,
        });
    }

    Ok(clips)
}

struct PayloadWriter {
    buf: Vec<u8>,
}

impl PayloadWriter {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn write_bytes(&mut self, b: &[u8]) {
        self.write_u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    fn write_bitmask(&mut self, mask: &[bool]) {
        self.write_u32(mask.len() as u32);
        let packed_len = mask.len().div_ceil(8);
        let mut packed = vec![0u8; packed_len];
        for (i, &bit) in mask.iter().enumerate() {
            if bit {
                packed[i / 8] |= 1u8 << (i % 8);
            }
        }
        self.buf.extend_from_slice(&packed);
    }

    fn write_parallax_layer(&mut self, layer: &ParallaxLayer) {
        self.write_u32(layer.layer_index as u32);
        self.write_f32(layer.inverse_depth);
        self.write_i32(layer.atlas.origin[0]);
        self.write_i32(layer.atlas.origin[1]);
        self.write_u32(layer.atlas.image.width as u32);
        self.write_u32(layer.atlas.image.height as u32);
        self.write_bytes(&layer.atlas.image.rgb);
        self.write_bytes(&layer.alpha);
        self.write_u32(layer.trajectory.len() as u32);
        for &[tx, ty] in &layer.trajectory {
            self.write_i32(tx);
            self.write_i32(ty);
        }
        self.write_u32(layer.y_band.0 as u32);
        self.write_u32(layer.y_band.1 as u32);
    }

    fn write_articulated_holon(&mut self, holon: &ArticulatedHolon) -> io::Result<()> {
        self.write_u32(holon.num_parts as u32);
        self.write_u32(holon.num_view_cells as u32);
        self.write_u32(holon.parent_indices.len() as u32);
        for &pi in &holon.parent_indices {
            self.write_u32(pi as u32);
        }
        self.write_u32(holon.part_colors.len() as u32);
        for &c in &holon.part_colors {
            self.write_raw(&c);
        }

        self.write_u32(holon.kernels.len() as u32);
        for vc_kernels in &holon.kernels {
            self.write_u32(vc_kernels.len() as u32);
            for k in vc_kernels {
                self.write_u32(k.res_w as u32);
                self.write_u32(k.res_h as u32);
                self.write_bytes(&k.dc_rgba);
                self.write_bytes(&k.harmonic_rgba);
                self.write_u32(k.cft_rgba_coeffs.len() as u32);
                for coeff in &k.cft_rgba_coeffs {
                    for &v in coeff {
                        self.write_f32(v);
                    }
                }
                self.write_u32(k.gaussian_envelope.len() as u32);
                for &g in &k.gaussian_envelope {
                    self.write_f32(g);
                }
            }
        }

        let poses_json = serde_json::to_vec(&holon.poses_per_obs)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;
        self.write_bytes(&poses_json);
        Ok(())
    }
}

struct PayloadReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PayloadReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_raw(&mut self, len: usize) -> io::Result<&'a [u8]> {
        let end = self.pos.saturating_add(len);
        if end > self.bytes.len() {
            return Err(io::Error::new(
                ErrorKind::UnexpectedEof,
                "Truncated enfolded binary payload",
            ));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        Ok(self.read_raw(1)?[0])
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read_raw(4)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.read_raw(4)?.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.read_raw(4)?.try_into().unwrap()))
    }

    fn read_bytes(&mut self) -> io::Result<&'a [u8]> {
        let len = self.read_u32()? as usize;
        self.read_raw(len)
    }

    fn read_str(&mut self) -> io::Result<String> {
        let b = self.read_bytes()?;
        std::str::from_utf8(b)
            .map(str::to_string)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))
    }

    fn read_bitmask(&mut self) -> io::Result<Vec<bool>> {
        let len = self.read_u32()? as usize;
        let packed_len = len.div_ceil(8);
        let packed = self.read_raw(packed_len)?;
        let mut out = vec![false; len];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = (packed[i / 8] & (1u8 << (i % 8))) != 0;
        }
        Ok(out)
    }

    fn read_parallax_layer(&mut self) -> io::Result<ParallaxLayer> {
        let layer_index = self.read_u32()? as usize;
        let inverse_depth = self.read_f32()?;
        let ox = self.read_i32()?;
        let oy = self.read_i32()?;
        let w = self.read_u32()? as usize;
        let h = self.read_u32()? as usize;
        let rgb = self.read_bytes()?.to_vec();
        let alpha = self.read_bytes()?.to_vec();
        let traj_len = self.read_u32()? as usize;
        let mut trajectory = Vec::with_capacity(traj_len);
        for _ in 0..traj_len {
            trajectory.push([self.read_i32()?, self.read_i32()?]);
        }
        let yb0 = self.read_u32()? as usize;
        let yb1 = self.read_u32()? as usize;

        Ok(ParallaxLayer {
            layer_index,
            inverse_depth,
            atlas: WorldAtlas {
                image: RgbImage8::from_raw(w, h, rgb),
                origin: [ox, oy],
            },
            alpha,
            trajectory,
            y_band: (yb0, yb1),
        })
    }

    fn read_articulated_holon(&mut self) -> io::Result<ArticulatedHolon> {
        let num_parts = self.read_u32()? as usize;
        let num_view_cells = self.read_u32()? as usize;
        let n_par = self.read_u32()? as usize;
        let mut parent_indices = Vec::with_capacity(n_par);
        for _ in 0..n_par {
            parent_indices.push(self.read_u32()? as usize);
        }
        let n_col = self.read_u32()? as usize;
        let mut part_colors = Vec::with_capacity(n_col);
        for _ in 0..n_col {
            let c = self.read_raw(3)?;
            part_colors.push([c[0], c[1], c[2]]);
        }

        let n_vc = self.read_u32()? as usize;
        let mut kernels = Vec::with_capacity(n_vc);
        for _ in 0..n_vc {
            let n_pk = self.read_u32()? as usize;
            let mut vc_kernels = Vec::with_capacity(n_pk);
            for _ in 0..n_pk {
                let res_w = self.read_u32()? as usize;
                let res_h = self.read_u32()? as usize;
                let dc_rgba = self.read_bytes()?.to_vec();
                let harmonic_rgba = self.read_bytes()?.to_vec();
                let n_cft = self.read_u32()? as usize;
                let mut cft_rgba_coeffs = Vec::with_capacity(n_cft);
                for _ in 0..n_cft {
                    cft_rgba_coeffs.push([
                        self.read_f32()?,
                        self.read_f32()?,
                        self.read_f32()?,
                        self.read_f32()?,
                        self.read_f32()?,
                    ]);
                }
                let n_env = self.read_u32()? as usize;
                let mut gaussian_envelope = Vec::with_capacity(n_env);
                for _ in 0..n_env {
                    gaussian_envelope.push(self.read_f32()?);
                }
                vc_kernels.push(WavepacketPartKernel {
                    res_w,
                    res_h,
                    dc_rgba,
                    harmonic_rgba,
                    cft_rgba_coeffs,
                    gaussian_envelope,
                });
            }
            kernels.push(vc_kernels);
        }

        let poses_bytes = self.read_bytes()?;
        let poses_per_obs: Vec<Vec<SubHolonPose>> = serde_json::from_slice(poses_bytes)
            .map_err(|e| io::Error::new(ErrorKind::InvalidData, e.to_string()))?;

        Ok(ArticulatedHolon {
            num_parts,
            num_view_cells,
            parent_indices,
            part_colors,
            kernels,
            poses_per_obs,
        })
    }
}
