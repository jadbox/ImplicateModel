//! Prompt & `DzogLang` Scene Compiler for Novel Unfoldment (Track R4.3, Whitepaper §6.1).
//!
//! Compiles text / `DzogLang` scene specifications combining entities, emotions, Jungian archetypes,
//! camera-locked HUD holons, PGA boundary blades, and relational verbs (`"patrol"`, `"jump_over"`,
//! `"pursue"`, `"flee_from"`, `"orbit"`) across `pacman`, `smb3`, `raptor`, and *The Secret of NIMH*.

use serde::{Deserialize, Serialize};

/// Specification of a single Holon actor in an unfolded scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptHolonSpec {
    pub name: String,
    /// Semantic role / archetype / emotion clause (e.g. `"Mrs_Brisby hero courage mouse"`).
    pub semantic_clause: String,
    pub initial_pos: [f32; 2],
    pub initial_vel: [f32; 2],
    pub radius: f32,
    pub rho: f32,
    pub bulk_modulus_k: f32,
    pub base_color: [u8; 3],
    pub luminescence_emission: [f32; 3],
    /// Number of articulated sub-holon parts ($P = 1$ rigid, $P = 4$ quadruped/character, $P = 6$ winged).
    pub num_parts: usize,
    /// True if this holon is a camera-locked HUD interface holon ($M_{\text{hud}}(t) = M_{\text{cam}}(t)$).
    pub is_camera_locked_hud: bool,
    /// Griffith surface cohesion $C \in [0, 1]$ ($C \ge 0.5$ solid matter, $C < 0.5$ field soliton).
    #[serde(default = "default_cohesion")]
    pub cohesion_c: f32,
    /// Static level fixture ($\mathbf{v} = \mathbf{0}$, $Z \to \infty$): already present in the enfolded
    /// background atlas, so it acts as a boundary/anchor rather than a splatted sprite.
    #[serde(default)]
    pub is_static_fixture: bool,
    /// True when `base_color` was stated by the author; false when it is a placeholder. Only explicit
    /// appearance participates in dispatch to enfolded tracks.
    #[serde(default)]
    pub appearance_explicit: bool,
}

fn default_cohesion() -> f32 {
    1.0
}

/// Specification of a relational verb binding `(subject, verb, target)` over normalized or frame time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptVerbBinding {
    pub subject: String,
    pub verb: String,
    pub target: String,
}

/// Specification of an oriented PGA boundary blade $B_k \in \mathbb{G}_{3,0,1}$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptBoundaryBlade {
    pub point: [f64; 2],
    pub normal: [f64; 2],
}

/// Complete compiled scene specification ready for single-pass causal unfoldment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnfoldSceneSpec {
    pub scene_id: String,
    pub prompt_text: String,
    pub width: u32,
    pub height: u32,
    pub num_frames: usize,
    pub camera_scroll_vel: [f32; 2],
    pub background_top_rgb: [u8; 3],
    pub background_bottom_rgb: [u8; 3],
    pub enable_cliffordnet_refiner: bool,
    pub holons: Vec<PromptHolonSpec>,
    pub verb_bindings: Vec<PromptVerbBinding>,
    pub boundaries: Vec<PromptBoundaryBlade>,
    /// True when `width x height` was stated by the author. Otherwise the scene adopts the native
    /// resolution of the enfolded clip it unfolds from (see `adapt_to_resolution`).
    #[serde(default)]
    pub resolution_explicit: bool,
}

impl UnfoldSceneSpec {
    /// Compile the 4 novel unseen re-prompting benchmark scenes across `pacman`, `smb3`, `raptor`,
    /// and *The Secret of NIMH* for Milestone `R4.3`.
    pub fn canonical_r4_3_novel_scenes() -> Vec<Self> {
        vec![
            // Scene 1: Pac-Man novel maze configuration ("flee_from" + "pursue" + "orbit" + HUD holon)
            Self {
                scene_id: "pacman_novel_talisman_orbit".to_string(),
                prompt_text: "@scene pacman_novel_talisman_orbit: Pacman (hero, fear) executes flee_from Blinky (shadow, anger) while Ghost_Inky executes orbit around Power_Pellet and HUD_Score tracks state.".to_string(),
                width: 240,
                height: 308,
                num_frames: 60,
                camera_scroll_vel: [0.0, 0.0],
                background_top_rgb: [6, 8, 28],
                background_bottom_rgb: [12, 16, 48],
                enable_cliffordnet_refiner: false,
                holons: vec![
                    PromptHolonSpec {
                        name: "Power_Pellet".to_string(),
                        semantic_clause: "Power_Pellet self talisman glowing orb".to_string(),
                        initial_pos: [120.0, 154.0],
                        initial_vel: [0.0, 0.0],
                        radius: 4.0,
                        rho: 1.5,
                        bulk_modulus_k: 3.0,
                        base_color: [255, 220, 180],
                        luminescence_emission: [2.5, 2.0, 1.0],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 0.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Pacman".to_string(),
                        semantic_clause: "Pacman hero fear yellow runner".to_string(),
                        initial_pos: [88.0, 154.0],
                        initial_vel: [-12.0, 4.0],
                        radius: 6.0,
                        rho: 1.0,
                        bulk_modulus_k: 1.2,
                        base_color: [255, 235, 0],
                        luminescence_emission: [0.6, 0.5, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Blinky_Ghost".to_string(),
                        semantic_clause: "Blinky shadow anger red pursuer".to_string(),
                        initial_pos: [62.0, 154.0],
                        initial_vel: [14.0, 0.0],
                        radius: 6.0,
                        rho: 0.9,
                        bulk_modulus_k: 1.1,
                        base_color: [255, 40, 40],
                        luminescence_emission: [1.2, 0.1, 0.1],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Inky_Ghost".to_string(),
                        semantic_clause: "Inky trickster curiosity cyan orbiter".to_string(),
                        initial_pos: [92.0, 154.0],
                        initial_vel: [0.0, -15.0],
                        radius: 6.0,
                        rho: 0.9,
                        bulk_modulus_k: 1.1,
                        base_color: [0, 230, 255],
                        luminescence_emission: [0.1, 0.9, 1.2],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "HUD_Score_Bar".to_string(),
                        semantic_clause: "entity_hud_interface score lives status bar".to_string(),
                        initial_pos: [120.0, 288.0],
                        initial_vel: [0.0, 0.0],
                        radius: 5.0,
                        rho: 1.0,
                        bulk_modulus_k: 1.0,
                        base_color: [240, 240, 255],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: true,
                        cohesion_c: 0.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                ],
                verb_bindings: vec![
                    PromptVerbBinding {
                        subject: "Pacman".to_string(),
                        verb: "flee_from".to_string(),
                        target: "Power_Pellet".to_string(),
                    },
                    PromptVerbBinding {
                        subject: "Blinky_Ghost".to_string(),
                        verb: "pursue".to_string(),
                        target: "Power_Pellet".to_string(),
                    },
                    PromptVerbBinding {
                        subject: "Inky_Ghost".to_string(),
                        verb: "orbit".to_string(),
                        target: "Power_Pellet".to_string(),
                    },
                ],
                boundaries: vec![
                    PromptBoundaryBlade {
                        point: [12.0, 12.0],
                        normal: [0.0, 1.0],
                    },
                    PromptBoundaryBlade {
                        point: [12.0, 296.0],
                        normal: [0.0, -1.0],
                    },
                    PromptBoundaryBlade {
                        point: [12.0, 12.0],
                        normal: [1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [228.0, 12.0],
                        normal: [-1.0, 0.0],
                    },
                ],
                resolution_explicit: true,
            },
            // Scene 2: Super Mario Bros. 3 novel vault & sentinel patrol with scrolling camera & HUD holon
            Self {
                scene_id: "smb3_novel_vault_and_patrol".to_string(),
                prompt_text: "@scene smb3_novel_vault_and_patrol: Mario (hero, joy) executes jump_over Koopa_Block while Goomba_Sentinel (shadow) executes patrol and camera-locked HUD holon updates.".to_string(),
                width: 256,
                height: 240,
                num_frames: 60,
                camera_scroll_vel: [18.0, 0.0],
                background_top_rgb: [110, 180, 255],
                background_bottom_rgb: [180, 225, 255],
                enable_cliffordnet_refiner: false,
                holons: vec![
                    PromptHolonSpec {
                        name: "Koopa_Block".to_string(),
                        semantic_clause: "Koopa_Block threshold_guardian golden block".to_string(),
                        initial_pos: [138.0, 144.0],
                        initial_vel: [0.0, 0.0],
                        radius: 6.0,
                        rho: 2.0,
                        bulk_modulus_k: 4.5,
                        base_color: [245, 180, 40],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: true,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Mario".to_string(),
                        semantic_clause: "Mario hero joy jumper".to_string(),
                        initial_pos: [86.0, 178.0],
                        initial_vel: [22.0, -24.0],
                        radius: 6.5,
                        rho: 1.1,
                        bulk_modulus_k: 1.6,
                        base_color: [240, 45, 45],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 4,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Goomba_Sentinel".to_string(),
                        semantic_clause: "Goomba shadow anger sentinel walker".to_string(),
                        initial_pos: [98.0, 178.0],
                        initial_vel: [12.0, 0.0],
                        radius: 5.5,
                        rho: 1.0,
                        bulk_modulus_k: 1.2,
                        base_color: [165, 85, 35],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "SMB3_HUD_Panel".to_string(),
                        semantic_clause: "entity_hud_interface bottom status panel cards coins".to_string(),
                        initial_pos: [128.0, 216.0],
                        initial_vel: [18.0, 0.0],
                        radius: 6.0,
                        rho: 1.0,
                        bulk_modulus_k: 1.0,
                        base_color: [250, 230, 190],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: true,
                        cohesion_c: 0.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                ],
                verb_bindings: vec![
                    PromptVerbBinding {
                        subject: "Mario".to_string(),
                        verb: "jump_over".to_string(),
                        target: "Koopa_Block".to_string(),
                    },
                    PromptVerbBinding {
                        subject: "Goomba_Sentinel".to_string(),
                        verb: "patrol".to_string(),
                        target: "Koopa_Block".to_string(),
                    },
                ],
                boundaries: vec![
                    PromptBoundaryBlade {
                        point: [0.0, 192.0],
                        normal: [0.0, -1.0],
                    },
                    PromptBoundaryBlade {
                        point: [8.0, 0.0],
                        normal: [1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [248.0, 0.0],
                        normal: [-1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [0.0, 8.0],
                        normal: [0.0, 1.0],
                    },
                ],
                resolution_explicit: true,
            },
            // Scene 3: Raptor Call of the Shadows novel aerial intercept & glowing plasma orbit
            Self {
                scene_id: "raptor_novel_plasma_intercept".to_string(),
                prompt_text: "@scene raptor_novel_plasma_intercept: Raptor_Jet (hero, determination) executes pursue on Command_Carrier while Plasma_Drone executes orbit with glowing blue luminescence.".to_string(),
                width: 320,
                height: 240,
                num_frames: 60,
                camera_scroll_vel: [0.0, 24.0],
                background_top_rgb: [18, 28, 42],
                background_bottom_rgb: [36, 52, 68],
                enable_cliffordnet_refiner: false,
                holons: vec![
                    PromptHolonSpec {
                        name: "Command_Carrier".to_string(),
                        semantic_clause: "Command_Carrier shadow boss warship".to_string(),
                        initial_pos: [160.0, 148.0],
                        initial_vel: [0.0, 8.0],
                        radius: 8.0,
                        rho: 2.2,
                        bulk_modulus_k: 4.8,
                        base_color: [140, 155, 170],
                        luminescence_emission: [0.8, 0.4, 0.2],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Raptor_Jet".to_string(),
                        semantic_clause: "Raptor_Jet hero courage fighter aircraft".to_string(),
                        initial_pos: [118.0, 148.0],
                        initial_vel: [18.0, 12.0],
                        radius: 6.5,
                        rho: 1.4,
                        bulk_modulus_k: 2.8,
                        base_color: [210, 220, 235],
                        luminescence_emission: [0.6, 1.4, 2.2],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Plasma_Drone".to_string(),
                        semantic_clause: "Plasma_Drone trickster glowing energy orbiter".to_string(),
                        initial_pos: [132.0, 148.0],
                        initial_vel: [0.0, -18.0],
                        radius: 4.5,
                        rho: 0.6,
                        bulk_modulus_k: 2.0,
                        base_color: [90, 220, 255],
                        luminescence_emission: [1.2, 3.4, 4.8],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 0.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                ],
                verb_bindings: vec![
                    PromptVerbBinding {
                        subject: "Raptor_Jet".to_string(),
                        verb: "pursue".to_string(),
                        target: "Command_Carrier".to_string(),
                    },
                    PromptVerbBinding {
                        subject: "Plasma_Drone".to_string(),
                        verb: "orbit".to_string(),
                        target: "Command_Carrier".to_string(),
                    },
                ],
                boundaries: vec![
                    PromptBoundaryBlade {
                        point: [12.0, 12.0],
                        normal: [1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [308.0, 12.0],
                        normal: [-1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [12.0, 12.0],
                        normal: [0.0, 1.0],
                    },
                    PromptBoundaryBlade {
                        point: [12.0, 228.0],
                        normal: [0.0, -1.0],
                    },
                ],
                resolution_explicit: true,
            },
            // Scene 4: The Secret of NIMH novel articulated multi-plane scene (Mrs. Brisby P=4 + Jeremy P=6 + Red Amulet + CliffordNet @rolpa)
            Self {
                scene_id: "nimh_novel_amulet_sanctuary".to_string(),
                prompt_text: "@scene nimh_novel_amulet_sanctuary: Articulated Mrs_Brisby (P=4, hero, courage) executes jump_over toward glowing Red_Amulet (self, awe) while Jeremy_Crow (P=6, trickster, joy) executes orbit and CliffordNet-Lite refines @rolpa.".to_string(),
                width: 320,
                height: 180,
                num_frames: 60,
                camera_scroll_vel: [6.0, 2.0],
                background_top_rgb: [16, 22, 46],
                background_bottom_rgb: [38, 28, 52],
                enable_cliffordnet_refiner: true,
                holons: vec![
                    PromptHolonSpec {
                        name: "Red_Amulet".to_string(),
                        semantic_clause: "Red_Amulet self awe glowing talisman stone".to_string(),
                        initial_pos: [160.0, 78.0],
                        initial_vel: [0.0, 0.0],
                        radius: 5.5,
                        rho: 2.4,
                        bulk_modulus_k: 5.4,
                        base_color: [255, 65, 45],
                        luminescence_emission: [4.5, 1.2, 0.6],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                        cohesion_c: 0.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Mrs_Brisby".to_string(),
                        semantic_clause: "Mrs_Brisby hero courage mother mouse in red cape".to_string(),
                        initial_pos: [116.0, 78.0],
                        initial_vel: [16.0, 22.0],
                        radius: 6.5,
                        rho: 1.0,
                        bulk_modulus_k: 1.4,
                        base_color: [220, 50, 60],
                        luminescence_emission: [0.6, 0.2, 0.2],
                        num_parts: 4,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                    PromptHolonSpec {
                        name: "Jeremy_Crow".to_string(),
                        semantic_clause: "Jeremy_Crow trickster joy friendly crow bird".to_string(),
                        initial_pos: [126.0, 78.0],
                        initial_vel: [0.0, -18.0],
                        radius: 7.5,
                        rho: 0.85,
                        bulk_modulus_k: 1.1,
                        base_color: [55, 65, 95],
                        luminescence_emission: [0.2, 0.3, 0.5],
                        num_parts: 6,
                        is_camera_locked_hud: false,
                        cohesion_c: 1.0,
                        is_static_fixture: false,
                        appearance_explicit: true,
                    },
                ],
                verb_bindings: vec![
                    PromptVerbBinding {
                        subject: "Mrs_Brisby".to_string(),
                        verb: "jump_over".to_string(),
                        target: "Red_Amulet".to_string(),
                    },
                    PromptVerbBinding {
                        subject: "Jeremy_Crow".to_string(),
                        verb: "orbit".to_string(),
                        target: "Red_Amulet".to_string(),
                    },
                ],
                boundaries: vec![
                    PromptBoundaryBlade {
                        point: [0.0, 24.0],
                        normal: [0.0, 1.0],
                    },
                    PromptBoundaryBlade {
                        point: [10.0, 24.0],
                        normal: [1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [310.0, 24.0],
                        normal: [-1.0, 0.0],
                    },
                ],
                resolution_explicit: true,
            },
        ]
    }

    /// Re-project an implicit-resolution scene onto `(width, height)`: pixel-space positions,
    /// velocities, radii, and boundary blades scale with the viewport so layout is preserved.
    pub fn adapt_to_resolution(&mut self, width: u32, height: u32) {
        if self.resolution_explicit || width == 0 || height == 0 {
            return;
        }
        let sx = width as f32 / self.width.max(1) as f32;
        let sy = height as f32 / self.height.max(1) as f32;
        let s_iso = sx.min(sy);
        for h in &mut self.holons {
            h.initial_pos = [h.initial_pos[0] * sx, h.initial_pos[1] * sy];
            h.initial_vel = [h.initial_vel[0] * sx, h.initial_vel[1] * sy];
            h.radius *= s_iso;
        }
        self.camera_scroll_vel = [self.camera_scroll_vel[0] * sx, self.camera_scroll_vel[1] * sy];
        for b in &mut self.boundaries {
            b.point = [b.point[0] * sx as f64, b.point[1] * sy as f64];
        }
        self.width = width;
        self.height = height;
        self.resolution_explicit = true;
    }

    /// Parse a structured `DzogLang` script:
    ///
    /// ```text
    /// @scene <id> [WxH] [frames=N] [cam_vel=(vx, vy)] [refiner=true|false]
    /// @holon <Name> [pos=(x, y)] [vel=(vx, vy)] [parts=P] [color=(r, g, b)] [radius=r]
    ///               [cohesion=C] [rho=..] [K=..] [fixture] [<free attribute words>...]
    /// @verb <Subject> <verb> <Target>
    /// @hud <Name>
    /// ```
    ///
    /// Positions/velocities/radii may be pixels or normalized $[0, 1]^2$ coordinates.
    pub fn parse_dzog_script(script: &str) -> Result<Self, String> {
        let mut scene_id = "dzog_scene".to_string();
        let (mut width, mut height, mut resolution_explicit) = (320u32, 240u32, false);
        let mut num_frames = 60usize;
        let mut camera_scroll_vel = [0.0f32, 0.0];
        let mut enable_cliffordnet_refiner = false;
        let mut holons: Vec<PromptHolonSpec> = Vec::new();
        let mut verb_bindings: Vec<PromptVerbBinding> = Vec::new();
        let mut hud_names: Vec<String> = Vec::new();

        for (ln, raw) in script.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let (head, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            let (kv, words) = parse_dzog_attrs(rest);
            let get = |k: &str| kv.iter().find(|(key, _)| key.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str());
            match head {
                "@scene" => {
                    if let Some(id) = words.first() {
                        scene_id = sanitize_ident(id);
                    }
                    for w in &words {
                        if let (w_, h_, true) = parse_prompt_resolution(&w.to_ascii_lowercase()) {
                            (width, height, resolution_explicit) = (w_, h_, true);
                        }
                    }
                    if let Some(n) = get("frames").and_then(|v| v.parse().ok()) {
                        num_frames = n;
                    }
                    if let Some(v) = get("cam_vel").and_then(parse_dzog_vec::<2>) {
                        camera_scroll_vel = [v[0] as f32, v[1] as f32];
                    }
                    if let Some(v) = get("refiner") {
                        enable_cliffordnet_refiner = v.eq_ignore_ascii_case("true");
                    }
                }
                "@holon" => {
                    let Some(name) = words.first().map(|w| sanitize_ident(w)) else {
                        return Err(format!("line {}: @holon requires a name", ln + 1));
                    };
                    let attrs = words[1..].join(" ");
                    let attr_lower = attrs.to_ascii_lowercase();
                    let name_lower = name.to_ascii_lowercase();
                    let explicit_color = get("color").and_then(parse_dzog_vec::<3>).map(|c| c.map(|v| v.clamp(0.0, 255.0) as u8));
                    let (base_color, appearance_explicit) = match explicit_color.or_else(|| infer_actor_color(&name_lower, &attr_lower)) {
                        Some(c) => (c, true),
                        None => (placeholder_color(holons.len()), false),
                    };
                    let num_parts = get("parts")
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or_else(|| parse_part_count(&attr_lower, &name_lower))
                        .clamp(1, 8);
                    let cohesion_c = get("cohesion").and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    let lum = get("glow").and_then(parse_dzog_vec::<3>).map(|v| v.map(|x| x as f32));
                    holons.push(PromptHolonSpec {
                        name: name.clone(),
                        semantic_clause: format!("{name} {attrs}").trim().to_string(),
                        initial_pos: get("pos").and_then(parse_dzog_vec::<2>).map(|v| [v[0] as f32, v[1] as f32]).unwrap_or([0.5, 0.5]),
                        initial_vel: get("vel").and_then(parse_dzog_vec::<2>).map(|v| [v[0] as f32, v[1] as f32]).unwrap_or([0.0, 0.0]),
                        radius: get("radius").and_then(|v| v.parse().ok()).unwrap_or(6.0),
                        rho: get("rho").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                        bulk_modulus_k: get("k").and_then(|v| v.parse().ok()).unwrap_or(1.5),
                        base_color,
                        luminescence_emission: lum.unwrap_or([0.0; 3]),
                        num_parts,
                        is_camera_locked_hud: false,
                        cohesion_c,
                        is_static_fixture: words.iter().any(|w| w.eq_ignore_ascii_case("fixture")),
                        appearance_explicit,
                    });
                }
                "@verb" => {
                    if words.len() < 3 {
                        return Err(format!("line {}: @verb expects '<Subject> <verb> <Target>'", ln + 1));
                    }
                    verb_bindings.push(PromptVerbBinding {
                        subject: sanitize_ident(&words[0]),
                        verb: words[1].clone(),
                        target: sanitize_ident(&words[2]),
                    });
                }
                "@hud" => {
                    hud_names.push(words.first().map(|w| sanitize_ident(w)).unwrap_or_else(|| "HUD".to_string()));
                }
                other => return Err(format!("line {}: unknown directive '{other}'", ln + 1)),
            }
        }

        for vb in &verb_bindings {
            for n in [&vb.subject, &vb.target] {
                if !holons.iter().any(|h| &h.name == n) {
                    return Err(format!("@verb references undeclared holon '{n}'"));
                }
            }
        }
        if verb_bindings.is_empty() {
            return Err("DzogLang script declares no @verb bindings".to_string());
        }
        for name in hud_names {
            holons.push(PromptHolonSpec {
                name,
                semantic_clause: "entity_hud_interface camera_locked_hud score status bar".to_string(),
                // Pixel units: camera-locked holons are re-anchored to the camera motor every frame.
                initial_pos: [width as f32 * 0.5, height as f32 * 0.05],
                initial_vel: camera_scroll_vel,
                radius: 0.02 * width.min(height) as f32,
                rho: 1.0,
                bulk_modulus_k: 1.0,
                base_color: [245, 235, 200],
                luminescence_emission: [0.0; 3],
                num_parts: 1,
                is_camera_locked_hud: true,
                cohesion_c: 0.0,
                is_static_fixture: false,
                appearance_explicit: false,
            });
        }

        // Viewport boundary blades inset by a fraction of the shorter side.
        let inset = 0.04 * width.min(height) as f64;
        let (wf, hf) = (width as f64, height as f64);
        let boundaries = vec![
            PromptBoundaryBlade { point: [inset, inset], normal: [0.0, 1.0] },
            PromptBoundaryBlade { point: [inset, hf - inset], normal: [0.0, -1.0] },
            PromptBoundaryBlade { point: [inset, inset], normal: [1.0, 0.0] },
            PromptBoundaryBlade { point: [wf - inset, inset], normal: [-1.0, 0.0] },
        ];

        Ok(Self {
            scene_id,
            prompt_text: script.to_string(),
            width,
            height,
            num_frames,
            camera_scroll_vel,
            background_top_rgb: [16, 22, 46],
            background_bottom_rgb: [38, 28, 52],
            enable_cliffordnet_refiner,
            holons,
            verb_bindings,
            boundaries,
            resolution_explicit,
        })
    }

    /// Parse a free-form text or `DzogLang` scene specification into an `UnfoldSceneSpec`.
    ///
    /// Supports `@scene <id>:` headers, parenthetical holon attributes `(P=4, hero, courage)`,
    /// relational verb bindings (`patrol`, `jump_over`, `pursue`, `flee_from`, `orbit`),
    /// camera-locked HUD holons (`HUD` / `score` / `status`), and `@rolpa` / `CliffordNet` refinement flags.
    pub fn parse_prompt(input: &str) -> Result<Self, String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err("Empty prompt string".to_string());
        }
        if trimmed.lines().any(|l| {
            let l = l.trim_start();
            l.starts_with("@holon") || l.starts_with("@verb")
        }) {
            return Self::parse_dzog_script(trimmed);
        }

        // Extract optional `@scene <id>:` header
        let mut scene_id = "custom_prompt_scene".to_string();
        let mut body = trimmed;
        if let Some(rest) = trimmed.strip_prefix("@scene") {
            let rest = rest.trim_start();
            if let Some((head, tail)) = rest.split_once(':') {
                let id_token = head.split_whitespace().next().unwrap_or("custom_prompt_scene");
                let clean_id: String = id_token
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if !clean_id.is_empty() {
                    scene_id = clean_id;
                }
                body = tail.trim();
            }
        }

        let lower_all = trimmed.to_ascii_lowercase();
        let (width, height, resolution_explicit) = parse_prompt_resolution(&lower_all);
        let num_frames = 60usize;
        let enable_cliffordnet_refiner = lower_all.contains("cliffordnet")
            || lower_all.contains("@rolpa")
            || lower_all.contains("refiner")
            || lower_all.contains("glowing")
            || lower_all.contains("sanctuary");

        let camera_scroll_vel = if lower_all.contains("scroll") || (lower_all.contains("patrol") && lower_all.contains("vault")) {
            [16.0f32, 0.0]
        } else if lower_all.contains("intercept") || lower_all.contains("carrier") {
            [0.0f32, 24.0]
        } else if lower_all.contains("sanctuary") || lower_all.contains("drift") {
            [6.0f32, 2.0]
        } else {
            [0.0f32, 0.0]
        };

        let (background_top_rgb, background_bottom_rgb) = if lower_all.contains("sky")
            || lower_all.contains("day")
            || lower_all.contains("vault")
        {
            ([110, 180, 255], [180, 225, 255])
        } else if lower_all.contains("dark") || lower_all.contains("maze") || lower_all.contains("talisman") {
            ([6, 8, 28], [12, 16, 48])
        } else {
            ([16, 22, 46], [38, 28, 52])
        };

        // Split body into sub-clauses around ";", "while", and "and"
        let normalized_body = body
            .replace(";", " | ")
            .replace(" while ", " | ")
            .replace(" and ", " | ")
            .replace("jumps over", "jump_over")
            .replace("flees from", "flee_from")
            .replace("pursues", "pursue")
            .replace("orbits", "orbit")
            .replace("patrols", "patrol");

        let verb_list = ["jump_over", "flee_from", "pursue", "orbit", "patrol"];
        let mut holons: Vec<PromptHolonSpec> = Vec::new();
        let mut verb_bindings: Vec<PromptVerbBinding> = Vec::new();
        let mut wants_hud = lower_all.contains("hud") || lower_all.contains("score");

        let center_x = (width as f32) * 0.5;
        let center_y = (height as f32) * 0.46;

        let ensure_holon = |holons: &mut Vec<PromptHolonSpec>,
                            raw_desc: &str,
                            is_target: bool,
                            bind_idx: usize,
                            verb: &str|
         -> String {
            let (name, attrs) = parse_actor_and_attrs(raw_desc);
            if let Some(existing) = holons.iter().find(|h| h.name.eq_ignore_ascii_case(&name)) {
                return existing.name.clone();
            }
            let attr_lower = attrs.to_ascii_lowercase();
            let name_lower = name.to_ascii_lowercase();
            let num_parts = parse_part_count(&attr_lower, &name_lower);
            let is_glow = attr_lower.contains("glow")
                || attr_lower.contains("awe")
                || attr_lower.contains("talisman")
                || attr_lower.contains("soliton")
                || attr_lower.contains("energy")
                || attr_lower.contains("orb")
                || name_lower.contains("orb");
            let is_terrain = attr_lower.contains("fixture")
                || attr_lower.contains("block")
                || attr_lower.contains("terrain")
                || name_lower.contains("block")
                || name_lower.contains("wall");
            let cohesion_c = if is_glow { 0.0 } else { 1.0 };
            let (rho, bulk_modulus_k) = if is_terrain {
                (2.0, 4.5)
            } else if is_target {
                (2.0, 4.2)
            } else {
                (1.1, 1.6)
            };

            let (pos, vel) = if is_target {
                ([center_x, center_y], [0.0f32, 0.0])
            } else {
                let offset_x = 34.0 + (bind_idx as f32) * 8.0;
                match verb {
                    "jump_over" => ([center_x - offset_x, center_y], [18.0, 24.0]),
                    "flee_from" => ([center_x - offset_x, center_y], [-14.0, 4.0]),
                    "pursue" => ([center_x - offset_x, center_y], [16.0, 6.0]),
                    "orbit" => ([center_x - offset_x, center_y], [0.0, -18.0]),
                    _ => ([center_x - offset_x, center_y], [12.0, 0.0]),
                }
            };

            let (base_color, appearance_explicit) =
                match infer_actor_color(&name_lower, &attr_lower) {
                    Some(c) => (c, true),
                    None => (placeholder_color(bind_idx + holons.len()), false),
                };
            let luminescence_emission = if is_glow {
                [3.2, 1.4, 0.8]
            } else {
                [
                    (base_color[0] as f32) / 255.0 * 0.6,
                    (base_color[1] as f32) / 255.0 * 0.6,
                    (base_color[2] as f32) / 255.0 * 0.6,
                ]
            };

            let semantic_clause = if attrs.is_empty() {
                format!("{name} hero courage actor")
            } else {
                format!("{name} {attrs}")
            };

            holons.push(PromptHolonSpec {
                name: name.clone(),
                semantic_clause,
                initial_pos: pos,
                initial_vel: vel,
                radius: if is_target { 6.0 } else { 6.5 },
                rho,
                bulk_modulus_k,
                base_color,
                luminescence_emission,
                num_parts,
                is_camera_locked_hud: false,
                cohesion_c,
                is_static_fixture: is_terrain && is_target,
                appearance_explicit,
            });
            name
        };

        let mut last_target: Option<String> = None;

        for clause in normalized_body.split('|') {
            let clause = clause.trim();
            if clause.is_empty() {
                continue;
            }
            let cl_lower = clause.to_ascii_lowercase();
            if (cl_lower.contains("hud") || cl_lower.contains("score"))
                && !cl_lower.contains("executes ")
                && !verb_list.iter().any(|v| cl_lower.contains(v))
            {
                wants_hud = true;
                continue;
            }

            let mut matched_verb: Option<&str> = None;
            let mut split_pair: Option<(&str, &str)> = None;
            if let Some((lhs, exec_rhs)) = clause.split_once("executes ") {
                let exec_rhs = exec_rhs.trim();
                if let Some((v_tok, tail)) = exec_rhs.split_once(' ') {
                    matched_verb = Some(v_tok.trim());
                    split_pair = Some((lhs.trim(), tail.trim()));
                } else if !exec_rhs.is_empty() {
                    matched_verb = Some(exec_rhs);
                    split_pair = Some((lhs.trim(), ""));
                }
            } else {
                for &v in &verb_list {
                    if let Some((lhs, rhs)) = clause.split_once(v) {
                        matched_verb = Some(v);
                        split_pair = Some((lhs.trim(), rhs.trim()));
                        break;
                    }
                }
            }

            let Some(verb) = matched_verb else {
                continue;
            };
            let (lhs, rhs) = split_pair.unwrap();
            let lhs_clean = lhs
                .trim_end_matches("executes")
                .trim_start_matches("Articulated")
                .trim();
            let rhs_clean = rhs
                .trim_start_matches("on ")
                .trim_start_matches("around ")
                .trim_start_matches("toward ")
                .trim_start_matches("from ")
                .trim();

            let bind_idx = verb_bindings.len();
            let target_name = if rhs_clean.is_empty() || rhs_clean.starts_with("with ") {
                last_target.clone().unwrap_or_else(|| {
                    ensure_holon(&mut holons, "Anchor_Talisman (self, awe)", true, 0, verb)
                })
            } else {
                let t_desc = rhs_clean.split(" with ").next().unwrap_or(rhs_clean).trim();
                let t = ensure_holon(&mut holons, t_desc, true, bind_idx, verb);
                last_target = Some(t.clone());
                t
            };

            let subject_name = ensure_holon(&mut holons, lhs_clean, false, bind_idx, verb);
            if subject_name != target_name {
                verb_bindings.push(PromptVerbBinding {
                    subject: subject_name,
                    verb: verb.to_string(),
                    target: target_name,
                });
            }
        }

        if verb_bindings.is_empty() {
            return Err(format!(
                "No recognized verb bindings (patrol, jump_over, pursue, flee_from, orbit) found in prompt: '{trimmed}'"
            ));
        }

        if wants_hud {
            holons.push(PromptHolonSpec {
                name: "HUD_Status_Bar".to_string(),
                semantic_clause: "entity_hud_interface camera_locked_hud score status bar"
                    .to_string(),
                initial_pos: [center_x, 22.0],
                initial_vel: camera_scroll_vel,
                radius: 5.5,
                rho: 1.0,
                bulk_modulus_k: 1.0,
                base_color: [245, 235, 200],
                luminescence_emission: [0.0, 0.0, 0.0],
                num_parts: 1,
                is_camera_locked_hud: true,
                cohesion_c: 0.0,
                is_static_fixture: false,
                appearance_explicit: false,
            });
        }

        let boundaries = vec![
            PromptBoundaryBlade {
                point: [10.0, 20.0],
                normal: [0.0, 1.0],
            },
            PromptBoundaryBlade {
                point: [10.0, (height as f64) - 12.0],
                normal: [0.0, -1.0],
            },
            PromptBoundaryBlade {
                point: [10.0, 20.0],
                normal: [1.0, 0.0],
            },
            PromptBoundaryBlade {
                point: [(width as f64) - 10.0, 20.0],
                normal: [-1.0, 0.0],
            },
        ];

        Ok(Self {
            scene_id,
            prompt_text: trimmed.to_string(),
            width,
            height,
            num_frames,
            camera_scroll_vel,
            background_top_rgb,
            background_bottom_rgb,
            enable_cliffordnet_refiner,
            holons,
            verb_bindings,
            boundaries,
            resolution_explicit,
        })
    }
}

/// Split a DzogLang directive tail into `key=value` pairs (values may be parenthesized tuples
/// containing spaces) and bare words.
fn parse_dzog_attrs(rest: &str) -> (Vec<(String, String)>, Vec<String>) {
    let (mut kv, mut words) = (Vec::new(), Vec::new());
    let mut chars = rest.chars().peekable();
    let mut tok = String::new();
    let mut depth = 0i32;
    let flush = |tok: &mut String, kv: &mut Vec<(String, String)>, words: &mut Vec<String>| {
        let t = tok.trim().trim_matches(',').to_string();
        if !t.is_empty() {
            match t.split_once('=') {
                Some((k, v)) => kv.push((k.trim().to_string(), v.trim().to_string())),
                None => words.push(t.trim_matches(|c| c == '(' || c == ')').to_string()),
            }
        }
        tok.clear();
    };
    while let Some(c) = chars.next() {
        match c {
            '(' => {
                depth += 1;
                tok.push(c);
            }
            ')' => {
                depth -= 1;
                tok.push(c);
            }
            c if c.is_whitespace() && depth <= 0 => flush(&mut tok, &mut kv, &mut words),
            _ => tok.push(c),
        }
    }
    flush(&mut tok, &mut kv, &mut words);
    words.retain(|w| !w.is_empty());
    (kv, words)
}

/// Parse an `N`-tuple such as `(1.0, 2.0)`.
fn parse_dzog_vec<const N: usize>(v: &str) -> Option<[f64; N]> {
    let inner = v.trim().trim_start_matches('(').trim_end_matches(')');
    let parts: Vec<f64> = inner.split(',').map(|p| p.trim().parse::<f64>()).collect::<Result<_, _>>().ok()?;
    parts.try_into().ok()
}

fn parse_actor_and_attrs(raw: &str) -> (String, String) {
    let cleaned = raw
        .trim()
        .trim_end_matches('.')
        .trim_start_matches("glowing ")
        .trim_start_matches("Articulated ")
        .trim();
    if let Some((name_part, rest)) = cleaned.split_once('(') {
        let attrs = rest
            .split(')')
            .next()
            .unwrap_or("")
            .replace(',', " ")
            .trim()
            .to_string();
        let name = sanitize_ident(name_part);
        (name, attrs)
    } else {
        let tokens: Vec<&str> = cleaned.split_whitespace().collect();
        let name = tokens
            .first()
            .map(|s| sanitize_ident(s))
            .unwrap_or_else(|| "Actor".to_string());
        let attrs = if tokens.len() > 1 {
            tokens[1..].join(" ")
        } else {
            String::new()
        };
        (name, attrs)
    }
}

fn sanitize_ident(raw: &str) -> String {
    let s: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if s.is_empty() {
        "Actor".to_string()
    } else {
        s
    }
}

/// Parse an explicit `WxH` resolution token. Without one, a neutral default is returned and
/// flagged as implicit so the scene can adopt its enfolded clip's native resolution.
fn parse_prompt_resolution(text: &str) -> (u32, u32, bool) {
    for part in text.split(|c: char| c == '_' || c.is_whitespace() || c == ':') {
        if let Some((w_str, h_str)) = part.split_once('x') {
            if let (Ok(w), Ok(h)) = (w_str.parse::<u32>(), h_str.parse::<u32>()) {
                if w >= 32 && w <= 4096 && h >= 32 && h <= 4096 {
                    return (w, h, true);
                }
            }
        }
    }
    (320, 240, false)
}

fn parse_part_count(attr_lower: &str, name_lower: &str) -> usize {
    for tok in attr_lower
        .split_whitespace()
        .chain(name_lower.split(|c: char| c == '_' || c.is_whitespace()))
    {
        if let Some(num_str) = tok.strip_prefix("p=")
            && let Ok(p) = num_str.parse::<usize>()
        {
            return p.clamp(1, 8);
        }
    }
    // Morphology words only; archetypes ("hero") and roles carry no body-plan information.
    let combined = format!("{attr_lower} {name_lower}");
    if ["winged", "avian", "bird"].iter().any(|w| combined.contains(w)) {
        6
    } else if ["articulated", "quadruped", "biped", "humanoid"].iter().any(|w| combined.contains(w)) {
        4
    } else {
        1
    }
}

/// Basic colour vocabulary of the prompt language: the words authors can use, and the category
/// sites against which enfolded holon palettes are measured during dispatch.
pub const NAMED_COLORS: &[(&[&str], [u8; 3])] = &[
    (&["yellow", "lemon", "blonde"], [255, 235, 0]),
    (&["red", "crimson", "scarlet", "ruby"], [235, 55, 55]),
    (&["cyan", "azure", "teal", "blue"], [70, 215, 255]),
    (&["pink", "magenta", "rose"], [255, 170, 220]),
    (&["gold", "golden", "amber"], [245, 190, 50]),
    (&["orange"], [255, 150, 40]),
    (&["brown", "tan", "copper", "earth"], [150, 85, 40]),
    (&["green", "emerald", "lime"], [60, 210, 75]),
    (&["navy", "indigo", "violet", "purple"], [70, 60, 170]),
    (&["black", "dark"], [20, 20, 28]),
    (&["gray", "grey", "slate"], [128, 128, 136]),
    (&["white", "silver"], [240, 240, 250]),
];

/// Colour stated in the holon's name or attributes, if any (first vocabulary match).
fn infer_actor_color(name_lower: &str, attr_lower: &str) -> Option<[u8; 3]> {
    let words: Vec<&str> = attr_lower
        .split(|c: char| !c.is_alphanumeric())
        .chain(name_lower.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    NAMED_COLORS
        .iter()
        .find(|(terms, _)| words.iter().any(|w| terms.contains(w)))
        .map(|(_, rgb)| *rgb)
}

/// Distinguishable placeholder used only for procedural rendering when no colour was stated.
fn placeholder_color(idx: usize) -> [u8; 3] {
    const PALETTE: &[[u8; 3]] = &[
        [235, 55, 55],
        [70, 215, 255],
        [255, 235, 0],
        [165, 85, 35],
        [210, 220, 240],
        [95, 175, 235],
    ];
    PALETTE[idx % PALETTE.len()]
}
