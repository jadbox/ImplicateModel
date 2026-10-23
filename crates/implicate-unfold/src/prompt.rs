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
                        initial_pos: [128.0, 72.0],
                        initial_vel: [0.0, 0.0],
                        radius: 6.0,
                        rho: 2.0,
                        bulk_modulus_k: 4.5,
                        base_color: [245, 180, 40],
                        luminescence_emission: [0.8, 0.6, 0.1],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                    },
                    PromptHolonSpec {
                        name: "Mario".to_string(),
                        semantic_clause: "Mario hero joy jumper".to_string(),
                        initial_pos: [86.0, 72.0],
                        initial_vel: [22.0, 28.0],
                        radius: 6.5,
                        rho: 1.1,
                        bulk_modulus_k: 1.6,
                        base_color: [240, 45, 45],
                        luminescence_emission: [0.3, 0.1, 0.1],
                        num_parts: 4,
                        is_camera_locked_hud: false,
                    },
                    PromptHolonSpec {
                        name: "Goomba_Sentinel".to_string(),
                        semantic_clause: "Goomba shadow anger sentinel walker".to_string(),
                        initial_pos: [94.0, 72.0],
                        initial_vel: [12.0, 0.0],
                        radius: 5.5,
                        rho: 1.0,
                        bulk_modulus_k: 1.2,
                        base_color: [165, 85, 35],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: false,
                    },
                    PromptHolonSpec {
                        name: "SMB3_HUD_Panel".to_string(),
                        semantic_clause: "entity_hud_interface bottom status panel cards coins".to_string(),
                        initial_pos: [128.0, 24.0],
                        initial_vel: [18.0, 0.0],
                        radius: 6.0,
                        rho: 1.0,
                        bulk_modulus_k: 1.0,
                        base_color: [250, 230, 190],
                        luminescence_emission: [0.0, 0.0, 0.0],
                        num_parts: 1,
                        is_camera_locked_hud: true,
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
                        point: [0.0, 42.0],
                        normal: [0.0, 1.0],
                    },
                    PromptBoundaryBlade {
                        point: [8.0, 42.0],
                        normal: [1.0, 0.0],
                    },
                    PromptBoundaryBlade {
                        point: [248.0, 42.0],
                        normal: [-1.0, 0.0],
                    },
                ],
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
            },
        ]
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
        let (width, height) = if lower_all.contains("pacman") {
            (240u32, 308u32)
        } else if lower_all.contains("smb") || lower_all.contains("mario") {
            (256u32, 240u32)
        } else if lower_all.contains("raptor") {
            (320u32, 240u32)
        } else {
            (320u32, 180u32)
        };
        let num_frames = 60usize;
        let enable_cliffordnet_refiner = lower_all.contains("cliffordnet")
            || lower_all.contains("@rolpa")
            || lower_all.contains("refiner")
            || lower_all.contains("glowing")
            || lower_all.contains("amulet");

        let camera_scroll_vel = if lower_all.contains("scroll") || lower_all.contains("mario") {
            [16.0f32, 0.0]
        } else if lower_all.contains("raptor") {
            [0.0f32, 20.0]
        } else if lower_all.contains("nimh") || lower_all.contains("brisby") {
            [6.0f32, 2.0]
        } else {
            [0.0f32, 0.0]
        };

        let (background_top_rgb, background_bottom_rgb) = if lower_all.contains("mario")
            || lower_all.contains("sky")
        {
            ([110, 180, 255], [180, 225, 255])
        } else if lower_all.contains("pacman") {
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
                || name_lower.contains("amulet")
                || name_lower.contains("pellet")
                || name_lower.contains("plasma");

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

            let base_color = infer_actor_color(&name_lower, &attr_lower, is_target, bind_idx);
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
                rho: if is_target { 2.0 } else { 1.1 },
                bulk_modulus_k: if is_target { 4.2 } else { 1.6 },
                base_color,
                luminescence_emission,
                num_parts,
                is_camera_locked_hud: false,
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
        })
    }
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

fn parse_part_count(attr_lower: &str, name_lower: &str) -> usize {
    for tok in attr_lower.split_whitespace() {
        if let Some(num_str) = tok.strip_prefix("p=")
            && let Ok(p) = num_str.parse::<usize>()
        {
            return p.clamp(1, 8);
        }
    }
    if name_lower.contains("crow") || name_lower.contains("jeremy") || name_lower.contains("bird") {
        6
    } else if name_lower.contains("brisby") || name_lower.contains("mario") {
        4
    } else {
        1
    }
}

fn infer_actor_color(name_lower: &str, attr_lower: &str, is_target: bool, idx: usize) -> [u8; 3] {
    if name_lower.contains("pacman") || attr_lower.contains("yellow") {
        [255, 235, 0]
    } else if name_lower.contains("mario")
        || name_lower.contains("brisby")
        || name_lower.contains("blinky")
        || name_lower.contains("amulet")
        || attr_lower.contains("red")
    {
        [235, 55, 55]
    } else if name_lower.contains("inky")
        || name_lower.contains("plasma")
        || attr_lower.contains("cyan")
        || attr_lower.contains("blue")
    {
        [70, 215, 255]
    } else if name_lower.contains("koopa") || attr_lower.contains("gold") || is_target {
        [245, 190, 50]
    } else if idx % 2 == 0 {
        [210, 220, 240]
    } else {
        [95, 175, 235]
    }
}
