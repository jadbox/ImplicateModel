//! GBNF Schema & Wording-Space Spinor Projection (`implicate-lexicon`).

use std::collections::HashMap;
use implicate_math::{Spinor16, project_gram_svd};
use serde::{Deserialize, Serialize};

/// Strict GBNF grammar constraining `Qwen2.5-VL-7B-Instruct` outputs to valid `ShotLexicon` JSON.
pub const IMPLICATE_LEXICON_GBNF: &str = r#"root   ::= "{" ws "\"milieu\":" ws string "," ws "\"holons\":" ws "[" ws holon ("," ws holon)* ws "]," ws "\"verbs\":" ws "[" ws verb ("," ws verb)* ws "]" ws "}" ws
holon  ::= "{" ws "\"id\":" ws number "," ws "\"noun\":" ws string "," ws "\"appearance\":" ws string "," ws "\"location\":" ws string ws "}"
verb   ::= "{" ws "\"time_range\":" ws "\"[" timestamp " - " timestamp "]\"," ws "\"subject_id\":" ws number "," ws "\"verb\":" ws string "," ws "\"target\":" ws string ws "}"
timestamp ::= [0-9][0-9] ":" [0-9][0-9] ":" [0-9][0-9] ("." [0-9][0-9][0-9])?
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= [0-9]+
ws     ::= [ \t\n\r]*
"#;

/// Structured output emitted by `Qwen2.5-VL-7B-Instruct` (`Q4_K_M`) for a video shot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShotLexicon {
    pub milieu: String,
    pub holons: Vec<VlmHolonRecord>,
    pub verbs: Vec<VlmVerbRecord>,
}

/// A holon entity identified by the VLM within a shot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VlmHolonRecord {
    pub id: u32,
    pub noun: String,
    pub appearance: String,
    pub location: String,
}

/// An action verb event with `[HH:MM:SS - HH:MM:SS]` timestamp range identified by the VLM.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VlmVerbRecord {
    pub time_range: String,
    pub subject_id: u32,
    pub verb: String,
    pub target: String,
}

/// Ontological stratum of a concept cluster in the implicate lexicon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConceptKind {
    KinematicAction,
    PhysicalEntity,
    AffectiveEmotion,
    JungianArchetype,
    LearnedVisualConcept,
}

/// A semantic-morphological cluster mapping a family of stems to a spinor-space axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConceptCluster {
    pub label: String,
    pub kind: ConceptKind,
    pub terms: Vec<String>,
    pub weight: f64,
}

/// Dynamic, open-ended concept lexicon that seeds physical, emotional, and Jungian priors
/// and expands online as `Qwen2.5-VL` classifies novel objects, actions, and behaviors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConceptLexicon {
    pub clusters: Vec<ConceptCluster>,
}

impl Default for ConceptLexicon {
    fn default() -> Self {
        use ConceptKind::*;
        let seeds: &[(&str, ConceptKind, &[&str], f64)] = &[
            // Kinematic & physical actions
            ("jump_vault", KinematicAction, &["jump", "jumps", "leap", "leaps", "hop", "hops", "vault", "vaults", "bounce", "bounces", "across", "over"], 3.5),
            ("run_locomote", KinematicAction, &["run", "runs", "running", "sprint", "sprints", "dash", "dashes", "scurry", "scurries", "walk", "walks"], 3.5),
            ("fly_soar", KinematicAction, &["fly", "flies", "flying", "soar", "soars", "swoop", "swoops", "glide", "glides", "hover", "hovers"], 3.5),
            ("shatter_cleave", KinematicAction, &["shatter", "shatters", "break", "breaks", "cleave", "cleaves", "fracture", "fractures", "smash", "smashes", "explode"], 3.5),
            ("sleep_stillness", KinematicAction, &["sleep", "sleeps", "rest", "rests", "idle", "stand", "stands", "wait", "waits", "still", "stationary"], 3.5),
            ("shoot_project", KinematicAction, &["shoot", "shoots", "fire", "fires", "launch", "launches", "emit", "emits", "blast", "blasts", "projectile"], 3.5),
            ("flee_evade", KinematicAction, &["flee", "flees", "escape", "escapes", "retreat", "retreats", "evade", "evades", "avoid", "avoids"], 3.5),
            ("pursue_seek", KinematicAction, &["pursue", "pursues", "chase", "chases", "follow", "follows", "seek", "seeks", "reach", "reaches", "approach"], 3.5),
            ("orbit_revolve", KinematicAction, &["orbit", "orbits", "circle", "circles", "revolve", "revolves", "spin", "spins", "rotate", "rotates", "turn"], 3.5),
            ("glow_radiate", KinematicAction, &["glow", "glows", "radiate", "radiates", "shine", "shines", "illuminate", "illuminates", "sparkle"], 3.5),
            ("collide_strike", KinematicAction, &["stomp", "stomps", "crush", "crushes", "squash", "squashes", "collide", "collides", "strike", "strikes"], 3.5),
            // Baseline entities
            ("entity_rodent", PhysicalEntity, &["mouse", "small_mammal", "rodent", "shrew", "rat"], 3.0),
            ("entity_avian", PhysicalEntity, &["crow", "raptor_bird", "bird", "owl", "winged", "avian"], 3.0),
            ("entity_vessel", PhysicalEntity, &["jet", "ship", "aircraft", "fighter", "vessel", "tractor", "plow", "carrier"], 3.0),
            ("entity_hud_interface", PhysicalEntity, &["hud", "interface", "scoreboard", "meter", "gauge", "counter", "status", "overlay", "cockpit", "radar", "panel"], 3.0),
            // Chromatic appearance
            ("color_chroma_red", PhysicalEntity, &["red", "crimson", "scarlet", "ruby", "vermilion"], 3.0),
            ("color_chroma_yellow", PhysicalEntity, &["yellow", "lemon", "blonde"], 3.0),
            ("color_chroma_gold", PhysicalEntity, &["gold", "golden", "amber"], 3.0),
            ("color_chroma_blue_cyan", PhysicalEntity, &["blue", "cyan", "azure", "sapphire", "teal"], 3.0),
            ("color_chroma_green", PhysicalEntity, &["green", "emerald", "lime", "olive"], 3.0),
            ("color_chroma_brown", PhysicalEntity, &["brown", "tan", "chestnut", "earth"], 3.0),
            ("color_chroma_orange", PhysicalEntity, &["orange", "copper", "rust", "bronze"], 3.0),
            ("color_chroma_dark_black", PhysicalEntity, &["black", "dark", "shadowy", "charcoal", "obsidian"], 3.0),
            ("color_chroma_bright_white", PhysicalEntity, &["white", "bright", "silver", "pale", "light"], 3.0),
            // Morphology & structural anatomy
            ("morphology_rigid_single", PhysicalEntity, &["rigid", "single", "orb", "ball", "sphere", "disc", "block", "p1"], 3.0),
            ("morphology_articulated_limbs", PhysicalEntity, &["articulated", "limbs", "quadruped", "biped", "humanoid", "walker", "creature", "p4"], 3.0),
            ("morphology_winged_aerodynamic", PhysicalEntity, &["winged", "avian", "bird", "wings", "flight", "soaring", "vessel", "aircraft", "p6", "p8"], 3.0),
            // Constitutive physical categories
            ("constitutive_field_soliton", PhysicalEntity, &["soliton", "plasma", "exciton", "luminescence", "beam", "laser", "energy", "radiance"], 3.2),
            ("constitutive_terrain_boundary", PhysicalEntity, &["terrain", "fixture", "wall", "ground", "obstacle", "barrier", "platform"], 3.2),
            // Affective emotions
            ("emotion_fear_dread", AffectiveEmotion, &["fear", "terrified", "dread", "panic", "anxious", "frightened", "tremble", "trembling", "cower", "cowering"], 3.2),
            ("emotion_courage_resolve", AffectiveEmotion, &["courage", "brave", "resolute", "defiant", "determined", "steadfast", "undaunted", "bold"], 3.2),
            ("emotion_grief_sorrow", AffectiveEmotion, &["grief", "sorrow", "sad", "weeping", "mourn", "mourning", "despair", "forlorn", "melancholy"], 3.2),
            ("emotion_joy_wonder", AffectiveEmotion, &["joy", "delight", "wonder", "playful", "cheerful", "ecstatic", "relief", "rejoice", "happy"], 3.2),
            ("emotion_rage_menace", AffectiveEmotion, &["rage", "wrath", "angry", "fierce", "menacing", "hostile", "ferocious", "snarl", "snarling", "furious"], 3.2),
            ("emotion_awe_numinous", AffectiveEmotion, &["awe", "reverence", "mystical", "numinous", "sacred", "trance", "solemn", "enchanted"], 3.2),
            // Jungian archetypes
            ("archetype_self_talisman", JungianArchetype, &["self", "wholeness", "amulet", "talisman", "mandala", "stone", "medallion", "nexus"], 3.4),
            ("archetype_shadow_beast", JungianArchetype, &["shadow", "predator", "dragon", "cat", "monster", "beast", "stalker", "adversary", "villain", "fiend"], 3.4),
            ("archetype_hero_protector", JungianArchetype, &["hero", "heroine", "seeker", "protector", "rescuer", "champion", "savior", "martyr", "protagonist", "player"], 3.4),
            ("archetype_sage_hierophant", JungianArchetype, &["sage", "elder", "patriarch", "oracle", "wizard", "mystic", "mentor", "hierophant"], 3.4),
            ("archetype_trickster_fool", JungianArchetype, &["trickster", "jester", "fool", "clumsy", "tangled", "comic", "mischief", "buffoon"], 3.4),
            ("archetype_threshold_guardian", JungianArchetype, &["guardian", "sentinel", "gatekeeper", "watcher", "warden", "guard", "threshold"], 3.4),
            ("archetype_anima_guide", JungianArchetype, &["anima", "animus", "guide", "muse", "companion", "ally", "psychopomp"], 3.4),
            ("archetype_great_mother_sanctuary", JungianArchetype, &["mother", "maternal", "sanctuary", "hearth", "nest", "burrow", "shelter", "cradle", "nurture"], 3.4),
        ];

        let clusters = seeds
            .iter()
            .map(|&(label, kind, terms, weight)| ConceptCluster {
                label: label.to_string(),
                kind,
                terms: terms.iter().map(|&t| t.to_string()).collect(),
                weight,
            })
            .collect();
        Self { clusters }
    }
}

impl ConceptLexicon {
    /// Extract normalized content tokens from a free-form clause, stripping spatial/syntactic stopwords.
    pub fn tokenize_content(clause: &str) -> Vec<String> {
        const STOPWORDS: &[&str] = &[
            "the", "and", "with", "from", "into", "onto", "near", "towards", "toward", "through",
            "under", "above", "below", "left", "right", "top", "bottom", "center", "middle",
            "background", "foreground", "screen", "frame", "side", "area", "part", "small", "large",
        ];
        let lower: String = clause
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        lower
            .split_whitespace()
            .filter(|t| t.len() >= 3 && !STOPWORDS.contains(t) && !t.chars().all(|c| c.is_numeric()))
            .map(|t| t.to_string())
            .collect()
    }

    fn token_matches(tok: &str, term: &str) -> bool {
        tok == term || (tok.len() >= 4 && term.len() >= 4 && (tok.starts_with(term) || term.starts_with(tok)))
    }

    /// Assimilate a visual classification clause (`noun + appearance` or `verb + target`).
    /// Returns `true` if a new `LearnedVisualConcept` cluster was created or an existing cluster expanded.
    pub fn learn_clause(&mut self, primary: &str, context: &str, weight: f64) -> bool {
        let primary_tokens = Self::tokenize_content(primary);
        if primary_tokens.is_empty() {
            return false;
        }
        let context_tokens = Self::tokenize_content(context);

        // Check if any primary token already belongs to an existing concept cluster.
        let mut matched_idx = None;
        for (idx, cluster) in self.clusters.iter().enumerate() {
            if primary_tokens.iter().any(|pt| {
                cluster.terms.iter().any(|ct| Self::token_matches(pt, ct))
            }) {
                matched_idx = Some(idx);
                break;
            }
        }

        if let Some(idx) = matched_idx {
            let mut modified = false;
            for pt in &primary_tokens {
                if !self.clusters[idx].terms.iter().any(|ct| Self::token_matches(pt, ct)) {
                    self.clusters[idx].terms.push(pt.clone());
                    modified = true;
                }
            }
            modified
        } else {
            let label = format!("learned_{}", primary_tokens.join("_"));
            let mut terms = primary_tokens;
            for ct in context_tokens.into_iter().take(3) {
                if !terms.iter().any(|t| Self::token_matches(&ct, t)) {
                    terms.push(ct);
                }
            }
            self.clusters.push(ConceptCluster {
                label,
                kind: ConceptKind::LearnedVisualConcept,
                terms,
                weight,
            });
            true
        }
    }

    /// Dynamically expand the concept lexicon from a `ShotLexicon` harvested by `Qwen2.5-VL`.
    /// Returns the number of newly created or expanded concept clusters.
    pub fn learn_from_shot(&mut self, shot: &ShotLexicon) -> usize {
        let mut updates = 0;
        if self.learn_clause(&shot.milieu, "", 2.8) {
            updates += 1;
        }
        for h in &shot.holons {
            if self.learn_clause(&h.noun, &h.appearance, 3.0) {
                updates += 1;
            }
        }
        for v in &shot.verbs {
            // The verb phrase carries action + manner; target is a distinct entity, not a verb synonym.
            if self.learn_clause(&v.verb, "", 3.5) {
                updates += 1;
            }
            if self.learn_clause(&v.target, "", 3.0) {
                updates += 1;
            }
        }
        updates
    }

    /// Project a slice of wording clauses into $(\mathbb{G}_{2,0}^+)^{16}$ spinors using
    /// the current (seed + learned) concept clusters and character trigram backoff.
    pub fn embed_clauses(&self, clauses: &[&str]) -> Vec<Spinor16> {
        let num_clusters = self.clusters.len();
        const NGRAM_BINS: usize = 32;
        let dim = num_clusters * 2 + NGRAM_BINS;

        let mut embeddings = Vec::with_capacity(clauses.len());
        for &clause in clauses {
            let mut vec = vec![0.0_f64; dim];
            let lower: String = clause
                .to_lowercase()
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { ' ' })
                .collect();
            let tokens: Vec<&str> = lower.split_whitespace().collect();

            for &tok in &tokens {
                for (axis, cluster) in self.clusters.iter().enumerate() {
                    if cluster.terms.iter().any(|w| Self::token_matches(tok, w)) {
                        vec[axis] += cluster.weight;
                        vec[num_clusters + axis] += cluster.weight * 0.7;
                    }
                }
                let bytes = tok.as_bytes();
                for win in bytes.windows(3.min(bytes.len().max(1))) {
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &b in win {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    let idx = num_clusters * 2 + ((h as usize) % NGRAM_BINS);
                    vec[idx] += 0.12;
                }
            }
            if vec.iter().all(|&x| x == 0.0) {
                vec[num_clusters * 2] = 1.0;
            }
            embeddings.push(vec);
        }

        project_gram_svd(&embeddings)
    }
}

/// Canonical verb/noun/emotion/archetype feature extractor projecting wording clauses
/// into $(\mathbb{G}_{2,0}^+)^{16}$ spinors via `ConceptLexicon::default()`.
pub fn embed_clauses_to_spinors(clauses: &[&str]) -> Vec<Spinor16> {
    ConceptLexicon::default().embed_clauses(clauses)
}

/// Build a Spinor Lexicon lookup table mapping canonical noun/verb strings to $(\mathbb{G}_{2,0}^+)^{16}$,
/// dynamically learning any novel visual concepts present in `shot`.
pub fn build_shot_spinor_map(shot: &ShotLexicon) -> HashMap<String, Spinor16> {
    let mut lexicon = ConceptLexicon::default();
    lexicon.learn_from_shot(shot);
    build_shot_spinor_map_with_lexicon(shot, &lexicon)
}

/// Build a Spinor Lexicon lookup table using an explicit (shared/accumulated) `ConceptLexicon`.
pub fn build_shot_spinor_map_with_lexicon(
    shot: &ShotLexicon,
    lexicon: &ConceptLexicon,
) -> HashMap<String, Spinor16> {
    let mut keys: Vec<String> = Vec::new();
    for h in &shot.holons {
        if !keys.contains(&h.noun) {
            keys.push(h.noun.clone());
        }
    }
    for v in &shot.verbs {
        if !keys.contains(&v.verb) {
            keys.push(v.verb.clone());
        }
    }
    let refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let spinors = lexicon.embed_clauses(&refs);
    keys.into_iter().zip(spinors).collect()
}
