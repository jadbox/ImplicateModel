//! Lexicon binding between `Spinor16` semantic verbs and scale-invariant Bohm-Hiley
//! Clifford Quantum Potentials $\mathcal{Q}_{\text{Cl},v}$ (Track R3.1, Whitepaper §5.3).

use implicate_math::{
    HolonPose2D, LearnedVerbPotential, Spinor16, normalized_frechet_distance, wrap_angle_pi,
};
use serde::{Deserialize, Serialize};

use crate::schema::ConceptLexicon;

/// Canonical 5 core action verbs across the Tier-1 and Tier-2 benchmark corpus.
pub const CORE_VERBS: [&str; 5] = ["patrol", "jump_over", "pursue", "flee_from", "orbit"];

/// A lexicon entry pairing a semantic verb `Spinor16` with its learned $\mathcal{Q}_{\text{Cl},v}$ potential.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerbPotentialEntry {
    pub verb: String,
    pub spinor: Spinor16,
    pub potential: LearnedVerbPotential,
}

/// Collection of learned Bohm-Hiley verb potentials indexed by wording-space spinors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerbPotentialLexicon {
    pub entries: Vec<VerbPotentialEntry>,
}

impl VerbPotentialLexicon {
    /// Train a `VerbPotentialLexicon` from labeled trajectory ensembles using `ConceptLexicon`.
    pub fn from_labeled_ensembles(
        lexicon: &ConceptLexicon,
        ensembles: &[(&str, Vec<(Vec<HolonPose2D>, Vec<HolonPose2D>)>)],
    ) -> Self {
        let clauses: Vec<&str> = ensembles.iter().map(|(v, _)| *v).collect();
        let spinors = lexicon.embed_clauses(&clauses);
        let mut entries = Vec::with_capacity(ensembles.len());
        for ((verb, clips), spinor) in ensembles.iter().zip(spinors.into_iter()) {
            let potential = LearnedVerbPotential::fit_from_trajectories(*verb, clips);
            entries.push(VerbPotentialEntry {
                verb: (*verb).to_string(),
                spinor,
                potential,
            });
        }
        Self { entries }
    }

    /// Retrieve a verb potential by exact name.
    pub fn get(&self, verb_name: &str) -> Option<&VerbPotentialEntry> {
        self.entries.iter().find(|e| e.verb == verb_name)
    }

    /// Resolve a verb clause by exact name first, falling back to maximum `Spinor16`
    /// wording-space coherence $\langle \psi_q \widetilde{\psi}_e \rangle_0$.
    pub fn resolve(&self, lexicon: &ConceptLexicon, verb_clause: &str) -> Option<&VerbPotentialEntry> {
        if let Some(exact) = self.get(verb_clause) {
            return Some(exact);
        }
        let query_spinor = lexicon.embed_clauses(&[verb_clause]).into_iter().next()?;
        self.entries.iter().max_by(|a, b| {
            let ca = query_spinor.coherence(&a.spinor);
            let cb = query_spinor.coherence(&b.spinor);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Insert or update a learned verb potential entry in the lexicon.
    pub fn upsert(&mut self, verb: String, spinor: Spinor16, potential: LearnedVerbPotential) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.verb == verb) {
            existing.spinor = spinor;
            existing.potential = potential;
        } else {
            self.entries.push(VerbPotentialEntry {
                verb,
                spinor,
                potential,
            });
        }
    }
}

/// Generate a ground-truth trajectory pair $(A(t), B(t))$ for one of the 5 core corpus verbs
/// under a given world translation `offset`, characteristic scale `scale_l`, global rotation `rot_rad`,
/// and trajectory sample noise `jitter_amp`.
pub fn synthesize_corpus_verb_clip(
    verb: &str,
    num_steps: usize,
    offset: [f64; 2],
    scale_l: f64,
    rot_rad: f64,
    jitter_amp: f64,
    seed: u64,
) -> (Vec<HolonPose2D>, Vec<HolonPose2D>) {
    let n = num_steps.max(8);
    let mut traj_a = Vec::with_capacity(n);
    let mut traj_b = Vec::with_capacity(n);
    let (sr, cr) = rot_rad.sin_cos();
    let pi = std::f64::consts::PI;

    for k in 0..n {
        let tau = (k as f64) / ((n - 1) as f64);
        // Deterministic pseudo-random observation noise (zero at k=0 so initial anchor is well-defined)
        let n_x = if k == 0 {
            0.0
        } else {
            jitter_amp * pseudo_normal(seed.wrapping_add((k as u64) * 31))
        };
        let n_y = if k == 0 {
            0.0
        } else {
            jitter_amp * pseudo_normal(seed.wrapping_add((k as u64) * 31 + 17))
        };

        // Canonical motion of B and relative motion of A in unrotated unit frame where A(0) = (-1, 0), B(0) = (0, 0)
        let (bx_loc, by_loc, b_theta_loc, ax_rel, ay_rel, a_theta_rel) = match verb {
            "patrol" => {
                // Back-and-forth horizontal sentinel patrol across anchor B (Goomba / guard mouse)
                let x = -(2.0 * pi * tau).cos();
                let y = 0.08 * (2.0 * pi * tau).sin();
                let vx = 2.0 * pi * (2.0 * pi * tau).sin();
                let vy = 0.16 * pi * (2.0 * pi * tau).cos();
                (0.0, 0.0, 0.0, x, y, vy.atan2(vx.abs().max(0.2)))
            }
            "jump_over" => {
                // Ballistic parabolic vault over obstacle B at origin from (-1, 0) to (+1, 0) (Mario / Brisby)
                let x = -1.0 + 2.0 * tau;
                let y = 4.0 * 0.85 * tau * (1.0 - tau);
                let vx = 2.0;
                let vy = 3.4 * (1.0 - 2.0 * tau);
                (0.0, 0.0, 0.0, x, y, vy.atan2(vx))
            }
            "pursue" => {
                // Target B moves forward while pursuer A closes distance from (-1, 0) to (-0.12, 0)
                let bx = 0.65 * tau;
                let by = 0.15 * (pi * tau).sin();
                let x = -1.0 + 0.88 * (1.0 - (1.0 - tau).powi(2));
                let y = 0.12 * (pi * tau).sin();
                let vx = 1.76 * (1.0 - tau) + 0.65;
                let vy = 0.27 * pi * (pi * tau).cos();
                (bx, by, 0.0, x, y, vy.atan2(vx))
            }
            "flee_from" => {
                // Predator B advances while subject A accelerates away from (-1, 0) to (-2.4, 0.45)
                let bx = -0.35 * tau;
                let by = 0.0;
                let x = -1.0 - 1.40 * (tau + 0.35 * tau * tau);
                let y = 0.45 * (0.5 * pi * tau).sin();
                let vx = -1.40 * (1.0 + 0.70 * tau);
                let vy = 0.225 * pi * (0.5 * pi * tau).cos();
                (bx, by, pi, x, y, vy.atan2(vx))
            }
            "orbit" | _ => {
                // Circular/elliptical orbital circuit around central talisman/target B (Crow / Amulet)
                let ang = pi + 2.0 * pi * tau;
                let x = ang.cos();
                let y = ang.sin();
                let vx = -2.0 * pi * ang.sin();
                let vy = 2.0 * pi * ang.cos();
                (0.0, 0.0, 0.0, x, y, vy.atan2(vx))
            }
        };

        let ax_loc = bx_loc + ax_rel + n_x;
        let ay_loc = by_loc + ay_rel + n_y;

        let wx_b = offset[0] + scale_l * (bx_loc * cr - by_loc * sr);
        let wy_b = offset[1] + scale_l * (bx_loc * sr + by_loc * cr);
        let wtheta_b = wrap_angle_pi(b_theta_loc + rot_rad);

        let wx_a = offset[0] + scale_l * (ax_loc * cr - ay_loc * sr);
        let wy_a = offset[1] + scale_l * (ax_loc * sr + ay_loc * cr);
        let wtheta_a = wrap_angle_pi(a_theta_rel + rot_rad);

        traj_b.push(HolonPose2D::new(wx_b, wy_b, wtheta_b));
        traj_a.push(HolonPose2D::new(wx_a, wy_a, wtheta_a));
    }
    (traj_a, traj_b)
}

/// Evaluate the normalized Fréchet distance of `LearnedVerbPotential::unfold_trajectory`
/// against a ground-truth clip `(gt_a, gt_b)`.
pub fn evaluate_verb_unfold_frechet(
    potential: &LearnedVerbPotential,
    gt_a: &[HolonPose2D],
    gt_b: &[HolonPose2D],
) -> f64 {
    if gt_a.is_empty() || gt_b.is_empty() {
        return 1.0;
    }
    let pred_a = potential.unfold_trajectory(gt_a[0], gt_b);
    let span_ab = (gt_a[0].x - gt_b[0].x)
        .hypot(gt_a[0].y - gt_b[0].y)
        .max(1e-3);
    normalized_frechet_distance(&pred_a, gt_a, span_ab)
}

#[inline]
fn pseudo_normal(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    let u = ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
    2.0 * u - 1.0
}
