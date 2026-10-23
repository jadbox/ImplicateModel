//! `implicate-lexicon`: Local VLM Semantic Harvester (`llama.cpp` `Qwen2.5-VL-7B-Instruct` @ `Q4_K_M`),
//! GBNF Schema Enforcement, CFT Kinematic Timestamp Snapping, and Wording-Space Spinor Projection.

pub mod cft_snap;
pub mod gatr;
pub mod harvester;
pub mod schema;
pub mod verb_potential;

pub use cft_snap::{
    KinematicImpulseBoundary, detect_cft_kinematic_impulses, format_timestamp_range,
    parse_timestamp_range, parse_timestamp_sec, snap_verb_window_to_cft_impulses, temporal_iou,
};
pub use gatr::{
    GATR_BLOCKS, GATR_MV_CHANNELS, GATR_SCALAR_CHANNELS, GatrHolonOutput, GatrHolonToken,
    GatrVerbBinder, NonEquivariantMlpBaseline, VerbReadoutRule,
};
pub use harvester::{
    DEFAULT_QWEN_VL_MMPROJ, DEFAULT_QWEN_VL_MODEL, QwenVlHarvesterConfig, content_hash_hex,
    extract_json_object, harvest_clip_with_qwen_vl,
};
pub use schema::{
    ConceptCluster, ConceptKind, ConceptLexicon, IMPLICATE_LEXICON_GBNF, ShotLexicon,
    VlmHolonRecord, VlmVerbRecord, build_shot_spinor_map, build_shot_spinor_map_with_lexicon,
    embed_clauses_to_spinors,
};
pub use verb_potential::{
    CORE_VERBS, VerbPotentialEntry, VerbPotentialLexicon, evaluate_verb_unfold_frechet,
    synthesize_corpus_verb_clip,
};

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use super::*;

    #[test]
    fn r1_2_timestamp_parsing_and_cft_kinematic_snapping_tiou() {
        let gt_intervals = [
            (0.40, 4.63),
            (5.73, 8.20),
            (8.67, 10.00),
            (0.00, 5.23),
            (5.60, 11.87),
        ];
        let raw_vlm_ranges = [
            "[00:00:00 - 00:00:05]",
            "[00:00:06 - 00:00:08]",
            "[00:00:09 - 00:00:10]",
            "[00:00:00 - 00:00:05]",
            "[00:00:06 - 00:00:12]",
        ];
        let cft_impulses_smb3 = [
            KinematicImpulseBoundary {
                t_start: 0.43,
                t_end: 4.60,
                delta_momentum: 6.2,
            },
            KinematicImpulseBoundary {
                t_start: 5.70,
                t_end: 8.17,
                delta_momentum: 3.4,
            },
            KinematicImpulseBoundary {
                t_start: 8.70,
                t_end: 9.97,
                delta_momentum: 4.1,
            },
        ];
        let cft_impulses_nimh = [
            KinematicImpulseBoundary {
                t_start: 0.03,
                t_end: 5.20,
                delta_momentum: 1.8,
            },
            KinematicImpulseBoundary {
                t_start: 5.63,
                t_end: 11.83,
                delta_momentum: 4.9,
            },
        ];

        let mut raw_tiou_sum = 0.0;
        let mut snapped_tiou_sum = 0.0;

        for i in 0..5 {
            let vlm_win = parse_timestamp_range(raw_vlm_ranges[i]).expect("valid timestamp");
            let raw_tiou = temporal_iou(vlm_win, gt_intervals[i]);
            let impulses: &[KinematicImpulseBoundary] = if i < 3 {
                &cft_impulses_smb3
            } else {
                &cft_impulses_nimh
            };
            let snapped_win = snap_verb_window_to_cft_impulses(vlm_win, impulses, 1.25);
            let snapped_tiou = temporal_iou(snapped_win, gt_intervals[i]);

            raw_tiou_sum += raw_tiou;
            snapped_tiou_sum += snapped_tiou;
        }

        let mean_raw_tiou = raw_tiou_sum / 5.0;
        let mean_snapped_tiou = snapped_tiou_sum / 5.0;

        assert!(
            mean_snapped_tiou > 0.85,
            "expected CFT-snapped tIoU > 0.85, got {mean_snapped_tiou:.4} (raw: {mean_raw_tiou:.4})"
        );
        assert!(mean_snapped_tiou > mean_raw_tiou + 0.10);
    }

    #[test]
    fn r1_3_gbnf_validity_and_wording_spinor_coherence() {
        let clauses = [
            "leaps_over",
            "jumps_across",
            "sleeps",
            "shatters",
            "flies_upward",
            "soars_skyward",
            "Mrs_Brisby_mouse",
            "small_brown_mouse",
        ];
        let spinors = embed_clauses_to_spinors(&clauses);

        assert!(spinors[0].coherence(&spinors[1]) > 0.82);
        assert!(spinors[4].coherence(&spinors[5]) > 0.82);
        assert!(spinors[6].coherence(&spinors[7]) > 0.82);
        assert!(spinors[2].coherence(&spinors[3]).abs() < 0.15);
        assert!(spinors[0].coherence(&spinors[2]).abs() < 0.15);
    }

    #[test]
    fn r1_3_emotions_jungian_archetypes_and_open_ended_concept_learning() {
        let mut lexicon = ConceptLexicon::default();
        let initial_len = lexicon.clusters.len();

        // 1. Emotions & Jungian archetypes coherence
        let archetypal_clauses = [
            "terrified_trembling_mouse",
            "frightened_cowering_rodent",
            "nicodemus_wise_elder_sage",
            "mystic_hierophant_oracle",
            "shadow_predator_dragon_cat",
            "menacing_stalker_beast",
        ];
        let sp = lexicon.embed_clauses(&archetypal_clauses);
        assert!(sp[0].coherence(&sp[1]) > 0.80, "fear/rodent coherence");
        assert!(sp[2].coherence(&sp[3]) > 0.82, "Sage/Hierophant archetype coherence");
        assert!(sp[4].coherence(&sp[5]) > 0.82, "Shadow/Beast archetype coherence");
        assert!(sp[0].coherence(&sp[2]).abs() < 0.20, "orthogonal archetypes");

        // 2. Dynamic open-ended concept learning from novel Qwen visual classifications
        let novel_shot = ShotLexicon {
            milieu: "bioluminescent_subterranean_cavern".to_string(),
            holons: vec![
                VlmHolonRecord {
                    id: 1,
                    noun: "clockwork_automaton".to_string(),
                    appearance: "brass_gears_pendulum_chassis".to_string(),
                    location: "center".to_string(),
                },
                VlmHolonRecord {
                    id: 2,
                    noun: "crystalline_obelisk".to_string(),
                    appearance: "violet_prismatic_monolith".to_string(),
                    location: "right".to_string(),
                },
            ],
            verbs: vec![VlmVerbRecord {
                time_range: "[00:00:01 - 00:00:04]".to_string(),
                subject_id: 1,
                verb: "calibrates_chronometer".to_string(),
                target: "violet_prismatic_monolith".to_string(),
            }],
        };

        let learned = lexicon.learn_from_shot(&novel_shot);
        assert!(learned >= 3);
        assert!(lexicon.clusters.len() > initial_len);

        // Newly learned concepts now cohere across visual description & noun variants
        let post_clauses = [
            "clockwork_automaton",
            "brass_gears_pendulum",
            "crystalline_obelisk",
            "violet_prismatic_monolith",
            "calibrates_chronometer",
        ];
        let post_sp = lexicon.embed_clauses(&post_clauses);
        assert!(
            post_sp[0].coherence(&post_sp[1]) > 0.80,
            "learned object + visual appearance attributes cohere"
        );
        assert!(
            post_sp[2].coherence(&post_sp[3]) > 0.80,
            "learned target + appearance attributes cohere"
        );
        assert!(
            post_sp[0].coherence(&post_sp[2]).abs() < 0.20,
            "distinct learned concepts remain orthogonal"
        );
    }

    #[test]
    fn r1_3_live_qwen_vl_gbnf_harvest_and_cache() {
        let clip_path = Path::new("/tmp/opencode/r1_test/smb3_10s.mp4");
        if !clip_path.exists() || !Path::new(DEFAULT_QWEN_VL_MODEL).exists() {
            return;
        }
        let cfg = QwenVlHarvesterConfig {
            cache_dir: PathBuf::from("/tmp/opencode/r1_test/.implicate_cache"),
            ..Default::default()
        };
        let lexicon = harvest_clip_with_qwen_vl(clip_path, 10, &cfg)
            .expect("Qwen2.5-VL GBNF harvest should succeed");
        assert!(!lexicon.milieu.is_empty());
        assert!(!lexicon.holons.is_empty());
        assert!(!lexicon.verbs.is_empty());
        for v in &lexicon.verbs {
            assert!(parse_timestamp_range(&v.time_range).is_some());
        }
        let spinor_map = build_shot_spinor_map(&lexicon);
        assert!(!spinor_map.is_empty());
    }
}
