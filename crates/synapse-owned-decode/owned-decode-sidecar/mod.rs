//! Request-scoped semantic-sidecar result normalization and hint banking.
//!
//! The module is pure data handling: it does not start sidecar transports, touch
//! routing, or commit target decode state. An integration layer may call these
//! contracts after an eligible request completes a sidecar job and before a
//! worker performs its own non-blocking suffix-match pickup.

pub mod bank;
pub mod client;
pub mod outcome;
pub mod renderer;
pub mod slotter;

pub use bank::{
    build_default_hint_bank, build_hint_bank, find_hint_continuation, HintBankError,
    HintContinuation, SidecarWorkBounds, TargetTokenizer, MAX_HINT_PROPOSAL_TOKENS,
    MAX_SUFFIX_MATCH_TOKENS,
};
pub use client::{
    prepare_per_field, prepare_whole_object, PreparedSidecarResult, SidecarResultError,
};
pub use outcome::{SidecarBankEffect, SidecarOutcome, SidecarOutcomeEvents, SpanClass};
pub use renderer::{
    render_object, FrozenObjectSchema, FrozenProperty, FrozenScalarType, FrozenSchema, RenderError,
    RenderPolicy, SchemaValidationError,
};
pub use slotter::{
    PerFieldFallbackContract, PerFieldPlan, PerFieldPlanError, PerFieldResult, PerFieldSlot,
    PerFieldSlotResult,
};

#[cfg(test)]
mod tests {
    use std::fmt;

    use serde_json::json;
    use synapse_core::{SidecarHintBank, SidecarOutcome, SidecarOutcomeEvents};

    use super::*;

    #[derive(Default)]
    struct ByteTokenizer;

    #[derive(Debug)]
    struct TokenizerFailure;

    impl fmt::Display for TokenizerFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("tokenizer failed")
        }
    }

    impl TargetTokenizer for ByteTokenizer {
        type Error = TokenizerFailure;

        fn tokenize_target_view(&self, view: &str) -> Result<Vec<u32>, Self::Error> {
            Ok(view.bytes().map(u32::from).collect())
        }
    }

    fn schema() -> FrozenObjectSchema {
        FrozenObjectSchema {
            schema_identity: "schema-sha256".to_string(),
            properties: vec![
                FrozenProperty {
                    name: "first".to_string(),
                    required: true,
                    schema: FrozenSchema::Scalar {
                        scalar_type: FrozenScalarType::String,
                        enumeration: None,
                    },
                },
                FrozenProperty {
                    name: "state".to_string(),
                    required: true,
                    schema: FrozenSchema::Scalar {
                        scalar_type: FrozenScalarType::String,
                        enumeration: Some(vec![json!("ready"), json!("done")]),
                    },
                },
                FrozenProperty {
                    name: "optional".to_string(),
                    required: false,
                    schema: FrozenSchema::Array {
                        items: Box::new(FrozenSchema::Scalar {
                            scalar_type: FrozenScalarType::Integer,
                            enumeration: None,
                        }),
                    },
                },
            ],
        }
    }

    fn contract() -> PerFieldFallbackContract {
        PerFieldFallbackContract {
            prompt_template_revision: "field-prompt-v1".to_string(),
            max_new_tokens: 16,
            presence_representation_revision: "present-value-v1".to_string(),
            validation_revision: "validation-v1".to_string(),
            joining_revision: "join-v1".to_string(),
        }
    }

    #[test]
    fn rendering_owns_property_order_and_layout_not_sidecar_structure() {
        let policy = RenderPolicy {
            leading_whitespace: " ".to_string(),
            colon_whitespace: " ".to_string(),
            comma_whitespace: " ".to_string(),
            trailing_whitespace: "\n".to_string(),
        };
        let raw = b"{\"state\":\"ready\",\"optional\":[1,2],\"first\":\"Ada\"} \t";

        let prepared = prepare_whole_object(&schema(), raw, &policy).expect("valid whole object");

        assert_eq!(
            prepared.rendered_view,
            " {\"first\": \"Ada\", \"state\": \"ready\", \"optional\": [1, 2]}\n"
        );
        assert_eq!(
            policy.digest(),
            policy.digest(),
            "policy digest is deterministic"
        );
    }

    #[test]
    fn strict_whole_object_path_rejects_non_single_and_invalid_results() {
        let policy = RenderPolicy::compact();
        for raw in [
            b"".as_slice(),
            b"```json\n{}\n```".as_slice(),
            b"{}{}".as_slice(),
            b"[]".as_slice(),
            b"{\"first\":\"Ada\",\"state\":\"not-an-enum\"}".as_slice(),
            b"{\"first\":\"Ada\",\"state\":\"ready\",\"unknown\":true}".as_slice(),
        ] {
            assert!(
                prepare_whole_object(&schema(), raw, &policy).is_err(),
                "raw={raw:?}"
            );
        }
        assert!(prepare_whole_object(&schema(), &[b'{', 0xff, b'}'], &policy).is_err());
    }

    #[test]
    fn frozen_field_slots_validate_independently_and_join_in_schema_order() {
        let schema = schema();
        let plan = PerFieldPlan::freeze(&schema, contract()).expect("freeze field plan");
        let raw_slots: [&[u8]; 3] = [
            br#"{"present":true,"value":"Ada"}"#,
            br#"{"present":true,"value":"ready"}"#,
            br#"{"present":true,"value":[1,"bad"]}"#,
        ];

        let slots = PerFieldResult::decode(&plan, &raw_slots).expect("decode independently");
        assert!(matches!(slots.slots[0], PerFieldSlotResult::Usable { .. }));
        assert!(matches!(slots.slots[1], PerFieldSlotResult::Usable { .. }));
        assert!(matches!(slots.slots[2], PerFieldSlotResult::Invalid { .. }));
        assert_eq!(
            slots
                .join_object(&plan)
                .expect("invalid optional field is omitted"),
            json!({"first":"Ada","state":"ready"})
        );
        assert!(plan.validate_frozen().is_ok());
        assert!(
            plan.slots[1].prompt.contains("ready"),
            "enum stays in frozen slot data"
        );
    }

    #[test]
    fn mutated_frozen_field_plan_is_rejected_before_result_handling() {
        let schema = schema();
        let mut plan = PerFieldPlan::freeze(&schema, contract()).expect("freeze field plan");
        plan.contract.joining_revision = "join-v2".to_string();

        assert_eq!(
            plan.validate_frozen(),
            Err(PerFieldPlanError::DigestMismatch)
        );
    }

    #[test]
    fn bank_keeps_views_separate_and_uses_target_tokens_only() {
        let schema = schema();
        let policy = RenderPolicy::compact();
        let prepared =
            prepare_whole_object(&schema, br#"{"first":"Ada","state":"ready"}"#, &policy)
                .expect("prepared");
        let bank = build_hint_bank(
            &ByteTokenizer,
            schema.schema_identity,
            policy.digest(),
            &[
                prepared.rendered_view.clone(),
                " {\"first\":\"Ada\",\"state\":\"ready\"}".to_string(),
            ],
            SidecarWorkBounds {
                max_views: 2,
                max_rendered_bytes_per_view: 256,
                max_tokens_per_view: 256,
            },
            88,
        )
        .expect("build bank");

        assert_eq!(bank.views.len(), 2);
        assert_ne!(bank.views[0], bank.views[1]);
        assert!(bank.views.iter().all(|view| !view.is_empty()));
    }

    #[test]
    fn suffix_lookup_is_deterministic_and_never_crosses_view_boundaries() {
        let bank = SidecarHintBank {
            views: vec![vec![9, 4, 5, 70], vec![4, 5, 80, 81], vec![4, 5, 90]],
            schema_identity: "schema".to_string(),
            render_policy_digest: "policy".to_string(),
            built_at: 1,
        };

        let selected = find_hint_continuation(&bank, &[4, 5], 10, 10, 16).expect("match");
        assert_eq!(
            selected.view_index, 0,
            "lowest view wins equal suffix match"
        );
        assert_eq!(selected.bank_offset, 3, "lowest offset wins within a view");
        assert_eq!(
            selected.tokens,
            vec![70],
            "view boundary stops continuation"
        );

        let bounded = find_hint_continuation(&bank, &[4, 5], 1, 10, 16).expect("match");
        assert_eq!(bounded.tokens, vec![70]);
        assert!(find_hint_continuation(&bank, &[4, 5], 0, 10, 16).is_none());
    }

    #[test]
    fn bank_content_is_replayable_across_observational_timestamps() {
        let views = vec!["{\"first\":\"Ada\",\"state\":\"ready\"}".to_string()];
        let bounds = SidecarWorkBounds {
            max_views: 1,
            max_rendered_bytes_per_view: 256,
            max_tokens_per_view: 256,
        };
        let first = build_hint_bank(&ByteTokenizer, "schema", "policy", &views, bounds, 10)
            .expect("first bank");
        let later = build_hint_bank(&ByteTokenizer, "schema", "policy", &views, bounds, 20)
            .expect("later bank");

        let replay: SidecarHintBank =
            serde_json::from_str(&serde_json::to_string(&first).expect("bank serializes"))
                .expect("bank replays");

        assert_eq!(first.views, later.views);
        assert_eq!(first.content_digest(), later.content_digest());
        assert_eq!(first, replay);
        assert_eq!(first.content_digest(), replay.content_digest());
        assert_ne!(first.built_at, later.built_at);
    }

    #[test]
    fn core_outcome_contract_reports_usable_bank_effect() {
        assert!(SidecarOutcome::classify(SidecarOutcomeEvents {
            completed_valid: true,
            bank_used: false,
            ..SidecarOutcomeEvents::default()
        })
        .is_usable());
    }
}
