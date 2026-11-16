use synapse_speech::{normalize_words, qwen::align_words, Error, Word};

fn word(text: &str, start_ms: f64, end_ms: f64) -> Word {
    Word {
        word: text.into(),
        start_ms,
        end_ms,
    }
}

fn check(text: &str, anchors: &[Word], duration: f64, expected: &[Word]) {
    let words = align_words(text, anchors, duration).unwrap();
    assert_eq!(words, expected);
    let mut previous = 0.0;
    for w in words {
        assert!(w.start_ms.is_finite() && w.end_ms.is_finite());
        assert!(0.0 <= w.start_ms && w.start_ms <= w.end_ms && w.end_ms <= duration);
        assert!(w.start_ms >= previous);
        previous = w.start_ms;
    }
}

#[test]
fn listed_name_missing_from_companion_shares_gap() {
    check(
        "call Ada Lovelace now",
        &[word("call", 10.0, 20.0), word("now", 60.0, 80.0)],
        100.0,
        &[
            word("call", 10.0, 20.0),
            word("Ada", 20.0, 40.0),
            word("Lovelace", 40.0, 60.0),
            word("now", 60.0, 80.0),
        ],
    );
}

#[test]
fn all_words_align_with_normalization_for_matching_only() {
    check(
        "ＳＴＲＡＳＳＥ!\tHello,\nWORLD",
        &[
            word("Straße", 10.0, 20.0),
            word("hello", 30.0, 40.0),
            word("world", 50.0, 60.0),
        ],
        100.0,
        &[
            word("ＳＴＲＡＳＳＥ!", 10.0, 20.0),
            word("Hello,", 30.0, 40.0),
            word("WORLD", 50.0, 60.0),
        ],
    );
    assert_eq!(
        normalize_words("(Straße) ＨＥＬＬＯ a—b ①"),
        ["strasse", "hello", "ab", "1"]
    );
}

#[test]
fn leading_and_trailing_unaligned_runs_share_duration_edges() {
    check(
        "first two middle last two",
        &[word("middle", 40.0, 60.0)],
        100.0,
        &[
            word("first", 0.0, 20.0),
            word("two", 20.0, 40.0),
            word("middle", 40.0, 60.0),
            word("last", 60.0, 80.0),
            word("two", 80.0, 100.0),
        ],
    );
}

#[test]
fn empty_normalized_words_share_companion_span_without_aligning() {
    check(
        "... !!! —",
        &[word("speech", 10.0, 30.0), word("here", 50.0, 70.0)],
        100.0,
        &[
            word("...", 10.0, 30.0),
            word("!!!", 30.0, 50.0),
            word("—", 50.0, 70.0),
        ],
    );
}

#[test]
fn repeated_word_ties_choose_earliest_position() {
    check(
        "go go",
        &[word("go", 20.0, 40.0)],
        100.0,
        &[word("go", 20.0, 40.0), word("go", 40.0, 100.0)],
    );
    check(
        "go",
        &[word("go", 20.0, 40.0), word("go", 60.0, 80.0)],
        100.0,
        &[word("go", 20.0, 40.0)],
    );
}

#[test]
fn substitutions_win_edit_distance_ties_and_keep_anchor_times() {
    check(
        "left right",
        &[word("right", 10.0, 20.0), word("left", 30.0, 40.0)],
        100.0,
        &[word("left", 10.0, 20.0), word("right", 30.0, 40.0)],
    );
}

#[test]
fn overlapping_and_zero_duration_anchors_never_reverse_gap_words() {
    check(
        "one missing two",
        &[word("one", 10.0, 70.0), word("two", 30.0, 50.0)],
        100.0,
        &[
            word("one", 10.0, 70.0),
            word("missing", 30.0, 30.0),
            word("two", 30.0, 50.0),
        ],
    );
    check(
        "one ... two",
        &[word("one", 30.0, 30.0), word("two", 30.0, 30.0)],
        100.0,
        &[
            word("one", 30.0, 30.0),
            word("...", 30.0, 30.0),
            word("two", 30.0, 30.0),
        ],
    );
}

#[test]
fn invalid_companion_times_and_missing_words_are_refused() {
    for anchors in [
        vec![word("bad", f64::NAN, 30.0)],
        vec![word("bad", 0.0, f64::INFINITY)],
        vec![word("bad", -1.0, 20.0)],
        vec![word("bad", 40.0, 20.0)],
        vec![word("bad", 20.0, 101.0)],
        vec![word("one", 20.0, 30.0), word("two", 10.0, 40.0)],
        vec![],
    ] {
        assert!(matches!(
            align_words("text", &anchors, 100.0),
            Err(Error::InvalidWordTiming(_))
        ));
    }
    for duration in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(matches!(
            align_words("text", &[], duration),
            Err(Error::InvalidWordTiming(_))
        ));
    }
    assert!(align_words(" \t\n", &[], 100.0).unwrap().is_empty());
}
