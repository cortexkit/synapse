use synapse_speech::*;

fn wav(codec: u16, bits: u16, rate: u32, channels: u16, data: &[u8], extensible: bool) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&(if extensible { 0xfffe } else { codec }).to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
    fmt.extend_from_slice(&block.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    if extensible {
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt.extend_from_slice(&0u32.to_le_bytes());
        fmt.extend_from_slice(&u32::from(codec).to_le_bytes());
        fmt.extend_from_slice(&[0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]);
    }
    let mut bytes = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
    bytes.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&fmt);
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
    bytes.extend_from_slice(data);
    if data.len() % 2 == 1 {
        bytes.push(0);
    }
    let length = bytes.len() as u32 - 8;
    bytes[4..8].copy_from_slice(&length.to_le_bytes());
    bytes
}

#[test]
fn wav_scaling_downmix_and_extensible_subformats() {
    for extensible in [false, true] {
        let pcm: Vec<_> = [-32768i16, 32767, 16384, 0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let decoded = decode_wav(&wav(1, 16, 8000, 2, &pcm, extensible)).unwrap();
        assert_eq!(decoded.samples, [-1.0 / 65536.0, 0.25]);
        assert_eq!(decoded.duration_ms(), 1);
        let floats: Vec<_> = [2.0f32, 4.0, -2.0, -4.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let decoded = decode_wav(&wav(3, 32, 192000, 2, &floats, extensible)).unwrap();
        assert_eq!(decoded.samples, [3.0, -3.0]);
    }
    let pcm = [0; 16];
    assert_eq!(
        decode_wav(&wav(1, 16, 16000, 8, &pcm, false))
            .unwrap()
            .samples,
        [0.0]
    );
}

#[test]
fn c5_refusals_are_invalid_audio_before_features() {
    let frontend = Frontend::new();
    let mut refused = vec![];
    for bits in [8, 24, 32, 64] {
        refused.push(wav(1, bits, 16000, 1, &[0; 8], false));
    }
    for codec in [0, 2, 6, 7, 17, 0xffff] {
        refused.push(wav(codec, 16, 16000, 1, &[0; 8], false));
    }
    for rate in [0, 7999, 192001, 1000000] {
        refused.push(wav(1, 16, rate, 1, &[0; 8], false));
    }
    for channels in [0, 9] {
        refused.push(wav(1, 16, 16000, channels, &[0; 18], false));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for extensible in [false, true] {
            refused.push(wav(3, 32, 16000, 1, &value.to_le_bytes(), extensible));
        }
    }
    refused.push(wav(1, 16, 16000, 1, &[], false));
    refused.push(wav(3, 16, 16000, 1, &[0; 2], false));
    refused.push(wav(1, 16, 16000, 2, &[0; 2], false));
    let good = wav(1, 16, 16000, 1, &[0; 4], false);
    for len in 0..good.len() {
        refused.push(good[..len].to_vec());
    }
    for (offset, bytes) in [
        (0, b"RIFX".as_slice()),
        (8, b"AVI "),
        (16, &[255; 4]),
        (28, &[0; 4]),
        (32, &[1, 0]),
    ] {
        let mut bad = good.clone();
        bad[offset..offset + bytes.len()].copy_from_slice(bytes);
        refused.push(bad);
    }
    let ext = wav(1, 16, 16000, 1, &[0; 2], true);
    for offset in [36, 38, 44, 48, 52, 56] {
        let mut bad = ext.clone();
        bad[offset] ^= 0xff;
        refused.push(bad);
    }
    // Even a well-formed data chunk cannot hide a malformed trailing chunk.
    let mut trailing = good.clone();
    trailing.extend_from_slice(b"junk");
    let length = trailing.len() as u32 - 8;
    trailing[4..8].copy_from_slice(&length.to_le_bytes());
    refused.push(trailing);
    for (index, input) in refused.iter().enumerate() {
        assert!(
            matches!(frontend.process_wav(input), Err(Error::InvalidAudio(_))),
            "case {index}"
        );
    }
}

#[test]
fn riff_unknown_chunks_padding_duplicates_and_extensions() {
    let mut bytes = wav(1, 16, 16000, 1, &[0; 2], false);
    bytes.extend_from_slice(b"JUNK\x01\0\0\0x\0");
    let length = bytes.len() as u32 - 8;
    bytes[4..8].copy_from_slice(&length.to_le_bytes());
    assert!(decode_wav(&bytes).is_ok());
    bytes.pop();
    let length = bytes.len() as u32 - 8;
    bytes[4..8].copy_from_slice(&length.to_le_bytes());
    assert!(decode_wav(&bytes).is_err());
    let good = wav(1, 16, 16000, 1, &[0; 2], false);
    for chunk in [&good[12..36], &good[36..]] {
        let mut duplicate = good.clone();
        duplicate.extend_from_slice(chunk);
        let length = duplicate.len() as u32 - 8;
        duplicate[4..8].copy_from_slice(&length.to_le_bytes());
        assert!(decode_wav(&duplicate).is_err());
    }
}

#[test]
fn resampler_length_zero_edges_and_identity() {
    assert_eq!(resample(&[1.0, -2.0], 16000).unwrap(), [1.0, -2.0]);
    for rate in [8000, 44100, 48000, 192000] {
        assert_eq!(
            resample(&[0.0; 17], rate).unwrap(),
            vec![0.0; (17u64 * 16000).div_ceil(u64::from(rate)) as usize]
        );
    }
    // With a one-sample input every other source index contributes zero, not
    // a replicated edge or a normalized interpolation kernel.
    let output = resample(&[1.0], 8000).unwrap();
    assert_eq!(output[0], 0.99);
    let d = -0.5f64 * 0.99;
    let expected = 0.99 * (std::f64::consts::PI * d).sin() / (std::f64::consts::PI * d)
        * (d * std::f64::consts::PI / 12.0).cos().powi(2);
    assert!((output[1] - expected as f32).abs() < 1e-7);
    assert!(resample(&[f32::NAN], 16000).is_err());
}

struct Stub {
    tokens: Vec<f32>,
    durations: Vec<f32>,
    calls: usize,
    accepts: usize,
    resets: usize,
    positive_at_cap: bool,
    blank: bool,
}
impl Stub {
    fn new(positive_at_cap: bool, blank: bool) -> Self {
        let mut tokens = vec![-10.0; VOCAB_SIZE];
        tokens[if blank { BLANK as usize } else { 12 }] = 1.0;
        Self {
            tokens,
            durations: vec![0.0; 5],
            calls: 0,
            accepts: 0,
            resets: 0,
            positive_at_cap,
            blank,
        }
    }
}
impl Joint for Stub {
    fn reset(&mut self) -> Result<()> {
        self.calls = 0;
        self.accepts = 0;
        self.resets += 1;
        Ok(())
    }
    fn evaluate(&mut self, _: usize) -> Result<JointLogits<'_>> {
        self.calls += 1;
        self.durations.fill(-10.0);
        let d =
            usize::from(self.positive_at_cap && self.calls.is_multiple_of(MAX_SYMBOLS_PER_FRAME));
        self.durations[d] = 1.0;
        Ok(JointLogits {
            tokens: &self.tokens,
            durations: &self.durations,
        })
    }
    fn accept(&mut self, token: u32) -> Result<()> {
        assert!(!self.blank);
        assert_ne!(token, BLANK);
        self.accepts += 1;
        Ok(())
    }
}

#[test]
fn duration_zero_stub_emits_exactly_t_times_k_and_caps_each_frame() {
    let mut joint = Stub::new(false, false);
    let output = greedy(&mut joint, 3, true, |_| {}).unwrap();
    assert_eq!(joint.calls, 30);
    assert_eq!(joint.accepts, 30);
    assert_eq!(joint.resets, 1);
    assert_eq!(output.triples.len(), 30);
    for frame in 0..3 {
        assert_eq!(
            &output.triples[frame * 10..(frame + 1) * 10],
            &[[12, frame as u32, 0]; 10]
        );
        assert!(output.trace[frame * 10 + 9].cap_hit);
    }
    assert_eq!(output.trace.iter().filter(|x| x.cap_hit).count(), 3);
}

#[test]
fn duration_one_at_k_has_no_extra_advance_or_cap_hit() {
    let mut joint = Stub::new(true, false);
    let output = greedy(&mut joint, 3, true, |_| {}).unwrap();
    assert_eq!(joint.calls, 30);
    assert_eq!(output.triples[9], [12, 0, 1]);
    assert_eq!(output.triples[10], [12, 1, 0]);
    assert_eq!(output.triples[29], [12, 2, 1]);
    assert!(output.trace.iter().all(|x| !x.cap_hit));
}

#[test]
fn blank_preserves_predictor_ties_are_lowest_and_observe_is_passive() {
    let mut joint = Stub::new(false, true);
    let mut progress = vec![];
    let output = greedy(&mut joint, 251, true, |t| progress.push(t)).unwrap();
    assert!(output.triples.is_empty());
    assert_eq!(joint.accepts, 0);
    assert_eq!(joint.calls, 251);
    assert_eq!(progress, [125, 250]);
    let unobserved = greedy(&mut joint, 251, false, |_| {}).unwrap();
    assert_eq!(output.triples, unobserved.triples);
    assert_eq!(joint.calls, 251);
    assert!(unobserved.trace.is_empty());
    joint.tokens.fill(0.0);
    joint.blank = false;
    let tied = greedy(&mut joint, 1, true, |_| {}).unwrap();
    assert_eq!(tied.trace[0].token_top1, 0);
    assert_eq!(tied.trace[0].token_top2, 1);
    joint.tokens[5] = f32::NAN;
    assert!(matches!(
        greedy(&mut joint, 1, true, |_| {}),
        Err(Error::InvalidJoint(_))
    ));
    joint.tokens.clear();
    assert!(greedy(&mut joint, 1, false, |_| {}).is_err());
}

#[test]
fn words_clamp_final_frame_and_keep_zero_duration_pieces() {
    let tokenizer = Tokenizer::pinned();
    let word_id = (0..BLANK)
        .find(|&id| tokenizer.piece(id).unwrap() == "▁the")
        .unwrap();
    let piece = (0..BLANK)
        .find(|&id| tokenizer.piece(id).unwrap() == "s")
        .unwrap();
    let result = tokenizer
        .transcript(&[[word_id, 2, 0], [piece, 2, 0], [word_id, 3, 4]], 250.5)
        .unwrap();
    assert_eq!(result.text, "thes the");
    assert_eq!(
        result.words[0],
        Word {
            word: "thes".into(),
            start_ms: 160.0,
            end_ms: 240.0
        }
    );
    assert_eq!(result.words[1].end_ms, 250.5);
    assert!(tokenizer.transcript(&[], 0.0).unwrap().no_speech);
}

#[test]
fn parakeet_decode_groups_before_pad_filtering_without_skipping_other_specials() {
    let tokenizer = Tokenizer::pinned();
    let word = (0..BLANK)
        .find(|&id| tokenizer.piece(id).unwrap() == "▁the")
        .unwrap();
    assert_eq!(tokenizer.decode(&[word, word]).unwrap(), "the");
    assert_eq!(
        tokenizer.decode(&[word, PAD_TOKEN, word]).unwrap(),
        "the the"
    );
    assert_eq!(tokenizer.decode(&[PAD_TOKEN, word]).unwrap(), "the");
    assert_eq!(tokenizer.decode(&[0, 0]).unwrap(), "<unk>");
    assert!(tokenizer.decode(&[VOCAB_SIZE as u32]).is_err());
}

#[test]
fn frontend_masks_tail_and_preserves_reference_silence_rounding() {
    let frontend = Frontend::new();
    let features = frontend.process_samples(&vec![0.0; 16000]).unwrap();
    assert_eq!(features.mel_frames, 101);
    assert_eq!(features.valid_mel_frames, 100);
    assert_eq!(features.valid_encoder_frames, 13);
    for &value in &features.values[..100 * MEL_BINS] {
        assert!((value - -0.16005333).abs() < 1e-6);
    }
    assert!(features.values[100 * MEL_BINS..].iter().all(|&x| x == 0.0));
    for length in [1, 159, 160, 161] {
        let features = frontend.process_samples(&vec![0.0; length]).unwrap();
        assert!(features.values.iter().all(|x| x.is_finite()));
        assert_eq!(features.valid_mel_frames, length / 160);
    }
}
