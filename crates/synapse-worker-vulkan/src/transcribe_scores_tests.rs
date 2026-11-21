use super::*;

#[test]
fn transcribe_scores_stages_both_operands_in_shared_memory() {
    let shader = SHADERS
        .iter()
        .find(|(name, _)| *name == "transcribe_scores")
        .unwrap()
        .1;
    let code = ash::util::read_spv(&mut Cursor::new(shader)).unwrap();
    let mut offset = 5;
    let mut shared = 0;
    let mut barrier = false;
    while offset < code.len() {
        let instruction = code[offset];
        let opcode = instruction & 0xffff;
        // OpVariable's fourth word is the storage class; Workgroup is 4.
        if opcode == 59 && code[offset + 3] == 4 {
            shared += 1;
        }
        barrier |= opcode == 224; // OpControlBarrier
        offset += (instruction >> 16) as usize;
    }
    assert!(shared >= 2 && barrier, "score shader must fill and synchronize both shared tiles (found {shared} workgroup variables, barrier={barrier})");
}

fn legacy_pipeline(device: &Device) -> Result<vk::Pipeline> {
    // Use the production flags and the full old shader, not an extracted score
    // branch, so glslc sees the same surrounding speech operations as before.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let sdk = std::path::PathBuf::from(std::env::var_os("VULKAN_SDK").context("pinned build SDK")?);
    let temporary = tempfile::tempdir()?;
    let output = temporary.path().join("legacy.spv");
    let compilation =
        synapse_core::without_launch_nonce(std::process::Command::new(sdk.join("bin/glslc")))
            .args(["-O", "-fshader-stage=compute", "--target-env=vulkan1.2"])
            .arg(root.join("tests/transcribe_scores_legacy.comp"))
            .arg("-o")
            .arg(&output)
            .output()?;
    ensure!(
        compilation.status.success(),
        "compile legacy score shader: {}",
        String::from_utf8_lossy(&compilation.stderr)
    );
    compute_pipeline_from_spirv(&device.raw, device.pipeline_layout, &std::fs::read(output)?)
}

#[test]
#[ignore = "requires a software Vulkan adapter and the pinned Vulkan SDK"]
fn transcribe_scores_match_legacy_shader_byte_for_byte() -> Result<()> {
    let device = Device::software_for_tests()?;
    let mut tiled = SpeechBatch::new(Arc::clone(&device))?;
    let mut legacy = SpeechBatch::new(Arc::clone(&device))?;
    let unused = std::mem::replace(&mut legacy.scores, legacy_pipeline(&device)?);
    unsafe { device.raw.destroy_pipeline(unused, None) };
    // Keep both dispatches pending so the legacy run can use its original grid.
    legacy.dispatch_limit_override = Some(2);
    tiled.dispatch_limit_override = Some(2);

    let mut seed = 0xf140_40a1u32;
    for (seq, rows, window, offset) in [
        (1, 1, 1, 0),
        (16, 16, 16, 0),
        (19, 10, 15, 9),
        (33, 17, 31, 16),
        (37, 8, 16, 24),
        // A full 128-query block at seq 8,192, the size the encoder dispatches,
        // needs more than 65,535 workgroups, so the grid spills into a second
        // row. Some scores
        // remain unmasked in the second grid row, so its indices are exercised.
        (8192, 128, 70, 0),
    ] {
        let count = rows * 8 * seq;
        let output_size = (count as usize + 1) * 4;
        let plan = [
            ("q", seq * 1024),
            ("dummy", 1),
            ("scores", count + 1),
            ("k", seq * 1024),
            ("bias_u", 1024),
            ("bias_v", 1024),
            ("relative", (2 * seq - 1) * 1024),
        ];
        let buffers = plan.map(|(name, elements)| PlannedBuffer {
            name: name.into(),
            bytes: u64::from(elements) * 4,
            per_sub_batch: false,
        });
        let arena = Arena::new(Arc::clone(&device), &buffers, arena_bytes(&buffers))?;
        for (name, elements) in plan {
            if name == "scores" || name == "dummy" {
                continue;
            }
            let mut bytes = Vec::with_capacity(elements as usize * 4);
            for _ in 0..elements {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = ((seed >> 8) as f32 - 8_388_608.0) / 13_000_003.0;
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            arena.transfer(name, &mut bytes, true)?;
        }
        let params = Params {
            op: SPEECH_SCORE_OP,
            rows,
            seq,
            offset,
            window,
            ..Default::default()
        };
        let bindings = plan.map(|(name, _)| name);
        let mut expected = vec![0x5a; output_size];
        let mut actual = vec![0xa5; output_size];
        arena.transfer("scores", &mut expected, true)?;
        legacy.push(&arena, params, bindings, count)?;
        legacy.pending[0].groups = count.div_ceil(64);
        legacy.flush()?;
        arena.transfer("scores", &mut expected, false)?;
        arena.transfer("scores", &mut actual, true)?;
        tiled.push(&arena, params, bindings, count)?;
        tiled.flush()?;
        arena.transfer("scores", &mut actual, false)?;

        let end = output_size - 4;
        assert_eq!(&expected[end..], &[0x5a; 4], "legacy wrote past output");
        assert_eq!(&actual[end..], &[0xa5; 4], "tiled wrote past output");
        let expected = expected[..end].as_chunks::<4>().0;
        let actual = actual[..end].as_chunks::<4>().0;
        if let Some(i) = expected.iter().zip(actual).position(|(a, b)| a != b) {
            let query = (i as u32) / (8 * seq) + offset;
            let head = (i as u32 / seq) % 8;
            let key = i as u32 % seq;
            panic!("seq={seq} rows={rows} window={window} offset={offset}, query={query} head={head} key={key}: legacy={:#010x} tiled={:#010x}", u32::from_le_bytes(expected[i]), u32::from_le_bytes(actual[i]));
        }
    }
    Ok(())
}
