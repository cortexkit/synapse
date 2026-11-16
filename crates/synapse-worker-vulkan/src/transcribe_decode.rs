//! The Parakeet predictor and joint on the GPU, one submission per decode step.
//!
//! The greedy decode rule stays on the host, in `synapse_speech::greedy`. Each
//! `evaluate` submits the predictor updates that `reset` and `accept` left
//! pending, the joint head for one encoder frame and a reduction to the head's
//! two best tokens. It reads back only those two tokens with their logits and
//! the five duration logits, never the 8,198-wide head output. The LSTM state
//! and the encoder projection stay on the device.
use super::{compute_pipeline, Arena, Device};
use anyhow::{ensure, Context, Result};
use ash::vk;
use std::sync::Arc;
use synapse_speech::{Error, Joint, JointLogits, BLANK, DURATIONS, VOCAB_SIZE};

const HIDDEN: usize = 640;
const HEAD_ROWS: usize = VOCAB_SIZE + DURATIONS.len();

/// Encoder scratch buffers that are idle while the decoder runs. Each
/// request's encoder overwrites them, so every decode first clears the state.
pub(super) const STATE_BUFFER: &str = "q";
pub(super) const RESULT_BUFFER: &str = "k";

// State-buffer layout, in f32 elements.
/// The hidden state of `layer`. Two copies alternate so that a layer never
/// overwrites the state that other workgroups are still reading.
const fn hidden(copy: usize, layer: usize) -> usize {
    (copy * 2 + layer) * HIDDEN
}
/// Cell states of layers 0 and 1, updated in place.
const CELL: usize = 4 * HIDDEN;
const PREDICTOR: usize = 6 * HIDDEN;
const LOGITS: usize = 7 * HIDDEN;
const STATE_ELEMENTS: usize = LOGITS + HEAD_ROWS;

/// Words written by the top-two reduction: first and second token, their
/// logits' bits and a flag for any non-finite token logit.
const TOP_TWO_WORDS: usize = 5;
/// Byte offset of the five duration logits in the readback buffer.
const DURATION_BYTES: usize = 32;
const READBACK_BYTES: u64 = 64;

const OP_LSTM: u32 = 120;
const OP_PROJECTION: u32 = 121;
const OP_HEAD: u32 = 122;
const OP_TOP_TWO: u32 = 123;
/// Rows of a matrix operation handled by one 64-thread workgroup.
const ROWS_PER_GROUP: usize = 4;

/// Push constants of shaders/transcribe_decode.comp, in declaration order.
#[derive(Clone, Copy, Default)]
struct StepParams {
    op: u32,
    rows: u32,
    input_offset: u32,
    input_in_state: u32,
    recurrent: u32,
    hidden: u32,
    cell: u32,
    predictor: u32,
    output_offset: u32,
}

impl StepParams {
    fn bytes(self) -> Vec<u8> {
        [
            self.op,
            self.rows,
            self.input_offset,
            self.input_in_state,
            self.recurrent,
            self.hidden,
            self.cell,
            self.predictor,
            self.output_offset,
        ]
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect()
    }
}

/// Descriptor sets, one per kind of dispatch; their bindings never change.
#[derive(Clone, Copy)]
enum Set {
    Layer0 = 0,
    Layer1,
    Projection,
    Head,
    TopTwo,
}

/// The value of every token logit that the device did not report. It is below
/// any finite second-best logit, or equal to it only when every logit but the
/// best already has this value.
const FILLER: f32 = f32::MIN;

/// A token-logit vector rebuilt from the device's two best tokens, so that the
/// shared decode rule picks exactly those two tokens with their logits.
pub(super) struct TokenScores {
    logits: Vec<f32>,
    marked: [usize; 2],
}

impl TokenScores {
    pub(super) fn new() -> Self {
        Self {
            logits: vec![FILLER; VOCAB_SIZE],
            marked: [0, 1],
        }
    }

    /// Rebuilds the vector from the top-two words. A non-finite logit becomes
    /// a NaN, which the shared rule refuses as it would the full vector.
    pub(super) fn publish(&mut self, words: [u32; TOP_TWO_WORDS]) -> synapse_speech::Result<()> {
        for index in self.marked {
            self.logits[index] = FILLER;
        }
        let [first, second, first_bits, second_bits, nonfinite] = words;
        if nonfinite != 0 {
            self.logits[0] = f32::NAN;
            self.marked = [0, 0];
            return Ok(());
        }
        let (first, second) = (first as usize, second as usize);
        let (high, low) = (f32::from_bits(first_bits), f32::from_bits(second_bits));
        let ordered = high > low || (high == low && first < second);
        if first >= VOCAB_SIZE
            || second >= VOCAB_SIZE
            || first == second
            || !ordered
            || !high.is_finite()
            || !low.is_finite()
        {
            self.marked = [0, 1];
            return Err(Error::InvalidJoint("device top two is inconsistent"));
        }
        self.logits[first] = high;
        self.logits[second] = low;
        self.marked = [first, second];
        Ok(())
    }

    pub(super) fn logits(&self) -> &[f32] {
        &self.logits
    }
}

pub(super) struct GpuJoint {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    arena_buffer: vk::Buffer,
    state: vk::DescriptorBufferInfo,
    result: vk::DescriptorBufferInfo,
    readback: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    mapped: *const u8,
    /// Valid encoder frames of the current request.
    pub(super) frames: usize,
    /// Which hidden-state copy holds the current state.
    copy: usize,
    clear: bool,
    pending: Vec<u32>,
    scores: TokenScores,
    durations: [f32; DURATIONS.len()],
    /// The device failure behind the last `InvalidJoint` error, if any.
    device_error: Option<anyhow::Error>,
}

impl Drop for GpuJoint {
    fn drop(&mut self) {
        // Every submission waited for the queue to go idle, so nothing here is
        // still in use. Destroying null handles is a no-op in Vulkan.
        unsafe {
            let raw = &self.device.raw;
            raw.destroy_pipeline(self.pipeline, None);
            raw.destroy_descriptor_pool(self.pool, None);
            raw.destroy_buffer(self.readback, None);
            raw.free_memory(self.readback_memory, None);
        }
    }
}

// SAFETY: `mapped` points into an allocation that this value owns and frees,
// and it is only read through `&mut self`; moving the value to another thread
// moves that exclusive access with it.
unsafe impl Send for GpuJoint {}

impl GpuJoint {
    /// Prepares the GPU decoder over the arena's decoder weights, projected
    /// encoder output and idle scratch buffers.
    pub(super) fn new(arena: &Arena) -> Result<Self> {
        let device = Arc::clone(&arena.device);
        let state = arena.descriptor(STATE_BUFFER)?;
        let result = arena.descriptor(RESULT_BUFFER)?;
        ensure!(
            state.range >= (STATE_ELEMENTS * 4) as u64
                && result.range >= (TOP_TWO_WORDS * 4) as u64,
            "decoder scratch buffers are too small"
        );
        let mut joint = Self {
            device: Arc::clone(&device),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            arena_buffer: arena.buffer,
            state,
            result,
            readback: vk::Buffer::null(),
            readback_memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null(),
            frames: 0,
            copy: 0,
            clear: true,
            pending: Vec::new(),
            scores: TokenScores::new(),
            durations: [0.; DURATIONS.len()],
            device_error: None,
        };
        let raw = &device.raw;
        joint.pipeline = compute_pipeline(raw, device.pipeline_layout, "transcribe_decode")?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(7 * 5)];
        joint.pool = unsafe {
            raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(5)
                    .pool_sizes(&sizes),
                None,
            )
        }?;
        let weight = |name: &str| format!("weight:{name}");
        let layer = |index: usize| {
            [
                if index == 0 {
                    weight("decoder.embedding.weight")
                } else {
                    "dummy".into()
                },
                weight(&format!("decoder.lstm.weight_ih_l{index}")),
                STATE_BUFFER.into(),
                weight(&format!("decoder.lstm.weight_hh_l{index}")),
                weight(&format!("decoder.lstm.bias_ih_l{index}")),
                weight(&format!("decoder.lstm.bias_hh_l{index}")),
                "dummy".into(),
            ]
        };
        let bindings: [[String; 7]; 5] = [
            layer(0),
            layer(1),
            [
                "dummy".into(),
                weight("decoder.decoder_projector.weight"),
                STATE_BUFFER.into(),
                "dummy".into(),
                weight("decoder.decoder_projector.bias"),
                "dummy".into(),
                "dummy".into(),
            ],
            [
                "projected".into(),
                weight("joint.head.weight"),
                STATE_BUFFER.into(),
                "dummy".into(),
                weight("joint.head.bias"),
                "dummy".into(),
                "dummy".into(),
            ],
            [
                "dummy".into(),
                "dummy".into(),
                STATE_BUFFER.into(),
                "dummy".into(),
                "dummy".into(),
                "dummy".into(),
                RESULT_BUFFER.into(),
            ],
        ];
        let layouts = [device.descriptor_layout; 5];
        joint.sets = unsafe {
            raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(joint.pool)
                    .set_layouts(&layouts),
            )
        }?;
        for (names, &set) in bindings.iter().zip(&joint.sets) {
            let descriptors = names
                .iter()
                .map(|name| arena.descriptor(name))
                .collect::<Result<Vec<_>>>()?;
            let writes: Vec<_> = descriptors
                .iter()
                .enumerate()
                .map(|(binding, descriptor)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(descriptor))
                })
                .collect();
            unsafe { raw.update_descriptor_sets(&writes, &[]) };
        }
        joint.readback = unsafe {
            raw.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(READBACK_BYTES)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let requirements = unsafe { raw.get_buffer_memory_requirements(joint.readback) };
        let properties = unsafe {
            device
                .instance
                .raw
                .get_physical_device_memory_properties(device.physical)
        };
        let visible =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        // Prefer cached host memory: the host reads this buffer every step.
        let memory_type = properties.memory_types[..properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                requirements.memory_type_bits & (1 << i) != 0 && m.property_flags.contains(visible)
            })
            .max_by_key(|(_, m)| {
                m.property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_CACHED)
            })
            .context("no host-visible memory for the decode readback")?
            .0 as u32;
        joint.readback_memory = unsafe {
            raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }?;
        unsafe {
            raw.bind_buffer_memory(joint.readback, joint.readback_memory, 0)?;
            joint.mapped = raw
                .map_memory(
                    joint.readback_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
                .cast::<u8>()
                .cast_const();
        }
        Ok(joint)
    }

    /// Takes the device failure behind the last decode error, if there was one.
    pub(super) fn take_device_error(&mut self) -> Option<anyhow::Error> {
        self.device_error.take()
    }

    /// Records the pending predictor updates, the joint head for `frame` and
    /// the top-two reduction in one submission, waits for it and reads back the
    /// two best tokens and the duration logits.
    fn step(&mut self, frame: usize) -> Result<[u32; TOP_TWO_WORDS]> {
        let clear = std::mem::take(&mut self.clear);
        let tokens = std::mem::take(&mut self.pending);
        let start_copy = self.copy;
        let device = &self.device;
        let raw = &device.raw;
        let state = self.state;
        let result = self.result;
        let barrier = |command, source_stage, source, target_stage, target| unsafe {
            raw.cmd_pipeline_barrier(
                command,
                source_stage,
                target_stage,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(source)
                    .dst_access_mask(target)],
                &[],
                &[],
            );
        };
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let read_write = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
        let dispatch = |command, set: Set, params: StepParams, groups: usize| unsafe {
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                device.pipeline_layout,
                0,
                &[self.sets[set as usize]],
                &[],
            );
            raw.cmd_push_constants(
                command,
                device.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &params.bytes(),
            );
            raw.cmd_dispatch(command, groups as u32, 1, 1);
            barrier(
                command,
                compute,
                vk::AccessFlags::SHADER_WRITE,
                compute | transfer,
                read_write | vk::AccessFlags::TRANSFER_READ,
            );
        };
        device.submit(|command| unsafe {
            // Earlier submissions (the encoder, uploads) must be visible.
            barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE,
                compute | transfer,
                read_write | vk::AccessFlags::TRANSFER_WRITE,
            );
            if clear {
                // Zero both hidden-state copies and both cell states.
                raw.cmd_fill_buffer(
                    command,
                    self.arena_buffer,
                    state.offset,
                    (PREDICTOR * 4) as u64,
                    0,
                );
                barrier(
                    command,
                    transfer,
                    vk::AccessFlags::TRANSFER_WRITE,
                    compute,
                    read_write,
                );
            }
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            let mut copy = start_copy;
            for &token in &tokens {
                let next = 1 - copy;
                for layer in 0..2 {
                    dispatch(
                        command,
                        if layer == 0 { Set::Layer0 } else { Set::Layer1 },
                        StepParams {
                            op: OP_LSTM,
                            input_offset: if layer == 0 {
                                token * HIDDEN as u32
                            } else {
                                hidden(next, 0) as u32
                            },
                            input_in_state: u32::from(layer == 1),
                            recurrent: hidden(copy, layer) as u32,
                            hidden: hidden(next, layer) as u32,
                            cell: (CELL + layer * HIDDEN) as u32,
                            ..Default::default()
                        },
                        HIDDEN,
                    );
                }
                dispatch(
                    command,
                    Set::Projection,
                    StepParams {
                        op: OP_PROJECTION,
                        rows: HIDDEN as u32,
                        input_offset: hidden(next, 1) as u32,
                        input_in_state: 1,
                        output_offset: PREDICTOR as u32,
                        ..Default::default()
                    },
                    HIDDEN / ROWS_PER_GROUP,
                );
                copy = next;
            }
            dispatch(
                command,
                Set::Head,
                StepParams {
                    op: OP_HEAD,
                    rows: HEAD_ROWS as u32,
                    input_offset: (frame * HIDDEN) as u32,
                    predictor: PREDICTOR as u32,
                    output_offset: LOGITS as u32,
                    ..Default::default()
                },
                HEAD_ROWS.div_ceil(ROWS_PER_GROUP),
            );
            dispatch(
                command,
                Set::TopTwo,
                StepParams {
                    op: OP_TOP_TWO,
                    rows: VOCAB_SIZE as u32,
                    output_offset: LOGITS as u32,
                    ..Default::default()
                },
                1,
            );
            raw.cmd_copy_buffer(
                command,
                self.arena_buffer,
                self.readback,
                &[
                    vk::BufferCopy::default()
                        .src_offset(result.offset)
                        .dst_offset(0)
                        .size((TOP_TWO_WORDS * 4) as u64),
                    vk::BufferCopy::default()
                        .src_offset(state.offset + ((LOGITS + VOCAB_SIZE) * 4) as u64)
                        .dst_offset(DURATION_BYTES as u64)
                        .size((DURATIONS.len() * 4) as u64),
                ],
            );
            barrier(
                command,
                transfer,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::HOST,
                vk::AccessFlags::HOST_READ,
            );
        })?;
        // The predictor ran on the device; its state now lives in the other copy.
        if tokens.len() % 2 == 1 {
            self.copy = 1 - self.copy;
        }
        let mut bytes = [0u8; READBACK_BYTES as usize];
        // SAFETY: the mapping covers the whole coherent allocation, at least
        // READBACK_BYTES long, and the submission above has completed.
        unsafe { std::ptr::copy_nonoverlapping(self.mapped, bytes.as_mut_ptr(), bytes.len()) };
        let word =
            |index: usize| u32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
        for (index, duration) in self.durations.iter_mut().enumerate() {
            *duration = f32::from_bits(word(DURATION_BYTES / 4 + index));
        }
        Ok(std::array::from_fn(word))
    }

    #[cfg(test)]
    pub(super) fn debug_state(&self, arena: &Arena) -> Result<Vec<f32>> {
        let mut bytes = vec![0; STATE_ELEMENTS * 4];
        arena.transfer(STATE_BUFFER, &mut bytes, false)?;
        Ok(synapse_core::decode_f32_frame(&bytes)?)
    }

    /// The current predictor output.
    #[cfg(test)]
    pub(super) fn debug_predictor(&self, arena: &Arena) -> Result<Vec<f32>> {
        Ok(self.debug_state(arena)?[PREDICTOR..PREDICTOR + HIDDEN].to_vec())
    }

    /// The head output of the last evaluation.
    #[cfg(test)]
    pub(super) fn debug_head(&self, arena: &Arena) -> Result<Vec<f32>> {
        Ok(self.debug_state(arena)?[LOGITS..].to_vec())
    }

    /// Replaces the predictor output and drops pending predictor updates, so
    /// that the next evaluation runs the joint head on exactly this vector.
    #[cfg(test)]
    pub(super) fn debug_set_predictor(&mut self, arena: &Arena, predictor: &[f32]) -> Result<()> {
        let mut state = self.debug_state(arena)?;
        state[PREDICTOR..PREDICTOR + HIDDEN].copy_from_slice(predictor);
        let mut bytes = synapse_core::encode_f32_frame(&state);
        arena.transfer(STATE_BUFFER, &mut bytes, true)?;
        self.pending.clear();
        self.clear = false;
        Ok(())
    }

    /// Runs only the top-two reduction over `tokens` as the token logits.
    #[cfg(test)]
    pub(super) fn debug_top_two(
        &mut self,
        arena: &Arena,
        tokens: &[f32],
    ) -> Result<[u32; TOP_TWO_WORDS]> {
        assert_eq!(tokens.len(), VOCAB_SIZE);
        let mut state = self.debug_state(arena)?;
        state[LOGITS..LOGITS + VOCAB_SIZE].copy_from_slice(tokens);
        let mut bytes = synapse_core::encode_f32_frame(&state);
        arena.transfer(STATE_BUFFER, &mut bytes, true)?;
        let raw = &self.device.raw;
        self.device.submit(|command| unsafe {
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            raw.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            raw.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.device.pipeline_layout,
                0,
                &[self.sets[Set::TopTwo as usize]],
                &[],
            );
            raw.cmd_push_constants(
                command,
                self.device.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &StepParams {
                    op: OP_TOP_TWO,
                    rows: VOCAB_SIZE as u32,
                    output_offset: LOGITS as u32,
                    ..Default::default()
                }
                .bytes(),
            );
            raw.cmd_dispatch(command, 1, 1, 1);
            raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ)],
                &[],
                &[],
            );
        })?;
        let mut words = vec![0; TOP_TWO_WORDS * 4];
        arena.transfer(RESULT_BUFFER, &mut words, false)?;
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes(words[i * 4..i * 4 + 4].try_into().unwrap())
        }))
    }
}

impl Joint for GpuJoint {
    fn reset(&mut self) -> synapse_speech::Result<()> {
        // The zeroed state and the initial blank prediction run at the start
        // of the first evaluation's submission.
        self.clear = true;
        self.copy = 0;
        self.pending.clear();
        self.pending.push(BLANK);
        self.device_error = None;
        Ok(())
    }

    fn evaluate(&mut self, frame: usize) -> synapse_speech::Result<JointLogits<'_>> {
        if frame >= self.frames {
            return Err(Error::InvalidJoint("encoder frame out of range"));
        }
        let words = self.step(frame).map_err(|error| {
            self.device_error = Some(error);
            Error::InvalidJoint("device decode step failed")
        })?;
        self.scores.publish(words)?;
        Ok(JointLogits {
            tokens: self.scores.logits(),
            durations: &self.durations,
        })
    }

    fn accept(&mut self, token: u32) -> synapse_speech::Result<()> {
        if token >= BLANK {
            return Err(Error::InvalidJoint("invalid emitted token"));
        }
        self.pending.push(token);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcribe::joint::HostJoint;
    use std::collections::BTreeMap;
    use synapse_speech::TraceStep;

    /// The shader's dot product for row `r` of a `rows`-row matrix, operation
    /// for operation: sixteen strided partial sums, then partials j and j+8
    /// added, then those eight sums; rows past the last multiple of four in
    /// plain column order.
    fn shader_dot(row: &[f32], input: &[f32], r: usize, rows: usize) -> f32 {
        if r >= rows / 4 * 4 {
            return row.iter().zip(input).fold(0., |sum, (w, x)| sum + w * x);
        }
        let mut partial = [0f32; 16];
        for (k, (w, x)) in row.iter().zip(input).enumerate() {
            partial[k % 16] += w * x;
        }
        let mut total = partial[0] + partial[8];
        for j in 1..8 {
            total += partial[j] + partial[8 + j];
        }
        total
    }

    /// The shader's top-two reduction: 64 threads scan strided indices, then a
    /// pairwise tree merges their pairs.
    fn shader_top_two(values: &[f32]) -> [u32; TOP_TWO_WORDS] {
        let better = |a: f32, ia: u32, b: f32, ib: u32| a > b || (a == b && ia < ib);
        let floor = f32::from_bits(0xff7f_ffff);
        let mut lanes = vec![(floor, u32::MAX, floor, u32::MAX, false); 64];
        for (lane, state) in lanes.iter_mut().enumerate() {
            for i in (lane..values.len()).step_by(64) {
                let (v, i) = (values[i], i as u32);
                state.4 |= !v.is_finite();
                if better(v, i, state.0, state.1) {
                    (state.2, state.3) = (state.0, state.1);
                    (state.0, state.1) = (v, i);
                } else if better(v, i, state.2, state.3) {
                    (state.2, state.3) = (v, i);
                }
            }
        }
        let mut half = 32;
        while half > 0 {
            for lane in 0..half {
                let (a, b) = (lanes[lane], lanes[lane + half]);
                let merged = if better(a.0, a.1, b.0, b.1) {
                    if better(a.2, a.3, b.0, b.1) {
                        (a.0, a.1, a.2, a.3)
                    } else {
                        (a.0, a.1, b.0, b.1)
                    }
                } else if better(a.0, a.1, b.2, b.3) {
                    (b.0, b.1, a.0, a.1)
                } else {
                    (b.0, b.1, b.2, b.3)
                };
                lanes[lane] = (merged.0, merged.1, merged.2, merged.3, a.4 | b.4);
            }
            half /= 2;
        }
        let best = lanes[0];
        [
            best.1,
            best.3,
            best.0.to_bits(),
            best.2.to_bits(),
            u32::from(best.4),
        ]
    }

    /// A joint that returns fixed logits, to read the shared rule's choice.
    struct Fixed {
        tokens: Vec<f32>,
    }
    impl Joint for Fixed {
        fn reset(&mut self) -> synapse_speech::Result<()> {
            Ok(())
        }
        fn evaluate(&mut self, _: usize) -> synapse_speech::Result<JointLogits<'_>> {
            // The largest duration advances past the only frame.
            Ok(JointLogits {
                tokens: &self.tokens,
                durations: &[0., 0., 0., 0., 1.],
            })
        }
        fn accept(&mut self, _: u32) -> synapse_speech::Result<()> {
            Ok(())
        }
    }

    /// The shared decode rule's trace step for one token-logit vector.
    fn shared_rule(tokens: &[f32]) -> synapse_speech::Result<TraceStep> {
        let mut joint = Fixed {
            tokens: tokens.to_vec(),
        };
        Ok(synapse_speech::greedy(&mut joint, 1, true, |_| {})?.trace[0].clone())
    }

    fn assert_same_choice(words: [u32; TOP_TWO_WORDS], step: &TraceStep, case: &str) {
        assert_eq!(words[4], 0, "{case}: unexpected non-finite flag");
        assert_eq!(
            (words[0], words[1], words[2], words[3]),
            (
                step.token_top1,
                step.token_top2,
                step.token_logit1.to_bits(),
                step.token_logit2.to_bits()
            ),
            "{case}"
        );
    }

    struct Random(u32);
    impl Random {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 as f64 / u32::MAX as f64 * 2. - 1.) as f32
        }
        fn vector(&mut self, count: usize, scale: f32) -> Vec<f32> {
            (0..count).map(|_| self.next() * scale).collect()
        }
    }

    /// Token-logit vectors with ties for the best and second-best logits in
    /// the same and in different reduction lanes, signed zeros and extremes.
    fn tie_cases() -> Vec<(&'static str, Vec<f32>)> {
        let mut random = Random(0x7e57_0001);
        let base = random.vector(VOCAB_SIZE, 1.);
        let with = |pairs: &[(usize, f32)]| {
            let mut v = base.clone();
            for &(i, x) in pairs {
                v[i] = x;
            }
            v
        };
        let mut zeros = vec![-1f32; VOCAB_SIZE];
        zeros[300] = 0.;
        zeros[200] = -0.;
        let mut floor = vec![f32::MIN; VOCAB_SIZE];
        floor[4097] = 2.;
        vec![
            ("random", base.clone()),
            (
                "best tie across lanes",
                with(&[(8192, 5.), (7, 5.), (4000, 5.)]),
            ),
            (
                "best tie in one lane",
                with(&[(130, 5.), (66, 5.), (2, 5.)]),
            ),
            (
                "second tie",
                with(&[(100, 9.), (8191, 3.), (128, 3.), (64, 3.)]),
            ),
            ("blank and last token tie", with(&[(8192, 7.), (8191, 7.)])),
            ("all equal", vec![0.5; VOCAB_SIZE]),
            ("signed zeros", zeros),
            ("most negative floor", floor),
            ("all most negative", vec![f32::MIN; VOCAB_SIZE]),
        ]
    }

    #[test]
    fn transcribe_decode_dot_product_order_matches_the_host_kernel_on_real_shapes() {
        let mut random = Random(0x5eed_1234);
        let gemv = crate::transcribe_gemv::Gemv::detect();
        #[cfg(target_arch = "x86_64")]
        let avx = std::arch::is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;
        for rows in [2560, 640, HEAD_ROWS] {
            let matrix = random.vector(rows * HIDDEN, 1.);
            let input = random.vector(HIDDEN, 1.);
            let mut host = vec![0.; rows];
            gemv.apply(&matrix, &input, &mut host);
            for (r, row) in matrix.as_chunks::<HIDDEN>().0.iter().enumerate() {
                let device = shader_dot(row, &input, r, rows);
                let exact: f64 = row
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum();
                let magnitude: f64 = row
                    .iter()
                    .zip(&input)
                    .map(|(w, x)| (f64::from(*w) * f64::from(*x)).abs())
                    .sum();
                assert!(
                    (f64::from(device) - exact).abs() <= 8. * f64::from(f32::EPSILON) * magnitude,
                    "{rows} rows, row {r}: {device} vs {exact}"
                );
                // On x86-64 with AVX the host kernel sums in the same order, so
                // the device and host results are the same float.
                if avx {
                    assert_eq!(device.to_bits(), host[r].to_bits(), "{rows} rows, row {r}");
                }
            }
        }
    }

    #[test]
    fn transcribe_decode_top_two_matches_the_shared_rule_on_constructed_ties() {
        for (case, tokens) in tie_cases() {
            let step = shared_rule(&tokens).unwrap();
            let words = shader_top_two(&tokens);
            assert_same_choice(words, &step, case);
            // The rebuilt vector leads the shared rule to the same step.
            let mut scores = TokenScores::new();
            scores.publish(words).unwrap();
            let rebuilt = shared_rule(scores.logits()).unwrap();
            assert_same_choice(words, &rebuilt, case);
            assert_eq!(rebuilt.emitted, step.emitted, "{case}");
        }
        // Lowest index wins on equal values.
        let ties = &tie_cases()[1].1;
        assert_eq!(shader_top_two(ties)[..2], [7, 4000]);
    }

    #[test]
    fn transcribe_decode_scores_reuse_one_vector_and_refuse_bad_words() {
        let mut scores = TokenScores::new();
        for (case, tokens) in tie_cases() {
            let words = shader_top_two(&tokens);
            scores.publish(words).unwrap();
            // Only the two reported entries differ from the filler.
            let set: Vec<_> = scores
                .logits()
                .iter()
                .enumerate()
                .filter(|(_, v)| v.to_bits() != FILLER.to_bits())
                .map(|(i, _)| i as u32)
                .collect();
            assert!(
                set.iter().all(|i| *i == words[0] || *i == words[1]),
                "{case}"
            );
        }
        let mut tokens = vec![0.25f32; VOCAB_SIZE];
        tokens[17] = f32::INFINITY;
        let words = shader_top_two(&tokens);
        assert_eq!(words[4], 1);
        scores.publish(words).unwrap();
        let error = shared_rule(scores.logits()).unwrap_err().to_string();
        assert_eq!(error, shared_rule(&tokens).unwrap_err().to_string());
        // The next publication restores the poisoned entry.
        scores.publish(shader_top_two(&[0.5; VOCAB_SIZE])).unwrap();
        assert!(scores.logits().iter().all(|v| v.is_finite()));
        for bad in [
            [3, 3, 1f32.to_bits(), 0f32.to_bits(), 0],
            [3, 4, 0f32.to_bits(), 1f32.to_bits(), 0],
            [4, 3, 1f32.to_bits(), 1f32.to_bits(), 0],
            [VOCAB_SIZE as u32, 3, 1f32.to_bits(), 0f32.to_bits(), 0],
            [3, 4, f32::NAN.to_bits(), 0f32.to_bits(), 0],
        ] {
            assert!(scores.publish(bad).is_err(), "{bad:?}");
        }
    }

    const TENSORS: [(&str, usize, f32); 11] = [
        ("decoder.embedding.weight", VOCAB_SIZE * HIDDEN, 1.),
        ("decoder.lstm.weight_ih_l0", 4 * HIDDEN * HIDDEN, 0.08),
        ("decoder.lstm.weight_hh_l0", 4 * HIDDEN * HIDDEN, 0.08),
        ("decoder.lstm.bias_ih_l0", 4 * HIDDEN, 0.1),
        ("decoder.lstm.bias_hh_l0", 4 * HIDDEN, 0.1),
        ("decoder.lstm.weight_ih_l1", 4 * HIDDEN * HIDDEN, 0.08),
        ("decoder.lstm.weight_hh_l1", 4 * HIDDEN * HIDDEN, 0.08),
        ("decoder.lstm.bias_ih_l1", 4 * HIDDEN, 0.1),
        ("decoder.lstm.bias_hh_l1", 4 * HIDDEN, 0.1),
        ("decoder.decoder_projector.weight", HIDDEN * HIDDEN, 0.08),
        ("decoder.decoder_projector.bias", HIDDEN, 0.1),
    ];

    struct Fixture {
        arena: Arena,
        gpu: GpuJoint,
        host: HostJoint,
        frames: usize,
    }

    /// The GPU and host decoders over the same random weights, on the first
    /// software Vulkan adapter.
    fn fixture() -> Fixture {
        use synapse_parity::vulkan::{arena_bytes, Buffer};
        let frames = 48;
        let mut random = Random(0x0dec_0de5);
        let mut weights: BTreeMap<String, Vec<f32>> = TENSORS
            .iter()
            .map(|(name, count, scale)| (name.to_string(), random.vector(*count, *scale)))
            .collect();
        weights.insert(
            "joint.head.weight".into(),
            random.vector(HEAD_ROWS * HIDDEN, 0.08),
        );
        let mut head_bias = random.vector(HEAD_ROWS, 0.1);
        // Make blanks common enough that both emitted and blank steps occur.
        head_bias[BLANK as usize] += 1.0;
        weights.insert("joint.head.bias".into(), head_bias);
        let projected = random.vector(frames * HIDDEN, 1.);
        let mut plan: Vec<Buffer> = weights
            .iter()
            .map(|(name, values)| Buffer {
                name: format!("weight:{name}"),
                bytes: values.len() as u64 * 4,
                per_sub_batch: false,
            })
            .collect();
        for (name, elements) in [
            ("projected", projected.len()),
            (STATE_BUFFER, STATE_ELEMENTS),
            (RESULT_BUFFER, 8),
            ("dummy", 1),
        ] {
            plan.push(Buffer {
                name: name.into(),
                bytes: elements as u64 * 4,
                per_sub_batch: false,
            });
        }
        let device = Device::software_for_tests().expect("software Vulkan adapter");
        let arena = Arena::new(device, &plan, arena_bytes(&plan)).unwrap();
        for (name, values) in &weights {
            let mut bytes = synapse_core::encode_f32_frame(values);
            arena
                .transfer(&format!("weight:{name}"), &mut bytes, true)
                .unwrap();
        }
        let mut bytes = synapse_core::encode_f32_frame(&projected);
        arena.transfer("projected", &mut bytes, true).unwrap();
        let mut gpu = GpuJoint::new(&arena).unwrap();
        gpu.frames = frames;
        Fixture {
            arena,
            gpu,
            host: HostJoint::from_weights(weights, projected),
            frames,
        }
    }

    fn max_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0., f32::max)
    }

    #[test]
    #[ignore = "requires a software Vulkan adapter (lavapipe) and VULKAN_SDK for the shaders"]
    fn transcribe_decode_steps_on_software_vulkan_match_the_host_decoder() {
        let Fixture {
            arena,
            mut gpu,
            mut host,
            frames,
        } = fixture();
        #[cfg(target_arch = "x86_64")]
        let avx = std::arch::is_x86_feature_detected!("avx");
        #[cfg(not(target_arch = "x86_64"))]
        let avx = false;

        // The joint head on the host's own predictor output: the same float
        // per row on an x86-64 AVX host, and the same two best tokens.
        host.reset().unwrap();
        for frame in [0, 5, frames - 1] {
            let host_step = {
                let logits = host.evaluate(frame).unwrap();
                shared_rule(logits.tokens).unwrap()
            };
            gpu.debug_set_predictor(&arena, host.predictor()).unwrap();
            let gpu_tokens = gpu.evaluate(frame).unwrap().tokens.to_vec();
            let gpu_head = gpu.debug_head(&arena).unwrap();
            if avx {
                for (row, (a, b)) in gpu_head.iter().zip(host.head()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "frame {frame} row {row}");
                }
            }
            println!(
                "head frame {frame}: max |gpu - host| = {:e}",
                max_difference(&gpu_head, host.head())
            );
            assert!(max_difference(&gpu_head, host.head()) <= 1e-5);
            assert!(max_difference(&gpu.durations, &host.head()[VOCAB_SIZE..]) <= 1e-5);
            let gpu_step = shared_rule(&gpu_tokens).unwrap();
            assert_eq!(
                (gpu_step.token_top1, gpu_step.token_top2),
                (host_step.token_top1, host_step.token_top2),
                "frame {frame}"
            );
            if avx {
                assert_eq!(gpu_step, host_step, "frame {frame}");
            }
        }

        // The predictor: initial blank, then a few emitted tokens.
        host.reset().unwrap();
        gpu.reset().unwrap();
        let mut worst = 0f32;
        for (frame, token) in [(0, 17u32), (1, 4000), (1, 8191), (2, 0), (3, 255)] {
            host.evaluate(frame).unwrap();
            gpu.evaluate(frame).unwrap();
            let difference =
                max_difference(&gpu.debug_predictor(&arena).unwrap(), host.predictor());
            worst = worst.max(difference);
            assert!(
                difference <= 2e-5,
                "frame {frame}: predictor differs by {difference}"
            );
            host.accept(token).unwrap();
            gpu.accept(token).unwrap();
        }
        println!("predictor: max |gpu - host| = {worst:e}");

        // Whole greedy decodes, twice to show that reset clears device state.
        for _ in 0..2 {
            let host_output = synapse_speech::greedy(&mut host, frames, true, |_| {}).unwrap();
            let started = std::time::Instant::now();
            let gpu_output = synapse_speech::greedy(&mut gpu, frames, true, |_| {}).unwrap();
            println!(
                "greedy: {} steps, {} tokens, {:.2} ms per step on this adapter",
                gpu_output.trace.len(),
                gpu_output.triples.len(),
                started.elapsed().as_secs_f64() * 1e3 / gpu_output.trace.len() as f64
            );
            assert_eq!(gpu_output.triples, host_output.triples);
            assert!(gpu_output.trace.iter().any(|s| s.emitted));
            assert!(gpu_output.trace.iter().any(|s| !s.emitted));
            assert_eq!(gpu_output.trace.len(), host_output.trace.len());
            for (a, b) in gpu_output.trace.iter().zip(&host_output.trace) {
                assert_eq!(
                    (
                        a.frame_index,
                        a.token_top1,
                        a.token_top2,
                        a.duration_top1,
                        a.duration_top2
                    ),
                    (
                        b.frame_index,
                        b.token_top1,
                        b.token_top2,
                        b.duration_top1,
                        b.duration_top2
                    )
                );
                for (x, y) in [
                    (a.token_logit1, b.token_logit1),
                    (a.token_logit2, b.token_logit2),
                    (a.duration_logit1, b.duration_logit1),
                    (a.duration_logit2, b.duration_logit2),
                ] {
                    assert!((x - y).abs() <= 1e-4, "{x} vs {y}");
                }
            }
        }
        assert!(gpu.evaluate(frames).is_err());
        assert!(gpu.accept(BLANK).is_err());

        // The device reduction on constructed ties.
        for (case, tokens) in tie_cases() {
            let words = gpu.debug_top_two(&arena, &tokens).unwrap();
            assert_same_choice(words, &shared_rule(&tokens).unwrap(), case);
        }
        let mut tokens = vec![0.25f32; VOCAB_SIZE];
        tokens[6000] = f32::NAN;
        assert_eq!(gpu.debug_top_two(&arena, &tokens).unwrap()[4], 1);
    }
}
