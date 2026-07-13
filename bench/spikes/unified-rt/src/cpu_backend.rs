use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::ValueEnum;
use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
use half::f16;
use rayon::prelude::*;
use serde::Serialize;

use super::{matmul_impl, BLayout};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum CpuGemm {
    /// The platform BLAS on macOS and the scalar compatibility path elsewhere.
    Platform,
    /// faer f32 GEMM using the model's resident f32 weights.
    Faer,
    /// Experimental f16 weight storage expanded to f32 before faer GEMM.
    FaerF16,
    /// Owned f16-weight microkernel with runtime AVX-512/AVX2 dispatch.
    Hand,
}

#[derive(Copy, Clone, Debug, Default, Serialize)]
pub(super) struct CpuProfile {
    pub(super) gemm_wall_s: f64,
    pub(super) gemm_calls: u64,
    pub(super) gemm_dispatches: u64,
    pub(super) static_pack_wall_s: f64,
    pub(super) static_pack_count: u64,
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum HandIsa {
    Avx512,
    Avx2,
    Scalar,
}

impl HandIsa {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
            {
                return Self::Avx512;
            }
            if std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("f16c")
                && std::is_x86_feature_detected!("fma")
            {
                return Self::Avx2;
            }
        }
        Self::Scalar
    }

    fn label(self) -> &'static str {
        match self {
            Self::Avx512 => "avx512f+f16c+fma",
            Self::Avx2 => "avx2+f16c+fma",
            Self::Scalar => "scalar",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct StaticRhsKey {
    address: usize,
    len: usize,
    n: usize,
    k: usize,
    layout: BLayout,
}

pub(super) struct CpuBackend {
    kind: CpuGemm,
    threads: usize,
    pool: rayon::ThreadPool,
    hand_isa: HandIsa,
    packed_static_rhs: HashMap<StaticRhsKey, Arc<[u16]>>,
    f32_scratch: Vec<f32>,
    profile: CpuProfile,
}

impl CpuBackend {
    pub(super) fn new(kind: CpuGemm, threads: Option<usize>) -> Result<Self> {
        let default_threads = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1)
            .div_ceil(2)
            .max(1);
        let threads = threads.unwrap_or(default_threads);
        anyhow::ensure!(threads > 0, "--cpu-threads must be at least one");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("owned-cpu-{index}"))
            .build()
            .context("build owned CPU thread pool")?;
        Ok(Self {
            kind,
            threads,
            pool,
            hand_isa: HandIsa::detect(),
            packed_static_rhs: HashMap::new(),
            f32_scratch: Vec::new(),
            profile: CpuProfile::default(),
        })
    }

    pub(super) fn name(&self) -> &'static str {
        match self.kind {
            CpuGemm::Platform => {
                if cfg!(target_os = "macos") {
                    "cpu-accelerate"
                } else {
                    "cpu-scalar"
                }
            }
            CpuGemm::Faer => "cpu-faer-f32",
            CpuGemm::FaerF16 => "cpu-faer-f16",
            CpuGemm::Hand => match self.hand_isa {
                HandIsa::Avx512 => "cpu-hand-f16-avx512",
                HandIsa::Avx2 => "cpu-hand-f16-avx2",
                HandIsa::Scalar => "cpu-hand-f16-scalar",
            },
        }
    }

    pub(super) fn details(&self) -> String {
        match self.kind {
            CpuGemm::Platform => format!("cpu_gemm=platform, cpu_threads={}", self.threads),
            CpuGemm::Faer => format!(
                "cpu_gemm=faer-f32, cpu_threads={}, faer_parallelism=rayon",
                self.threads
            ),
            CpuGemm::FaerF16 => format!(
                "cpu_gemm=faer-f16-storage, cpu_threads={}, faer_compute=f32, unpack=per-call",
                self.threads
            ),
            CpuGemm::Hand => format!(
                "cpu_gemm=hand-f16-storage, cpu_threads={}, hand_isa={}",
                self.threads,
                self.hand_isa.label()
            ),
        }
    }

    pub(super) fn reset_profile(&mut self) {
        self.profile = CpuProfile::default();
    }

    pub(super) fn profile(&self) -> CpuProfile {
        self.profile
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pack_attention_heads(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        q_heads: &mut [f32],
        k_heads: &mut [f32],
        v_heads: &mut [f32],
    ) {
        let hidden = heads * head_dim;
        let head_matrix_len = seq * head_dim;
        self.pool.install(|| {
            q_heads
                .par_chunks_mut(head_matrix_len)
                .zip(k_heads.par_chunks_mut(head_matrix_len))
                .zip(v_heads.par_chunks_mut(head_matrix_len))
                .enumerate()
                .for_each(|(head_batch, ((q_out, k_out), v_out))| {
                    let batch_index = head_batch / heads;
                    let head = head_batch % heads;
                    for position in 0..seq {
                        let source = (batch_index * seq + position) * hidden + head * head_dim;
                        let destination = position * head_dim;
                        q_out[destination..destination + head_dim]
                            .copy_from_slice(&q[source..source + head_dim]);
                        k_out[destination..destination + head_dim]
                            .copy_from_slice(&k[source..source + head_dim]);
                        v_out[destination..destination + head_dim]
                            .copy_from_slice(&v[source..source + head_dim]);
                    }
                });
        });
        debug_assert_eq!(q_heads.len(), batch * heads * head_matrix_len);
    }

    pub(super) fn masked_softmax(
        &self,
        scores: &mut [f32],
        attention_mask: &[u8],
        seq: usize,
        heads: usize,
        scale: f32,
    ) {
        self.pool.install(|| {
            scores
                .par_chunks_mut(seq)
                .enumerate()
                .for_each(|(score_row, row)| {
                    let head_batch = score_row / seq;
                    let batch_index = head_batch / heads;
                    for key_position in 0..seq {
                        row[key_position] = if attention_mask[batch_index * seq + key_position] == 0
                        {
                            -10_000.0
                        } else {
                            row[key_position] * scale
                        };
                    }
                    super::softmax(row);
                });
        });
    }

    pub(super) fn unpack_attention_heads(
        &self,
        context_heads: &[f32],
        batch: usize,
        seq: usize,
        heads: usize,
        head_dim: usize,
        context: &mut [f32],
    ) {
        let hidden = heads * head_dim;
        self.pool.install(|| {
            context
                .par_chunks_mut(hidden)
                .enumerate()
                .for_each(|(row, output)| {
                    let batch_index = row / seq;
                    let position = row % seq;
                    for head in 0..heads {
                        let source = ((batch_index * heads + head) * seq + position) * head_dim;
                        let destination = head * head_dim;
                        output[destination..destination + head_dim]
                            .copy_from_slice(&context_heads[source..source + head_dim]);
                    }
                });
        });
        debug_assert_eq!(context.len(), batch * seq * hidden);
    }

    pub(super) fn layer_norm(
        &self,
        hidden: usize,
        data: &mut [f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
    ) {
        self.pool.install(|| {
            data.par_chunks_mut(hidden).for_each(|row| {
                let mean = row.iter().copied().sum::<f32>() / hidden as f32;
                let variance = row
                    .iter()
                    .map(|value| {
                        let centered = *value - mean;
                        centered * centered
                    })
                    .sum::<f32>()
                    / hidden as f32;
                let inverse_stddev = 1.0 / (variance + eps).sqrt();
                for column in 0..hidden {
                    row[column] =
                        (row[column] - mean) * inverse_stddev * weight[column] + bias[column];
                }
            });
        });
    }

    pub(super) fn add_bias(&self, columns: usize, data: &mut [f32], bias: &[f32]) {
        self.pool.install(|| {
            data.par_chunks_mut(columns).for_each(|row| {
                for column in 0..columns {
                    row[column] += bias[column];
                }
            });
        });
    }

    pub(super) fn add_in_place(&self, destination: &mut [f32], source: &[f32]) {
        self.pool.install(|| {
            destination
                .par_iter_mut()
                .zip(source.par_iter())
                .for_each(|(destination, source)| *destination += *source);
        });
    }

    pub(super) fn gelu(&self, values: &mut [f32]) {
        self.pool.install(|| {
            values
                .par_iter_mut()
                .for_each(|value| *value = super::gelu(*value));
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
        static_rhs: bool,
    ) {
        let started = Instant::now();
        match self.kind {
            CpuGemm::Platform => matmul_impl(m, n, k, a, b, b_layout, c),
            CpuGemm::Faer => self.faer_f32(m, n, k, a, b, b_layout, c),
            CpuGemm::FaerF16 => {
                let packed = self.packed_rhs(n, k, b, b_layout, static_rhs);
                self.f32_scratch.clear();
                self.f32_scratch.reserve(packed.len());
                self.f32_scratch
                    .extend(packed.iter().map(|bits| f16::from_bits(*bits).to_f32()));
                let unpacked = &self.f32_scratch;
                faer_f32_in_pool(
                    &self.pool,
                    self.threads,
                    m,
                    n,
                    k,
                    a,
                    unpacked,
                    BLayout::RowMajorKn,
                    c,
                );
            }
            CpuGemm::Hand => {
                let packed = self.packed_rhs(n, k, b, b_layout, static_rhs);
                hand_f16_in_pool(
                    &self.pool,
                    self.threads,
                    self.hand_isa,
                    m,
                    n,
                    k,
                    a,
                    &packed,
                    c,
                );
            }
        }
        self.profile.gemm_wall_s += started.elapsed().as_secs_f64();
        self.profile.gemm_calls += 1;
        self.profile.gemm_dispatches += 1;
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_batched(
        &mut self,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) {
        let started = Instant::now();
        match self.kind {
            CpuGemm::Platform => {
                for batch in 0..batches {
                    matmul_impl(
                        m,
                        n,
                        k,
                        &a[batch * m * k..(batch + 1) * m * k],
                        &b[batch * n * k..(batch + 1) * n * k],
                        b_layout,
                        &mut c[batch * m * n..(batch + 1) * m * n],
                    );
                }
            }
            CpuGemm::Faer => {
                faer_f32_batched_in_pool(&self.pool, batches, m, n, k, a, b, b_layout, c)
            }
            CpuGemm::FaerF16 => {
                let packed = pack_rhs_f16_batched(&self.pool, batches, n, k, b, b_layout);
                self.f32_scratch.clear();
                self.f32_scratch.reserve(packed.len());
                self.f32_scratch
                    .extend(packed.iter().map(|bits| f16::from_bits(*bits).to_f32()));
                faer_f32_batched_in_pool(
                    &self.pool,
                    batches,
                    m,
                    n,
                    k,
                    a,
                    &self.f32_scratch,
                    BLayout::RowMajorKn,
                    c,
                );
            }
            CpuGemm::Hand => {
                let packed = pack_rhs_f16_batched(&self.pool, batches, n, k, b, b_layout);
                hand_f16_batched_in_pool(
                    &self.pool,
                    self.hand_isa,
                    batches,
                    m,
                    n,
                    k,
                    a,
                    &packed,
                    c,
                );
            }
        }
        self.profile.gemm_wall_s += started.elapsed().as_secs_f64();
        self.profile.gemm_calls += batches as u64;
        self.profile.gemm_dispatches += 1;
    }

    #[allow(clippy::too_many_arguments)]
    fn faer_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) {
        faer_f32_in_pool(&self.pool, self.threads, m, n, k, a, b, b_layout, c);
    }

    fn packed_rhs(
        &mut self,
        n: usize,
        k: usize,
        b: &[f32],
        layout: BLayout,
        static_rhs: bool,
    ) -> Arc<[u16]> {
        if !static_rhs {
            return pack_rhs_f16(n, k, b, layout);
        }
        let key = StaticRhsKey {
            address: b.as_ptr() as usize,
            len: b.len(),
            n,
            k,
            layout,
        };
        if let Some(packed) = self.packed_static_rhs.get(&key) {
            return Arc::clone(packed);
        }
        let started = Instant::now();
        let packed = pack_rhs_f16(n, k, b, layout);
        self.profile.static_pack_wall_s += started.elapsed().as_secs_f64();
        self.profile.static_pack_count += 1;
        self.packed_static_rhs.insert(key, Arc::clone(&packed));
        packed
    }
}

#[allow(clippy::too_many_arguments)]
fn faer_f32_in_pool(
    pool: &rayon::ThreadPool,
    threads: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    pool.install(|| faer_f32_kernel(m, n, k, a, b, b_layout, c, Par::rayon(threads)));
}

#[allow(clippy::too_many_arguments)]
fn faer_f32_batched_in_pool(
    pool: &rayon::ThreadPool,
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    pool.install(|| {
        c.par_chunks_mut(m * n)
            .enumerate()
            .take(batches)
            .for_each(|(batch, c)| {
                faer_f32_kernel(
                    m,
                    n,
                    k,
                    &a[batch * m * k..(batch + 1) * m * k],
                    &b[batch * n * k..(batch + 1) * n * k],
                    b_layout,
                    c,
                    Par::Seq,
                );
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn faer_f32_kernel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
    parallelism: Par,
) {
    let a = MatRef::from_row_major_slice(a, m, k);
    let b = match b_layout {
        BLayout::RowMajorKn => MatRef::from_row_major_slice(b, k, n),
        BLayout::RowMajorNkTransposed => MatRef::from_row_major_slice(b, n, k).transpose(),
    };
    let c = MatMut::from_row_major_slice_mut(c, m, n);
    matmul(c, Accum::Replace, a, b, 1.0f32, parallelism);
}

fn pack_rhs_f16_batched(
    pool: &rayon::ThreadPool,
    batches: usize,
    n: usize,
    k: usize,
    b: &[f32],
    layout: BLayout,
) -> Vec<u16> {
    let mut packed = vec![0u16; batches * k * n];
    pool.install(|| {
        packed
            .par_chunks_mut(k * n)
            .enumerate()
            .for_each(|(batch, destination)| {
                let source = &b[batch * n * k..(batch + 1) * n * k];
                pack_rhs_f16_into(n, k, source, layout, destination);
            });
    });
    packed
}

fn pack_rhs_f16(n: usize, k: usize, b: &[f32], layout: BLayout) -> Arc<[u16]> {
    let mut packed = vec![0u16; k * n];
    pack_rhs_f16_into(n, k, b, layout, &mut packed);
    packed.into()
}

fn pack_rhs_f16_into(n: usize, k: usize, b: &[f32], layout: BLayout, packed: &mut [u16]) {
    match layout {
        BLayout::RowMajorKn => {
            for (destination, source) in packed.iter_mut().zip(b) {
                *destination = f16::from_f32(*source).to_bits();
            }
        }
        BLayout::RowMajorNkTransposed => {
            for inner in 0..k {
                for output in 0..n {
                    packed[inner * n + output] = f16::from_f32(b[output * k + inner]).to_bits();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn hand_f16_in_pool(
    pool: &rayon::ThreadPool,
    threads: usize,
    isa: HandIsa,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
) {
    let rows_per_task = m.div_ceil(threads).max(4).next_multiple_of(4);
    pool.install(|| {
        c.par_chunks_mut(rows_per_task * n)
            .enumerate()
            .for_each(|(chunk_index, c_chunk)| {
                let first_row = chunk_index * rows_per_task;
                let chunk_rows = c_chunk.len() / n;
                let a_chunk = &a[first_row * k..(first_row + chunk_rows) * k];
                let vector_columns = match isa {
                    HandIsa::Avx512 => n / 16 * 16,
                    HandIsa::Avx2 => n / 8 * 8,
                    HandIsa::Scalar => 0,
                };
                run_hand_kernel(isa, chunk_rows, n, k, a_chunk, b, c_chunk);
                for row in 0..chunk_rows {
                    for column in vector_columns..n {
                        let mut sum = 0.0f32;
                        for inner in 0..k {
                            sum += a_chunk[row * k + inner]
                                * f16::from_bits(b[inner * n + column]).to_f32();
                        }
                        c_chunk[row * n + column] = sum;
                    }
                }
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn hand_f16_batched_in_pool(
    pool: &rayon::ThreadPool,
    isa: HandIsa,
    batches: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
) {
    pool.install(|| {
        c.par_chunks_mut(m * n)
            .enumerate()
            .take(batches)
            .for_each(|(batch, c)| {
                let a = &a[batch * m * k..(batch + 1) * m * k];
                let b = &b[batch * k * n..(batch + 1) * k * n];
                run_hand_kernel(isa, m, n, k, a, b, c);
                let vector_columns = match isa {
                    HandIsa::Avx512 => n / 16 * 16,
                    HandIsa::Avx2 => n / 8 * 8,
                    HandIsa::Scalar => 0,
                };
                for row in 0..m {
                    for column in vector_columns..n {
                        let mut sum = 0.0f32;
                        for inner in 0..k {
                            sum +=
                                a[row * k + inner] * f16::from_bits(b[inner * n + column]).to_f32();
                        }
                        c[row * n + column] = sum;
                    }
                }
            });
    });
}

#[allow(clippy::too_many_arguments)]
fn run_hand_kernel(
    isa: HandIsa,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", not(target_env = "msvc")))]
    unsafe {
        match isa {
            HandIsa::Avx512 => {
                synapse_hand_f16_gemm_avx512(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
            }
            HandIsa::Avx2 => {
                synapse_hand_f16_gemm_avx2(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr())
            }
            HandIsa::Scalar => {}
        }
    }
    #[cfg(not(all(target_arch = "x86_64", not(target_env = "msvc"))))]
    let _ = (isa, m, n, k, a, b, c);
}

#[cfg(all(target_arch = "x86_64", not(target_env = "msvc")))]
unsafe extern "C" {
    fn synapse_hand_f16_gemm_avx512(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const u16,
        c: *mut f32,
    );
    fn synapse_hand_f16_gemm_avx2(
        m: usize,
        n: usize,
        k: usize,
        a: *const f32,
        b: *const u16,
        c: *mut f32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patterned(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 17 % 31) as f32 - 15.0) * scale)
            .collect()
    }

    fn reference(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; m * n];
        matmul_impl(m, n, k, a, b, BLayout::RowMajorNkTransposed, &mut output);
        output
    }

    #[test]
    fn faer_matches_platform_for_transposed_rhs() {
        let (m, n, k) = (7, 19, 13);
        let a = patterned(m * k, 0.03125);
        let b = patterned(n * k, 0.015625);
        let expected = reference(m, n, k, &a, &b);
        let mut actual = vec![0.0; m * n];
        let mut backend = CpuBackend::new(CpuGemm::Faer, Some(2)).unwrap();
        backend.matmul(
            m,
            n,
            k,
            &a,
            &b,
            BLayout::RowMajorNkTransposed,
            &mut actual,
            true,
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn hand_kernel_tracks_f16_reference_across_vector_tails() {
        let (m, n, k) = (9, 23, 17);
        let a = patterned(m * k, 0.03125);
        let b = patterned(n * k, 0.015625);
        let quantized = b
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect::<Vec<_>>();
        let expected = reference(m, n, k, &a, &quantized);
        let mut actual = vec![0.0; m * n];
        let mut backend = CpuBackend::new(CpuGemm::Hand, Some(2)).unwrap();
        backend.matmul(
            m,
            n,
            k,
            &a,
            &b,
            BLayout::RowMajorNkTransposed,
            &mut actual,
            true,
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-4);
        }
    }
}
