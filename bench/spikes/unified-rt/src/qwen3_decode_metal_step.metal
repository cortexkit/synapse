#include <metal_stdlib>
using namespace metal;

constant uint Q8_BLOCK_BYTES = 34;
constant uint Q8_BLOCK_ELEMENTS = 32;
constant uint METAL_STEP_SIMD_WIDTH = 32;

struct NormConfig {
    uint width;
    float epsilon;
};

struct MatvecConfig {
    uint input_width;
    uint output_width;
    uint quantized;
    uint add_residual;
};

struct ResidualNormConfig {
    uint width;
    float epsilon;
};

struct QkvConfig {
    uint input_width;
    uint query_width;
    uint kv_width;
    uint head_dim;
    uint capacity;
    uint position;
    uint quantized;
};

struct QkConfig {
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    float epsilon;
    uint capacity;
    uint position;
};

struct AttentionConfig {
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint capacity;
    uint position;
};

struct ArgmaxConfig {
    uint vocab;
    uint partials;
};

struct EmbeddingConfig {
    uint hidden;
};

inline float matrix_dot_q8_chunk(
    const device half *input_block,
    const device uchar *block_q8,
    uint lane_start
) {
    const device half *scale = (const device half *)block_q8;
    const device char *values = (const device char *)((const device uchar *)scale + 2);
    half4 input_values = *(const device half4 *)(input_block + lane_start);
    char4 quant_values = *(const device char4 *)(values + lane_start);
    float4 products = float4(input_values) * float4(quant_values) * (float)scale[0];
    return (float)products[0] + (float)products[1] + (float)products[2] + (float)products[3];
}

inline float matrix_dot_q8_partial(
    const device half *input,
    const device uchar *q8,
    uint row,
    uint input_width,
    uint sub_lane
) {
    float partial = 0.0f;
    // Eight sub-lanes each consume a contiguous char4/half4 slice of this row,
    // in exactly the same ascending block order as before. Only the reduction
    // moved to the caller so four independent rows can share one simdgroup.
    {
        uint lane_start = sub_lane * 4;
        uint block_count = input_width / Q8_BLOCK_ELEMENTS;
        const device uchar *row_q8 = q8 + row * block_count * Q8_BLOCK_BYTES;
        const device half *input_block = input;
        uint block = 0;
        // Four block addresses remain independent, while each lane now issues
        // wider aligned loads from its block slice.
        for (; block + 4 <= block_count; block += 4) {
            partial += matrix_dot_q8_chunk(input_block, row_q8, lane_start);
            partial += matrix_dot_q8_chunk(input_block + Q8_BLOCK_ELEMENTS, row_q8 + Q8_BLOCK_BYTES, lane_start);
            partial += matrix_dot_q8_chunk(input_block + Q8_BLOCK_ELEMENTS * 2, row_q8 + Q8_BLOCK_BYTES * 2, lane_start);
            partial += matrix_dot_q8_chunk(input_block + Q8_BLOCK_ELEMENTS * 3, row_q8 + Q8_BLOCK_BYTES * 3, lane_start);
            input_block += Q8_BLOCK_ELEMENTS * 4;
            row_q8 += Q8_BLOCK_BYTES * 4;
        }
        for (; block < block_count; ++block) {
            partial += matrix_dot_q8_chunk(input_block, row_q8, lane_start);
            input_block += Q8_BLOCK_ELEMENTS;
            row_q8 += Q8_BLOCK_BYTES;
        }
    }
    return partial;
}

// Four independent rows per simdgroup: lanes [8g, 8g+8) own row_base + g.
// Each row is still reduced by one simd_sum over a 32-lane vector whose other
// 24 entries are exactly zero; the nonzero window only shifts by whole groups
// of eight lanes, so every row's reduction network and f32 addition order are
// identical to the single-row-per-simdgroup form.
inline float metal_step_pack_sum(float partial, uint group) {
    float value = 0.0f;
    for (uint target = 0; target < 4; ++target) {
        float summed = simd_sum(group == target ? partial : 0.0f);
        if (group == target) value = summed;
    }
    return value;
}

inline float matrix_dot_q8_simd(
    const device half *input,
    const device uchar *q8,
    uint row,
    uint input_width,
    uint lane
) {
    float partial = 0.0f;
    // Eight lanes each consume a contiguous char4/half4 slice. The remaining
    // lanes stay active with zero partials so simd_sum remains well-defined.
    if (lane < 8) {
        partial = matrix_dot_q8_partial(input, q8, row, input_width, lane);
    }
    return simd_sum(partial);
}

inline float matrix_dot_f16(
    const device half *input,
    const device half *fp16,
    uint row,
    uint input_width
) {
    float sum = 0.0f;
    const device half *weight = fp16 + row * input_width;
    uint col = 0;
    // Keep one serial accumulator per output row, but expose two adjacent
    // half4 loads at a time. Every product is still rounded to half and added
    // to the f32 accumulator in column order, so this only removes loop and
    // address-generation overhead from the exact wave-1 dot.
    for (; col + 8 <= input_width; col += 8) {
        half4 input_values0 = *(const device half4 *)(input + col);
        half4 weight_values0 = *(const device half4 *)(weight + col);
        half4 input_values1 = *(const device half4 *)(input + col + 4);
        half4 weight_values1 = *(const device half4 *)(weight + col + 4);
        half4 products0 = input_values0 * weight_values0;
        half4 products1 = input_values1 * weight_values1;
        sum += (float)products0[0];
        sum += (float)products0[1];
        sum += (float)products0[2];
        sum += (float)products0[3];
        sum += (float)products1[0];
        sum += (float)products1[1];
        sum += (float)products1[2];
        sum += (float)products1[3];
    }
    for (; col + 4 <= input_width; col += 4) {
        half4 input_values = *(const device half4 *)(input + col);
        half4 weight_values = *(const device half4 *)(weight + col);
        half4 products = input_values * weight_values;
        sum += (float)products[0];
        sum += (float)products[1];
        sum += (float)products[2];
        sum += (float)products[3];
    }
    for (; col < input_width; ++col) {
        half product = (half)((half)input[col] * weight[col]);
        sum += (float)product;
    }
    return sum;
}

// One simdgroup owns the short vector: each lane keeps its local products
// serial, then simd_sum combines the fixed lane partitions. This removes the
// one-thread dispatch while keeping the elementwise output work disjoint;
// generation exactness is still checked because the reduction tree changed.
inline void rmsnorm_body_simd(
    const device half *input,
    device half *output,
    const device half *weight,
    constant NormConfig &config,
    uint lane
) {
    float partial = 0.0f;
    for (uint i = lane; i < config.width; i += METAL_STEP_SIMD_WIDTH) {
        float value = (float)input[i];
        partial += value * value;
    }
    float sum = simd_sum(partial);
    float inv = rsqrt(sum / (float)config.width + config.epsilon);
    for (uint i = lane; i < config.width; i += METAL_STEP_SIMD_WIDTH) {
        output[i] = (half)((float)input[i] * inv * (float)weight[i]);
    }
}

kernel void metal_step_rmsnorm(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    const device half *weight [[buffer(2)]],
    constant NormConfig &config [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (tid >= METAL_STEP_SIMD_WIDTH) return;
    rmsnorm_body_simd(input, output, weight, config, lane);
}

kernel void metal_step_qkv_matvec(
    const device half *input [[buffer(0)]],
    device half *query [[buffer(1)]],
    device half *key [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    const device half *q_fp16 [[buffer(4)]],
    const device uchar *q_q8 [[buffer(5)]],
    const device half *k_fp16 [[buffer(6)]],
    const device uchar *k_q8 [[buffer(7)]],
    const device half *v_fp16 [[buffer(8)]],
    const device uchar *v_q8 [[buffer(9)]],
    constant QkvConfig &config [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (config.quantized == 0) {
        if (gid < config.query_width) {
            query[gid] = (half)matrix_dot_f16(input, q_fp16, gid, config.input_width);
        }
        if (gid < config.kv_width) {
            key[gid] = (half)matrix_dot_f16(input, k_fp16, gid, config.input_width);
            uint head = gid / config.head_dim;
            uint dimension = gid % config.head_dim;
            value_cache[(head * config.capacity + config.position) * config.head_dim + dimension] =
                (half)matrix_dot_f16(input, v_fp16, gid, config.input_width);
        }
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint row = gid / METAL_STEP_SIMD_WIDTH;
    uint total_rows = config.query_width + 2 * config.kv_width;
    if (row >= total_rows) return;
    if (row < config.query_width) {
        float result = matrix_dot_q8_simd(input, q_q8, row, config.input_width, lane);
        if (lane == 0) query[row] = (half)result;
        return;
    }
    row -= config.query_width;
    if (row < config.kv_width) {
        float result = matrix_dot_q8_simd(input, k_q8, row, config.input_width, lane);
        if (lane == 0) key[row] = (half)result;
        return;
    }
    row -= config.kv_width;
    float result = matrix_dot_q8_simd(input, v_q8, row, config.input_width, lane);
    if (lane == 0) {
        uint head = row / config.head_dim;
        uint dimension = row % config.head_dim;
        value_cache[(head * config.capacity + config.position) * config.head_dim + dimension] = (half)result;
    }
}

kernel void metal_step_qk_norm_rope(
    device half *query [[buffer(0)]],
    device half *key [[buffer(1)]],
    const device half *query_weight [[buffer(2)]],
    const device half *key_weight [[buffer(3)]],
    const device half *rope_cos [[buffer(4)]],
    const device half *rope_sin [[buffer(5)]],
    device half *key_cache [[buffer(6)]],
    constant QkConfig &config [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = gid / METAL_STEP_SIMD_WIDTH;
    uint half_dim = config.head_dim / 2;
    if (head < config.query_heads) {
        uint start = head * config.head_dim;
        float inv = 0.0f;
        if (lane == 0) {
            float sum = 0.0f;
            for (uint i = 0; i < config.head_dim; i += 4) {
                half4 q4 = *reinterpret_cast<const device half4 *>(query + start + i);
                float v0 = (float)q4.x;
                float v1 = (float)q4.y;
                float v2 = (float)q4.z;
                float v3 = (float)q4.w;
                sum += v0 * v0;
                sum += v1 * v1;
                sum += v2 * v2;
                sum += v3 * v3;
            }
            inv = rsqrt(sum / (float)config.head_dim + config.epsilon);
        }
        inv = simd_broadcast(inv, 0);
        for (uint i = lane; i < half_dim; i += METAL_STEP_SIMD_WIDTH) {
            half first = (half)((float)query[start + i] * inv * (float)query_weight[i]);
            half second = (half)((float)query[start + half_dim + i] * inv * (float)query_weight[half_dim + i]);
            half cosine = rope_cos[i];
            half sine = rope_sin[i];
            query[start + i] = (half)((half)(first * cosine) - (half)(second * sine));
            query[start + half_dim + i] = (half)((half)(second * cosine) + (half)(first * sine));
        }
    }
    if (head < config.kv_heads) {
        uint start = head * config.head_dim;
        float inv = 0.0f;
        if (lane == 0) {
            float sum = 0.0f;
            for (uint i = 0; i < config.head_dim; i += 4) {
                half4 k4 = *reinterpret_cast<const device half4 *>(key + start + i);
                float v0 = (float)k4.x;
                float v1 = (float)k4.y;
                float v2 = (float)k4.z;
                float v3 = (float)k4.w;
                sum += v0 * v0;
                sum += v1 * v1;
                sum += v2 * v2;
                sum += v3 * v3;
            }
            inv = rsqrt(sum / (float)config.head_dim + config.epsilon);
        }
        inv = simd_broadcast(inv, 0);
        for (uint i = lane; i < half_dim; i += METAL_STEP_SIMD_WIDTH) {
            half first = (half)((float)key[start + i] * inv * (float)key_weight[i]);
            half second = (half)((float)key[start + half_dim + i] * inv * (float)key_weight[half_dim + i]);
            half cosine = rope_cos[i];
            half sine = rope_sin[i];
            half first_rotated = (half)((half)(first * cosine) - (half)(second * sine));
            half second_rotated = (half)((half)(second * cosine) + (half)(first * sine));
            key[start + i] = first_rotated;
            key[start + half_dim + i] = second_rotated;
            uint cache_start = (head * config.capacity + config.position) * config.head_dim;
            key_cache[cache_start + i] = first_rotated;
            key_cache[cache_start + half_dim + i] = second_rotated;
        }
    }
}

kernel void metal_step_attention(
    const device half *query [[buffer(0)]],
    const device half *key_cache [[buffer(1)]],
    const device half *value_cache [[buffer(2)]],
    device half *output [[buffer(3)]],
    device float *scores [[buffer(4)]],
    constant AttentionConfig &config [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint query_head = gid / METAL_STEP_SIMD_WIDTH;
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    if (query_head >= config.query_heads) return;
    uint head_dim = config.head_dim;
    uint group = config.query_heads / config.kv_heads;
    uint kv_head = query_head / group;
    uint q_start = query_head * head_dim;
    uint score_start = query_head * config.capacity;
    float scale = 1.0f / sqrt((float)head_dim);

    // Each lane owns independent KV positions and computes each QK dot in the
    // reference serial order. Parallelizing across positions cannot change a
    // dot product's accumulation order; only the cheap softmax reductions stay
    // serial on lane zero below.
    for (uint position = lane; position <= config.position; position += METAL_STEP_SIMD_WIDTH) {
        float score = 0.0f;
        uint key_start = (kv_head * config.capacity + position) * head_dim;
        if ((head_dim & 3u) == 0u) {
            // One wide half4 load per four elements; the f32 accumulation
            // still adds each product in the original ascending serial order,
            // so the dot result is bit-identical to the scalar loop.
            for (uint i = 0; i < head_dim; i += 4) {
                half4 q4 = *reinterpret_cast<const device half4 *>(query + q_start + i);
                half4 k4 = *reinterpret_cast<const device half4 *>(key_cache + key_start + i);
                score += (float)q4.x * (float)k4.x;
                score += (float)q4.y * (float)k4.y;
                score += (float)q4.z * (float)k4.z;
                score += (float)q4.w * (float)k4.w;
            }
        } else {
            for (uint i = 0; i < head_dim; ++i) {
                score += (float)query[q_start + i] * (float)key_cache[key_start + i];
            }
        }
        scores[score_start + position] = score * scale;
    }
    simdgroup_barrier(mem_flags::mem_device);

    float maximum = -3.402823466e+38f;
    float denominator = 0.0f;
    if (lane == 0) {
        for (uint position = 0; position <= config.position; ++position) {
            maximum = max(maximum, scores[score_start + position]);
        }
        for (uint position = 0; position <= config.position; ++position) {
            denominator += exp(scores[score_start + position] - maximum);
        }
    }
    maximum = simd_broadcast(maximum, 0);
    denominator = simd_broadcast(denominator, 0);
    if (head_dim == METAL_STEP_SIMD_WIDTH * 4) {
        float result0 = 0.0f;
        float result1 = 0.0f;
        float result2 = 0.0f;
        float result3 = 0.0f;
        for (uint position = 0; position <= config.position; ++position) {
            uint value_start = (kv_head * config.capacity + position) * head_dim + lane;
            half probability = (half)(exp(scores[score_start + position] - maximum) / denominator);
            result0 += (float)probability * (float)value_cache[value_start];
            result1 += (float)probability * (float)value_cache[value_start + METAL_STEP_SIMD_WIDTH];
            result2 += (float)probability * (float)value_cache[value_start + METAL_STEP_SIMD_WIDTH * 2];
            result3 += (float)probability * (float)value_cache[value_start + METAL_STEP_SIMD_WIDTH * 3];
        }
        output[q_start + lane] = (half)result0;
        output[q_start + lane + METAL_STEP_SIMD_WIDTH] = (half)result1;
        output[q_start + lane + METAL_STEP_SIMD_WIDTH * 2] = (half)result2;
        output[q_start + lane + METAL_STEP_SIMD_WIDTH * 3] = (half)result3;
    } else {
        for (uint i = lane; i < head_dim; i += METAL_STEP_SIMD_WIDTH) {
            float result = 0.0f;
            for (uint position = 0; position <= config.position; ++position) {
                uint value_index = (kv_head * config.capacity + position) * head_dim + i;
                half probability = (half)(exp(scores[score_start + position] - maximum) / denominator);
                result += (float)probability * (float)value_cache[value_index];
            }
            output[q_start + i] = (half)result;
        }
    }
}

kernel void metal_step_matvec_residual(
    const device half *input [[buffer(0)]],
    const device half *residual [[buffer(1)]],
    device half *output [[buffer(2)]],
    const device half *fp16 [[buffer(3)]],
    const device uchar *q8 [[buffer(4)]],
    constant MatvecConfig &config [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        float value = matrix_dot_f16(input, fp16, gid, config.input_width);
        output[gid] = (half)(value + (config.add_residual != 0 ? (float)residual[gid] : 0.0f));
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float partial = 0.0f;
    if (row < config.output_width) {
        partial = matrix_dot_q8_partial(input, q8, row, config.input_width, sub_lane);
    }
    float value = metal_step_pack_sum(partial, group);
    if (sub_lane == 0 && row < config.output_width) {
        output[row] = (half)(value + (config.add_residual != 0 ? (float)residual[row] : 0.0f));
    }
}

kernel void metal_step_residual_rmsnorm(
    device half *projection [[buffer(0)]],
    const device half *residual [[buffer(1)]],
    device half *normalized [[buffer(2)]],
    const device half *weight [[buffer(3)]],
    constant ResidualNormConfig &config [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (tid >= METAL_STEP_SIMD_WIDTH) return;
    // Lane zero keeps the residual-add sum-of-squares in the original
    // ascending element order; no single reduction is split. The independent
    // per-element residual store and normalize outputs are then divided across
    // the simdgroup, and every stored half is bit-identical to the one-thread
    // version because the same float expression produces it.
    float inv = 0.0f;
    if (lane == 0) {
        float sum = 0.0f;
        uint i = 0;
        for (; i + 4 <= config.width; i += 4) {
            half4 projection_values = *(const device half4 *)(projection + i);
            half4 residual_values = *(const device half4 *)(residual + i);
            float value0 = (float)projection_values[0] + (float)residual_values[0];
            float value1 = (float)projection_values[1] + (float)residual_values[1];
            float value2 = (float)projection_values[2] + (float)residual_values[2];
            float value3 = (float)projection_values[3] + (float)residual_values[3];
            sum += value0 * value0;
            sum += value1 * value1;
            sum += value2 * value2;
            sum += value3 * value3;
        }
        for (; i < config.width; ++i) {
            float value = (float)projection[i] + (float)residual[i];
            sum += value * value;
        }
        inv = rsqrt(sum / (float)config.width + config.epsilon);
    }
    // Lane zero's reads all retire before any lane republishes projection.
    threadgroup_barrier(mem_flags::mem_device);
    // Only lane zero holds a non-zero inverse norm, so the maximum is an exact
    // bit-preserving broadcast.
    inv = simd_max(inv);
    uint vector_width = config.width & ~3u;
    for (uint i = lane * 4; i + 4 <= config.width; i += METAL_STEP_SIMD_WIDTH * 4) {
        half4 projection_values = *(const device half4 *)(projection + i);
        half4 residual_values = *(const device half4 *)(residual + i);
        half4 weight_values = *(const device half4 *)(weight + i);
        half4 updated;
        updated[0] = (half)((float)projection_values[0] + (float)residual_values[0]);
        updated[1] = (half)((float)projection_values[1] + (float)residual_values[1]);
        updated[2] = (half)((float)projection_values[2] + (float)residual_values[2]);
        updated[3] = (half)((float)projection_values[3] + (float)residual_values[3]);
        *(device half4 *)(projection + i) = updated;
        half4 normalized_values;
        normalized_values[0] = (half)((float)updated[0] * inv * (float)weight_values[0]);
        normalized_values[1] = (half)((float)updated[1] * inv * (float)weight_values[1]);
        normalized_values[2] = (half)((float)updated[2] * inv * (float)weight_values[2]);
        normalized_values[3] = (half)((float)updated[3] * inv * (float)weight_values[3]);
        *(device half4 *)(normalized + i) = normalized_values;
    }
    for (uint i = vector_width + lane; i < config.width; i += METAL_STEP_SIMD_WIDTH) {
        half value = (half)((float)projection[i] + (float)residual[i]);
        projection[i] = value;
        normalized[i] = (half)((float)value * inv * (float)weight[i]);
    }
}

kernel void metal_step_gate_up_swiglu(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    const device half *gate_fp16 [[buffer(2)]],
    const device uchar *gate_q8 [[buffer(3)]],
    const device half *up_fp16 [[buffer(4)]],
    const device uchar *up_q8 [[buffer(5)]],
    constant MatvecConfig &config [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        float gate = matrix_dot_f16(input, gate_fp16, gid, config.input_width);
        float up = matrix_dot_f16(input, up_fp16, gid, config.input_width);
        output[gid] = (half)((gate / (1.0f + exp(-gate))) * up);
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    if (row < config.output_width) {
        gate_partial = matrix_dot_q8_partial(input, gate_q8, row, config.input_width, sub_lane);
        up_partial = matrix_dot_q8_partial(input, up_q8, row, config.input_width, sub_lane);
    }
    float gate = metal_step_pack_sum(gate_partial, group);
    float up = metal_step_pack_sum(up_partial, group);
    if (sub_lane == 0 && row < config.output_width) {
        output[row] = (half)((gate / (1.0f + exp(-gate))) * up);
    }
}

kernel void metal_step_lm_head(
    const device half *input [[buffer(0)]],
    device float *logits [[buffer(1)]],
    const device half *fp16 [[buffer(2)]],
    const device uchar *q8 [[buffer(3)]],
    constant MatvecConfig &config [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        logits[gid] = matrix_dot_f16(input, fp16, gid, config.input_width);
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float partial = 0.0f;
    if (row < config.output_width) {
        partial = matrix_dot_q8_partial(input, q8, row, config.input_width, sub_lane);
    }
    float value = metal_step_pack_sum(partial, group);
    if (sub_lane == 0 && row < config.output_width) logits[row] = value;
}

// Map a float to a strictly increasing UNSIGNED integer key so a plain unsigned
// `>` reproduces the host sampler's f32::total_cmp ordering exactly, including
// the sign of zero. This is the canonical IEEE-754 radix-order transform: for a
// non-negative float set the top bit (keys sit in the upper half, larger float
// = larger key); for a negative float flip every bit (keys sit in the lower
// half, more-negative float = smaller key). Larger float always yields a larger
// unsigned key, and -0.0 sorts just below +0.0 exactly as total_cmp requires.
inline uint total_order_key(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The host greedy sampler picks the highest logit and breaks ties toward the
// LOWEST token id (see logit_precedes in qwen3_decode.rs). A candidate beats
// the incumbent when its ordered key is strictly greater, or the keys tie and
// its token id is strictly smaller. Both stages of the reduction use this rule
// so the chained decode's device-side token selection is byte-identical to the
// per-token host argmax.
inline bool argmax_candidate_wins(int candidate_key, uint candidate_id, int best_key, uint best_id) {
    return candidate_key > best_key || (candidate_key == best_key && candidate_id < best_id);
}

// Stage one: one thread per threadgroup scans a contiguous vocab slice in
// ascending id order and writes one (key, id) partial. A serial ascending scan
// makes the lowest-id tie rule exact without depending on cross-lane shuffle
// semantics; the two-stage split keeps each scan short. Sentinel key INT_MIN
// with id UINT_MAX loses to every real candidate, so an empty slice is inert.
kernel void metal_step_argmax_partial(
    const device float *logits [[buffer(0)]],
    device uint *partial_keys [[buffer(1)]],
    device uint *partial_ids [[buffer(2)]],
    constant ArgmaxConfig &config [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    if (group_id >= config.partials || lane != 0) return;
    // Even slices across the vocabulary; the final slice absorbs the remainder.
    uint slice = (config.vocab + config.partials - 1) / config.partials;
    uint start = group_id * slice;
    uint end = min(start + slice, config.vocab);
    uint best_key = 0u;
    uint best_id = 0xffffffffu;
    bool have = false;
    for (uint index = start; index < end; ++index) {
        uint key = total_order_key(logits[index]);
        // Ascending index scan with strict-greater wins: the first (lowest) id
        // holding the maximum key is kept, matching the host lowest-id tie rule.
        if (!have || key > best_key) {
            best_key = key;
            best_id = index;
            have = true;
        }
    }
    partial_keys[group_id] = best_key;
    partial_ids[group_id] = best_id;
}

// Stage two: one thread folds the partials into a single token id with the same
// ascending-scan / strict-greater rule, so partial boundaries never change the
// selected id.
kernel void metal_step_argmax_final(
    const device uint *partial_keys [[buffer(0)]],
    const device uint *partial_ids [[buffer(1)]],
    device uint *token_out [[buffer(2)]],
    constant ArgmaxConfig &config [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]]) {
    if (lane != 0) return;
    uint best_key = 0u;
    uint best_id = 0xffffffffu;
    bool have = false;
    for (uint index = 0; index < config.partials; ++index) {
        uint key = partial_keys[index];
        uint id = partial_ids[index];
        // Partials are emitted in ascending vocab order; strict-greater on the
        // key with a lowest-id tie-break keeps the lowest id among equal maxima.
        if (!have || key > best_key || (key == best_key && id < best_id)) {
            best_key = key;
            best_id = id;
            have = true;
        }
    }
    token_out[0] = best_id;
}

// Gather one row of the tied f16 embedding table into the step's input buffer,
// indexed by the previous step's argmax output. This lets a chained step read
// its input token device-side without a host round trip. The embedding row and
// the host embedding() slice are the same f16 bits, so the fed activation is
// identical to the per-token path.
kernel void metal_step_embedding_gather(
    const device half *embeddings [[buffer(0)]],
    const device uint *token_in [[buffer(1)]],
    device half *input [[buffer(2)]],
    constant EmbeddingConfig &config [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= config.hidden) return;
    uint token = token_in[0];
    input[gid] = embeddings[(ulong)token * config.hidden + gid];
}

// ---------------------------------------------------------------------------
// Batched verification kernels.
//
// Speculative verification feeds K already-known draft tokens through the
// transformer in ONE forward pass instead of K dependent single-token steps.
// The bandwidth win comes from streaming each projection weight row once and
// applying it to all K position activations (a mat-mat with K columns), rather
// than re-streaming it once per token.
//
// Exactness law: batching parallelizes ACROSS positions (independent
// reductions); it never reorders the accumulation WITHIN one dot product. Each
// (output row, position) dot below walks the weight in the same ascending
// column/block order and adds products in the same order as the single-token
// kernels above, so every position's logits are bit-identical to a sequential
// single-token step at that position. The K columns simply share the weight
// load. K is bounded by 16 (METAL_STEP_MAX_BATCH_K on the host) and rounded up
// to a power-of-two template instantiation in each kernel's entry below.
// ---------------------------------------------------------------------------

struct BatchQkvConfig {
    uint input_width;
    uint query_width;
    uint kv_width;
    uint head_dim;
    uint capacity;
    uint position;   // position of batch column 0
    uint quantized;
    uint batch;      // K columns
};

struct BatchMatvecConfig {
    uint input_width;
    uint output_width;
    uint quantized;
    uint add_residual;
    uint batch;
};

// F16 mat-mat dot for a COMPILE-TIME column count N. The weight row is streamed
// once and applied to all N column activations. Crucially, N is a template
// parameter so the column loop unrolls and the N accumulators stay in registers
// indexed by constants; each column's `sums[k] += p0; sums[k] += p1; ...` chain
// is then preserved in the exact ascending order of matrix_dot_f16. (A runtime
// column count spills the accumulators to memory and lets the compiler fold the
// eight products before adding, which changes the rounding and breaks the
// byte-exact gate.) Column k is therefore bit-identical to
// matrix_dot_f16(inputs + k*input_width, fp16, row, input_width).
template <uint N>
inline void matrix_dot_f16_batch_n(
    const device half *inputs,
    const device half *fp16,
    uint row,
    uint input_width,
    thread float (&sums)[N]
) {
    const device half *weight = fp16 + row * input_width;
    for (uint k = 0; k < N; ++k) sums[k] = 0.0f;
    uint col = 0;
    for (; col + 8 <= input_width; col += 8) {
        half4 weight_values0 = *(const device half4 *)(weight + col);
        half4 weight_values1 = *(const device half4 *)(weight + col + 4);
        for (uint k = 0; k < N; ++k) {
            const device half *input = inputs + (uint64_t)k * input_width + col;
            half4 input_values0 = *(const device half4 *)(input);
            half4 input_values1 = *(const device half4 *)(input + 4);
            half4 products0 = input_values0 * weight_values0;
            half4 products1 = input_values1 * weight_values1;
            sums[k] += (float)products0[0];
            sums[k] += (float)products0[1];
            sums[k] += (float)products0[2];
            sums[k] += (float)products0[3];
            sums[k] += (float)products1[0];
            sums[k] += (float)products1[1];
            sums[k] += (float)products1[2];
            sums[k] += (float)products1[3];
        }
    }
    for (; col + 4 <= input_width; col += 4) {
        half4 weight_values = *(const device half4 *)(weight + col);
        for (uint k = 0; k < N; ++k) {
            const device half *input = inputs + (uint64_t)k * input_width + col;
            half4 input_values = *(const device half4 *)(input);
            half4 products = input_values * weight_values;
            sums[k] += (float)products[0];
            sums[k] += (float)products[1];
            sums[k] += (float)products[2];
            sums[k] += (float)products[3];
        }
    }
    for (; col < input_width; ++col) {
        half weight_value = weight[col];
        for (uint k = 0; k < N; ++k) {
            half product = (half)((half)inputs[(uint64_t)k * input_width + col] * weight_value);
            sums[k] += (float)product;
        }
    }
}

// Q8 mat-mat chunk for N columns: dequantize one weight block slice ONCE (scale
// + char4 read a single time) and apply it to every column's half4. Per column
// this is the same four-product ascending sum as matrix_dot_q8_chunk.
template <uint N>
inline void matrix_dot_q8_chunk_batch_n(
    const device half *inputs,
    uint input_width,
    uint col,
    const device uchar *block_q8,
    uint lane_start,
    thread float (&partials)[N]
) {
    const device half *scale = (const device half *)block_q8;
    const device char *values = (const device char *)((const device uchar *)scale + 2);
    char4 quant_values = *(const device char4 *)(values + lane_start);
    float dequant_scale = (float)scale[0];
    float4 quant_float = float4(quant_values);
    for (uint k = 0; k < N; ++k) {
        half4 input_values = *(const device half4 *)(inputs + (uint64_t)k * input_width + col + lane_start);
        float4 products = float4(input_values) * quant_float * dequant_scale;
        partials[k] += (float)products[0] + (float)products[1] + (float)products[2] + (float)products[3];
    }
}

// Q8 mat-mat partial for one (row, sub-lane) across N columns: walk the row's
// blocks once in the same ascending order as matrix_dot_q8_partial. N is a
// template parameter so the column accumulators stay register-resident and keep
// the single-token reduction order (see matrix_dot_f16_batch_n).
template <uint N>
inline void matrix_dot_q8_partial_batch_n(
    const device half *inputs,
    uint input_width,
    const device uchar *q8,
    uint row,
    uint sub_lane,
    thread float (&partials)[N]
) {
    for (uint k = 0; k < N; ++k) partials[k] = 0.0f;
    uint lane_start = sub_lane * 4;
    uint block_count = input_width / Q8_BLOCK_ELEMENTS;
    const device uchar *row_q8 = q8 + row * block_count * Q8_BLOCK_BYTES;
    uint col = 0;
    uint block = 0;
    for (; block + 4 <= block_count; block += 4) {
        matrix_dot_q8_chunk_batch_n<N>(inputs, input_width, col, row_q8, lane_start, partials);
        matrix_dot_q8_chunk_batch_n<N>(inputs, input_width, col + Q8_BLOCK_ELEMENTS, row_q8 + Q8_BLOCK_BYTES, lane_start, partials);
        matrix_dot_q8_chunk_batch_n<N>(inputs, input_width, col + Q8_BLOCK_ELEMENTS * 2, row_q8 + Q8_BLOCK_BYTES * 2, lane_start, partials);
        matrix_dot_q8_chunk_batch_n<N>(inputs, input_width, col + Q8_BLOCK_ELEMENTS * 3, row_q8 + Q8_BLOCK_BYTES * 3, lane_start, partials);
        col += Q8_BLOCK_ELEMENTS * 4;
        row_q8 += Q8_BLOCK_BYTES * 4;
    }
    for (; block < block_count; ++block) {
        matrix_dot_q8_chunk_batch_n<N>(inputs, input_width, col, row_q8, lane_start, partials);
        col += Q8_BLOCK_ELEMENTS;
        row_q8 += Q8_BLOCK_BYTES;
    }
}

// Each batched projection kernel is templated on the compile-time column count
// N and dispatched through a runtime switch that rounds config.batch up to the
// next power of two in {1,2,4,8,16}. Columns k >= config.batch are computed but
// not written, so any batch <= 16 is exact while the common power-of-two spans
// use a tight instantiation.

template <uint N>
inline void metal_step_qkv_matvec_batch_body(
    const device half *input,
    device half *query,
    device half *key,
    device half *value_cache,
    const device half *q_fp16,
    const device uchar *q_q8,
    const device half *k_fp16,
    const device uchar *k_q8,
    const device half *v_fp16,
    const device uchar *v_q8,
    constant BatchQkvConfig &config,
    uint gid
) {
    uint batch = config.batch;
    if (config.quantized == 0) {
        if (gid < config.query_width) {
            float sums[N];
            matrix_dot_f16_batch_n<N>(input, q_fp16, gid, config.input_width, sums);
            for (uint k = 0; k < N; ++k) {
                if (k < batch) query[(uint64_t)k * config.query_width + gid] = (half)sums[k];
            }
        }
        if (gid < config.kv_width) {
            float key_sums[N];
            float value_sums[N];
            matrix_dot_f16_batch_n<N>(input, k_fp16, gid, config.input_width, key_sums);
            matrix_dot_f16_batch_n<N>(input, v_fp16, gid, config.input_width, value_sums);
            uint head = gid / config.head_dim;
            uint dimension = gid % config.head_dim;
            for (uint k = 0; k < N; ++k) {
                if (k >= batch) continue;
                key[(uint64_t)k * config.kv_width + gid] = (half)key_sums[k];
                value_cache[(head * config.capacity + config.position + k) * config.head_dim + dimension] =
                    (half)value_sums[k];
            }
        }
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint row = gid / METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint total_rows = config.query_width + 2 * config.kv_width;
    if (row >= total_rows) return;
    if (row < config.query_width) {
        float partials[N];
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, q_q8, row, sub_lane, partials);
        for (uint k = 0; k < N; ++k) {
            float result = metal_step_pack_sum(partials[k], group);
            if (sub_lane == 0 && k < batch) query[(uint64_t)k * config.query_width + row] = (half)result;
        }
        return;
    }
    row -= config.query_width;
    if (row < config.kv_width) {
        float partials[N];
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, k_q8, row, sub_lane, partials);
        for (uint k = 0; k < N; ++k) {
            float result = metal_step_pack_sum(partials[k], group);
            if (sub_lane == 0 && k < batch) key[(uint64_t)k * config.kv_width + row] = (half)result;
        }
        return;
    }
    row -= config.kv_width;
    float partials[N];
    matrix_dot_q8_partial_batch_n<N>(input, config.input_width, v_q8, row, sub_lane, partials);
    uint head = row / config.head_dim;
    uint dimension = row % config.head_dim;
    for (uint k = 0; k < N; ++k) {
        float result = metal_step_pack_sum(partials[k], group);
        if (sub_lane == 0 && k < batch) {
            value_cache[(head * config.capacity + config.position + k) * config.head_dim + dimension] = (half)result;
        }
    }
}

kernel void metal_step_qkv_matvec_batch(
    const device half *input [[buffer(0)]],
    device half *query [[buffer(1)]],
    device half *key [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    const device half *q_fp16 [[buffer(4)]],
    const device uchar *q_q8 [[buffer(5)]],
    const device half *k_fp16 [[buffer(6)]],
    const device uchar *k_q8 [[buffer(7)]],
    const device half *v_fp16 [[buffer(8)]],
    const device uchar *v_q8 [[buffer(9)]],
    constant BatchQkvConfig &config [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    uint b = config.batch;
    if (b <= 1) {
        metal_step_qkv_matvec_batch_body<1>(input, query, key, value_cache, q_fp16, q_q8, k_fp16, k_q8, v_fp16, v_q8, config, gid);
    } else if (b <= 2) {
        metal_step_qkv_matvec_batch_body<2>(input, query, key, value_cache, q_fp16, q_q8, k_fp16, k_q8, v_fp16, v_q8, config, gid);
    } else if (b <= 4) {
        metal_step_qkv_matvec_batch_body<4>(input, query, key, value_cache, q_fp16, q_q8, k_fp16, k_q8, v_fp16, v_q8, config, gid);
    } else if (b <= 8) {
        metal_step_qkv_matvec_batch_body<8>(input, query, key, value_cache, q_fp16, q_q8, k_fp16, k_q8, v_fp16, v_q8, config, gid);
    } else {
        metal_step_qkv_matvec_batch_body<16>(input, query, key, value_cache, q_fp16, q_q8, k_fp16, k_q8, v_fp16, v_q8, config, gid);
    }
}

template <uint N>
inline void metal_step_matvec_residual_batch_body(
    const device half *input,
    const device half *residual,
    device half *output,
    const device half *fp16,
    const device uchar *q8,
    constant BatchMatvecConfig &config,
    uint gid
) {
    uint batch = config.batch;
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        float sums[N];
        matrix_dot_f16_batch_n<N>(input, fp16, gid, config.input_width, sums);
        for (uint k = 0; k < N; ++k) {
            if (k >= batch) continue;
            float value = sums[k] + (config.add_residual != 0 ? (float)residual[(uint64_t)k * config.output_width + gid] : 0.0f);
            output[(uint64_t)k * config.output_width + gid] = (half)value;
        }
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float partials[N];
    if (row < config.output_width) {
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, q8, row, sub_lane, partials);
    } else {
        for (uint k = 0; k < N; ++k) partials[k] = 0.0f;
    }
    for (uint k = 0; k < N; ++k) {
        float value = metal_step_pack_sum(partials[k], group);
        if (sub_lane == 0 && row < config.output_width && k < batch) {
            float with_residual = value + (config.add_residual != 0 ? (float)residual[(uint64_t)k * config.output_width + row] : 0.0f);
            output[(uint64_t)k * config.output_width + row] = (half)with_residual;
        }
    }
}

kernel void metal_step_matvec_residual_batch(
    const device half *input [[buffer(0)]],
    const device half *residual [[buffer(1)]],
    device half *output [[buffer(2)]],
    const device half *fp16 [[buffer(3)]],
    const device uchar *q8 [[buffer(4)]],
    constant BatchMatvecConfig &config [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint b = config.batch;
    if (b <= 1) {
        metal_step_matvec_residual_batch_body<1>(input, residual, output, fp16, q8, config, gid);
    } else if (b <= 2) {
        metal_step_matvec_residual_batch_body<2>(input, residual, output, fp16, q8, config, gid);
    } else if (b <= 4) {
        metal_step_matvec_residual_batch_body<4>(input, residual, output, fp16, q8, config, gid);
    } else if (b <= 8) {
        metal_step_matvec_residual_batch_body<8>(input, residual, output, fp16, q8, config, gid);
    } else {
        metal_step_matvec_residual_batch_body<16>(input, residual, output, fp16, q8, config, gid);
    }
}

template <uint N>
inline void metal_step_gate_up_swiglu_batch_body(
    const device half *input,
    device half *output,
    const device half *gate_fp16,
    const device uchar *gate_q8,
    const device half *up_fp16,
    const device uchar *up_q8,
    constant BatchMatvecConfig &config,
    uint gid
) {
    uint batch = config.batch;
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        float gate_sums[N];
        float up_sums[N];
        matrix_dot_f16_batch_n<N>(input, gate_fp16, gid, config.input_width, gate_sums);
        matrix_dot_f16_batch_n<N>(input, up_fp16, gid, config.input_width, up_sums);
        for (uint k = 0; k < N; ++k) {
            if (k >= batch) continue;
            float gate = gate_sums[k];
            float up = up_sums[k];
            output[(uint64_t)k * config.output_width + gid] = (half)((gate / (1.0f + exp(-gate))) * up);
        }
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float gate_partials[N];
    float up_partials[N];
    if (row < config.output_width) {
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, gate_q8, row, sub_lane, gate_partials);
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, up_q8, row, sub_lane, up_partials);
    } else {
        for (uint k = 0; k < N; ++k) {
            gate_partials[k] = 0.0f;
            up_partials[k] = 0.0f;
        }
    }
    for (uint k = 0; k < N; ++k) {
        float gate = metal_step_pack_sum(gate_partials[k], group);
        float up = metal_step_pack_sum(up_partials[k], group);
        if (sub_lane == 0 && row < config.output_width && k < batch) {
            output[(uint64_t)k * config.output_width + row] = (half)((gate / (1.0f + exp(-gate))) * up);
        }
    }
}

kernel void metal_step_gate_up_swiglu_batch(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    const device half *gate_fp16 [[buffer(2)]],
    const device uchar *gate_q8 [[buffer(3)]],
    const device half *up_fp16 [[buffer(4)]],
    const device uchar *up_q8 [[buffer(5)]],
    constant BatchMatvecConfig &config [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    uint b = config.batch;
    if (b <= 1) {
        metal_step_gate_up_swiglu_batch_body<1>(input, output, gate_fp16, gate_q8, up_fp16, up_q8, config, gid);
    } else if (b <= 2) {
        metal_step_gate_up_swiglu_batch_body<2>(input, output, gate_fp16, gate_q8, up_fp16, up_q8, config, gid);
    } else if (b <= 4) {
        metal_step_gate_up_swiglu_batch_body<4>(input, output, gate_fp16, gate_q8, up_fp16, up_q8, config, gid);
    } else if (b <= 8) {
        metal_step_gate_up_swiglu_batch_body<8>(input, output, gate_fp16, gate_q8, up_fp16, up_q8, config, gid);
    } else {
        metal_step_gate_up_swiglu_batch_body<16>(input, output, gate_fp16, gate_q8, up_fp16, up_q8, config, gid);
    }
}

template <uint N>
inline void metal_step_lm_head_batch_body(
    const device half *input,
    device float *logits,
    const device half *fp16,
    const device uchar *q8,
    constant BatchMatvecConfig &config,
    uint gid
) {
    uint batch = config.batch;
    if (config.quantized == 0) {
        if (gid >= config.output_width) return;
        float sums[N];
        matrix_dot_f16_batch_n<N>(input, fp16, gid, config.input_width, sums);
        for (uint k = 0; k < N; ++k) {
            if (k < batch) logits[(uint64_t)k * config.output_width + gid] = sums[k];
        }
        return;
    }
    uint lane = gid % METAL_STEP_SIMD_WIDTH;
    uint group = lane / 8;
    uint sub_lane = lane % 8;
    uint row_base = (gid / METAL_STEP_SIMD_WIDTH) * 4;
    if (row_base >= config.output_width) return;
    uint row = row_base + group;
    float partials[N];
    if (row < config.output_width) {
        matrix_dot_q8_partial_batch_n<N>(input, config.input_width, q8, row, sub_lane, partials);
    } else {
        for (uint k = 0; k < N; ++k) partials[k] = 0.0f;
    }
    for (uint k = 0; k < N; ++k) {
        float value = metal_step_pack_sum(partials[k], group);
        if (sub_lane == 0 && row < config.output_width && k < batch) {
            logits[(uint64_t)k * config.output_width + row] = value;
        }
    }
}

kernel void metal_step_lm_head_batch(
    const device half *input [[buffer(0)]],
    device float *logits [[buffer(1)]],
    const device half *fp16 [[buffer(2)]],
    const device uchar *q8 [[buffer(3)]],
    constant BatchMatvecConfig &config [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    uint b = config.batch;
    if (b <= 1) {
        metal_step_lm_head_batch_body<1>(input, logits, fp16, q8, config, gid);
    } else if (b <= 2) {
        metal_step_lm_head_batch_body<2>(input, logits, fp16, q8, config, gid);
    } else if (b <= 4) {
        metal_step_lm_head_batch_body<4>(input, logits, fp16, q8, config, gid);
    } else if (b <= 8) {
        metal_step_lm_head_batch_body<8>(input, logits, fp16, q8, config, gid);
    } else {
        metal_step_lm_head_batch_body<16>(input, logits, fp16, q8, config, gid);
    }
}
