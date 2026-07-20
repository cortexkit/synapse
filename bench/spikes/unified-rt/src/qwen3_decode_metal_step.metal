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

inline float matrix_dot_q8_chunk(
    const device half *input,
    const device uchar *q8,
    uint row_start,
    uint block_start,
    uint lane_start
) {
    const device half *scale = (const device half *)(q8 +
        (row_start + block_start) / Q8_BLOCK_ELEMENTS * Q8_BLOCK_BYTES);
    const device char *values = (const device char *)((const device uchar *)scale + 2);
    half4 input_values = *(const device half4 *)(input + block_start + lane_start);
    char4 quant_values = *(const device char4 *)(values + lane_start);
    float4 products = float4(input_values) * float4(quant_values) * (float)scale[0];
    return (float)products[0] + (float)products[1] + (float)products[2] + (float)products[3];
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
        uint row_start = row * input_width;
        uint lane_start = lane * 4;
        uint block_count = input_width / Q8_BLOCK_ELEMENTS;
        uint block = 0;
        // Four block addresses remain independent, while each lane now issues
        // wider aligned loads from its block slice.
        for (; block + 4 <= block_count; block += 4) {
            uint block0 = block * Q8_BLOCK_ELEMENTS;
            uint block1 = block0 + Q8_BLOCK_ELEMENTS;
            uint block2 = block1 + Q8_BLOCK_ELEMENTS;
            uint block3 = block2 + Q8_BLOCK_ELEMENTS;
            partial += matrix_dot_q8_chunk(input, q8, row_start, block0, lane_start);
            partial += matrix_dot_q8_chunk(input, q8, row_start, block1, lane_start);
            partial += matrix_dot_q8_chunk(input, q8, row_start, block2, lane_start);
            partial += matrix_dot_q8_chunk(input, q8, row_start, block3, lane_start);
        }
        for (; block < block_count; ++block) {
            partial += matrix_dot_q8_chunk(
                input,
                q8,
                row_start,
                block * Q8_BLOCK_ELEMENTS,
                lane_start
            );
        }
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

inline void rmsnorm_body(
    const device half *input,
    device half *output,
    const device half *weight,
    constant NormConfig &config
) {
    float sum = 0.0f;
    uint i = 0;
    // Wider aligned reads reduce address-generation overhead while scalar
    // component accumulation preserves the reference order exactly.
    for (; i + 4 <= config.width; i += 4) {
        half4 input_values = *(const device half4 *)(input + i);
        float value0 = (float)input_values[0];
        float value1 = (float)input_values[1];
        float value2 = (float)input_values[2];
        float value3 = (float)input_values[3];
        sum += value0 * value0;
        sum += value1 * value1;
        sum += value2 * value2;
        sum += value3 * value3;
    }
    for (; i < config.width; ++i) {
        float value = (float)input[i];
        sum += value * value;
    }
    float inv = rsqrt(sum / (float)config.width + config.epsilon);
    i = 0;
    for (; i + 4 <= config.width; i += 4) {
        half4 input_values = *(const device half4 *)(input + i);
        half4 weight_values = *(const device half4 *)(weight + i);
        output[i] = (half)((float)input_values[0] * inv * (float)weight_values[0]);
        output[i + 1] = (half)((float)input_values[1] * inv * (float)weight_values[1]);
        output[i + 2] = (half)((float)input_values[2] * inv * (float)weight_values[2]);
        output[i + 3] = (half)((float)input_values[3] * inv * (float)weight_values[3]);
    }
    for (; i < config.width; ++i) {
        output[i] = (half)((float)input[i] * inv * (float)weight[i]);
    }
}

kernel void metal_step_rmsnorm(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    const device half *weight [[buffer(2)]],
    constant NormConfig &config [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    rmsnorm_body(input, output, weight, config);
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
    uint head [[thread_position_in_grid]]) {
    uint half_dim = config.head_dim / 2;
    if (head < config.query_heads) {
        uint start = head * config.head_dim;
        float sum = 0.0f;
        for (uint i = 0; i < config.head_dim; ++i) {
            float value = (float)query[start + i];
            sum += value * value;
        }
        float inv = rsqrt(sum / (float)config.head_dim + config.epsilon);
        for (uint i = 0; i < half_dim; ++i) {
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
        float sum = 0.0f;
        for (uint i = 0; i < config.head_dim; ++i) {
            float value = (float)key[start + i];
            sum += value * value;
        }
        float inv = rsqrt(sum / (float)config.head_dim + config.epsilon);
        for (uint i = 0; i < half_dim; ++i) {
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
        for (uint i = 0; i < head_dim; ++i) {
            score += (float)query[q_start + i] * (float)key_cache[key_start + i];
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
    uint row = gid / METAL_STEP_SIMD_WIDTH;
    if (row >= config.output_width) return;
    float value = matrix_dot_q8_simd(input, q8, row, config.input_width, lane);
    if (lane == 0) {
        output[row] = (half)(value + (config.add_residual != 0 ? (float)residual[row] : 0.0f));
    }
}

kernel void metal_step_residual_rmsnorm(
    device half *projection [[buffer(0)]],
    const device half *residual [[buffer(1)]],
    device half *normalized [[buffer(2)]],
    const device half *weight [[buffer(3)]],
    constant ResidualNormConfig &config [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float sum = 0.0f;
    uint i = 0;
    // Keep the residual add and sum in element order, but fetch both inputs
    // with aligned half4 loads to reduce scalar memory instructions.
    for (; i + 4 <= config.width; i += 4) {
        half4 projection_values = *(const device half4 *)(projection + i);
        half4 residual_values = *(const device half4 *)(residual + i);
        float value0 = (float)projection_values[0] + (float)residual_values[0];
        float value1 = (float)projection_values[1] + (float)residual_values[1];
        float value2 = (float)projection_values[2] + (float)residual_values[2];
        float value3 = (float)projection_values[3] + (float)residual_values[3];
        projection[i] = (half)value0;
        projection[i + 1] = (half)value1;
        projection[i + 2] = (half)value2;
        projection[i + 3] = (half)value3;
        sum += value0 * value0;
        sum += value1 * value1;
        sum += value2 * value2;
        sum += value3 * value3;
    }
    for (; i < config.width; ++i) {
        float value = (float)projection[i] + (float)residual[i];
        projection[i] = (half)value;
        sum += value * value;
    }
    float inv = rsqrt(sum / (float)config.width + config.epsilon);
    i = 0;
    for (; i + 4 <= config.width; i += 4) {
        half4 projection_values = *(const device half4 *)(projection + i);
        half4 weight_values = *(const device half4 *)(weight + i);
        normalized[i] = (half)((float)projection_values[0] * inv * (float)weight_values[0]);
        normalized[i + 1] = (half)((float)projection_values[1] * inv * (float)weight_values[1]);
        normalized[i + 2] = (half)((float)projection_values[2] * inv * (float)weight_values[2]);
        normalized[i + 3] = (half)((float)projection_values[3] * inv * (float)weight_values[3]);
    }
    for (; i < config.width; ++i) {
        normalized[i] = (half)((float)projection[i] * inv * (float)weight[i]);
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
    uint row = gid / METAL_STEP_SIMD_WIDTH;
    if (row >= config.output_width) return;
    float gate = matrix_dot_q8_simd(input, gate_q8, row, config.input_width, lane);
    float up = matrix_dot_q8_simd(input, up_q8, row, config.input_width, lane);
    if (lane == 0) output[row] = (half)((gate / (1.0f + exp(-gate))) * up);
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
    uint row = gid / METAL_STEP_SIMD_WIDTH;
    if (row >= config.output_width) return;
    float value = matrix_dot_q8_simd(input, q8, row, config.input_width, lane);
    if (lane == 0) logits[row] = value;
}
