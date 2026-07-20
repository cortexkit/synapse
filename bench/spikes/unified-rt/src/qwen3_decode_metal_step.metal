#include <metal_stdlib>
using namespace metal;

constant uint Q8_BLOCK_BYTES = 34;
constant uint Q8_BLOCK_ELEMENTS = 32;

struct NormConfig {
    uint width;
    float epsilon;
};

struct MatvecConfig {
    uint input_width;
    uint output_width;
    uint quantized;
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
};

struct AttentionConfig {
    uint query_heads;
    uint kv_heads;
    uint head_dim;
    uint capacity;
    uint position;
};

inline float matrix_value(
    const device half *fp16,
    const device uchar *q8,
    uint quantized,
    uint row,
    uint col,
    uint input_width
) {
    if (quantized == 0) {
        return (float)fp16[row * input_width + col];
    }
    uint linear = row * input_width + col;
    uint block = linear / Q8_BLOCK_ELEMENTS;
    uint lane = linear % Q8_BLOCK_ELEMENTS;
    const device half *scale_ptr = (const device half *)(q8 + block * Q8_BLOCK_BYTES);
    const device char *value_ptr = (const device char *)(q8 + block * Q8_BLOCK_BYTES + 2);
    return (float)scale_ptr[0] * (float)value_ptr[lane];
}

inline float matrix_dot(
    const device half *input,
    const device half *fp16,
    const device uchar *q8,
    uint quantized,
    uint row,
    uint input_width
) {
    float sum = 0.0f;
    for (uint col = 0; col < input_width; ++col) {
        sum += (float)input[col] * matrix_value(fp16, q8, quantized, row, col, input_width);
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
    for (uint i = 0; i < config.width; ++i) {
        float value = (float)input[i];
        sum += value * value;
    }
    float inv = rsqrt(sum / (float)config.width + config.epsilon);
    for (uint i = 0; i < config.width; ++i) {
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
    if (gid < config.query_width) {
        query[gid] = (half)matrix_dot(input, q_fp16, q_q8, config.quantized, gid, config.input_width);
    }
    if (gid < config.kv_width) {
        key[gid] = (half)matrix_dot(input, k_fp16, k_q8, config.quantized, gid, config.input_width);
        uint head = gid / config.head_dim;
        uint dimension = gid % config.head_dim;
        value_cache[(head * config.capacity + config.position) * config.head_dim + dimension] =
            (half)matrix_dot(input, v_fp16, v_q8, config.quantized, gid, config.input_width);
    }
}

kernel void metal_step_qk_norm_rope(
    device half *query [[buffer(0)]],
    device half *key [[buffer(1)]],
    const device half *query_weight [[buffer(2)]],
    const device half *key_weight [[buffer(3)]],
    const device half *rope_cos [[buffer(4)]],
    const device half *rope_sin [[buffer(5)]],
    constant QkConfig &config [[buffer(6)]],
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
            float first = (float)query[start + i] * inv * (float)query_weight[i];
            float second = (float)query[start + half_dim + i] * inv * (float)query_weight[half_dim + i];
            float cosine = (float)rope_cos[i];
            float sine = (float)rope_sin[i];
            query[start + i] = (half)(first * cosine - second * sine);
            query[start + half_dim + i] = (half)(second * cosine + first * sine);
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
            float first = (float)key[start + i] * inv * (float)key_weight[i];
            float second = (float)key[start + half_dim + i] * inv * (float)key_weight[half_dim + i];
            float cosine = (float)rope_cos[i];
            float sine = (float)rope_sin[i];
            key[start + i] = (half)(first * cosine - second * sine);
            key[start + half_dim + i] = (half)(second * cosine + first * sine);
        }
    }
}

kernel void metal_step_attention(
    const device half *query [[buffer(0)]],
    const device half *key_cache [[buffer(1)]],
    const device half *value_cache [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant AttentionConfig &config [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]]) {
    uint query_head = threadgroup_position.x;
    if (query_head >= config.query_heads) return;
    uint head_dim = config.head_dim;
    uint group = config.query_heads / config.kv_heads;
    uint kv_head = query_head / group;
    uint q_start = query_head * head_dim;
    float scale = rsqrt((float)head_dim);
    threadgroup float scores[2048];
    threadgroup float maximum;
    threadgroup float denominator;

    for (uint position = 0; position <= config.position; ++position) {
        float partial = 0.0f;
        uint key_start = (kv_head * config.capacity + position) * head_dim;
        for (uint i = lane; i < head_dim; i += 32) {
            partial += (float)query[q_start + i] * (float)key_cache[key_start + i];
        }
        float score = simd_sum(partial) * scale;
        if (lane == 0) scores[position] = score;
    }
    float partial_max = -INFINITY;
    for (uint position = lane; position <= config.position; position += 32) {
        partial_max = max(partial_max, scores[position]);
    }
    float max_value = simd_max(partial_max);
    if (lane == 0) maximum = max_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial_denominator = 0.0f;
    for (uint position = lane; position <= config.position; position += 32) {
        partial_denominator += exp(scores[position] - maximum);
    }
    float sum_value = simd_sum(partial_denominator);
    if (lane == 0) denominator = sum_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < head_dim; i += 32) {
        float result = 0.0f;
        for (uint position = 0; position <= config.position; ++position) {
            uint value_index = (kv_head * config.capacity + position) * head_dim + i;
            result += exp(scores[position] - maximum) / denominator * (float)value_cache[value_index];
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
    if (gid >= config.output_width) return;
    float value = matrix_dot(input, fp16, q8, config.quantized, gid, config.input_width);
    output[gid] = (half)((float)residual[gid] + value);
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
    for (uint i = 0; i < config.width; ++i) {
        float value = (float)projection[i] + (float)residual[i];
        projection[i] = (half)value;
        sum += value * value;
    }
    float inv = rsqrt(sum / (float)config.width + config.epsilon);
    for (uint i = 0; i < config.width; ++i) {
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
    if (gid >= config.output_width) return;
    float gate = matrix_dot(input, gate_fp16, gate_q8, config.quantized, gid, config.input_width);
    float up = matrix_dot(input, up_fp16, up_q8, config.quantized, gid, config.input_width);
    output[gid] = (half)((gate / (1.0f + exp(-gate))) * up);
}

kernel void metal_step_lm_head(
    const device half *input [[buffer(0)]],
    device float *logits [[buffer(1)]],
    const device half *fp16 [[buffer(2)]],
    const device uchar *q8 [[buffer(3)]],
    constant MatvecConfig &config [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= config.output_width) return;
    logits[gid] = matrix_dot(input, fp16, q8, config.quantized, gid, config.input_width);
}
