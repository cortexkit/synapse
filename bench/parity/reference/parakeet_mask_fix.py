"""synapse-parakeet-mask-fix-v1: define fully masked eager attention rows.

Replaces transformers 5.16.1 modeling_parakeet.eager_attention_forward (241-263).
The upstream pairwise mask (615-620) makes padded query rows entirely -inf.
Ordinary softmax then creates NaNs. NeMo uses a finite sentinel before softmax
and zeroes masked attention probabilities afterwards. No other operator changes.
"""

REVISION = "synapse-parakeet-mask-fix-v1"


def eager_attention_forward(module, query, key, value, attention_mask,
                            scaling, dropout=0.0, **kwargs):
    import torch
    from transformers.models.parakeet.modeling_parakeet import repeat_kv

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    masked = None
    if attention_mask is not None:
        masked = torch.isneginf(attention_mask)
        scores = scores + attention_mask
        scores = scores.masked_fill(masked, -10000.0)
    weights = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if masked is not None:
        weights = weights.masked_fill(masked, 0.0)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    output = torch.matmul(weights, value_states).transpose(1, 2).contiguous()
    return output, weights
