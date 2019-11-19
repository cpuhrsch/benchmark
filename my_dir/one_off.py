import torch
import gc
import time

# Notes
# Why isn't batch-size first?
# What's the point of embed_dim_to_check?
# Add docs that documents the shape in_proj_weight needs to be
# Indexing ops are expensive
# How can bias k and bias v be optional before a positional argument?
# Why is nn.MultiHeadAttention using Linear for bias and weight instead of what it does for in_proj_weight


def run_bench(*args, **kwargs):
    mha = gen_run_one_bench(*args, **kwargs)
    for _ in range(3):  # Warmup
        mha()
    times = []
    for _ in range(20):  # Collect data
        gc.collect()
        gc.collect()
        ti = time.time()
        mha()
        times.append(time.time() - ti)
        gc.collect()
        gc.collect()
    times = torch.tensor(times)
    times = times * 1e6
    return int(times.mean().item()), int(times.std().item())


def gen_run_one_bench(L, N, embed_dim, num_heads, use_separate_proj_weight, qkv_same, kv_same, device):
    def my_rand(*args):
        return torch.randn(*args, device=device)
    query = my_rand(L, N, embed_dim)
    key = my_rand(L, N, embed_dim)
    value = my_rand(L, N, embed_dim)
    if qkv_same:
        key = query.clone()
        value = query.clone()
    if kv_same:
        key = value.clone()
    in_proj_weight = my_rand(3 * embed_dim, embed_dim)  # Definitely wrong
    in_proj_bias = my_rand(3 * embed_dim)
    out_proj_weight = my_rand(embed_dim, embed_dim)  # Defoutitely wrong
    out_proj_bias = my_rand(embed_dim)
    bias_k = None
    bias_v = None
    add_zero_attn = False
    dropout = 0.5
    training=True
    key_padding_mask = None
    need_weights = False
    attn_mask = None
    q_proj_weight = my_rand(embed_dim, embed_dim)
    k_proj_weight = my_rand(embed_dim, embed_dim)
    v_proj_weight = my_rand(embed_dim, embed_dim)
    def run_mha():
        torch.nn.functional.multi_head_attention_forward(
            query, key, value, embed_dim, num_heads,
            in_proj_weight, in_proj_bias,
            bias_k, bias_v, add_zero_attn,
            dropout, out_proj_weight, out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight)
        if device == 'cuda':
            torch.cuda.synchronize()
    return run_mha


keys = ["L", "N", "embed_dim", "num_heads", "use_separate_proj_weight", "qkv_same", "kv_same", "device"]
configs = [
    (4, 40, 256, 16, True, True, False,'cpu'),
    (4, 40, 256, 16, True, False, True,'cpu'),
    (4, 40, 256, 16, True, False, False,'cpu'),
    (4, 40, 256, 16, False, True, False,'cpu'),
    (4, 40, 256, 16, False, False, True,'cpu'),
    (4, 40, 256, 16, False, False, False,'cpu'),
    (4, 40, 256, 16, True, True, False,'cuda'),
    (4, 40, 256, 16, True, False, True,'cuda'),
    (4, 40, 256, 16, True, False, False,'cuda'),
    (4, 40, 256, 16, False, True, False,'cuda'),
    (4, 40, 256, 16, False, False, True,'cuda'),
    (4, 40, 256, 16, False, False, False,'cuda'),
]
print(",".join(keys + ["avg(us)", "std(us)"]))
for config in configs:
    print(",".join(map(str, config + run_bench(*config))))
