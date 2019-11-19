import torch
import argparse
import gc
import time
import sys

# Notes
# Why isn't batch-size first?
# What's the point of embed_dim_to_check?
# Add docs that documents the shape in_proj_weight needs to be
# Indexing ops are expensive
# How can bias k and bias v be optional before a positional argument?
# Why is nn.MultiHeadAttention using Linear for bias and weight instead of what it does for in_proj_weight


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_bench(*args, **kwargs):
    mha = gen_run_one_bench(*args, **kwargs)
    all_time = 0.0
    while(all_time < 1.0):  # Warmup for 1s
        ti = time.time()
        mha()
        all_time += time.time() - ti
    times = []
    all_time = 0.0
    while(all_time < 2.0):  # Run for 2s
        # gc.collect() TODO: Test for additional stability
        ti = time.time()
        mha()
        ti = time.time() - ti
        times.append(ti)
        all_time += ti
    times = torch.tensor(times)
    times = times * 1e6
    return int(times.mean().item()), int(times.std().item()), len(times)


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
    training = True
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


def tuple_prod(tup):
    r = 1
    for t in tup:
        r *= t
    return float(r)

basic_configs = []
for L in [16, 32]:
    for N in [1, 64, 256]:
        for embed_dim in [16, 256, 1024]:
            for num_heads in [1, 8, 16]:
                basic_configs.append((L, N, embed_dim, num_heads))
extra_configs = [
    (True, True, False, 'cpu'),
    (True, False, True, 'cpu'),
    (True, False, False, 'cpu'),
    (False, True, False, 'cpu'),
    (False, False, True, 'cpu'),
    (False, False, False, 'cpu'),
    (True, True, False, 'cuda'),
    (True, False, True, 'cuda'),
    (True, False, False, 'cuda'),
    (False, True, False, 'cuda'),
    (False, False, True, 'cuda'),
    (False, False, False, 'cuda'),
]
keys = ["L", "N", "embed_dim", "num_heads", "use_separate_proj_weight", "qkv_same", "kv_same", "device"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('numthreads')
    args = parser.parse_args()
    torch.set_num_threads(int(args.numthreads))
    eprint("Running {} experiments".format(len(basic_configs) * len(extra_configs)))
    print(",".join(keys + ["avg(us)", "std(us)", "num_runs", "data/avg(us)"]))
    for extra_config in extra_configs:
        for basic_config in basic_configs:
            config = basic_config + extra_config
            avg_time, std_time, num_runs = run_bench(*config)
            dp_per_time = int(tuple_prod(basic_config) / avg_time)
            print(",".join(map(str, config + (avg_time, std_time, num_runs, dp_per_time))))
