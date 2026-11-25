# MegaMoE integration recipe (DSv4.1-Flash NVFP4, gen-B DEP32 and gen-C DEP8)

Ops in this directory (`models/deepseek_v41/ops/moe`): `megamoe_expert_slots`, `megamoe_pack_experts`, `megamoe_shared_expert`, `megamoe_stage_inputs`,
`megamoe_workspace`, `cute_dsl_megamoe_nvfp4` (+ the shared catalog entry `quantization/fp4_quantize`). Verified on sm_103 (GB300) at EP4.

## Deployment facts (from the benchmark configs and the stock code)

- gen-B: EP32, `load_balancer.num_slots: 416` (13 slots/rank, 32 copies), `max_num_tokens 150`, bucket ladder `[150]`.
  gen-C: EP8, 384 slots (one copy), `max_num_tokens 1280`, ladder `[256, 1024, 1280]`. Both attention-DP, pure DEP
  (WORLD == EP, required by the kernel's barrier fence), top-6, hidden 5120, intermediate 2304, `swiglu_limit` from the config.
- `layer_updates_per_iter: 0` = static routing: no per-iteration balancer. The backend maps expert ids to slots itself from
  the table built from `initial_global_assignments`; no statistics are needed to run.
- `use_low_precision_moe_combine: true` has NO effect on MegaMoE (it only configures the other comm strategies). The kernel
  reads env `MEGAMOE_COMBINE_FORMAT` (default `bf16`), unset in the configs, and the fused shared expert requires bf16.
  So the stock gen configs run a bf16 combine. `32e4m3xe8m0` / `16e2m1xbf16` exist (see the docstring of `cute_dsl_megamoe_nvfp4`) but are not what stock runs.
- The stock shared-expert fusion (`megamoe_fuse_shared_expert`) is on only for sm_100/103, attention DP, `moe_tp_size 1`, one shared expert.

## Build time (every rank, same order, outside any forward/capture)

1. `pg = megamoe_ep_process_group(mapping)`.
2. `buckets = megamoe_token_buckets(max_num_tokens, 103)`.
3. Placement: `assign = megamoe_default_placement(384, num_slots, ep)`; `megamoe_check_placement(...)`; local slots are
   `assign[ep_rank*S : (ep_rank+1)*S]`, `S = num_slots // ep`.
4. Per layer, per local slot with expert `e` (checkpoint names `layers.N.ffn.experts.e.{w1,w2,w3}.{weight,scale?,...}`; in the
   engine loader's naming `weight`, `weight_scale`, `weight_scale_2`, `input_scale`): stack `w1` (gate), `w3` (up) `[S,I,H/2]` u8 with
   `weight_scale` `[S,I,H/16]` e4m3, and `w2` `[S,H,I/2]`, `[S,H,I/16]`. `scalars = megamoe_expert_scalars(i1, isc2, ws2_w1, ws2_w3, ws2_w2)`
   with `i1 = max over ALL 384 experts of max(w1.input_scale, w3.input_scale)` (the existing TRTLLM loader already computes it) and
   per-slot vectors for the others. `fc1, fc1_sf, fc2, fc2_sf = megamoe_pack_experts(..., scalars["w3_scale_fold"])`. Keep `fc1_alpha`,
   `fc2_alpha`, `fc1_norm_const`, `input_global_scale` as buffers. A copy slot gets the same bytes as its twin. Pack in chunks (done internally, 16 slots).
   Note: `megamoe_pack_experts` needs the scales' `view(uint8)`-able dtype; the checkpoint's expert scales are e4m3.
5. Shared expert (FP8 block-32 in the checkpoint: `layers.N.ffn.shared_experts.{w1,w3,w2}.{weight,scale}`, scale `[R/32, C/32]` UE8M0):
   `megamoe_pack_shared_expert(w1, s1, w3, s3, w2, s2, cfg.swiglu_limit, (32, 32))` (keep fp8 weights; do NOT dequantize them to bf16 for
   this path); `megamoe_shared_activation_buffers(b, H, dev)` for every bucket `b`. The unfused path (bf16 dequantized weights, as today) stays valid when the fusion is off.
6. `shared_ws, _ = megamoe_shared_workspace_bytes(...)` and `regions[b] = megamoe_symm_regions(...)` for every bucket, plus
   `idx[b] = megamoe_topk_idx_buffer(b, 6)`; `table = megamoe_slot_table(assign, 384, ep_rank)` (one per layer, rewritable in place).
   All layers share `regions[b]`, `idx[b]` and the activation buffers.
7. `gate_up_clamp = swiglu_limit` as a plain float in REAL units (NOT divided by alpha as for `fp4_block_scale_moe_runner`).

## Forward (per layer; routing stays the target's own: sqrtsoftplus, bias-selected, `norm_topk_prob`, `routed_scaling_factor`)

```python
n = x.shape[0]; b = megamoe_pick_bucket(buckets, max(all_rank_num_tokens)); r = regions[b]
ids, w = route(x)                                   # int32 [n,6] ids, f32 weights (target's routing, unchanged)
slots = megamoe_experts_to_slots(table, ids, 384)   # int32 [n,6]
if n: x_fp4, x_sf = fp4_quantize(x, input_global_scale, 16, False, False); x_sf = x_sf.view(n, H // 16)
      if fused_shared: megamoe_mxfp8_quantize(x, x8[b], x8_sf[b])
else: empty [0, H/2] and [0, H/16] u8
megamoe_stage_inputs(slots, w.float().contiguous(), idx[b], r.topk_weights, x_fp4, x_sf, r.activation, r.activation_sf)
y = cute_dsl_megamoe_nvfp4(r.activation, r.activation_sf, idx[b], r.topk_weights, fc1, fc1_sf, fc2, fc2_sf,
        fc1_alpha, fc2_alpha, fc1_norm_const, r.shared_workspace, r.peer_offsets, world_size=ep, local_rank=ep_rank,
        max_tokens_per_rank=b, num_tokens=n, gate_up_clamp=swiglu_limit, **shared_kwargs)   # [n, H] bf16
```
With attention DP every rank passes its own `n` rows (no allgather/reducescatter: the kernel does the exchange and returns each rank's rows).
`y` already includes the shared expert when fused: do not add it again (and do not run the separate shared expert).
Every rank must execute the call every layer, also with `n == 0`. This is the sequence the test matrix runs (`tests/unittest/_torch/modeling_v2/models/deepseek_v41/_cute_dsl_megamoe_nvfp4_op_matrix.py`, `Layer.forward`).

## Placement: is a warmup pass needed?

- To RUN and be correct: no. Any valid placement gives the same function (copies hold identical bytes; verified: replicated vs one-copy
  rel-RMS 0, a shuffled placement applied in place matches the reference). `megamoe_default_placement` needs only (E, num_slots, ep).
- What stock does with `warmup_placement` (+ `layer_updates_per_iter: 0`): the layers start at exactly that default placement. A controller counts the
  routed picks of live rows of real requests (no ADP dummies / graph padding; generation rows with >= `min_pos` tokens; every `count_every_steps`=4
  forwards from `start_step` 0), and every `interval_steps` (12000 gen-B, 4000 gen-C) once `min_rows` (1,048,576) new rows are counted, rank 0 re-plans
  (`planner: cost` = modeled MegaMoE cost per expert on gen-B; `lpt` = counted picks on gen-C) by swapping slots between heavy and light ranks (at most
  `max_moved_fraction` 0.1 of a layer's slots, `min_gain` 0.01, `min_round_gain` 0.5, counters decayed by 0.5), with extra slots (gen-B's 32) turned into
  copies of hot experts; the moved per-slot tensors travel GPU to GPU over NCCL in place, the slot table is rewritten in place (so captured graphs stay valid).
  `max_migrations` 100000: it keeps following traffic for the whole run. So the learned placement is NOT derivable without runtime statistics; the default
  placement is the uniform start. A target without the controller runs the default placement: same outputs, possible rank imbalance (a throughput difference to measure).
- The dep8 / dep32 targets reuse the stock controller (`ops/moe/megamoe_placement.py`): each `_MegaMoe` registers from its constructor
  (`megamoe_placement_register`: a view exposing the per-slot tensors under the names the controller moves, `mega_fc1_weight`,
  `mega_fc1_weight_sf`, `mega_fc2_weight`, `mega_fc2_weight_sf`, `fc31_alpha`, `fc2_alpha`, `fc1_norm_const`, every other tensor of the layer under
  its own name so an unknown per-slot tensor makes the controller refuse the layer, and `on_expert_placement_changed` rewriting the slot table in
  place with `megamoe_slot_table`); the route launch counts the picks (`dsv4_route_nvfp4_quant(counters=placement.counters())`). The engine
  balancer's own device table is not used (the target routes with its slot table), so the layer is registered with a no-op table holder.
  Test: `test_modeling_v2_megamoe_placement.py` (one round over 4 EP ranks, both targets).

## DSpark draft

`speculative_config.moe_backend: MEGAMOE_CUTEDSL` -> `_get_requested_draft_moe_backend` -> `DSv41DSparkDraftModel(draft_moe_backend=...)`: the draft's routed
experts (checkpoint MXFP4, 128 experts, top-3) are re-encoded as NVFP4 at load (`remap_dspark_v41_draft_keys(nvfp4_experts=True)`; exact: E8M0 scales are powers of
two, FC1/FC2 activation scales are model-derived bounds) and its MoE is created with the MegaMoE backend; its load balancer config is the target's (shared slots,
copies) or its own one-copy placement when the target learns placement in warmup (the "draft" row pass). Our targets use the engine's draft class as a black box, so
the draft path is selected by that config alone; a hand-written draft would need `megamoe_*` on its 128-expert bank with `num_slots` divisible by EP (a separate
workspace: top-3 and `block_size x requests` rows; the symmetric layout key includes top-k, so the draft gets its own regions).

## Caveats to carry into the target

- Graph capture: warm every bucket eagerly (a normal engine capture warm-up does); do not interleave kernel variants over the same buffers after the first capture.
- The output of the op is a fresh tensor, not an alias of a symmetric buffer (certified path), so it is safe to keep across the next layer.
- First launch per bucket compiles the kernel (CuTe-DSL JIT, tens of seconds); set `CUTE_DSL_CACHE_DIR` like the configs do.
- sm_103 only is certified here; K<=16, EP-only (moe_tp 1), no TEP.
