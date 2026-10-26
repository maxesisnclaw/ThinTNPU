# Device detection selection

```python
torch.ops.load_library("thin_backend/libselection_flow.so")
selected = torch.ops.thin_native_flow.detection_select(decoded, max_det, agnostic=False)
```

Input: inference-only PrivateUse1 FP16 `[B,A,4+nc]`, `nc>0`, arbitrary nonnegative strides (including the usual transpose of `[B,4+nc,A]`); coordinates are already decoded. `max_det>=0`. Output: contiguous FP32 `[B,min(max_det,A),6]`, rows `[box0,box1,box2,box3,score,class]`. Empty batch/anchor axis and zero `max_det` are supported.

Class-aware selection performs class maximum, anchor top-k, score gather, flattened anchor/class top-k, index mapping and box gather. Duplicate anchors with different classes are valid. Agnostic selection performs first class argmax and one anchor top-k. No threshold, IoU, NMS, clipping or coordinate rescaling is added.

## Tie contract

Equal anchor scores are ordered by original anchor index. Equal flattened scores are ordered by their index in gathered `[K,nc]`. Class maxima choose the first equal class. NaNs sort above numerical values; class reduction retains the first NaN. Signed zeros compare equal. Score bits survive selection unchanged; finite half values (including subnormals/signed zero) expand exactly to FP32, while NaNs preserve sign/payload and are quieted. Labels convert to FP32.

**This is not a claim of CPU `torch.topk` tie-index or output parity.** PyTorch explicitly does not promise stable tied indices. Different valid ties at either cutoff can produce different boxes, especially with FP16 sigmoid scores. Device eager/capture bit equality is a separate invariant. Exact official-reference parity can be required for tie-free inputs, but must not be inferred from exact score equality on tied inputs.

## Execution

AscendC sorts 128-record runs with `RpSort16` and `MrgSort`, then merges sorted runs. Tile size is a UB granularity, not a model/shape table or maximum top-k. Scratch scales with anchors and gathered pairs. All reductions, comparisons, gathers, index arithmetic and packing run on the device. Host work is shape/stride calculation, allocation and launch only.

Initial tiles sort by the exact integer key `score_key * 128 + local_index` represented in FP32. The complete key, including padding, is below `2^24`. Hardware sort stability is not assumed. The original FP16 score bits, index and label/anchor remain integer payloads, permuted with uint32 Gather; scores are never rounded or reconstructed from sorting keys.

For `K<=512`, each merged run retains its first `ceil(K/16)*16` records. A record below that prefix already has at least K better records in its own run and cannot enter the global top K. Each two-way merge adds a left/right discriminator to the exact score key: the left run covers a lower original-index interval, and each stream preserves its existing order. Larger K uses full merge-path sorting. Both paths implement the same total order; the capacity is a UB working-set choice, not an API limit.

For anchor-contiguous input, class maxima use vector comparisons of exact half-bit ordering keys and preserve the first winning class and its original payload. Aligned channel planes load 16 classes at once and reduce their keyed class ranks. Unaligned channel planes use per-class vectors; other strided layouts use direct scalar reads. Selected-score gathering uses a bounded UB working set when the anchor plane and gathered-score matrix fit, otherwise it reads directly from the supplied strides. These choices depend on layout and working-set size, not model identity or observed data values.

Scratch uses the backend capture-retained allocator. Kernels use `thin::stream()` and capture-aware `thin::sync()`. There is no CPU selection, scalar readback, `torch_npu` or host-initialized tensor data. The extension does not register generic ATen top-k/gather and does not alter model adapters.

## Build

Build from the ThinTNPU source root with the CANN environment enabled:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build_selection_flow.sh
```

The build expects `thin_backend/libthin_backend.so`; `THIN_BACKEND_LIB_DIR` may point to an existing baseline library directory when compiling a standalone worktree. Load the matching baseline before this extension.

## Stage inspection and verification

`detection_select_trace(decoded, max_det, agnostic=False)` returns device tensors in this order:

1. `output`: FP32 `[B,K,6]`, identical to the production interface.
2. `class_max`: FP16 `[B,A]`, captured before sorting the anchor runs.
3. `stage1_indices`: int64 `[B,K]`, original anchor indices.
4. `gathered_scores`: FP16 `[B,K,nc]`, captured before sorting pair runs; `[B,0,nc]` in agnostic mode.
5. `top_values`: FP16 `[B,K]`.
6. `stage2_indices`: int64 `[B,K]`, flattened gathered-pair indices; `[B,0]` in agnostic mode.
7. `anchor_indices`: int64 `[B,K]`, final mapped original anchors.
8. `labels`: int64 `[B,K]`.

Trace views may be noncontiguous. Trace snapshots are device-to-device writes from the actual selection buffers, not a CPU recomputation or a different selection formula. The production interface does not allocate these trace buffers.

```bash
python3 verify_selection_flow.py --cpu-only
python3 verify_selection_flow.py --save-dir evidence-selection
python3 verify_selection_flow.py --case unique
```

The CPU-only mode does not import `thin_backend` or use an NPU. It exhaustively checks all 65,536 half-bit comparison/conversion patterns, exercises bitonic/merge-path partitions and quantifies the two tie policies on the same inputs. These checks do not constitute device execution acceptance. The stable CPU reference gathers raw half bits through int16 views because CPU half gather quiets signaling NaNs.

Device mode validates every stage against the documented stable reference, verifies that final tuples exist in the input, requires exact official `topk` parity on tie-free cases, and reports official index/row differences for tied cases. It also checks production-versus-trace equality, eager-versus-capture bit equality and changed-input replay. Its 47 fixtures cover arbitrary strides, offset views, multiple batches, empty axes, `K>A`, duplicate-anchor selections, sort-tile boundaries, K=511/512/513, class-block boundaries, large class counts, all 65,536 half patterns (both as class candidates and a complete ordered output), subnormals, infinities, NaNs and two full detection shapes. `--save-dir` saves inputs and all device/reference stages as `.pt` fixtures for independent analysis. `--library` selects a compiled implementation for independent baseline runs; its linked backend must match the already-loaded backend.
