# Scope and frozen-graph contract

## Supported execution path

`WholeBlocked` is an opt-in, inference-only integration for classic YOLOv5, YOLOv8, YOLO11 and YOLO26, not a complete PyTorch device backend. It accepts fused evaluation models with frozen FP16 parameters, batch 1 and three-level non-export Detect heads. Classic v5 uses anchors/objectness; v8/v11 use anchor-free DFL with 16 bins. The supported YOLO26 dictionary head uses direct ltrb distances (`reg_max=1`), no DFL, and a fused one-to-one end-to-end branch. Supported blocks include C3/C2f/C3k2/C3k, ordinary/residual SPPF, nearest-neighbor upsample, C2PSA and nested PSABlock attention, subject to their channel/layout contracts.

Dense undilated convolution and multiplier-one depthwise are distinct supported cases. The model adapter accepts depthwise only when `groups == in_channels == out_channels`, channels are C0-aligned, and the kernel is k3/s1/p1 with SiLU or Identity. **General grouped convolution and depthwise multiplier >1 are unsupported.** YOLO11n's eight-channel intermediate layouts are not supported. Training, gradients, arbitrary modules/batches, Focus/reorg stems, export/segmentation and other unsupported head contracts and arbitrary dilated convolution are outside scope.

External inputs must be allocation-base, contiguous FP16 NCHW `[1,3,H,W]` tensors on the NPU. H and W must be positive multiples of 32. Internal features use NC1HWC0; Detect terminals restore NCHW. Each graph has one fixed shape/dtype/device/stride contract. Capture a separate graph for another shape; there is no H/W-only graph cache.

The shape-generalized contract belongs to `WholeBlocked`. The separate `BlockedC2f` pilot retains its original fixed-region contract, and legacy prepared-NCHW kernels still contain narrower spatial fast paths. Do not infer that these older entry points support arbitrary /32 whole-model inputs.

## Freeze → capture → replay without mutation

1. Load a trusted checkpoint, select the intended branch before fusion (YOLO26: `model.end2end = True`), fuse batch normalization, select eval mode, disable gradients and move the model to FP16 NPU tensors under `torch.no_grad()`. Do **not** construct parameters under `inference_mode()`: frozen-state validation requires version counters.
2. Run `prepare_model(model, fuse_silu=True)`, construct and enable `WholeBlocked` with the original frontend's attention scaling order (YOLO26: `attention_scale_before_matmul=True`), then capture `FrozenGraph(blocked, blocked.run, example)`.
3. Keep the model, prepared weights, padded biases, wrapper and captured graph alive. Replay on the creating thread and stream, without concurrent execution or mutation. `graph(value)` validates input metadata, copies NPU input into the graph's static buffer and replays. `graph.replay()` reuses that static input.
4. Returned outputs are borrowed graph-owned buffers and are overwritten by later replays. Copy/clone results before retaining them. Operations are asynchronous; synchronize when timing or consuming results outside the NPU stream. Moving to CPU provides the synchronization used in the example.
5. Close the graph on its owner thread before destroying the model/wrapper. To change weights, model structure, operator configuration or Detect output mode, close the old graph, reconstruct/reprepare the model as needed, and capture again. A graph is not an automatically invalidated model cache.

## Guards are not a proof of immutability

The guard checks tracked module structure, forward bindings, tensor identities/metadata/version counters, and selected convolution/pool/upsample configuration. Shape, dtype, device and strides are checked on copied graph inputs; resource checks reject unsupported kernel plans. **Not every possible mutation is guarded.** In particular:

- Writes through `.data` aliases can evade the tracked tensor's version counter. Prepared weights or graph state may then be stale without an exception. Do not mutate parameters or buffers through `.data`, storage aliases, pointers, or external device writes.
- `Detect.export` is included in the unified attribute guard. Untracked attributes such as `Detect.format` are not covered: replay can silently use the old output contract even though a fresh eager forward would differ. Keep all Detect configuration fixed; close and recapture after changes. Tracking one attribute is not a proof of complete immutability.
- Validation is not a concurrency or memory-safety boundary against external code. Unsupported untracked mutations are caller contract violations, not automatically detected updates.

The intended behavior for rejected input/configuration/resource cases is fail-loud, without silently switching precision or substituting CPU execution. The known guard gaps above are disclosed limitations of that policy, not a claim that every invalid use currently fails loudly.

## Shape and numerical boundaries

The generalized kernels derive spatial plans from dimensions and hardware capacities, not a fixed image-size whitelist. They still have finite L1/UB/L0 buffers, DMA-field/index ranges and physical-memory limits. Extremely large /32 images may fail resource checks or allocation. Multiples of 32 do not promise unbounded image size.

Tail handling prioritizes correct values over speed. In the historical v8 shape sweep, 352×608 replay took about 10.78 ms versus about 4.67 ms for 640×640. These are single-sample observations, not the controlled five-round comparison used for the historical v8 table. Padding/compaction and generic DFL softmax can produce a performance cliff despite fewer pixels.

FP16 inference is not bitwise identical to a CPU FP32 model. Convolution accumulates in FP32 and computes bias/SiLU before the final FP16 cast, but inputs, parameters and features are quantized. Bitwise claims refer to Thin eager/capture/replay and each family's corresponding existing Thin fixture, not the CPU FP32 reference, another family's fixture or all NaN payloads. Keep CPU FP32 and FP16-parameter/activation reference comparisons separate.

The unified validation reproduces these errors on the same YOLOv8s fixture (see [family validation](benchmarks/family_validation.json)):

| Input | Coordinate max absolute error | Normalized coordinate max | Normalized coordinate mean | Class-probability max error |
|---|---:|---:|---:|---:|
| 640×640 | 0.511474609 | 0.000799179077 | 0.000124578772 | 0.000577796251 |
| 1280×1280 | 1.12023926 | 0.000875186920 | 0.000123525586 | 0.000328999013 |

Coordinates are normalized by `max(H, W, max(abs(CPU decoded coordinates)))`. The 1280² normalized maximum is about 9.5% higher; the normalized mean is slightly lower. This is not a non-increasing maximum-error guarantee, a detection-quality/mAP result, or a universal accuracy bound. Resize/letterbox and source-image coordinate restoration belong to preprocessing/postprocessing, not this backend.

## Generalization within YOLO

The shared implementation combines general layout/strided-bmm/depthwise primitives with explicit model-family adapters. Classic v5 decoding obtains anchors, class counts and strides from the supplied module; YOLO11/YOLO26 attention uses logical channel ranges over blocked QKV, not checkpoint or image-size dispatch. Public model definitions and weights remain separate inputs. No application-specific loader is provided.

The unified five-round synchronized capture measurements retain every round/sample in [family capture data](benchmarks/family_capture_640.json). Ratios are Thin/native latency, not speedup. Models have different class counts and weights; neither absolute cross-family comparisons nor a family name alone guarantees support for every size/revision. YOLO11n and general grouped convolution remain unsupported.

Attention uses sequential-order FP32 accumulation in vector bmm plus vector softmax, not a native fused attention engine. These shared primitives remain a latency floor for v11/v26. The v5/v11/v26 cases remain above native latency in the measured comparison; classic v5 has no attention and has a distinct residual convolution/layout floor. No portable latency bound follows from these observations.

## YOLO26 output and selection

Select one-to-one **before** fusing. Setting a flag after fusion cannot restore the removed branch. Q prescaling versus post-matmul scaling is a numerical frontend contract, not algebraic freedom; half rounding changes if they are interchanged. The guard tracks the selected attention/Detect options but is still not a complete immutability proof.

Direct ltrb distances decode as `(anchors - left_top, anchors + right_bottom) * strides`, without DFL, sigmoid or clipping on distances. Classification independently uses sigmoid. Supported NMS-free selection stays on-device: class maximum, anchor top-k, class-score gather, flattened candidate top-k, anchor mapping, box gather and output packing.

The measured YOLO26s 640² end-to-end output is FP32 `[1,300,6]` (`xyxy, score, class`). More generally it is `[B,min(max_det,anchors),6]` for the selection primitive; primitive multi-batch support does not broaden the batch-one whole-model contract. No confidence threshold, IoU suppression, clipping or original-image coordinate restoration is included. Packing promotes labels and the final tensor to FP32; coordinates/scores were computed in FP16.

Equal scores use deterministic **lower-index** ordering. This does not reproduce `torch.topk`'s unspecified tied-index order. Same-input selected tuples can therefore differ, especially with many tied low-confidence scores. Whole-model CPU-reference differences also include FP16 numerical changes before selection. Selection reference tests, unchanged prior-Thin tensor outputs and eager/capture equality do not imply equality to official final detections or validated mAP. Complete real-image outputs, including low-confidence rows, remain part of regression coverage; reference errors are preserved in [validation data](benchmarks/family_validation.json).

Selection's K≤512 retained-prefix merge and larger-K complete merge-path sort use the same score/index total order. UB capacities are tiling choices, not model/image/class lookup tables. All 65,536 half encodings are exercised in selection tests; this is not a promise that every other operator preserves every NaN payload. See [selection](SELECTION_FLOW.md) and [coverage](SELECTION_GENERALIZATION.md).

Hardware validation is limited to Ascend 310P3 (`dav-2002`). No compatibility or performance claim is made for other Ascend chips. No chip-level parity claim follows from a particular model's synchronized graph latency.

Planner ranking coefficients are calibrated defaults for 310P3 in one CANN/container environment. They do not encode a shape whitelist or establish portable timing constants. Recalibrate when hardware, drivers or runtime environments change; legal plans still depend on the implemented capacity checks.
