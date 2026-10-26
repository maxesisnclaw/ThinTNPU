# Scope and frozen-graph contract

## Supported execution path

`WholeBlocked` is an opt-in, inference-only integration for classic YOLOv5, YOLOv8 and YOLO11, not a complete PyTorch device backend. It accepts fused evaluation models with frozen FP16 parameters, batch 1 and three-level non-export Detect heads. Classic v5 uses anchors/objectness; v8/v11 use anchor-free DFL with 16 bins. Supported blocks include C3/C2f/C3k2/C3k, SPPF, nearest-neighbor upsample and C2PSA attention, subject to their channel/layout contracts.

Dense undilated convolution and multiplier-one depthwise are distinct supported cases. The model adapter accepts depthwise only when `groups == in_channels == out_channels`, channels are C0-aligned, and the kernel is k3/s1/p1 with SiLU or Identity. **General grouped convolution and depthwise multiplier >1 are unsupported.** YOLO11n's eight-channel intermediate layouts are not supported. Training, gradients, arbitrary modules/batches, Focus/reorg stems, export/end-to-end/segmentation heads and arbitrary dilated convolution are outside scope.

External inputs must be allocation-base, contiguous FP16 NCHW `[1,3,H,W]` tensors on the NPU. H and W must be positive multiples of 32. Internal features use NC1HWC0; Detect terminals restore NCHW. Each graph has one fixed shape/dtype/device/stride contract. Capture a separate graph for another shape; there is no H/W-only graph cache.

The shape-generalized contract belongs to `WholeBlocked`. The separate `BlockedC2f` pilot retains its original fixed-region contract, and legacy prepared-NCHW kernels still contain narrower spatial fast paths. Do not infer that these older entry points support arbitrary /32 whole-model inputs.

## Freeze → capture → replay without mutation

1. Load a trusted checkpoint, fuse batch normalization, select eval mode, disable gradients and move the model to FP16 NPU tensors under `torch.no_grad()`. Do **not** construct parameters under `inference_mode()`: frozen-state validation requires version counters.
2. Run `prepare_model(model, fuse_silu=True)`, construct and enable `WholeBlocked`, then capture `FrozenGraph(blocked, blocked.run, example)`.
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

The shared implementation combines general layout/strided-bmm/depthwise primitives with explicit model-family adapters. Classic v5 decoding obtains anchors, class counts and strides from the supplied module; YOLO11 attention uses logical channel ranges over blocked QKV, not checkpoint or image-size dispatch. Public model definitions and weights remain separate inputs. No application-specific loader is provided.

The unified five-round synchronized capture measurement gives Thin/native latency ratios of 0.977× for v8s, 1.271× for v5m and 1.195× for v11s. V8 is near native; v5/v11 remain around a 1.2–1.3× floor for the measured fixtures. These are latency ratios, not universal performance bounds or evidence that integration caused a speedup. The models have different class counts and separately supplied weights. See [family capture data](benchmarks/family_capture_640.json) for every round and sample. Historical v8 data remains separate. Family names alone do not guarantee every size/revision is supported; v26 and other families are unverified.

Hardware validation is limited to Ascend 310P3 (`dav-2002`). No compatibility or performance claim is made for other Ascend chips. No chip-level parity claim follows from a particular model's synchronized graph latency.

Planner ranking coefficients are calibrated defaults for 310P3 in one CANN/container environment. They do not encode a shape whitelist or establish portable timing constants. Recalibrate when hardware, drivers or runtime environments change; legal plans still depend on the implemented capacity checks.
