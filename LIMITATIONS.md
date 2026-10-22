# Scope and frozen-graph contract

## Supported execution path

`WholeBlocked` is an opt-in, inference-only YOLO integration, not a complete PyTorch device backend. The tested full model is a fused YOLOv8s with a three-level, non-export, non-end-to-end Detect head and DFL with 16 bins. FP16 parameters and activations, batch 1, dense undilated convolutions, SiLU, SPPF and nearest-neighbor upsample form the validated operator family. The stem and blocked channels have architectural restrictions enforced by the implementation. Training, gradients, arbitrary modules, grouped/dilated convolutions and arbitrary batch sizes are not supported.

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
- Changing `Detect.export` or `Detect.format` after capture is not covered by the current attribute guard. Replay can silently use the old output contract/graph even though a fresh eager forward would behave differently. Keep all Detect configuration fixed; close and recapture after changes.
- Validation is not a concurrency or memory-safety boundary against external code. Unsupported untracked mutations are caller contract violations, not automatically detected updates.

The intended behavior for rejected input/configuration/resource cases is fail-loud, without silently switching precision or substituting CPU execution. The known guard gaps above are disclosed limitations of that policy, not a claim that every invalid use currently fails loudly.

## Shape and numerical boundaries

The generalized kernels derive spatial plans from dimensions and hardware capacities, not a fixed image-size whitelist. They still have finite L1/UB/L0 buffers, DMA-field/index ranges and physical-memory limits. Extremely large /32 images may fail resource checks or allocation. Multiples of 32 do not promise unbounded image size.

Tail handling prioritizes correct values over speed. In the current shape sweep, 352×608 replay took about 10.78 ms versus about 4.67 ms for 640×640. These are single-sample observations, not the controlled five-round comparison used for the headline. Padding/compaction and generic DFL softmax can produce a performance cliff despite fewer pixels.

FP16 inference is not bitwise identical to a CPU FP32 model. Convolution accumulates in FP32 and computes bias/SiLU before the final FP16 cast, but inputs, parameters and features are quantized. Tested eager/captured Thin outputs are bitwise equal; that statement does not apply to the CPU FP32 reference or all NaN payloads.

Coverage on the same YOLOv8s fixture reports:

| Input | Coordinate max absolute error | Normalized coordinate max | Normalized coordinate mean | Class-probability max error |
|---|---:|---:|---:|---:|
| 640×640 | 0.511474609 | 0.000799179077 | 0.000124578772 | 0.000577796251 |
| 1280×1280 | 1.12023926 | 0.000875186920 | 0.000123525586 | 0.000328999013 |

Coordinates are normalized by `max(H, W, max(abs(CPU decoded coordinates)))`. The 1280² normalized maximum is about 9.5% higher; the normalized mean is slightly lower. This is not a non-increasing maximum-error guarantee, a detection-quality/mAP result, or a universal accuracy bound. Resize/letterbox and source-image coordinate restoration belong to preprocessing/postprocessing, not this backend.

## Generalization within YOLO

The implementation is specialized to a YOLO operator domain rather than to one frozen image dimension. The same operator family can be useful for Ultralytics v5/v8/v11/v26. This is an operator-reuse direction, **not** tested end-to-end compatibility: those versions may require different blocks, heads, shape/channel contracts, or operators. Only YOLOv8s is covered by the current whole-model validation.

Hardware validation is limited to Ascend 310P3 (`dav-2002`). No compatibility or performance claim is made for other Ascend chips. No chip-level parity claim follows from a particular model's synchronized graph latency.

Planner ranking coefficients are calibrated defaults for 310P3 in one CANN/container environment. They do not encode a shape whitelist or establish portable timing constants. Recalibrate when hardware, drivers or runtime environments change; legal plans still depend on the implemented capacity checks.