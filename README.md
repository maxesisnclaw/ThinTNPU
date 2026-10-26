# ThinTNPU

**A thin PyTorch inference backend for the YOLO family on Ascend 310P.**

**Four generations, one shared implementation: YOLOv5 / YOLOv8 / YOLO11 / YOLO26.**

ThinTNPU implements PyTorch PrivateUse1 with ACL runtime calls and precompiled Ascend C kernels. Its inference process does not import or link `torch_npu`, and does not load aclnn/TBE or spawn a compiler-worker pool. It is a deliberately narrow, inference-only backend, not a general-purpose replacement for the Ascend PyTorch stack.

`WholeBlocked` provides architecture adapters for classic **YOLOv5**, **YOLOv8**, **YOLO11** and **YOLO26**. Backbone/neck features stay in NC1HWC0; Detect terminals restore NCHW. `FrozenGraph` captures the prepared model for fixed-input-contract replay. Model configuration is frozen; supported image dimensions are not a hardcoded shape list.

| Family | Architecture integration | Scope |
|---|---|---|
| YOLOv5 | C3, k6 stem, SPPF and classic anchor/objectness Detect | Three detection levels; anchors, classes and strides come from the supplied module. No bundled loader or checkpoint. |
| YOLOv8 | C2f and anchor-free DFL Detect | Near-native synchronized capture on the measured YOLOv8s fixture. |
| YOLO11 | C3k2/C3k, C2PSA attention and depthwise-separable Detect | C0-aligned intermediates; **YOLO11n is not supported**. |
| YOLO26 | C3k/PSABlock, residual SPPF, Q-prescaled attention, direct-distance Detect | Fused one-to-one branch, no DFL; device-resident NMS-free selection returns `[1,300,6]` for the measured 640² case. |

## Unified-family capture measurements

| Model topology | Thin replay ms | Native graph ms | Thin/native latency |
|---|---:|---:|---:|
| YOLOv8s, 17 classes | **4.57471** | 4.70555 | **0.972×** |
| Classic YOLOv5m, 11 classes | **7.98227** | 6.23399 | **1.280×** |
| YOLO11s, 80 classes | **6.90995** | 6.03161 | **1.146×** |
| YOLO26s, 80 classes, fused one-to-one NMS-free | **9.12324** | 7.18999 | **1.269×** |

The shared implementation was measured on Ascend 310P3, FP16 N=1 input `[1,3,640,640]`. Timing is synchronized **whole-model graph replay**, including Detect decoding for every family and the full device-resident two-stage selection/packing for YOLO26. It excludes preprocessing, transfers and external IoU NMS. Native is a real `torch_npu` graph-capture baseline, not eager execution. YOLO26's NMS-free `[1,300,6]` result is not a raw-head substitute; its device selection is inside both timed graphs.

Each family/backend has five fresh-process rounds, 30 warmup replays and 101 synchronized samples per round: **505 samples per family/backend, 40 processes and 4,040 timed samples**. Odd rounds run Thin then native; even rounds reverse that order. The headline is the median of five per-round medians, and the latency ratio divides those unrounded medians. Only the intended device was exposed; concurrent application services remained running, so this is not an exclusive-machine benchmark. Every sample, including outliers, is retained in [the family data](benchmarks/family_capture_640.json).

A ratio above one means Thin is slower, not faster. The four supplied models have different class counts/weights and different head work; absolute cross-family latency is not an equal-workload comparison. These are **engineering-level contemporaneous capture comparisons**, not chip-level parity or a universal performance/accuracy guarantee. The shared vector bmm and softmax paths remain an attention latency floor for v11/v26. Classic v5 itself has no attention; its residual convolution/layout overhead is a distinct floor.

Earlier separate-branch latency ratios were approximately v8 1.0×, v5m 1.29×, v11s 1.20× and v26s 1.296×. They are historical context, not replacements for the unified-source measurements above and not evidence that integration caused a speedup. Historical v8 [capture](benchmarks/capture_640.json) and [host overhead](benchmarks/host_overhead_reference.json) records remain separate; memory/startup were not remeasured as a four-family comparison.

## Unified-family validation

All 40 measured processes produced finite outputs bitwise equal to their own eager/capture execution. All four Thin families retain their corresponding prior Thin fixture outputs. The Thin processes exclude torch_npu, aclnn, TBE and the operator compiler.

V8 passes twelve independent-process shape comparisons with two inputs per shape, plus twelve eager/capture shape checks. V5 retains anchors and decoded/raw outputs at 640×640 and 352×608; v11 retains whole-output and attention snapshots at 640×640 and 320×512. Their shape-edge checks also pass. YOLO26's 47 selection fixtures cover arbitrary strides, ties, NaNs, all 65,536 half encodings, K=511/512/513 and changed-input capture replay. Together with complete outputs for three synthetic shapes and two real images, **52 prior-Thin snapshot files and 1,174 tensor leaves match raw bits**, including all detection rows, not only high-confidence rows.

The operator checks include the 311-case backend suite, anchor decode, bmm/layout bmm, depthwise, 44 materialization cases, 54 strided-concat cases, 58 flow-convolution cases and 62 convolution-plan comparisons. These suites have different reference contracts; completion does not mean CPU FP32 bit-exactness. The 20-case anchor-convolution CPU report retains **one bit-equal and 19 non-bit-equal cases** rather than relabeling completion as CPU parity.

[Family validation data](benchmarks/family_validation.json) retains shapes, booleans, mismatch counts and separate CPU-reference errors. **Bitwise claims mean Thin eager/capture/replay or corresponding prior Thin fixtures, not CPU FP32, native-backend equality or mAP.** YOLO26's deterministic lower-index tie order intentionally differs from `torch.topk`'s unspecified tied-index order. Official same-input/final errors remain reported; a replay pass is not a detection-quality acceptance or equality to the official CPU output.

## General primitives and architecture adapters

- General layout handling includes partial-C0 input padding, strided materialization/binary operations and strided concatenation. Dense convolution supports capacity-derived tiling and nonresident pointwise weights; k6 extends the spatial kernel set without checkpoint-specific constants.
- `bmm_flow.cpp` handles rank-three FP16 operands with nonnegative strides, transposes and storage offsets. The blocked-layout entry interprets logical channel ranges and head groups directly, avoiding contiguous Q/K operand copies. FP32 products accumulate in sequential reduction order before the FP16 cast.
- `depthwise_flow.cpp` supplies multiplier-one channelwise convolution. The model adapter admits C0-aligned k3/s1/p1 depthwise with SiLU or Identity; general grouped convolution and depthwise multiplier greater than one remain unsupported.
- Architecture adapters encode classic anchor/objectness, DFL and direct-distance heads, C2PSA/PSABlock attention, and the selected one-to-one branch. Attention Q scaling order is explicit: pre-matmul for the supplied v26 frontend, post-matmul by default for v11. These FP16 rounding orders are not interchangeable.
- `selection_flow.cpp` implements device class-max, vector `RpSort16`/`MrgSort`, exact stable top-k prefixes, score/box gather and FP32 output packing; no host selection or IoU NMS is inserted. The architecture adapters are not an arbitrary PyTorch graph compiler, a model-name dispatch table or a per-image-size lookup.

See [blocked execution](WHOLE_BLOCKED.md), [classic YOLOv5](YOLO_ANCHOR_BASED.md), [YOLO11](YOLO11.md), [YOLO26](YOLO26.md) and [device selection](SELECTION_FLOW.md) for output layouts, supported module contracts and verification commands.

## Capacity-driven planning, not a shape lookup table


The whole-model path has **no shape lookup table** and no 640-only dispatch. It computes plans from the actual dimensions and L1/UB/L0 capacity constraints. Spatial convolution enumerates legal M/TN/TK tiles, derives input-channel chunks and buffering arrangements from capacity, then ranks candidates using an arithmetic/data-movement cost model. The stem derives multi-row tiles and DMA strips from the input dimensions; SPPF pooling derives its multi-row tiles from UB capacity. Calibration measurements are not loaded at runtime.

The planner coefficients are **310P3 defaults calibrated in one specific CANN/container environment**, not universal hardware constants. Recalibrate them when changing hardware, drivers or runtime/container environments. The fitted defaults came from 429 legal candidates across 10 spatial-convolution cases; their role is candidate ranking, not a claim about fixed physical pipeline latencies. [Calibration metadata and coefficients](benchmarks/planner_calibration.json) document the environment, fit and units. End-to-end measured latency, not the fitted score, is the performance evidence.

## Scope and accuracy

- External input is allocation-base contiguous FP16 NPU `[1,3,H,W]`, with positive H/W divisible by 32. Each shape needs its own capture. Capacity, DMA/index ranges and physical memory remain finite; /32 does not imply unbounded inputs.
- **Bitwise equality means Thin eager versus Thin capture/replay, or comparison against the corresponding existing Thin fixture. It does not mean CPU FP32 bit-exactness.** Family fixtures are separate; one family does not validate another. CPU FP32 and FP16-parameter/activation reference errors must be reported separately.
- The unified v8 validation covers 12 shapes: 640×640, 1280×1280, 1280×704, 640×1920, 320×320, 352×608, 32×32, 32×3200, 3200×32, 320×1280, 1280×320 and 1024×1024. Those eager/capture pairs and corresponding prior Thin fixtures matched bitwise. See [family validation data](benchmarks/family_validation.json); [validation.json](benchmarks/validation.json) retains the older v8-only record.
- On the v8 fixture, normalized coordinate maximum error at 1280² is about 9.5% higher than at 640², while normalized mean error is slightly lower. This is not a mAP claim or a universal accuracy bound. Tail shapes can also be slower than larger aligned shapes.
- Training, arbitrary batch sizes, export/segmentation and other unsupported head contracts, general grouped or dilated convolution, depthwise multiplier >1, Focus/reorg stems and YOLO11n are outside the supported model contract. End-to-end inference is supported only for the documented dictionary-head/selected-branch contract, not arbitrary end-to-end model families.

Read [LIMITATIONS.md](LIMITATIONS.md) before use. In particular, `.data` aliases and untracked output-mode attributes such as `Detect.format` can leave stale prepared/captured state. `Detect.export` is tracked by the unified contract; this does not make the guard complete.

## Build


Tested unified-family toolchain: aarch64 Linux, CANN 9.0.0 / AscendC, PyTorch 2.8.0+cpu, Python 3.11.15 and an Ascend 310P3 (`dav-2002`). V5/v8/v11 use Ultralytics 8.3.233 plus separately supplied classic v5 definitions; v26 uses Ultralytics 8.4.189 in a separate frontend environment. The isolated native baseline uses torch_npu 2.8.0.post1. Obtain the CANN SDK and model dependencies separately. The kernel CMake SOC setting uses the 310P-family `ascend310p1` target; device validation here was performed on 310P3. Other devices/toolchains are unverified.

Use a CANN development container with a compatible PyTorch installation, compiler/CMake and separately installed Python dependencies. Expose only the intended NPU and management devices, and the matching host driver directory. For example, after setting `CANN_IMAGE` to your locally available development image and `CONTAINER` to a container name of your choice:

```bash
docker run -d --name "$CONTAINER" \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$PWD:/workspace/ThinTNPU" "$CANN_IMAGE" sleep infinity

docker exec -u 0 "$CONTAINER" bash -c '
  set -eo pipefail
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  cd /workspace/ThinTNPU
  bash build_all.sh
'
```

Build as container root: CANN kernel merging writes SDK-owned single-object/fatbin directories. Do not loosen their permissions just to build as an unprivileged user. `build_all.sh` builds the backend, region guard, blocked convolution and whole-model flow libraries, including strided bmm, depthwise convolution and device selection, from source. Generated binaries are not bundled.

In the container, from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export YOLO_CONFIG_DIR="$PWD/.yolo-config"
```

Import `thin_backend` before moving tensors to `npu`; it registers the PrivateUse1 name and device methods. Do **not** import `torch_npu` into that same process. The optional native comparison uses a separate process.

## Minimal inference

Supply model definitions and trusted weights separately, under their upstream licenses. No weights, images or application-specific loaders are bundled. Examples never download a checkpoint automatically.

For supported v8/v11 models, the same CLI uses the separately installed Ultralytics package and an existing local checkpoint. YOLO26 has a dedicated example for branch preparation and attention order:

```bash
python3 examples/infer.py --model /path/to/yolov8s.pt --height 640 --width 640
python3 examples/infer.py --model /path/to/yolo11s.pt --height 640 --width 640
python3 examples/infer_v26.py --model /path/to/yolo26s.pt --height 640 --width 640
```

YOLO26 requires a compatible newer frontend. Select `model.end2end = True` **before** `model.fuse()`; fusion removes the other branch. Its example uses `WholeBlocked(model, attention_scale_before_matmul=True)`, preserving the frontend's Q-prescale rounding order. Do not toggle branch or attention options after capture.

For classic v5, pass an upstream-constructed, fused `nn.Module` to `examples.infer_v5.infer(model, cpu_image)`. A public-architecture example (random weights; not a trained detector) is shown in [examples](examples/README.md). Checkpoint loading is the caller's responsibility; no custom loader is part of ThinTNPU.

The shared API, after constructing and fusing an appropriate model, is:

```python
import torch
import thin_backend
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
from blocked_graph import FrozenGraph

load_native_flow()
with torch.no_grad():
    model = model.eval().requires_grad_(False).half().to('npu')
    prepare_model(model, fuse_silu=True)
    blocked = WholeBlocked(model)
    blocked.enable()
    graph = FrozenGraph(blocked, blocked.run, example_npu_fp16)

try:
    with torch.inference_mode():
        output = graph(input_npu_fp16)
        retained_decoded = output[0].cpu().clone()
        graph.replay()
finally:
    graph.close()
```

Keep the model and wrapper unchanged for the graph's entire lifetime. Outputs are borrowed buffers. Close/reprepare/recapture to change model state or output modes; see the [contract](LIMITATIONS.md).

## Verification and comparison

For a unified four-family capture comparison, use [`benchmark_family_capture.py` and the forty-process runner](FAMILY_BENCHMARK.md). They accept external model/fixture inputs, keep Thin/native in independent processes, and default to 30 warmups and 101 synchronized replay samples. Classic v5 uses an upstream-constructed fused module, without a bundled checkpoint loader. The published family measurements and the historical v8 records remain separate datasets.

After building in the target CANN environment:

```bash
python3 verify_full.py
python3 verify_flow_conv.py
python3 verify_flow_ops.py
python3 verify_anchor_flow.py
python3 verify_anchor_decode.py
python3 verify_materialize.py
python3 verify_bmm_flow.py
python3 verify_bmm_layout.py
python3 verify_depthwise_flow.py
python3 verify_selection_flow.py
```

These are operator and architecture checks, not substitute whole-model benchmark results. The classic v5 decode test uses synthetic anchors; convolution tests report exact bit comparisons and CPU errors without implying CPU FP32 whole-model bit equality. See the family documents for additional fixture-based checks.

The retained v8 CLI is:

```bash
python3 examples/make_reference.py --model /path/to/yolov8s.pt --output reference.pt
python3 verify_shapes.py --model /path/to/yolov8s.pt --reference reference.pt
bash run_capture_comparison.sh /path/to/yolov8s.pt reference.pt capture-results
```

`verify_shapes.py` defaults to the historical 12 shapes. For cross-revision v8 bitwise comparisons, `verify_shape_parity.py` exports outputs using `--backend-root BASELINE_ROOT --model CHECKPOINT --fixture reference.pt --output baseline-outputs`; compare in a separate process with `--backend-root CURRENT_ROOT --model CHECKPOINT --fixture reference.pt --output comparison-output --reference baseline-outputs`. Both trees must be built separately and receive identical inputs. These v8 harnesses are not generic v5/v11 benchmarking entry points.

The native comparison requires a compatible `torch_npu` installation, used only in native subprocesses. Use an otherwise idle intended NPU; retain every round and full `samples_ms` array. Native capture success and baseline settings must be recorded per run. Independently supplied weights and fixtures will not reproduce identical numerical errors or latency.

## Source layout

- `thin_backend/`: backend registration and ACL graph wrapper.
- `backend.cpp`, `aten_adapter.cpp`, `ops.cpp`: allocator/runtime, ATen routing and operators.
- `conv_*.cpp`, `flow_ops.cpp`, `bmm_flow.cpp`, `depthwise_flow.cpp`: convolution, blocked flow, strided attention and depthwise kernels.
- `selection_*.h`, `selection_flow.cpp`: stable device selection, gather and final packing.
- `strided_binary.h`, `strided_concat.h`: general strided elementwise/copy layouts.
- `model_optimization.py`, `whole_blocked.py`: prepared weights and architecture adapters.
- `region_guard.cpp`, `region_contract.py`, `blocked_graph.py`: frozen-state checks and guarded replay.
- `verify_*.py`, `examples/`, `benchmark_backends.py`: verification, minimal use and isolated comparison.

## License

ThinTNPU's own code is licensed under [Apache-2.0](LICENSE). Copyright 2026 maxesisnclaw.

Dependencies and model assets retain their own terms. Ultralytics model classes and forwards for v5/v8/v11/v26 are subject to upstream AGPL-3.0 or separately obtained commercial terms; this repository's license does not override those obligations. Check the exact upstream revision you use. See [DEPENDENCIES.md](DEPENDENCIES.md).
