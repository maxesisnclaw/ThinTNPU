# ThinTNPU

**A thin PyTorch inference backend for the YOLO family on Ascend 310P.**

**640² synchronized capture replay: YOLOv8s 4.58 ms (0.977× native), YOLOv5m 7.95 ms (1.271×), YOLO11s 7.23 ms (1.195×).**

ThinTNPU implements PyTorch PrivateUse1 with ACL runtime calls and precompiled Ascend C kernels. Its inference process does not import or link `torch_npu`, and does not load aclnn/TBE or spawn a compiler-worker pool. It is a deliberately narrow, inference-only backend, not a general-purpose replacement for the Ascend PyTorch stack.

`WholeBlocked` provides architecture adapters for classic **YOLOv5**, **YOLOv8** and **YOLO11**. Backbone/neck features stay in NC1HWC0; Detect terminals restore NCHW. `FrozenGraph` captures the prepared model for fixed-input-contract replay. Model configuration is frozen; supported image dimensions are not a hardcoded shape list.

| Family | Architecture integration | Scope |
|---|---|---|
| YOLOv5 | C3, k6 stem, SPPF and classic anchor/objectness Detect | Three detection levels; anchors, classes and strides come from the supplied module. No bundled loader or checkpoint. |
| YOLOv8 | C2f and anchor-free DFL Detect | Near-native synchronized capture on the measured YOLOv8s fixture. |
| YOLO11 | C3k2/C3k, C2PSA attention and depthwise-separable Detect | C0-aligned intermediates; **YOLO11n is not supported**. |

## Unified-family capture measurements

The shared implementation was measured on Ascend 310P3 with FP16 N=1 input `[1,3,640,640]`. Timing is synchronized **whole-model graph replay**, including each family's Detect decode, and excludes preprocessing, NMS, input transfers/copies and output transfers. Native is a real `torch_npu` graph-capture baseline, not eager execution.

| Model topology | Thin replay ms | Native graph ms | Thin/native latency |
|---|---:|---:|---:|
| YOLOv8s, 17 classes | **4.57973** | 4.68540 | **0.977×** |
| Classic YOLOv5m, 11 classes | **7.95316** | 6.25951 | **1.271×** |
| YOLO11s, 80 classes | **7.23014** | 6.04915 | **1.195×** |

Each family/backend has five independent-process rounds, 30 warmup replays and 101 synchronized samples per round: **505 samples per family/backend, 30 processes and 3,030 timed samples in total**. Odd rounds run Thin then native; even rounds reverse that order. The headline is the median of the five per-round medians; the ratio divides the unrounded headline latencies. All samples, including outliers, are retained in [the per-round family data](benchmarks/family_capture_640.json).

A ratio above one means Thin is slower. V8 remains the near-native flagship; v5/v11 retain a roughly 1.2–1.3× native-latency floor in these cases. The models have different class counts and separately supplied weights, so their absolute latencies are not equal-workload cross-family comparisons or stock-checkpoint claims. These numbers describe this measurement, not a universal speed or accuracy guarantee. Differences from earlier runs do not establish that source integration caused a speedup.

This is an engineering-level contemporaneous capture comparison on one SDK/device environment, not chip-level absolute parity.

Earlier separate-branch results were v8s 4.75 ms / 1.016× native, v5m 8.04 ms / 1.289× and v11s 7.47 ms / 1.235×. The older v8 samples remain historical reference in [capture_640.json](benchmarks/capture_640.json), not the current family result. Historical v8 parent-process PSS/startup observations are retained in that file and [host_overhead_reference.json](benchmarks/host_overhead_reference.json); they were not remeasured as a unified three-family memory/startup comparison.

## Unified-family validation

All 30 measured processes produced finite outputs with their own eager/capture outputs bitwise equal. The three Thin family outputs—decoded output and all raw heads—also match their corresponding prior Thin fixtures bitwise. V5 additionally matches its prior fixture at 352×608; v11's whole-output and attention snapshots match the prior Thin implementation at 640×640 and 320×512. V8 passes the 12-shape independent-process fixture comparison with two inputs per shape, alongside the 12-shape eager/capture checks.

The operator checks include the 311-case backend suite, anchor decode, bmm, blocked/strided bmm layouts, depthwise, 44 materialization cases, 54 strided-concat cases, 58 flow-convolution cases and 62 convolution-plan bit comparisons. These suites have different reference contracts; a suite pass does not mean every operation is CPU FP32 bit-exact. In particular, the 20-case anchor-convolution CPU report has **one bit-equal case and 19 non-bit-equal cases**, while retaining the measured CPU errors rather than treating its completion marker as a bitwise pass.

[Family validation data](benchmarks/family_validation.json) contains shapes, booleans, mismatch counts and numerical errors without model assets or machine identifiers. **Bitwise claims refer to Thin eager/capture and the corresponding prior Thin fixtures—not CPU FP32 bit-exactness, Thin/native equality or detection-quality/mAP.** CPU FP32, rounded-FP32 and FP16 reference errors retain their separate labels.

## General primitives and architecture adapters

- General layout handling includes partial-C0 input padding, strided materialization/binary operations and strided concatenation. Dense convolution supports capacity-derived tiling and nonresident pointwise weights; k6 extends the spatial kernel set without checkpoint-specific constants.
- `bmm_flow.cpp` handles rank-three FP16 operands with nonnegative strides, transposes and storage offsets. The blocked-layout entry interprets logical channel ranges and head groups directly, avoiding contiguous Q/K operand copies. FP32 products accumulate in sequential reduction order before the FP16 cast.
- `depthwise_flow.cpp` supplies multiplier-one channelwise convolution. The model adapter admits C0-aligned k3/s1/p1 depthwise with SiLU or Identity; general grouped convolution and depthwise multiplier greater than one remain unsupported.
- Architecture adapters encode the classic anchor/objectness decoder, the DFL head, and C2PSA/Attention module semantics. They are not an arbitrary PyTorch graph compiler, a model-name dispatch table or a per-image-size lookup.

See [blocked execution](WHOLE_BLOCKED.md), [classic YOLOv5](YOLO_ANCHOR_BASED.md), and [YOLO11](YOLO11.md) for output layouts, supported module contracts and verification commands.

## Capacity-driven planning, not a shape lookup table


The whole-model path has **no shape lookup table** and no 640-only dispatch. It computes plans from the actual dimensions and L1/UB/L0 capacity constraints. Spatial convolution enumerates legal M/TN/TK tiles, derives input-channel chunks and buffering arrangements from capacity, then ranks candidates using an arithmetic/data-movement cost model. The stem derives multi-row tiles and DMA strips from the input dimensions; SPPF pooling derives its multi-row tiles from UB capacity. Calibration measurements are not loaded at runtime.

The planner coefficients are **310P3 defaults calibrated in one specific CANN/container environment**, not universal hardware constants. Recalibrate them when changing hardware, drivers or runtime/container environments. The fitted defaults came from 429 legal candidates across 10 spatial-convolution cases; their role is candidate ranking, not a claim about fixed physical pipeline latencies. [Calibration metadata and coefficients](benchmarks/planner_calibration.json) document the environment, fit and units. End-to-end measured latency, not the fitted score, is the performance evidence.

## Scope and accuracy

- External input is allocation-base contiguous FP16 NPU `[1,3,H,W]`, with positive H/W divisible by 32. Each shape needs its own capture. Capacity, DMA/index ranges and physical memory remain finite; /32 does not imply unbounded inputs.
- **Bitwise equality means Thin eager versus Thin capture/replay, or comparison against the corresponding existing Thin fixture. It does not mean CPU FP32 bit-exactness.** Family fixtures are separate; a v8 result does not validate v5 or v11. CPU FP32 and FP16-parameter/activation reference errors must be reported separately.
- The unified v8 validation covers 12 shapes: 640×640, 1280×1280, 1280×704, 640×1920, 320×320, 352×608, 32×32, 32×3200, 3200×32, 320×1280, 1280×320 and 1024×1024. Those eager/capture pairs and corresponding prior Thin fixtures matched bitwise. See [family validation data](benchmarks/family_validation.json); [validation.json](benchmarks/validation.json) retains the older v8-only record.
- On the v8 fixture, normalized coordinate maximum error at 1280² is about 9.5% higher than at 640², while normalized mean error is slightly lower. This is not a mAP claim or a universal accuracy bound. Tail shapes can also be slower than larger aligned shapes.
- Training, arbitrary batch sizes, export/end-to-end heads, general grouped or dilated convolution, depthwise multiplier >1, Focus/reorg stems and YOLO11n are outside the supported model contract. No claim is made for v26 or other families.

Read [LIMITATIONS.md](LIMITATIONS.md) before use. In particular, `.data` aliases and untracked output-mode attributes such as `Detect.format` can leave stale prepared/captured state. `Detect.export` is tracked by the unified contract; this does not make the guard complete.

## Build


Tested unified-family toolchain: aarch64 Linux, CANN 9.0.0 / AscendC, PyTorch 2.8.0+cpu, Python 3.11.15, Ultralytics 8.3.233 and an Ascend 310P3 (`dav-2002`). The isolated native baseline uses torch_npu 2.8.0.post1. Obtain the CANN SDK and model dependencies separately. The kernel CMake SOC setting uses the 310P-family `ascend310p1` target; device validation here was performed on 310P3. Other devices/toolchains are unverified.

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

Build as container root: CANN kernel merging writes SDK-owned single-object/fatbin directories. Do not loosen their permissions just to build as an unprivileged user. `build_all.sh` builds the backend, region guard, blocked convolution and whole-model flow libraries, including strided bmm and depthwise convolution, from source. Generated binaries are not bundled.

In the container, from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Import `thin_backend` before moving tensors to `npu`; it registers the PrivateUse1 name and device methods. Do **not** import `torch_npu` into that same process. The optional native comparison uses a separate process.

## Minimal inference

Supply model definitions and trusted weights separately, under their upstream licenses. No weights, images or application-specific loaders are bundled. Examples never download a checkpoint automatically.

For supported v8/v11 models, the same CLI uses the separately installed Ultralytics package and an existing local checkpoint:

```bash
python3 examples/infer.py --model /path/to/yolov8s.pt --height 640 --width 640
python3 examples/infer.py --model /path/to/yolo11s.pt --height 640 --width 640
```

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

For a unified three-family capture comparison, use [`benchmark_family_capture.py` and the thirty-process runner](FAMILY_BENCHMARK.md). They accept external model/fixture inputs, keep Thin/native in independent processes, and default to 30 warmups and 101 synchronized replay samples. Classic v5 uses an upstream-constructed fused module, without a bundled checkpoint loader. The published family measurements and the historical v8 records remain separate datasets.

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
- `strided_binary.h`, `strided_concat.h`: general strided elementwise/copy layouts.
- `model_optimization.py`, `whole_blocked.py`: prepared weights and architecture adapters.
- `region_guard.cpp`, `region_contract.py`, `blocked_graph.py`: frozen-state checks and guarded replay.
- `verify_*.py`, `examples/`, `benchmark_backends.py`: verification, minimal use and isolated comparison.

## License

ThinTNPU's own code is licensed under [Apache-2.0](LICENSE). Copyright 2026 maxesisnclaw.

Dependencies and model assets retain their own terms. Ultralytics model classes and forwards for v5/v8/v11 are subject to upstream AGPL-3.0 or separately obtained commercial terms; this repository's license does not override those obligations. Check the exact upstream revision you use. See [DEPENDENCIES.md](DEPENDENCIES.md).
