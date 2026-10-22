# ThinTNPU

**A YOLOv8-specialized thin PyTorch backend for Ascend 310P.**

**640² whole-model capture: 4.75 ms versus 4.67 ms native graph capture, approximately 1.016× native latency—near parity. Host parent-process PSS is about one fifth of native, with no observed Thin child processes and a 0.26 s versus 14.6 s first forward after model preparation.**

ThinTNPU implements PyTorch PrivateUse1 with ACL runtime calls and precompiled Ascend C kernels. Its inference process does not import or link `torch_npu`, and does not load aclnn/TBE or spawn a compiler-worker pool. It is a deliberately narrow, inference-only backend for the YOLO operator family—not a general-purpose replacement for the Ascend PyTorch stack.

The validated whole-model path is fused **YOLOv8s on Ascend 310P3**, batch 1, FP16. `WholeBlocked` keeps backbone/neck features in NC1HWC0 and restores NCHW at Detect terminals. `FrozenGraph` captures this path for fixed-input-contract replay. Model configuration is frozen; supported image dimensions are not a hardcoded shape list.

## Performance and measurement scope

The headline comparison uses synchronized **whole-model graph replay**, including Detect's DFL and box decode, excluding NMS, preprocessing, host-to-device transfer and source-image coordinate restoration. Input is `[1,3,640,640]`, FP16. The model is a YOLOv8s topology with a 17-class Detect head and separately supplied weights; numbers are **not** measurements of a stock COCO checkpoint or a promise for every YOLO variant.

| 640² FP16 N=1 | ThinTNPU | Native torch_npu graph |
|---|---:|---:|
| Graph replay wall | **4.74957 ms** | **4.67358 ms** |
| Thin/native replay latency | **1.016×** | 1.000× |
| NPU D2D input copy + replay | 4.99430 ms | 4.70907 ms |

| Round | Thin replay ms | Native replay ms | Thin/native |
|---|---:|---:|---:|
| 1 | 4.74587 | 4.66603 | 1.01711× |
| 2 | 4.73774 | 4.68882 | 1.01043× |
| 3 | 4.74957 | 4.67358 | 1.01626× |
| 4 | 4.76998 | 4.67299 | 1.02076× |
| 5 | 4.75752 | 4.69450 | 1.01342× |

Five fresh-process rounds alternate relative Thin/native order, with 30 warmups and 101 timed samples per mode per round. The acquisition also ran two earlier Thin implementations in the rotation; the JSON publishes all samples for the release Thin and native variants. Each replay includes synchronization; allocator/memory sampling stops before timing. Reported headline latency is the median of the five per-round medians; all 505 samples per variant per mode are retained, including outliers. Thin/native is a **latency ratio**, so a value greater than 1 means Thin is slower. The 1.016× ratio uses unrounded latencies. D2D-copy-plus-replay is reported separately and is not mixed into the headline. See [benchmark samples](benchmarks/capture_640.json).

This is an engineering comparison on one model, SDK and device—not chip-level absolute parity. Native graph capture is the baseline, not native eager execution. The Thin inference process has no observed children in these runs; avoiding a compiler-worker pool also reduces startup overhead, but this is not a universal cold-start guarantee. Memory figures below are host-process measurements, not device HBM or application-service capacity.

| Host process / startup metric | ThinTNPU | Native baseline |
|---|---:|---:|
| Steady parent-process PSS | 324.0 MiB | 1662.9 MiB |
| Startup sampled parent-process peak PSS | 424.9 MiB | 1703.5 MiB |
| First forward after model preparation | 260.4 ms | 14619.6 ms |
| Capture setup | 96.4 ms | 2374.2 ms |
| Observed direct child processes | 0 in every round | 3 in every round |

Steady parent PSS is about **one fifth** of native. An earlier controlled series measured **317 MiB versus 1665 MiB**; the current five-round series above measured 324 MiB versus 1663 MiB. The historical figures are retained separately in [host overhead reference](benchmarks/host_overhead_reference.json), not substituted for the current run. Startup sampled peak PSS is **not** one fifth (425 MiB versus 1704 MiB); native child memory is not included in either parent-process figure. The 0.26 s versus 14.6 s cold first-forward comparison starts after model preparation, not at process/service launch. Current values are medians across the five rounds, with per-round figures in the capture JSON.

## Capacity-driven planning, not a shape lookup table

The whole-model path has **no shape lookup table** and no 640-only dispatch. It computes plans from the actual dimensions and L1/UB/L0 capacity constraints. Spatial convolution enumerates legal M/TN/TK tiles, derives input-channel chunks and buffering arrangements from capacity, then ranks candidates using an arithmetic/data-movement cost model. The stem derives multi-row tiles and DMA strips from the input dimensions; SPPF pooling derives its multi-row tiles from UB capacity. Calibration measurements are not loaded at runtime.

The planner coefficients are **310P3 defaults calibrated in one specific CANN/container environment**, not universal hardware constants. Recalibrate them when changing hardware, drivers or runtime/container environments. The fitted defaults came from 429 legal candidates across 10 spatial-convolution cases; their role is candidate ranking, not a claim about fixed physical pipeline latencies. [Calibration metadata and coefficients](benchmarks/planner_calibration.json) document the environment, fit and units. End-to-end measured latency, not the fitted score, is the performance evidence.

## Scope and accuracy

- The generalized `WholeBlocked` entry accepts arbitrary positive H/W divisible by 32, subject to hardware/resource limits, as contiguous FP16 NPU `[1,3,H,W]` input. Each shape needs its own graph. The 12 tested shapes are 640×640, 1280×1280, 1280×704, 640×1920, 320×320, 352×608, 32×32, 32×3200, 3200×32, 320×1280, 1280×320 and 1024×1024. Finite coverage is not proof that every /32 input fits the hardware.
- Thin eager and capture outputs match **bit for bit** on two inputs per tested shape. The same 12 shapes also match an independent process running the preceding generalized implementation bitwise, across decoded output and all three raw heads; 640² additionally matches the prior prepared-NCHW baseline. **FP16 outputs are only approximately equal to CPU FP32**, not bitwise equal to that reference. Convolution accumulation/bias/SiLU use FP32 before the FP16 cast. See [validation data](benchmarks/validation.json).
- At 1280² the prior fixture's normalized coordinate maximum error is about **9.5% higher** than at 640², while normalized mean error is slightly lower. This is not a mAP or model-quality claim.
- Tail shapes can be much slower than larger aligned shapes: 352×608 has a known padding/compaction/generic-softmax performance cliff.
- L1/UB/L0 capacity, DMA/index ranges and physical memory impose real boundaries. Unsupported plans fail loudly; /32 does not mean unbounded size.
- YOLO-domain specialization and reuse within YOLO are compatible goals. Ultralytics **v5/v11/v26** share useful operator families, but are **not validated end to end** here and may require integration work. Only YOLOv8s currently has whole-model evidence.

Read [LIMITATIONS.md](LIMITATIONS.md) before use, especially the frozen-state contract and the known `.data` alias and post-capture `Detect.export`/`Detect.format` guard gaps. Those mutations can silently reuse a stale graph; current guards must not be mistaken for complete immutability enforcement.

## Build

Tested toolchain: aarch64 Linux, CANN 9.0.0 / AscendC, PyTorch 2.8.0, Python 3.11, Ultralytics 8.3.233 and an Ascend 310P3 (`dav-2002`). Obtain the CANN SDK and model dependencies separately. The kernel CMake SOC setting uses the 310P-family `ascend310p1` target; device validation here was performed on 310P3. Other devices/toolchains are unverified.

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

Build as container root: CANN kernel merging writes SDK-owned single-object/fatbin directories. Do not loosen their permissions just to build as an unprivileged user. `build_all.sh` builds the backend, region guard, blocked convolution and whole-model flow libraries from source. Generated binaries are not bundled.

In the container, from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

Import `thin_backend` before moving tensors to `npu`; it registers the PrivateUse1 name and device methods. Do **not** import `torch_npu` into that same process. The optional native comparison uses a separate process.

## Minimal inference

Supply a trusted, fused-compatible YOLOv8s checkpoint you have the right to use. No model weights or images are included, and the example requires an existing local file rather than downloading one.

```bash
python3 examples/infer.py --model /path/to/yolov8s.pt --height 640 --width 640
```

The core API is:

```python
import torch
from ultralytics import YOLO
import thin_backend
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
from blocked_graph import FrozenGraph

load_native_flow()
with torch.no_grad():
    model = YOLO(checkpoint).model.fuse().eval().requires_grad_(False).half().to('npu')
    prepare_model(model, fuse_silu=True)
    blocked = WholeBlocked(model)
    blocked.enable()
    graph = FrozenGraph(blocked, blocked.run, example_npu_fp16)

try:
    with torch.inference_mode():
        output = graph(input_npu_fp16)  # copied into the static input, then replayed
        retained_decoded = output[0].cpu().clone()
        graph.replay()                 # same static input; overwrites borrowed outputs
finally:
    graph.close()
```

Keep the model and wrapper unchanged for the graph's entire lifetime. Close/reprepare/recapture to change model state or output modes; see the [contract](LIMITATIONS.md).

## Verification and reproducing the comparison

```bash
python3 verify_full.py
python3 verify_flow_conv.py
python3 verify_flow_ops.py
mkdir -p evidence-shapes
python3 verify_flow_shapes.py
python3 verify_graph.py

python3 examples/make_reference.py --model /path/to/yolov8s.pt --output reference.pt
python3 verify_shapes.py --model /path/to/yolov8s.pt --reference reference.pt
bash run_capture_comparison.sh /path/to/yolov8s.pt reference.pt capture-results
```

`verify_shapes.py` defaults to the 12 shapes listed above. For cross-revision bitwise comparisons, `verify_shape_parity.py` can export outputs with `--backend-root BASELINE_ROOT --model CHECKPOINT --fixture reference.pt --output baseline-outputs`, then compare in a separate process with `--backend-root CURRENT_ROOT --model CHECKPOINT --fixture reference.pt --output comparison-output --reference baseline-outputs`. Both backend trees must be built separately, and model/fixture inputs must be identical.

The comparison additionally requires a compatible native `torch_npu` installation, used only in native subprocesses. Run on an otherwise idle intended NPU. Keep every round and full `samples_ms` array, not just the best latency. `verify_full.py` is the operator/backend suite (`FULL_ALL_PASS 311` in the validated revision); whole-model shape and lifecycle checks are separate. Native baseline settings and capture success are recorded per run. Independently supplied weights and fixtures will not reproduce identical numerical errors or latency.

## Source layout

- `thin_backend/`: backend registration and ACL graph wrapper.
- `backend.cpp`, `aten_adapter.cpp`, `ops.cpp`: allocator/runtime, ATen routing and operators.
- `conv_*.cpp`, `flow_ops.cpp`, other kernel sources: convolution, blocked flow and YOLO operators.
- `model_optimization.py`, `whole_blocked.py`: prepacked weights, leaf integration and blocked execution.
- `region_guard.cpp`, `region_contract.py`, `blocked_graph.py`: frozen-state checks and guarded replay.
- `verify_*.py`, `examples/`, `benchmark_backends.py`: verification, minimal use and isolated comparison.

## License

ThinTNPU's own code is licensed under [Apache-2.0](LICENSE). Copyright 2026 maxesisnclaw.

Dependencies and model assets retain their own terms. In particular, Ultralytics is AGPL-3.0 or commercially licensed; the Apache license here does not override those obligations for a combined application or service. See [DEPENDENCIES.md](DEPENDENCIES.md).