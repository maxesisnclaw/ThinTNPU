# YOLO family graph-replay comparison

`benchmark_family_capture.py` measures one fused model in one fresh process, using either ThinTNPU or the native `torch_npu` graph runtime. `run_family_capture_comparison.py` runs four families × five rounds × two backends, for 40 independent processes by default. A failed capture or correctness assertion fails the run; there is no eager-timing or alternate-loader fallback.

## Inputs and dependencies

Build the shared backend with `build_all.sh` and source `env.sh`. Install the corresponding public model definitions and the native baseline dependency separately. Set `PYTHONPATH` to the repository root and, for classic v5, the separately obtained upstream YOLOv5 source root. Do not import `torch_npu` in a Thin process.

Every worker requires explicit `--model` and `--fixture` files. These are trusted PyTorch artifacts supplied by the caller; loading serialized modules can execute code. Nothing is downloaded or bundled.

- `--model-format fused-module`: the file contains `{'model': fused_cpu_module}`. The module is already batch-normalization-fused, with its original upstream definitions importable. This is the classic v5 input route and also supports fused v8/v11/v26 modules.
- `--model-format ultralytics`: the installed public `ultralytics.YOLO` API opens the existing checkpoint, obtains `.model` and fuses it. This route is for v8/v11/v26, not classic v5. Format selection is explicit; a failed load is not retried through another route.
- A `v8s` fixture contains `x`; `v5m`, `v11s` and `v26s` fixtures contain `cases['640x640']['image_fp16']`. Only the input is consumed for this timing comparison. Reference predictions and CPU error metrics are not used to certify accuracy.
- The common input is `[1,3,640,640]`, converted to contiguous FP16 before transfer. The benchmark's `v5m`, `v8s`, `v11s` and `v26s` labels identify the requested comparison cases, not a topology detector; the caller must supply the matching architecture and record its upstream revision/class count.

No application-specific model loader is imported. To avoid serialized-module loading altogether, construct a fused `nn.Module` through its public upstream implementation and call the Python interface below in a fresh process.

## One worker

```bash
python3 benchmark_family_capture.py \
  --family v5m --backend thin \
  --model /path/to/v5-fused.pt --model-format fused-module \
  --fixture /path/to/v5-reference.pt --output results/v5m-thin
```

Use a separate invocation with `--backend native` and identical model/fixture inputs. For v8 use `--family v8s --model-format ultralytics`, or supply an already fused module with `fused-module`. YOLO11 uses `--family v11s` and either explicitly selected supported format.

For a caller-constructed fused CPU module and CPU image:

```python
from benchmark_family_capture import measure

result = measure(model, image, backend='thin', output='results/constructed-thin',
                 warmup=30, samples=101, metadata={'family': 'v5m'})
```

Run the native call in a different process with an independently constructed equivalent model. The function prepares/mutates the model; do not reuse that instance for the other backend. Do not switch backend registrations inside one process. For this direct API, record the model/fixture provenance yourself; the file CLI records both SHA-256 digests automatically.

## Forty-process comparison

```bash
python3 run_family_capture_comparison.py \
  --v5m-model /path/to/v5-fused.pt --v5m-fixture /path/to/v5-reference.pt \
  --v8s-model /path/to/yolov8s.pt --v8s-fixture /path/to/v8-reference.pt \
  --v11s-model /path/to/v11-fused.pt --v11s-fixture /path/to/v11-reference.pt \
  --v26s-model /path/to/v26-fused.pt --v26s-fixture /path/to/v26-reference.pt \
  --v26s-python /path/to/v26-environment/bin/python \
  --output family-results
```

The runner defaults to `fused-module` for v5/v11/v26 and `ultralytics` for v8. Explicit `--v8s-model-format` and `--v11s-model-format` select the other supported format when required. It runs families in v8/v5/v11/v26 order, alternates Thin/native order each round, runs workers sequentially, retains every sample, and writes `family_summary.json` under the chosen output directory. Each family's headline is the median of its five per-round medians; the latency ratio is Thin/native using those unrounded medians. A ratio above one means Thin is slower.

Run on an otherwise idle intended NPU. Device index 0 matches the backend's current execution contract; isolate the intended device in the environment. Do not mix concurrently running workloads into a controlled comparison. Default warmup/sample counts are 30/101; any CLI overrides are recorded and change the comparison protocol.

## Timing and correctness contract

Both backends use evaluation mode, frozen FP16 parameters and the same FP16 input. Preparation and the first eager forward occur before timing. Native capture receives ten untimed model warmups on its capture stream, then captures a real `torch.npu.NPUGraph`; Thin captures `FrozenGraph`. Each worker checks finite outputs and bitwise equality of **its own eager versus captured output** before the timed replay sequence.

The measured sequence is 30 warmup replays, each followed by synchronization, then 101 samples. Every timed sample synchronizes before starting the wall clock, replays the graph, and synchronizes before stopping the clock. Thin timing includes its guarded replay entry; native timing uses `NPUGraph.replay`. Capture setup, first eager execution, input transfer/copy, preprocessing, external IoU NMS and output transfer are excluded. The first eager duration is retained separately as `first_eager_ms_not_benchmark` and is not a process-start cold-start metric. No profiling or memory sampling occurs in the timed loop.

Per-process outputs are `result.json` with all `samples_ms`, plus local `eager.pt` and `capture.pt` output tensors. These are user-generated evidence, not bundled assets. Native and Thin files are generated by independent processes. The worker checks that the Thin process has not loaded `torch_npu`, aclnn, TBE or the operator compiler.

Eager/capture equality is not CPU FP32 bit-exactness, Thin/native bit-exactness or a comparison to an earlier Thin fixture. Cross-revision fixture checks and CPU FP32/FP16-reference error reports are separate validations. In particular, `verify_anchor_flow.py` reports CPU errors and bit comparisons; its completion marker does not mean all CPU bit comparisons passed.

The published [family capture data](benchmarks/family_capture_640.json) contains the unified-source five-round samples, while [family validation](benchmarks/family_validation.json) contains separate shape, fixture and CPU-error checks. The historical v8-only files do not measure the unified source; see the [benchmark index](benchmarks/README.md).

## Frontend versions and YOLO26

The v5/v8/v11 measurements use their compatible installed definitions; v26 uses Ultralytics 8.4.189 in a separate environment. `--v26s-python` selects that interpreter; each other family has the same optional `--FAMILY-python` flag. Install matching torch/native dependencies into each environment. A new frontend is not assumed to load old fused pickles with identical semantics.

For `v26s`, the checkpoint route selects `model.end2end = True` before fusion. A supplied fused module must already contain the intended one-to-one branch. The worker uses `attention_scale_before_matmul=True`, traverses dictionary output leaves and compares raw bytes including the final FP32 detections. For the direct Python API, explicitly pass `attention_scale_before_matmul=True` for the matching frontend. Do not reuse a mutated model across backends.

YOLO26 timing includes its entire NMS-free device selection and `[1,300,6]` packing; it is not a raw-head timing. The other models time their own Detect outputs without external NMS. The worker checks only each backend's own eager/capture equality. Stable tie semantics and whole-model reference errors are separate validation, not inferred from timing.
