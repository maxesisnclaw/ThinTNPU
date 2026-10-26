# Benchmarks and validation

## Unified YOLO family

- `family_capture_640.json`: synchronized whole-model graph-replay wall times for YOLOv8s, classic YOLOv5m, YOLO11s and YOLO26s on the shared source revision recorded in the file. Five rounds per family/backend, 30 warmups, 101 samples per round, alternating Thin/native order, no outlier removal. Each headline is the median of the five per-round medians; Thin/native is a latency ratio, with values above one meaning Thin is slower. Every sample and per-process eager/capture equality result is retained.
- `family_validation.json`: numerical, Boolean and shape-only validation results from the shared implementation. It separates prior Thin-fixture comparisons, each family's eager/capture checks, CPU-reference errors and operator-suite outcomes. No weights, checkpoint parameters, model paths, machine identifiers or raw process logs are included.

Timing includes YOLO26's NMS-free device selection and packing, and excludes preprocessing, external IoU NMS, input transfers/copies and output transfers. Native is graph capture, not eager execution. The model topologies have separately supplied weights and different class counts; the values are not stock-checkpoint guarantees or equal-workload cross-family comparisons. See [measurement protocol](../FAMILY_BENCHMARK.md) and the [README](../README.md).

### Reading validation data

`prior_thin_fixture` compares the legacy families' decoded/raw outputs against their existing Thin fixtures. YOLO26 records its full dictionary/end-to-end snapshots separately in `v26s`. Zero mismatches here do not imply CPU FP32 or Thin/native equality. V8 shape rows additionally record independent-process fixture equality and the number of tested inputs. V11 prior attention snapshot booleans cover the recorded input shapes.

CPU comparison names are retained separately: `decoded_vs_fp32`, `vs_fp32_rounded_half`, `vs_fp16_ref` and `vs_fp16_params_ref` are different references, not interchangeable accuracy claims. V11 `vs_cpu_fp32` and `vs_fp16_params_ref` contain decoded coordinates/classes, DFL, raw-head and attention errors. Shape and numerical summaries describe those supplied fixtures, not a mAP result or universal error bound. V5 anchor values are not published; only the fixture-equality Boolean is retained.

V11 attention-error arrays follow execution-hook order: `c2psa.cv1`, `attn.qkv`, `attn.pe`, `attn.proj`, `psa.attn`, `psa.ffn`, `c2psa.cv2`. Each record retains its tensor shape. Raw-head arrays retain detection-scale order.

Operator suite passes use the respective test's own numerical contract. The anchor-convolution CPU comparison is explicitly a report: one of its 20 cases is bitwise equal, and 19 are not. Its completion is not a CPU-bitexact pass.

## Historical YOLOv8s records

The following files retain the earlier v8-only experiments, not measurements of the unified family source:

- `capture_640.json`: historical capture rounds, samples and process overhead, including the 1.016× Thin/native latency ratio.
- `host_overhead_reference.json`: an earlier host-overhead reference series.
- `validation.json`: historical v8 shape and numerical checks.
- `planner_calibration.json`: planner calibration environment and coefficients.

Historical parent-process memory/startup data is not a new four-family memory comparison. Model assets are not included in either dataset.

YOLO26 rows retain CPU direct-head/final errors, complete selection-stage errors and selected anchor/class overlap separately from prior-Thin raw-bit equality. Stable lower-index ties intentionally differ from unspecified `torch.topk` tie indices. The complete three-shape/two-image prior snapshots and 47 selection cases are summarized without embedding any inputs or tensor assets.
