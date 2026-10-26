# Blocked YOLO inference

The shared execution path keeps FP16 activations in NC1HWC0 between the stem and Detect terminals. It uses ThinTNPU's Ascend C kernels and ACL graph runtime, without loading `torch_npu`, aclnn or TBE in the Thin process. Classic YOLOv5, YOLOv8 and YOLO11 use explicit architecture adapters over these primitives.

## Build

From the repository root in a compatible CANN development environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build_all.sh
source env.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

The flow libraries are `thin_backend/libconv_flow.so`, `thin_backend/libflow_ops.so`, `thin_backend/libbmm_flow.so` and `thin_backend/libdepthwise_flow.so`. `load_native_flow()` loads all four for every family; the bmm/depthwise libraries must be built even for a model that does not use their operators. The region guard is built by `build_blocked_c2f.sh`.

## Layout and numerical contract

- External input is allocation-base, contiguous FP16 NCHW `[1,3,H,W]`, with positive H/W divisible by 32.
- The stem emits NC1HWC0. Intermediate physical tensors have shape `[1,C/16,H,W,16]`; channel chunks, concatenations and residual additions operate on that representation.
- SPPF maxpool and nearest-neighbor upsample operate directly on blocked tensors.
- Classic v5 has three NCHW Detect terminals and anchor/objectness decoding. The v8/v11 DFL heads have three box and three class NCHW terminals. Returned decoded/raw tensors retain the corresponding family's public layout; these layouts are not interchangeable.
- Dense convolution accumulates, adds bias and applies the selected SiLU/Identity in FP32 before one FP16 cast. Depthwise follows its defined FP32 product/add order. Parameters and features remain FP16; this is not CPU FP32 inference.
- Dynamic tensor–tensor bmm accepts nonnegative strides, storage offsets and logical transposes. Blocked operands additionally describe logical channel ranges and head grouping. Q/K operands need not be unpacked to NCHW or materialized as contiguous matrices.
- The architecture adapter requires C0-aligned total intermediate channels. Attention head/key widths may cross C0 block boundaries; that does not remove the total-channel constraint or add YOLO11n support.

The graph freezes model state and one input contract, not an enumerated image-size table. Fresh eager execution and separate captures derive supported shapes from dimensions and hardware capacities. See [limitations](LIMITATIONS.md) for finite resources and the separate, narrower `BlockedC2f` pilot contract.

## Primitives and family-specific semantics

| Primitive / adapter | Contract |
|---|---|
| `stem` / `pack_input` | Dense three-channel model stem; general input padding handles partial C0 blocks. The retained specialized stem is k3 with 32 outputs; other supported aligned stems use packing plus planned convolution. |
| `conv` | Prepared dense weights; capacity-driven k1/k3/k6 kernels. Model body wrappers admit k1/k3; k6 is used by the supported stem. Blocked outputs require aligned channels; terminals restore NCHW. |
| `maxpool` / `upsample` | Blocked SPPF k5/s1/p2 and nearest 2× in the model adapter. |
| `layout_bmm` / `channel_select` | Strided or blocked logical operands; group-aware Q/K/V addressing and selected-channel compaction. FP32 sequential reduction, then FP16 output. |
| `depthwise` | Multiplier-one channelwise convolution. Model adapter: aligned channels, k3/s1/p1, SiLU or Identity. Not general grouped convolution. |
| Classic Detect | Three scales; anchors, strides, anchor/class counts supplied by the module; decoded xywh/objectness/classes. |
| Anchor-free Detect | Existing three-scale DFL/box decoding with 16 bins. |
| C2PSA/Attention | Ultralytics module semantics over shared bmm, softmax, depthwise and layout primitives. |

`prepare_model` prepackages weights; `WholeBlocked` adds padded biases and derived anchor state before freezing the contract. The caller supplies a fused, evaluation-mode model. No weights or model-specific loader are bundled.

## Capture

Construct versioned parameters under `torch.no_grad()`, not `inference_mode()`:

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
        retained = output[0].cpu().clone()
finally:
    graph.close()
```

The model/wrapper and its prepared weights must remain alive and unchanged. Outputs are graph-owned borrowed buffers. Capture/replay/close run on the creating thread and stream, without concurrent replay. Guards track selected state, not every possible mutation; see [the frozen-state contract](LIMITATIONS.md).

## Verification

```bash
python3 verify_flow_conv.py
python3 verify_flow_ops.py
python3 verify_anchor_flow.py
python3 verify_anchor_decode.py
python3 verify_materialize.py
python3 verify_bmm_flow.py
python3 verify_bmm_layout.py
python3 verify_depthwise_flow.py
```

[Classic v5](YOLO_ANCHOR_BASED.md) and [YOLO11](YOLO11.md) document the family-specific checks. Model/reference fixtures are supplied separately. Eager/capture and corresponding Thin-fixture bitwise comparisons do not assert CPU FP32 bit-exactness. The [benchmark index](benchmarks/README.md) separates unified-family capture/validation from historical v8 measurements.
