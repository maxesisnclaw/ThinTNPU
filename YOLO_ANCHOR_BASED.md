# YOLO anchor-based inference

`WholeBlocked` supports the classic three-scale YOLOv5 Detect contract in addition to the existing YOLOv8 DFL contract. The input is a fused evaluation model with frozen FP16 parameters; preprocessing and NMS remain outside the forward.

## Execution

Load the model through its upstream implementation or a matching architecture plus `state_dict`. The backend accepts the resulting `nn.Module`; it does not load checkpoints, import an application-specific model package, or substitute a different detection head.

```python
import torch
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
from blocked_graph import FrozenGraph

load_native_flow()
with torch.no_grad():
    model = model.eval().requires_grad_(False).half().to('npu')
    prepare_model(model, fuse_silu=True)
    flow = WholeBlocked(model)
    flow.enable()

with torch.inference_mode():
    prediction, raw = flow.run(image)
    graph = FrozenGraph(flow, model, image)
    prediction, raw = graph(image)
    graph.close()
```

`image` must be an allocation-base contiguous FP16 NPU tensor `[1,3,H,W]`, with positive H/W divisible by 32. A capture fixes that shape. Eager execution and separately constructed captures use the same shape-derived planner for other supported H/W.

The classic Detect module supplies `nl=3`, `na`, `nc`, `no=nc+5`, the three `m` terminal convolutions, normalized `anchors`, and `stride`. Its terminal convolutions restore NCHW before the decoder. The output is `(prediction, raw)`:

- `prediction`: `[1, na * sum(Hi*Wi), nc+5]`, ordered by scale, anchor, row, column.
- `raw[i]`: `[1,na,Hi,Wi,nc+5]`, before sigmoid.
- Prediction fields are `xywh`, objectness, and independent class probabilities. Objectness is not multiplied into the class probabilities in this output.

The original container forwards implement C3, residual addition, SPPF, concatenation, and nearest upsampling. Only convolution/pool/upsample leaves and the classic Detect forward are replaced. Disabling the replacements restores the original forwards; it does not make an unadapted classic Detect's `pow` operation supported.

## Anchors and decoding

Pixel anchors are derived from the loaded model's `Detect.anchors * Detect.stride`, not a YAML default or a built-in anchor table. They are prepared before the frozen-state contract is created. Grid constants depend on each actual head H/W and are materialized during warmup, before capture.

For `p = sigmoid(raw)`:

```text
xy = (2*p_xy - 0.5 + integer_grid) * stride
wh = (2*p_wh) * (2*p_wh) * pixel_anchor
objectness = p_obj
class_probability = p_cls
```

The square uses the existing multiplication primitive, not a general power kernel. Sigmoid, arithmetic, copies, and concatenation use the existing ThinTNPU operators.

## Generalization ledger

| Change | Classification | Shape/model-specific dispatch |
|---|---|---|
| NCHW-to-NC1HWC0 input packing with zero padding of a partial C0 block | General layout adapter | No channel-count or image-size table; spatial plane must satisfy its 16-element DMA alignment |
| FRACTAL_Z weight packing for k6 | General convolution extension | Adds k6 to the compiled kernel set; no stem-weight constants |
| Spatial planner and LoadData bounds use `kernel*kernel`, kernel-derived padding, and output dimensions | General convolution extension | Same capacity/cost search for k3 and k6; supported spatial kernels are k3/k6 with stride 1/2 |
| Nonresident pointwise L0B staging | General capacity handling | Resident versus streamed weights is selected by the L0B byte capacity, not by a model/layer name |
| Input-channel padding followed by planned convolution for other aligned stem widths | General stem adapter | No 48-channel special case; the pre-existing k3/32-output stem path is retained |
| Conv wrapper recognition and three NCHW terminal leaves | Model-graph adaptation | Uses module structure, not a checkpoint filename, class count, or layer-index table |
| Anchor/objectness Detect | Classic YOLOv5-specific output semantics | Three levels; `na`, `nc`, anchors, strides, and head H/W come from the model and inputs |
| Grid cache | Shape-derived decode constants | Keys are actual level/H/W; no enumerated image-size cases |

No new image-size table, checkpoint identifier, fixed class count, or v5m-only tiling plan is introduced. Arbitrary kernels, general grouped convolution, depthwise multiplier greater than one, Focus/reorg stems, segmentation heads, export-mode heads and general graph compilation remain unsupported. The shared YOLO11 integration separately supports multiplier-one depthwise k3/s1/p1 with aligned channels; see [YOLO11](YOLO11.md).

The pointwise change is needed when a convolution's weights no longer fit resident L0B. It retains the existing resident path when it fits, otherwise stages the same reduction tiles through two L0B banks. This also covers wide SPPF concatenations without a layer-specific exception.

## Numerical contract and verification

Parameters, features, and decoder tensors are FP16. Convolution accumulates and adds the FP16 bias in FP32; fused SiLU is evaluated before the output's FP16 cast. This is not the same computation as preserving FP32 parameters/features throughout a CPU model and casting only its final output.

The decoder reuses the existing FP16 sigmoid and arithmetic implementations. Their intermediate rounding and finite-range behavior can differ from CPU sigmoid's FP32 internal evaluation. CPU bit-exactness must therefore be measured, not inferred from an eager/capture bitwise match. No tolerance or reference precision is changed by enabling the anchor-based path.

```sh
bash build_all.sh
source env.sh
python3 verify_anchor_flow.py
python3 verify_anchor_decode.py
python3 verify_full.py
```

`verify_anchor_flow.py` reports exact bit comparisons and max/mean CPU-reference errors for convolution, including k6, non-default output widths, rectangular images, padded channels, and nonresident pointwise weights. It does not turn a failed bit comparison into a tolerance-based pass. `verify_anchor_decode.py` uses synthetic non-default anchors with different anchor/class counts and exactly representable zero-logit outputs to check decode layout and semantics independently of convolution error.

Checkpoint-level validation should separately report CPU FP32 versus ThinTNPU, FP16-parameter/feature CPU emulation versus ThinTNPU, and ThinTNPU eager versus capture/replay. Compare coordinates, objectness, class probabilities, and all raw heads. Keep checkpoint material and application-specific loaders outside the source tree.
