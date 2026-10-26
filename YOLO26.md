# Direct-distance end-to-end Detect

WholeBlocked lowers fused Ultralytics dictionary heads with direct distances (`reg_max=1`) or DFL16, while retaining the classic anchor/objectness and legacy DFL paths. Public checkpoints and CPU reference tensors are external inputs, not repository assets.

## Frontend preparation

Select the one-to-one branch **before** fusing a dual-head model. Fusion removes the unused branch and the other branch cannot be recovered by changing a flag afterward.

```python
model = YOLO(checkpoint).model.cpu().eval()
model.end2end = True
model.fuse()
model.requires_grad_(False)
model = model.half().to('npu')
prepare_model(model, fuse_silu=True)
pilot = WholeBlocked(model, attention_scale_before_matmul=True)
pilot.enable()
```

The attention option describes a numerical frontend contract, not a model identifier. Set it to match the original forward: `True` for `(q * scale).transpose(-2, -1) @ k`, `False` for `(q.transpose(-2, -1) @ k) * scale`. Their half-precision rounding points differ. The default preserves the previously supported post-matmul scaling order.

C3k, nested PSABlock and residual SPPF containers retain their original forward methods. Only convolution, pooling, upsampling, blocked channel split and attention leaves are lowered. The SPPF activation and residual behavior come from the actual module instance.

## Head and layout

The selected branch's terminals restore NCHW, including non-C0-aligned four-channel distance outputs. Dictionary heads return `boxes`, `scores` and blocked `feats`; end-to-end heads wrap them in `one2many`/`one2one` as the fused frontend does.

Grid coordinates are calculated from each feature's spatial dimensions and the instance's strides. Grid arithmetic uses half precision, including the half-cell offset. NC1HWC0's trailing channel lane is never interpreted as a spatial dimension. Each fresh spatial shape computes its own grid; captured graphs retain a fixed shape.

Direct distances are decoded without sigmoid, clipping or DFL:

```text
lower = anchors - distances[:, :2]
upper = anchors + distances[:, 2:]
xyxy = cat(lower, upper) * strides
```

Non-end-to-end `xywh` uses the frontend's `(lower + upper) / 2` and `upper - lower` before multiplying by strides. Classification logits pass through sigmoid independently.

## Selection boundary

End-to-end inference returns every row of `[B, min(max_det, anchors), 6]`, with columns `[x1, y1, x2, y2, score, class]`. It is not a raw decoded-head substitute. There is no confidence threshold, clipping, image-coordinate restoration or IoU suppression inside this output contract.

Selection must remain on the device and be replayable: class max, anchor top-k, score gather, flattened class/anchor top-k, index mapping and box gather. Packing promotes the final tensor to FP32 to match the frontend's floating class labels; box and score arithmetic remains FP16.

The selection contract's options, attention order and structural attributes are part of the existing region guard. A capture fixes input shape and storage; changed-input replay must be compared bit-for-bit with a separate eager execution.
