# Model examples

Run from the repository root after building and sourcing `env.sh`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

## YOLOv8 and YOLO11

`infer.py` uses a separately installed Ultralytics package and an existing local checkpoint. It does not download missing assets. Supply a compatible model under its upstream license:

```bash
python3 examples/infer.py --model /path/to/yolov8s.pt --height 640 --width 640
python3 examples/infer.py --model /path/to/yolo11s.pt --height 640 --width 640
```

The input is synthetic zeros, not an image-preprocessing pipeline. The example captures once and prints the decoded tensor shape. YOLO11n is not supported; model names alone do not guarantee a compatible layout or upstream revision.

`make_reference.py` generates the simple CPU reference used by the retained v8 validation/comparison CLI. It is not the richer fixture generator for `verify_yolo11.py`.

## Classic YOLOv5

`infer_v5.infer(model, image)` accepts a caller-constructed, batch-normalization-fused `torch.nn.Module` and CPU input. It neither loads checkpoints nor imports an application-specific package. It mutates the supplied model by preparing NPU weights and replacing forwards; use a dedicated model instance.

For a public-architecture smoke example, separately obtain a compatible upstream YOLOv5 tree and set `YOLOV5_ROOT` and `YOLOV5_CONFIG` to that tree and its public `yolov5m.yaml` configuration. This constructs random weights, not a useful trained detector:

```bash
export PYTHONPATH="$YOLOV5_ROOT:$PWD:${PYTHONPATH:-}"
python3 - <<'PY'
import os
import torch
from models.yolo import DetectionModel
from examples.infer_v5 import infer

with torch.no_grad():
    model = DetectionModel(os.environ['YOLOV5_CONFIG'], ch=3, nc=80).eval().fuse()
    image = torch.zeros(1, 3, 640, 640)
prediction, raw = infer(model, image)
print('Decoded:', tuple(prediction.shape))
print('Heads:', [tuple(head.shape) for head in raw])
PY
```

For a trained model, construct/load it through its upstream implementation outside this repository, fuse it, then pass that module to `infer`. Its original model definitions must remain importable. No weights, custom loader, checkpoint compatibility shim or alternate detection head are bundled. The accepted classic head returns xywh, objectness and independent class probabilities; it does not multiply objectness into the class scores.

## Numerical and lifecycle scope

All examples use FP16 N=1 on the NPU. Capture requires frozen versioned parameters prepared under `no_grad()`, not parameters created under `inference_mode()`. Returned results in the examples are copied to CPU before closing the graph. The underlying graph API returns borrowed buffers; see [LIMITATIONS.md](../LIMITATIONS.md).

For synchronized family timing, the [capture benchmark](../FAMILY_BENCHMARK.md) accepts external model/fixture CLI inputs or a caller-constructed fused `nn.Module`. Its Thin/native workers always run in separate processes.

Thin eager/capture or corresponding Thin-fixture bitwise equality is not CPU FP32 bit-exactness. Neither these examples nor a shape printout establish accuracy or benchmark results. Upstream model classes and supplied weights retain their separate licenses; see [DEPENDENCIES.md](../DEPENDENCIES.md).
## YOLO26s

Use a compatible separately installed Ultralytics frontend and trusted local weights:

```bash
export YOLO_CONFIG_DIR="$PWD/.yolo-config"
python3 examples/infer_v26.py --model /path/to/yolo26s.pt --height 640 --width 640
```

The example selects one-to-one before fusion, keeps Q-prescale attention order, and captures the full NMS-free output. At 640² it returns FP32 `[1,300,6]`, not only raw decoded heads. Stable lower-index ties, borrowed-output lifetime and external preprocessing/coordinate restoration are described in [limitations](../LIMITATIONS.md).
