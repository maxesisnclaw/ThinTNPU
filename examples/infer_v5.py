import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
from torch import nn

import thin_backend
from blocked_graph import FrozenGraph
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow


def copy_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return type(value)(copy_to_cpu(item) for item in value)


def infer(model: nn.Module, image: torch.Tensor):
    """Run a caller-supplied, batch-normalization-fused YOLOv5 module and CPU input."""
    load_native_flow()
    with torch.no_grad():
        model = model.eval().requires_grad_(False).half().to('npu')
        prepare_model(model, fuse_silu=True)
        blocked = WholeBlocked(model)
        blocked.enable()
        image = image.half().contiguous().to('npu')
        graph = FrozenGraph(blocked, blocked.run, image)
    try:
        with torch.inference_mode():
            return copy_to_cpu(graph(image))
    finally:
        graph.close()