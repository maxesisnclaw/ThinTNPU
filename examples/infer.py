import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

import thin_backend
from blocked_graph import FrozenGraph
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow


def copy_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return type(value)(copy_to_cpu(item) for item in value)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--height', type=int, default=640)
parser.add_argument('--width', type=int, default=640)
args = parser.parse_args()
if not args.model.is_file():
    raise FileNotFoundError(args.model)

torch.set_num_threads(1)
load_native_flow()
with torch.no_grad():
    model = YOLO(str(args.model)).model.fuse().eval().requires_grad_(False).half().to('npu')
    prepare_model(model, fuse_silu=True)
    blocked = WholeBlocked(model)
    blocked.enable()
    image = torch.zeros((1, 3, args.height, args.width), dtype=torch.float16).to('npu')
    graph = FrozenGraph(blocked, blocked.run, image)

try:
    with torch.inference_mode():
        result = copy_to_cpu(graph(image))
        graph.replay()
        print('Decoded output:', tuple(result[0].shape))
finally:
    graph.close()
