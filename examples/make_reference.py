import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if not args.model.is_file():
    raise FileNotFoundError(args.model)
torch.set_num_threads(1)
torch.manual_seed(42)
with torch.no_grad():
    model = YOLO(str(args.model)).model.fuse().eval().float()
    image = torch.rand(1, 3, 640, 640)
    output = model(image)
    torch.save({'x': image, 'out': output[0]}, args.output)
