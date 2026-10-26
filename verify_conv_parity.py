import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
import math
from pathlib import Path
import sys
import torch

p = argparse.ArgumentParser()
p.add_argument('--library', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.library.resolve().parent.parent))
import thin_backend
torch.ops.load_library(str(a.library.resolve()))
torch.set_num_threads(1)
torch.manual_seed(1931)
outputs = []
shapes = [(ci, co, 16, 32, k, s)
          for ci, co in ((16, 48), (48, 48), (48, 96), (96, 48), (96, 144),
                         (144, 96), (192, 192), (256, 80), (1536, 32))
          for k, s in ((1, 1), (3, 1), (3, 2))]
shapes += [(16, 48, 32, 64, 6, 2), (32, 64, 11, 19, 3, 1),
           (48, 48, 44, 76, 1, 1), (96, 144, 44, 76, 3, 1)]
with torch.inference_mode():
    for ci, co, h, w, k, s in shapes:
        image = torch.randn(1, ci // 16, h, w, 16).half().to('npu')
        weight = (torch.randn(co, ci, k, k) / math.sqrt(ci * k * k)).half().to('npu')
        packed = torch.ops.thin_backend.conv2d_prepare(weight)
        bias = torch.randn((co + 31) // 32 * 32).half().to('npu')
        for fused in (False, True):
            value = torch.ops.thin_native_flow.conv(image, packed, bias, co, k, s, fused, True)
            outputs.append(value.cpu())
    torch.save(outputs, a.output)
    if a.reference:
        expected = torch.load(a.reference, weights_only=False)
        assert len(expected) == len(outputs)
        for index, (x, y) in enumerate(zip(outputs, expected)):
            assert x.shape == y.shape
            assert torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16)), (index, shapes[index // 2])
print('CONV_PLAN_BITS_PASS' if a.reference else 'CONV_PLAN_REFERENCE_SAVED', len(outputs), flush=True)
