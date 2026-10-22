import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
from pathlib import Path
import torch
import thin_backend

parser = argparse.ArgumentParser()
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference', type=Path)
args = parser.parse_args()
torch.ops.load_library(str(args.library.resolve()))
torch.manual_seed(0)
outputs = {}
for c1, h, w in ((16, 20, 20), (32, 20, 20), (16, 40, 40), (8, 40, 40), (2, 13, 17)):
    x = torch.randn(1, c1, h, w, 16, dtype=torch.float16)
    outputs[f'pool-{c1}-{h}-{w}'] = torch.ops.thin_native_flow.maxpool(x.to('npu'), 5).cpu()
    bits = torch.randint(-32768, 32768, x.shape, dtype=torch.int16)
    outputs[f'upsample-{c1}-{h}-{w}'] = torch.ops.thin_native_flow.upsample(bits.view(torch.float16).to('npu')).cpu()
    expected = bits.repeat_interleave(2, 2).repeat_interleave(2, 3)
    assert torch.equal(outputs[f'upsample-{c1}-{h}-{w}'].view(torch.int16), expected)
if args.reference:
    expected = torch.load(args.reference, weights_only=False)
    for name, value in outputs.items():
        assert torch.equal(value.contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8)), name
    print('FLOW_RAW_BIT_PASS', len(outputs))
torch.save(outputs, args.output)
