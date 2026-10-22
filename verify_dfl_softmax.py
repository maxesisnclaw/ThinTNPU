import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--backend-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.backend_root.resolve()))
import thin_backend
from thin_backend.graph import CapturedForward

torch.set_num_threads(1)
torch.manual_seed(1928)
reference = torch.load(args.reference, weights_only=False) if args.reference else None
outputs, rows = {}, []

def bits(x):
    return x.contiguous().view(torch.uint8)


def check(label, base, view, half_to_float=False):
    device = base.to('npu')
    source = view(device)
    cpu = view(base)
    forward = lambda value: torch.ops.aten._softmax.default(view(value), 1, half_to_float)
    expected = torch.softmax(cpu.float() if half_to_float else cpu, 1)
    eager = forward(device).cpu()
    finite = torch.isfinite(expected)
    assert torch.equal(torch.isnan(eager), torch.isnan(expected)), label
    error = (eager.float()[finite] - expected.float()[finite]).abs().max().item() if finite.any() else 0.0
    assert error <= (1e-3 if eager.dtype == torch.float16 else 1e-5), (label, error)
    graph = CapturedForward(forward, device)
    for _ in range(3):
        actual = graph.replay().cpu()
        assert torch.equal(bits(actual), bits(eager)), label
    graph.close()
    outputs[label] = eager
    if reference is not None:
        assert eager.dtype == reference[label].dtype and eager.shape == reference[label].shape, label
        assert torch.equal(bits(eager), bits(reference[label])), (label, 'raw-bit mismatch')
    rows.append(dict(case=label, shape=list(source.shape), strides=list(source.stride()), dtype=str(source.dtype),
                     half_to_float=half_to_float, cpu_max_abs=error, capture_raw_bit=True,
                     baseline_raw_bit=reference is not None))


with torch.inference_mode():
    for dtype in (torch.float16, torch.float32):
        for width in (16, 80, 1040, 8400):
            cpu = (torch.randn(2, 4, 16, width) * 3).to(dtype)
            check(f'{dtype}-{width}-transpose', cpu, lambda x: x.transpose(1, 2))
            check(f'{dtype}-{width}-contiguous', cpu, lambda x: x.reshape(2, 16, 4, width))
            if dtype == torch.float16:
                check(f'{dtype}-{width}-half-to-float', cpu, lambda x: x.transpose(1, 2), True)
        cpu = torch.randn(2, 4, 16, 1120).to(dtype)
        check(f'{dtype}-offset', cpu, lambda x: x[:, :, :, 16:1056].transpose(1, 2))
        cpu = torch.randn(1, 4, 16, 80).to(dtype)
        check(f'{dtype}-broadcast', cpu, lambda x: x.expand(2, -1, -1, -1).transpose(1, 2))
        cpu = torch.randn(1, 4, 16, 17).to(dtype)
        check(f'{dtype}-unaligned', cpu, lambda x: x.transpose(1, 2))
        cpu = torch.randn(1, 4, 16, 80).to(dtype)
        for mode in ('nan', 'positive-inf', 'negative-inf', 'zeros', 'large-shift'):
            special = cpu.clone()
            if mode == 'nan':
                special[:, :, 5, 3] = float('nan')
            elif mode == 'positive-inf':
                special[:, :, 5, 3] = float('inf')
            elif mode == 'negative-inf':
                special[:, :, :, 3] = -float('inf')
            elif mode == 'zeros':
                special.zero_()
                special[:, :, 4, :] = -0.0
            else:
                special.add_(10000)
            check(f'{dtype}-{mode}', special, lambda x: x.transpose(1, 2))
            if dtype == torch.float16:
                check(f'{dtype}-{mode}-half-to-float', special, lambda x: x.transpose(1, 2), True)
    maps = Path('/proc/self/maps').read_text()
    assert not any(token in maps.lower() for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
    assert not any(name.startswith('torch_npu') for name in sys.modules)
args.output.mkdir(parents=True, exist_ok=True)
torch.save(outputs, args.output / 'outputs.pt')
(args.output / 'results.json').write_text(json.dumps(dict(cases=rows, count=len(rows)), indent=2) + '\n')
(args.output / 'maps.txt').write_text(maps)
print('DFL_SOFTMAX_PASS', len(rows), 'baseline_raw_bit', reference is not None)
