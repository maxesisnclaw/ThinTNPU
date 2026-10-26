import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
from pathlib import Path
import sys

import torch
import thin_backend
from thin_backend.graph import CapturedForward

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(2931)
outputs, rows = [], []


def same_bits(x, y):
    return torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))


with torch.inference_mode():
    for dtype in (torch.float16, torch.float32):
        for height, width in ((80, 80), (44, 76), (20, 20), (1, 16), (1, 2048), (1, 2064)):
            full = torch.randn(1, 3, height, width, 16, dtype=dtype).to('npu')
            grid = torch.randn(height, width, 2, dtype=dtype).to('npu')
            anchors = torch.randn(1, 3, 1, 1, 2, dtype=dtype).to('npu')
            contiguous = torch.randn(1, 3, height, width, 2, dtype=dtype).to('npu')
            scalar = torch.tensor(2, dtype=dtype).to('npu')
            cases = [
                ('slice-scalar', full[..., :2], scalar),
                ('offset-slice-scalar', full[..., 2:4], scalar),
                ('grid', contiguous, grid),
                ('anchors', contiguous, anchors),
                ('narrow-both', full[..., :2], full[..., 2:4]),
                ('narrow-swapped', scalar, full[..., 2:4]),
                ('narrow-broadcast', full[:, :1, ..., :2], full[..., :1]),
                ('lanes-four', full[..., 1:5], scalar),
                ('lanes-eight', full[..., 1:9], scalar),
            ]
            for label, x, y in cases:
                for operation in ('add', 'sub', 'mul', 'div'):
                    alphas = (1.0, 0.3) if operation in ('add', 'sub') else (1.0,)
                    for alpha in alphas:
                        def run(x=x, y=y, operation=operation, alpha=alpha):
                            if operation in ('add', 'sub'):
                                return getattr(torch, operation)(x, y, alpha=alpha)
                            return getattr(torch, operation)(x, y)
                        output = run().cpu()
                        graph = CapturedForward(lambda _: run(), x)
                        captured = graph.replay().cpu()
                        graph.close()
                        assert same_bits(output, captured)
                        outputs.append(output)
                        rows.append([str(dtype), height, width, label, operation, alpha])
    special = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1., -1., 2.] * 32,
                           dtype=torch.float16).to('npu').view(16, 16)
    for operation in ('add', 'sub', 'mul', 'div'):
        outputs.append(getattr(torch, operation)(special[:, 1:3], special[:, :1]).cpu())
    assert 'torch_npu' not in sys.modules
    torch.save(outputs, a.output)
    if a.reference:
        reference = torch.load(a.reference, weights_only=False)
        assert len(reference) == len(outputs)
        for i, (actual, expected) in enumerate(zip(outputs, reference)):
            assert same_bits(actual, expected), (i, rows[i] if i < len(rows) else 'special')
print('NARROW_BINARY_BITS_PASS' if a.reference else 'NARROW_BINARY_REFERENCE', len(outputs), flush=True)
