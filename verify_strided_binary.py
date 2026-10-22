import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
import json
from pathlib import Path
import torch
import thin_backend
from thin_backend.graph import CapturedForward

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(71)
outputs = []
rows = []
with torch.inference_mode():
    for dtype in (torch.float16, torch.float32):
        for width in (16, 48, 2048, 2064, 8400):
            data = torch.randn(4, width * 2 + 16, dtype=dtype).to('npu')
            full = torch.randn(4, width, dtype=dtype).to('npu')
            scalar = torch.tensor(2, dtype=dtype).to('npu')
            cases = [
                ('interleaved-left', data[:, 1:1 + 2 * width:2], full),
                ('interleaved-right', full, data[:, :2 * width:2]),
                ('unaligned', data[:, 1:1 + width], full),
                ('row-broadcast', full, full[:1]),
                ('column-broadcast', full, full[:, :1]),
                ('scalar', full, scalar),
                ('both-expanded', full[:1], full[:, :1]),
                ('rank-four', full.view(2, 2, 1, width), data[:, :2 * width:2].view(2, 2, 1, width)),
            ]
            for label, x, y in cases:
                for operation in ('add', 'sub', 'mul', 'div'):
                    for alpha in ((1.0, 0.3) if operation in ('add', 'sub') else (1.0,)):
                        def run(x=x, y=y, operation=operation, alpha=alpha):
                            if operation in ('add', 'sub'):
                                return getattr(torch, operation)(x, y, alpha=alpha)
                            return getattr(torch, operation)(x, y)
                        result = run().cpu()
                        graph = CapturedForward(lambda _: run(), x)
                        captured = graph.replay().cpu()
                        assert torch.equal(result.contiguous().view(torch.uint8), captured.contiguous().view(torch.uint8))
                        graph.close()
                        rows.append(dict(dtype=str(dtype), width=width, case=label, operation=operation, alpha=alpha))
                        outputs.append(result)
    special = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1., -1., 2.] * 4, dtype=torch.float16).to('npu')
    for operation in ('add', 'sub', 'mul', 'div'):
        outputs.append(getattr(torch, operation)(special[1:17], special[:1]).cpu())
torch.save(outputs, a.output)
if a.reference:
    reference = torch.load(a.reference, weights_only=False)
    assert len(outputs) == len(reference)
    for index, (actual, expected) in enumerate(zip(outputs, reference)):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)), (index, rows[index] if index < len(rows) else 'special')
print(json.dumps(dict(cases=len(outputs), eager_capture_equal=True, baseline_raw_bit_equal=bool(a.reference))))
