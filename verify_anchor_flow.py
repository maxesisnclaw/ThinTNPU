import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import thin_backend
from whole_blocked import load_native_flow

load_native_flow()
torch.set_num_threads(1)
torch.manual_seed(713)
rows = []
with torch.inference_mode():
    for ci, co, h, w, kernel, stride in (
            (3, 48, 32, 32, 6, 2), (3, 48, 640, 640, 6, 2),
            (3, 48, 32, 3200, 6, 2), (3, 48, 3200, 32, 6, 2),
            (3, 64, 352, 608, 6, 2), (17, 48, 64, 96, 6, 2),
            (16, 32, 64, 96, 6, 1), (3, 16, 64, 96, 3, 2),
            (1536, 768, 20, 20, 1, 1), (1536, 32, 4, 4, 1, 1)):
        image = torch.randn(1, ci, h, w).half()
        weight = (torch.randn(co, ci, kernel, kernel) / math.sqrt(ci * kernel * kernel)).half()
        bias = torch.randn(co).half()
        blocked = torch.ops.thin_native_flow.pack_input(image.to('npu'))
        restored = blocked.cpu().permute(0, 1, 4, 2, 3).reshape(1, -1, h, w)
        assert torch.equal(restored[:, :ci].contiguous().view(torch.int16), image.view(torch.int16))
        assert not restored[:, ci:].count_nonzero()
        packed = torch.ops.thin_backend.conv2d_prepare(weight.to('npu'))
        padded_bias = F.pad(bias, (0, (-co) % 32)).to('npu')
        reference = F.conv2d(image.float(), weight.float(), bias.float(), stride, (kernel - 1) // 2)
        for fused in (False, True):
            actual = torch.ops.thin_native_flow.conv(blocked, packed, padded_bias,
                                                    co, kernel, stride, fused, True).cpu()
            expected = (F.silu(reference) if fused else reference).half()
            error = (actual.float() - expected.float()).abs()
            row = dict(shape=[ci, co, h, w, kernel, stride], fused=fused,
                       bit_equal=torch.equal(actual.view(torch.int16), expected.view(torch.int16)),
                       max_abs=error.max().item(), mean_abs=error.mean().item(),
                       mismatched_bits=(actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
            assert torch.isfinite(actual).all(), row
            print(json.dumps(row), flush=True)
            rows.append(row)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(token in maps for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
print('ANCHOR_FLOW_COMPLETE', len(rows), flush=True)
