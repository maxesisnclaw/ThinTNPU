import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import thin_backend
from whole_blocked import load_native_flow
load_native_flow()
torch.set_num_threads(1)
torch.manual_seed(1729)
rows = []
with torch.inference_mode():
    for ci, co, h, w, k, s in ((16, 17, 3, 7, 1, 1), (32, 16, 6, 14, 3, 2), (16, 32, 1, 1, 3, 1), (32, 17, 11, 19, 3, 1), (16, 64, 13, 17, 3, 2), (32, 32, 8, 12, 3, 1), (32, 17, 80, 44, 1, 1), (32, 32, 16, 1600, 3, 1)):
        x = torch.randn(1, ci, h, w).half()
        weight = (torch.randn(co, ci, k, k) * 0.03).half()
        bias = (torch.randn(co) * 0.03).half()
        pb = torch.zeros((co + 31) // 32 * 32, dtype=torch.float16)
        pb[:co] = bias
        packed = torch.ops.thin_backend.conv2d_prepare(weight.to('npu'))
        blocked = x.reshape(1, ci // 16, 16, h, w).permute(0, 1, 3, 4, 2).contiguous().to('npu')
        for nchw in (True,) if co % 16 else (True, False):
            for fused in (False, True):
                out = torch.ops.thin_native_flow.conv(blocked, packed, pb.to('npu'), co, k, s, fused, nchw).cpu()
                if not nchw:
                    out = out.permute(0, 1, 4, 2, 3).reshape(1, co, out.shape[2], out.shape[3])
                ref = F.conv2d(x.float(), weight.float(), bias.float(), stride=s, padding=k // 2)
                if fused:
                    ref = F.silu(ref)
                assert out.shape == ref.shape and torch.isfinite(out).all()
                err = float((out.float() - ref).abs().max())
                assert torch.allclose(out.float(), ref, atol=0.002, rtol=0.002), (ci, co, h, w, k, s, nchw, fused, err)
                rows.append(dict(input=[1, ci, h, w], co=co, k=k, stride=s, nchw=nchw, fused=fused, max_abs=err, plan=torch.ops.thin_native_flow.plan(ci, co, h, w, k, s)))
Path('evidence-shapes/flow-cases.json').write_text(json.dumps(rows, indent=2) + '\n')
print('FLOW_SHAPES_PASS', len(rows))
