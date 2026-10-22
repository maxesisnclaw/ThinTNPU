import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import math
import torch
import torch.nn.functional as F
import thin_backend

torch.set_num_threads(1)
checks = 0
with torch.inference_mode():
    for co in (64, 128, 256, 512):
        for stride in (1, 2):
            torch.manual_seed(co + stride)
            ci, h = 256, 40
            bound = 1 / math.sqrt(ci * 9)
            x = torch.randn(1, ci, h, h).half()
            w = torch.empty(co, ci, 3, 3).uniform_(-bound, bound).half()
            b = torch.empty(co).uniform_(-bound, bound).half()
            xd, wd, bd = x.to('npu'), w.to('npu'), b.to('npu')
            packed = torch.ops.thin_backend.conv2d_prepare(wd)
            for has_bias in (False, True):
                ref = F.conv2d(x.float(), w.float(), b.float() if has_bias else None, stride, 1)
                for fused in (False, True):
                    expected = F.silu(ref) if fused else ref
                    for fp32 in (False, True):
                        actual = torch.ops.thin_backend.conv2d_prepared(
                            xd, wd, packed, bd if has_bias else None, stride, 1, fp32, fused).cpu().float()
                        torch.testing.assert_close(actual, expected, atol=1e-4 if fp32 else 2e-3, rtol=0)
                        checks += 1
            print('PASS', ci, h, co, stride, flush=True)
print('CONV_STREAM_ALL_PASS', checks, flush=True)
