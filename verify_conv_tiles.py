import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import math
import torch
import torch.nn.functional as F
import thin_backend

torch.set_num_threads(1)
cases = [
    (32, 160, 32, 3, 1),
    (32, 320, 64, 3, 2),
    (64, 80, 64, 3, 1),
    (128, 80, 256, 3, 2),
    (128, 40, 128, 3, 1),
    (256, 40, 512, 3, 2),
    (256, 40, 256, 3, 1),
    (256, 40, 128, 3, 1),
    (512, 20, 128, 3, 1),
    (64, 20, 64, 3, 1),
    (768, 40, 256, 1, 1),
    (512, 40, 256, 1, 1),
    (384, 40, 256, 1, 1),
    (144, 40, 128, 1, 1),
    (512, 20, 512, 1, 1),
]
with torch.inference_mode():
    for ci, size, co, kernel, stride in cases:
        torch.manual_seed(42)
        x = torch.randn(1, ci, size, size).half()
        bound = 1 / math.sqrt(ci * kernel * kernel)
        weight = torch.empty(co, ci, kernel, kernel).uniform_(-bound, bound).half()
        bias = torch.empty(co).uniform_(-bound, bound).half()
        xd, wd, bd = x.to("npu"), weight.to("npu"), bias.to("npu")
        packed = torch.ops.thin_backend.conv2d_prepare(wd)
        reference = F.conv2d(x.float(), weight.float(), bias.float(), stride, kernel // 2)
        for fused in (False, True):
            expected = F.silu(reference) if fused else reference
            for fp32 in (False, True):
                actual = torch.ops.thin_backend.conv2d_prepared(
                    xd, wd, packed, bd, stride, kernel // 2, fp32, fused).cpu().float()
                torch.testing.assert_close(actual, expected, atol=1e-4 if fp32 else 2e-3, rtol=0)
        print("PASS", ci, size, co, kernel, stride, flush=True)
print("CONV_TILES_ALL_PASS", len(cases), flush=True)
