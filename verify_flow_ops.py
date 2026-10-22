import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import thin_backend

torch.manual_seed(0)
torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libflow_ops.so'))


def to_blocked(x):
    n, c, h, w = x.shape
    assert n == 1 and c % 16 == 0
    return x.reshape(1, c // 16, 16, h, w).permute(0, 1, 3, 4, 2).contiguous()


def from_blocked(y):
    n, c1, h, w, c0 = y.shape
    assert n == 1 and c0 == 16
    return y.permute(0, 1, 4, 2, 3).reshape(1, c1 * 16, h, w).contiguous()


def run_maxpool(x_cpu, kernel):
    xb = to_blocked(x_cpu).to('npu')
    yb = torch.ops.thin_native_flow.maxpool(xb, kernel)
    return from_blocked(yb.cpu())


def ref_maxpool(x_cpu, kernel):
    # max_pool selects an existing fp16 value, so float math is bit-exact here.
    return F.max_pool2d(x_cpu.float(), kernel, stride=1, padding=kernel // 2).half()


def run_upsample(x_cpu):
    xb = to_blocked(x_cpu).to('npu')
    yb = torch.ops.thin_native_flow.upsample(xb)
    return from_blocked(yb.cpu())


def ref_upsample(x_cpu):
    return F.interpolate(x_cpu.float(), scale_factor=2, mode='nearest').half()


failures = []


def check(name, got, want):
    ok = got.shape == want.shape and torch.equal(got, want)
    if not ok:
        diff = (got.float() - want.float()).abs()
        failures.append(name)
        print(f'FAIL {name}: shape {tuple(got.shape)} vs {tuple(want.shape)}, '
              f'max_abs_diff={diff.max().item() if diff.numel() else float("nan")}')
    else:
        print(f'PASS {name}  shape={tuple(got.shape)}')


maxpool_shapes = [
    (256, 20, 20, 5),   # SPPF k5 on half spatial
    (128, 40, 40, 5),
    (512, 20, 20, 5),
    (32, 13, 17, 3),    # odd, non-square, smaller kernel
    (16, 7, 9, 7),      # kernel larger than one side's padding reach
    (48, 1, 31, 5),     # degenerate height
]
for c, h, w, k in maxpool_shapes:
    x = torch.randn(1, c, h, w, dtype=torch.float16)
    check(f'maxpool c{c} {h}x{w} k{k}', run_maxpool(x, k), ref_maxpool(x, k))

upsample_shapes = [
    (128, 40, 40),
    (256, 20, 20),
    (64, 13, 17),       # odd, non-square
    (16, 1, 33),
]
for c, h, w in upsample_shapes:
    x = torch.randn(1, c, h, w, dtype=torch.float16)
    check(f'upsample c{c} {h}x{w}', run_upsample(x), ref_upsample(x))

if failures:
    print(f'\n{len(failures)} case(s) failed: {failures}')
    sys.exit(1)
print('\nall flow ops cases passed')
