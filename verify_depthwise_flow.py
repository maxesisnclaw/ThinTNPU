import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import thin_backend

torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libdepthwise_flow.so'))
torch.set_num_threads(1)
torch.manual_seed(0)


def to_blocked(x):
    n, c, h, w = x.shape
    assert n == 1 and c % 16 == 0
    return x.reshape(1, c // 16, 16, h, w).permute(0, 1, 3, 4, 2).contiguous()


def from_blocked(y):
    n, c1, h, w, c0 = y.shape
    assert n == 1 and c0 == 16
    return y.permute(0, 1, 4, 2, 3).reshape(1, c1 * 16, h, w).contiguous()


def pack_weight(w):
    c, one, k, k2 = w.shape
    assert one == 1 and k == k2 and c % 16 == 0
    return w.reshape(c, k, k).reshape(c // 16, 16, k, k).permute(0, 2, 3, 1).contiguous()


def run(x_cpu, w_cpu, b_cpu, kernel, stride, fused, device_offset=False):
    pw = pack_weight(w_cpu).to('npu')
    bp = b_cpu.to('npu') if b_cpu is not None else None
    if device_offset:
        c = x_cpu.shape[1]
        pair = to_blocked(torch.cat([torch.randn_like(x_cpu), x_cpu], dim=1)).to('npu')
        xb = pair[:, c // 16:, :, :, :]
        assert xb.is_contiguous() and xb.storage_offset() % 16 == 0 and xb.storage_offset() != 0
    else:
        xb = to_blocked(x_cpu).to('npu')
    yb = torch.ops.thin_native_flow.depthwise(xb, pw, bp, kernel, stride, fused)
    return from_blocked(yb.cpu()).float()


def reference(x_cpu, w_cpu, b_cpu, kernel, stride, fused):
    c = x_cpu.shape[1]
    ref = F.conv2d(x_cpu.float(), w_cpu.float(), None if b_cpu is None else b_cpu.float(),
                   stride=stride, padding=kernel // 2, groups=c)
    return F.silu(ref) if fused else ref


failures = []


def check(name, got, want):
    ok = got.shape == want.shape
    diff = float('nan')
    if ok:
        d = (got - want).abs()
        diff = d.max().item() if d.numel() else 0.0
        ok = diff <= 2e-3
    if ok:
        print(f'PASS {name}  shape={tuple(got.shape)} max_abs_diff={diff:.2e}', flush=True)
    else:
        failures.append(name)
        print(f'FAIL {name}: shape {tuple(got.shape)} vs {tuple(want.shape)} max_abs_diff={diff:.2e}', flush=True)


def expect_raises(name, fn):
    try:
        fn()
    except (RuntimeError, ValueError, TypeError):
        print(f'PASS {name} (rejected)', flush=True)
        return
    failures.append(name)
    print(f'FAIL {name}: expected rejection but call succeeded', flush=True)


cases = [
    (16, 20, 20, 3, 1),
    (32, 40, 40, 3, 1),
    (64, 80, 80, 3, 1),
    (16, 13, 17, 3, 1),
    (128, 20, 20, 3, 1),
    (32, 40, 40, 3, 2),
    (64, 80, 80, 3, 2),
    (16, 33, 31, 3, 2),
    (16, 32, 32, 5, 1),
    (32, 20, 20, 5, 2),
    (48, 7, 9, 1, 1),
    (16, 1, 31, 3, 1),
    (256, 1, 100, 3, 1),
    (256, 1, 1, 3, 1),
    (256, 20, 20, 3, 1),
]

with torch.inference_mode():
    for c, h, w, k, s in cases:
        x = torch.randn(1, c, h, w, dtype=torch.float16)
        bound = 1.0 / math.sqrt(k * k)
        wt = torch.empty(c, 1, k, k).uniform_(-bound, bound).half()
        for fused in (False, True):
            for use_bias in (False, True):
                b = torch.empty(c).uniform_(-bound, bound).half() if use_bias else None
                tag = f'c{c} {h}x{w} k{k}s{s} fused={int(fused)} bias={int(use_bias)}'
                check(tag, run(x, wt, b, k, s, fused),
                      reference(x, wt, b, k, s, fused))

    c, h, w, k, s = 32, 40, 40, 3, 1
    x = torch.randn(1, c, h, w, dtype=torch.float16)
    wt = torch.empty(c, 1, k, k).uniform_(-0.33, 0.33).half()
    b = torch.empty(c).uniform_(-0.33, 0.33).half()
    check('device-offset c32 40x40 k3s1 fused bias',
          run(x, wt, b, k, s, True, device_offset=True),
          reference(x, wt, b, k, s, True))

    x = torch.randn(1, 256, 1, 100).half()
    wt = torch.randn(256, 1, 3, 3).half()
    bias = torch.randn(256).half().to('npu')
    xb = to_blocked(x).to('npu')
    pw = pack_weight(wt).to('npu')
    eager = torch.ops.thin_native_flow.depthwise(xb, pw, bias, 3, 1, False).cpu()
    for _ in range(3):
        repeated = torch.ops.thin_native_flow.depthwise(xb, pw, bias, 3, 1, False).cpu()
        assert torch.equal(eager.view(torch.int16), repeated.view(torch.int16))
    graph = torch.classes.thin_backend.RuntimeGraph()
    graph.begin()
    captured = torch.ops.thin_native_flow.depthwise(xb, pw, bias, 3, 1, False)
    graph.end()
    graph.replay()
    assert torch.equal(eager.view(torch.int16), captured.cpu().view(torch.int16))
    xb.copy_(to_blocked(x * 0.5).to('npu'))
    changed = torch.ops.thin_native_flow.depthwise(xb, pw, bias, 3, 1, False).cpu()
    graph.replay()
    assert torch.equal(changed.view(torch.int16), captured.cpu().view(torch.int16))
    graph.close()
    print('DEPTHWISE_CAPTURE_BITS_PASS', flush=True)

    c, h, w, k = 32, 20, 20, 3
    xb_ok = to_blocked(torch.randn(1, c, h, w, dtype=torch.float16)).to('npu')
    pw_ok = pack_weight(torch.empty(c, 1, k, k).uniform_(-0.3, 0.3).half()).to('npu')
    b_ok = torch.zeros(c).half().to('npu')
    expect_raises('reject fp32 input',
                  lambda: torch.ops.thin_native_flow.depthwise(
                      to_blocked(torch.randn(1, c, h, w)).to('npu').float(), pw_ok, b_ok, k, 1, True))
    expect_raises('reject even kernel',
                  lambda: torch.ops.thin_native_flow.depthwise(xb_ok, pw_ok, b_ok, 2, 1, True))
    expect_raises('reject stride 3',
                  lambda: torch.ops.thin_native_flow.depthwise(xb_ok, pw_ok, b_ok, k, 3, True))
    expect_raises('reject packed shape mismatch',
                  lambda: torch.ops.thin_native_flow.depthwise(
                      xb_ok, pack_weight(torch.zeros(c, 1, 5, 5).half()).to('npu'), b_ok, k, 1, True))
    expect_raises('reject bias length',
                  lambda: torch.ops.thin_native_flow.depthwise(
                      xb_ok, pw_ok, torch.zeros(c + 16).half().to('npu'), k, 1, True))

if failures:
    print(f'\n{len(failures)} case(s) failed: {failures}', flush=True)
    sys.exit(1)
print('\nDEPTHWISE_ALL_PASS', flush=True)
