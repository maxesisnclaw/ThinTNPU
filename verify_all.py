import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import gc
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import thin_backend

torch.set_num_threads(1)
torch.manual_seed(42)
evidence = Path('evidence-all')
evidence.mkdir(exist_ok=True)
results = []
stages = []


def descendants():
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            pass
    found = {os.getpid()}
    while True:
        new = {pid for pid, parent in parents.items() if parent in found} - found
        if not new:
            return sorted(found - {os.getpid()})
        found |= new


def snapshot(name):
    assert 'torch_npu' not in sys.modules
    assert not any(n.startswith('torch_npu.') for n in sys.modules)
    children = descendants()
    assert children == [], children
    rollup = Path('/proc/self/smaps_rollup').read_text()
    (evidence / (name + '-smaps_rollup.txt')).write_text(rollup)
    pss_kib = int(next(line.split()[1] for line in rollup.splitlines() if line.startswith('Pss:')))
    row = dict(stage=name, pid=os.getpid(), pss_kib=pss_kib,
               pss_mib=pss_kib / 1024, torch_npu=False, descendants=children)
    stages.append(row)
    print(json.dumps(row), flush=True)


def compare(op, shape, dtype, actual, expected, tolerance, **extra):
    assert actual.device.type == 'npu', actual.device
    cpu = actual.cpu()
    assert cpu.shape == expected.shape and cpu.dtype == expected.dtype
    error = (cpu.float() - expected.float()).abs().max().item() if cpu.numel() else 0.0
    assert torch.isfinite(cpu).all()
    assert error <= tolerance, (op, shape, dtype, error, tolerance)
    row = dict(op=op, shape=list(shape), dtype=str(dtype), max_abs=error,
               atol=tolerance, output_device=str(actual.device), **extra)
    results.append(row)
    print(json.dumps(row), flush=True)


snapshot('loaded')
for size in (5000, 65537, 1, 2047, 2048, 2049, 16385, 0):
    a, b = torch.randn(size), torch.randn(size)
    x, y = a.to('npu'), b.to('npu')
    compare('add', a.shape, a.dtype, torch.ops.thin_backend.add(x, y), a + b, 0.0)
for shape in ((), (3, 5, 7)):
    a, b = torch.randn(shape), torch.randn(shape)
    compare('add', shape, a.dtype, torch.ops.thin_backend.add(a.to('npu'), b.to('npu')), a+b, 0.0)
snapshot('add')
for dtype, tolerance in ((torch.float32, 1e-5), (torch.float16, 1e-3)):
    for size in (1, 5000, 65537):
        a = torch.linspace(-8, 8, size, dtype=dtype)
        compare('sigmoid', a.shape, dtype, torch.ops.thin_backend.sigmoid(a.to('npu')),
                torch.sigmoid(a), tolerance)
snapshot('sigmoid')
for dtype in (torch.float32, torch.float16):
    for shape in ((1, 2, 5, 7), (2, 3, 13, 11), (1, 64, 80, 80)):
        a = torch.randn(shape, dtype=dtype)
        compare('maxpool2d', shape, dtype, torch.ops.thin_backend.maxpool2d(a.to('npu')),
                F.max_pool2d(a, 5, 1, 2), 0.0)
snapshot('maxpool2d')
for dtype in (torch.float32, torch.float16):
    for shape in ((1, 2, 5, 7), (2, 3, 13, 11), (1, 64, 40, 40)):
        a = torch.randn(shape, dtype=dtype)
        compare('upsample_nearest2d', shape, dtype,
                torch.ops.thin_backend.upsample_nearest2d(a.to('npu')),
                F.interpolate(a, scale_factor=2, mode='nearest'), 0.0)
snapshot('upsample_nearest2d')
for cin, h, width, cout in ((16, 32, 32, 32), (64, 80, 80, 64), (256, 20, 20, 256)):
    a = torch.randn(1, cin, h, width)
    bound = 1 / math.sqrt(cin * 9)
    w = torch.empty(cout, cin, 3, 3).uniform_(-bound, bound)
    x, weight = a.to('npu'), w.to('npu')
    expected = F.conv2d(a, w, padding=1)
    actual = torch.ops.thin_backend.conv2d(x, weight)
    compare('conv2d', a.shape, a.dtype, actual, expected, 1e-4,
            weight_shape=list(w.shape), weight_bound=bound)
    del a, w, x, weight, expected, actual
snapshot('conv2d')
# Reuse all handles after the Cube object and its runtime have been loaded.
a = torch.randn(1, 16, 8, 8)
w = torch.empty(16, 16, 3, 3).uniform_(-1/12, 1/12)
y = torch.ops.thin_backend.conv2d(a.to('npu'), w.to('npu'))
y = torch.ops.thin_backend.maxpool2d(y)
y = torch.ops.thin_backend.upsample_nearest2d(y)
y = torch.ops.thin_backend.sigmoid(y)
y = torch.ops.thin_backend.add(y, y)
ref = 2 * torch.sigmoid(F.interpolate(F.max_pool2d(F.conv2d(a, w, padding=1), 5, 1, 2),
                                    scale_factor=2, mode='nearest'))
compare('five_op_chain', a.shape, a.dtype, y, ref, 1e-5)
for op, args in (
    ('add', (torch.empty(5000, device='meta'), torch.empty(5000, device='meta'))),
    ('sigmoid', (torch.empty(65537, device='meta'),)),
    ('maxpool2d', (torch.empty(1, 16, 8, 8, device='meta'),)),
    ('upsample_nearest2d', (torch.empty(1, 16, 8, 8, device='meta'),)),
    ('conv2d', (torch.empty(1, 16, 8, 8, device='meta'), torch.empty(32, 16, 3, 3, device='meta'))),
):
    assert getattr(torch.ops.thin_backend, op)(*args).device.type == 'meta'
    for key in ('PrivateUse1', 'AutogradPrivateUse1', 'Meta'):
        assert torch._C._dispatch_has_kernel_for_dispatch_key('thin_backend::' + op, key)

del a, w, y, ref
for name in ('x', 'weight', 'cpu', 'actual', 'expected'):
    globals().pop(name, None)
gc.collect()
snapshot('verified')
import ctypes
ctypes.CDLL(None).malloc_trim(0)
snapshot('released_cpu_test_heap')
maps = Path('/proc/self/maps').read_text()
(evidence / 'maps.txt').write_text(maps)
for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'):
    assert token not in maps, token
report = dict(results=results, stages=stages, torch_npu=False, descendants=descendants(),
              torch_version=torch.__version__, pid=os.getpid())
(evidence / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('ALL_PASS', flush=True)
if os.environ.get('THIN_HOLD'):
    (evidence / 'ready.pid').write_text(str(os.getpid()))
    with open(evidence / 'release.fifo') as release:
        release.read()
