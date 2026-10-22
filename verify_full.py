import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import gc
import ctypes
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np
import torch
import torch.nn.functional as F
import thin_backend

torch.set_num_threads(1)
torch.manual_seed(42)
root = Path('evidence-full')
root.mkdir(exist_ok=True)
rows, stages = [], []


def snapshot(name):
    assert not any(k == 'torch_npu' or k.startswith('torch_npu.') for k in sys.modules)
    parents = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                parents[int(p.name)] = int((p / 'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                pass
    found = {os.getpid()}
    while True:
        new = {p for p, parent in parents.items() if parent in found} - found
        if not new:
            break
        found |= new
    assert found == {os.getpid()}, found
    maps = Path('/proc/self/maps').read_text()
    assert not any(s in maps.lower() for s in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
    smaps = Path('/proc/self/smaps_rollup').read_text()
    pss = int(next(s.split()[1] for s in smaps.splitlines() if s.startswith('Pss:'))) / 1024
    row = dict(stage=name, pid=os.getpid(), pss_mib=pss, torch_npu=False, descendants=[])
    stages.append(row)
    (root / (name + '-smaps_rollup.txt')).write_text(smaps)
    (root / 'maps.txt').write_text(maps)
    (root / 'modules.txt').write_text('\n'.join(sorted(sys.modules)))
    print('STAGE ' + json.dumps(row), flush=True)


def compare(op, label, actual, expected, atol):
    assert actual.device.type == 'npu', actual.device
    got = actual.cpu()
    assert got.shape == expected.shape and got.dtype == expected.dtype
    assert torch.isfinite(got).all()
    error = (got.float() - expected.float()).abs().max().item() if got.numel() else 0.0
    if atol is not None:
        assert error <= atol, (op, label, error, atol)
    row = dict(op=op, label=label, dtype=str(actual.dtype), shape=list(actual.shape),
               max_abs=error, atol=atol, output_device=str(actual.device))
    rows.append(row)
    print(json.dumps(row), flush=True)


snapshot('loaded')
runpy.run_path('verify_all.py', run_name='__main__')
runpy.run_path('verify_ops.py', run_name='__main__')
snapshot('base_suites')

conv_cases = [
    ('stem16', 3, 640, 640, 16, 3, 2, True),
    ('point64', 64, 80, 80, 64, 1, 1, True),
    ('conv64', 64, 80, 80, 64, 3, 1, True),
    ('down128', 64, 80, 80, 128, 3, 2, True),
    ('cout17', 3, 64, 64, 17, 3, 2, True),
    ('cout31', 17, 20, 20, 31, 3, 1, True),
    ('point65', 19, 20, 20, 65, 1, 1, True),
    ('point_down', 64, 40, 40, 80, 1, 2, True),
    ('odd_down', 5, 13, 17, 23, 3, 2, True),
    ('odd_same', 3, 11, 13, 7, 3, 1, True),
    ('odd_point', 5, 11, 13, 9, 1, 1, False),
    ('tiny', 3, 1, 1, 5, 3, 2, True),
]
for dtype in (torch.float32, torch.float16):
    for label, c, h, width, cout, k, stride, has_bias in conv_cases:
        torch.manual_seed(42)
        x = torch.randn(1, c, h, width).to(dtype)
        bound = 1 / math.sqrt(c * k * k)
        w = torch.empty(cout, c, k, k).uniform_(-bound, bound).to(dtype)
        b = torch.empty(cout).uniform_(-bound, bound).to(dtype) if has_bias else None
        xd, wd = x.to('npu'), w.to('npu')
        bd = b.to('npu') if b is not None else None
        ref = F.conv2d(x.float(), w.float(), None if b is None else b.float(), stride=stride, padding=k // 2)
        actual = torch.ops.thin_backend.conv2d(xd, wd, bd, stride, k // 2)
        compare('conv2d', label, actual, ref.to(dtype), 1e-4 if dtype == torch.float32 else None)
        rows[-1]['fp32_reference_max_abs'] = (actual.cpu().float() - ref).abs().max().item()
        if dtype == torch.float16:
            accum = torch.ops.thin_backend.conv2d(xd, wd, bd, stride, k // 2, True)
            compare('conv2d_fp32_output', label, accum, ref, 1e-4)
        del x, w, b, xd, wd, bd, ref, actual
    snapshot('conv_' + str(dtype).split('.')[-1])

for dtype in (torch.float32, torch.float16):
    for shape, dim in (((5, 1000), -1), ((2, 1000, 17), 1),
                       ((1, 4, 16, 8400), 2), ((1, 16, 4, 8400), 1), ((2, 65537), -1)):
        x = (torch.randn(shape) * 3).to(dtype)
        compare('softmax', f'{shape}/dim={dim}', torch.softmax(x.to('npu'), dim),
                torch.softmax(x, dim), 1e-5 if dtype == torch.float32 else 1e-3)
    x = torch.randn(2, 16, 17).to(dtype)
    xd = x.to('npu').transpose(1, 2)
    compare('softmax', 'transpose', torch.softmax(xd, 1), torch.softmax(x.transpose(1, 2), 1),
            1e-5 if dtype == torch.float32 else 1e-3)
x = torch.randn(2, 16, 17).half()
compare('softmax', 'half_to_float', torch.ops.aten._softmax.default(x.to('npu'), 1, True),
        torch.softmax(x.float(), 1), 1e-5)
snapshot('softmax')


def nms_reference(boxes, scores, threshold, score_threshold, limit):
    order = np.argsort(-scores, kind='stable')
    order = order[scores[order].astype(np.float64) > score_threshold]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order) and len(keep) < limit:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        wh = np.maximum(np.minimum(boxes[i, 2:], boxes[rest, 2:]) -
                        np.maximum(boxes[i, :2], boxes[rest, :2]), 0)
        inter = wh[:, 0] * wh[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[~(iou.astype(np.float64) > threshold)]
    return keep


def check_nms(boxes, scores, threshold, score_threshold, limit, label):
    indices, count = torch.ops.thin_backend.nms(torch.from_numpy(boxes).to('npu'),
        torch.from_numpy(scores).to('npu'), threshold, score_threshold, limit)
    assert indices.device.type == count.device.type == 'npu'
    n = count.cpu().item()
    got = indices.cpu().tolist()
    expected = nms_reference(boxes, scores, threshold, score_threshold, limit)
    assert n == len(expected) and got[:n] == expected, (label, got[:n], expected)
    assert got[n:] == [-1] * (limit - n)
    row = dict(op='nms', label=label, m=len(scores), kept=n, max_out=limit,
               iou_threshold=threshold, score_threshold=score_threshold, exact_order=True, exact_count=True)
    rows.append(row)
    print(json.dumps(row), flush=True)


rng = np.random.default_rng(43)
for m in (0, 1, 7, 1023, 1024, 1025, 4096):
    xy = rng.uniform(-500, 500, (m, 2)).astype(np.float32)
    wh = rng.uniform(1, 350, (m, 2)).astype(np.float32)
    boxes = np.concatenate((xy, xy + wh), axis=1)
    scores = rng.permutation(np.arange(m, dtype=np.float32)) / max(m, 1)
    for threshold in (0.0, 0.5, 1.0):
        check_nms(boxes, scores, threshold, -float('inf'), m, 'random')
    check_nms(boxes, scores, 0.5, 0.2, min(300, m), 'threshold-cap')
boxes = np.array([[0, 0, 3, 1], [1, 0, 4, 1]], np.float32)
scores = np.array([0.75, 0.5], np.float32)
for threshold in (math.nextafter(0.5, 0.0), 0.5, math.nextafter(0.5, 1.0)):
    check_nms(boxes, scores, threshold, -float('inf'), 5, 'iou-double-boundary')
    check_nms(boxes, scores, 1.0, threshold, 5, 'score-double-boundary')
snapshot('nms')

for dtype in (torch.float32, torch.float16):
    torch.manual_seed(17)
    x = torch.randn(1, 3, 16, 16).to(dtype)
    w1 = torch.randn(7, 3, 3, 3).mul_(0.05).to(dtype)
    b1 = torch.randn(7).mul_(0.05).to(dtype)
    w2 = torch.randn(16, 14, 1, 1).mul_(0.05).to(dtype)
    b2 = torch.randn(16).mul_(0.05).to(dtype)
    y = torch.ops.thin_backend.conv2d(x.to('npu'), w1.to('npu'), b1.to('npu'), 2, 1)
    y = F.silu(y, inplace=True)
    y = torch.ops.thin_backend.maxpool2d(y)
    y = torch.ops.thin_backend.upsample_nearest2d(y)
    y = torch.cat((y, y), 1)
    y = torch.ops.thin_backend.conv2d(y, w2.to('npu'), b2.to('npu'), 1, 0)
    y = torch.softmax(y, 1)
    ref = F.conv2d(x.float(), w1.float(), b1.float(), stride=2, padding=1).to(dtype)
    ref = F.silu(ref, inplace=True)
    ref = F.interpolate(F.max_pool2d(ref, 5, 1, 2), scale_factor=2, mode='nearest')
    ref = torch.cat((ref, ref), 1)
    ref = F.conv2d(ref.float(), w2.float(), b2.float()).to(dtype)
    ref = torch.softmax(ref, 1)
    compare('mixed_chain', 'conv-silu-pool-upsample-cat-conv-softmax', y, ref,
            1e-4 if dtype == torch.float32 else 1e-3)
snapshot('all_checks_untrimmed')
gc.collect()
ctypes.CDLL(None).malloc_trim(0)
snapshot('test_heap_trimmed')

ops = ['aten::empty.memory_format', 'aten::empty_strided', 'aten::copy_', 'aten::as_strided',
       'aten::view', 'aten::split.Tensor', 'aten::split_with_sizes', 'aten::cat', 'aten::silu_',
       'aten::add.Tensor', 'aten::sub.Tensor', 'aten::mul.Tensor', 'aten::div.Tensor', 'aten::_softmax',
       'aten::transpose.int', 'aten::unsqueeze', 'aten::expand', 'aten::conv2d', 'aten::convolution',
       'aten::sigmoid', 'aten::max_pool2d', 'aten::upsample_nearest2d']
ops += ['thin_backend::' + s for s in ('add', 'sigmoid', 'maxpool2d', 'upsample_nearest2d', 'conv2d', 'nms')]
(root / 'dispatch.txt').write_text('\n\n'.join(op + '\n' + torch._C._dispatch_dump_table(op) for op in ops))
base = json.loads(Path('evidence-all/results.json').read_text())
extra = json.loads(Path('evidence-ops/results.json').read_text())
assert base['pid'] == extra['pid'] == os.getpid()
report = dict(pid=os.getpid(), torch_version=torch.__version__, autoload=os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],
              torch_npu=False, descendants=[], base_cases=base['results'], ops_cases=extra['results'],
              full_cases=rows, stages=stages,
              extension=str(Path(thin_backend.__file__).with_name('libthin_backend.so').resolve()),
              binary_environment={k: v for k, v in os.environ.items() if k.startswith('THIN_')})
(root / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('FULL_ALL_PASS', len(base['results']) + len(extra['results']) + len(rows), 'PID', os.getpid(), flush=True)
