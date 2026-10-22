import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import ctypes
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
evidence = Path('evidence-ops')
evidence.mkdir(exist_ok=True)
results, stages = [], []

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
        new = {p for p, parent in parents.items() if parent in found} - found
        if not new:
            return sorted(found - {os.getpid()})
        found |= new

def snapshot(stage):
    assert not any(k == 'torch_npu' or k.startswith('torch_npu.') for k in sys.modules)
    children = descendants()
    assert children == [], children
    maps = Path('/proc/self/maps').read_text()
    for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'):
        assert token not in maps, token
    rollup = Path('/proc/self/smaps_rollup').read_text()
    pss_kib = int(next(s.split()[1] for s in rollup.splitlines() if s.startswith('Pss:')))
    row = dict(stage=stage, pid=os.getpid(), pss_mib=pss_kib / 1024,
               torch_npu=False, descendants=children)
    stages.append(row)
    (evidence / (stage + '-smaps_rollup.txt')).write_text(rollup)
    (evidence / 'maps.txt').write_text(maps)
    print(json.dumps(row), flush=True)

def compare(op, actual, expected, atol=0, **extra):
    assert actual.device.type == 'npu'
    cpu = actual.cpu()
    assert cpu.shape == expected.shape and cpu.dtype == expected.dtype
    err = (cpu.float() - expected.float()).abs().max().item() if cpu.numel() else 0.0
    assert torch.isfinite(cpu).all()
    assert err <= atol, (op, actual.shape, actual.dtype, err, atol)
    row = dict(op=op, shape=list(cpu.shape), dtype=str(cpu.dtype), max_abs=err,
               atol=atol, output_device=str(actual.device), **extra)
    results.append(row)
    print(json.dumps(row), flush=True)

def alias(actual, original):
    assert actual.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()

snapshot('loaded')
for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 1e-3)):
    for shape in ((1,), (3,), (7,), (100,), (2047,), (2048,), (2049,), (16385,), (65537,), (0,), (), (3, 5, 7)):
        a = torch.rand(shape, dtype=dtype) * 2 - 1
        b = torch.rand(shape, dtype=dtype) + 0.5
        x, y = a.to('npu'), b.to('npu')
        for op in ('sub', 'mul', 'div'):
            compare(op, getattr(torch, op)(x, y), getattr(torch, op)(a, b), tol)
    a = torch.rand(2, 5, 7, dtype=dtype)
    b = torch.rand(2, 5, 7, dtype=dtype) + 0.5
    x, y = a.to('npu').transpose(0, 2), b.to('npu').transpose(0, 2)
    for op in ('sub', 'mul', 'div'):
        compare(op + '_strided', getattr(torch, op)(x, y), getattr(torch, op)(a.transpose(0, 2), b.transpose(0, 2)), tol)
    compare('sub_alpha', torch.sub(a.to('npu'), b.to('npu'), alpha=2), torch.sub(a, b, alpha=2), tol)
snapshot('binary')

for dtype in (torch.float32, torch.float16):
    for dim in (0, 1, 2, -1):
        axis = dim % 3
        shape_a, shape_b = [3, 5, 7], [3, 5, 7]
        shape_b[axis] = 4
        a, b = torch.randn(shape_a, dtype=dtype), torch.randn(shape_b, dtype=dtype)
        compare('cat', torch.cat((a.to('npu'), b.to('npu'), a.to('npu')), dim), torch.cat((a, b, a), dim), dim=dim)
    a = torch.randn(2, 5, 7, dtype=dtype)
    x = a.to('npu')
    for dim in (0, 1, 2, -1):
        parts, refs = torch.split(x, 2, dim), torch.split(a, 2, dim)
        assert len(parts) == len(refs)
        for i, (p, r) in enumerate(zip(parts, refs)):
            alias(p, x)
            compare('split', p, r, dim=dim, part=i, stride=list(p.stride()), offset=p.storage_offset())
        compare('split_cat', torch.cat(parts, dim), a, dim=dim)
        sizes = [1, 0, a.shape[dim] - 1]
        parts = torch.split(x, sizes, dim)
        for i, (p, r) in enumerate(zip(parts, torch.split(a, sizes, dim))):
            alias(p, x)
            compare('split_with_sizes', p, r, dim=dim, part=i, stride=list(p.stride()), offset=p.storage_offset())
    x = a.to('npu').transpose(0, 2)
    compare('cat_transposed', torch.cat((x, x), 1), torch.cat((a.transpose(0, 2), a.transpose(0, 2)), 1))
    empty = torch.empty(0, dtype=dtype).to('npu')
    compare('cat_empty', torch.cat((empty, a.to('npu'), empty), 1), a)
    compare('cat_all_empty', torch.cat((empty, empty)), torch.empty(0, dtype=dtype))
    for size in (0, 2):
        for p, r in zip(torch.split(empty, size), torch.split(torch.empty(0, dtype=dtype), size)):
            compare('split_empty', p, r)
snapshot('cat_split')

for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 1e-3)):
    for shape in ((1,), (3,), (7,), (100,), (2049,), (65537,), (0,), (), (2, 7, 5, 7)):
        count = math.prod(shape)
        a = torch.linspace(-4, 4, max(1, count), dtype=dtype)[:count].reshape(shape)
        x = a.to('npu')
        pointer = x.data_ptr()
        out = F.silu(x, inplace=True)
        assert out is x and x.data_ptr() == pointer
        compare('silu_', out, F.silu(a), tol)
    a = torch.randn(2, 7, 5, 7, dtype=dtype)
    x = a.to('npu')
    part = torch.split(x, [3, 4], 1)[1]
    refpart = torch.split(a, [3, 4], 1)[1]
    pointer = part.data_ptr()
    out = F.silu(part, inplace=True)
    F.silu(refpart, inplace=True)
    assert out is part and pointer == out.data_ptr()
    compare('silu_strided_parent', x, a, tol)
    for channels in (8, 7):
        a = torch.randn(2, channels, 13, 11, dtype=dtype) * 0.5
        x = a.to('npu')
        p, q = torch.split(x, [channels // 2, channels - channels // 2], 1)
        rp, rq = torch.split(a, [channels // 2, channels - channels // 2], 1)
        F.silu(q, inplace=True)
        F.silu(rq, inplace=True)
        z = torch.cat((p, torch.add(q, q)), 1)
        ref = torch.cat((rp, torch.add(rq, rq)), 1)
        compare('c2f_chain', z, ref, tol)
        compare('c2f_parent_alias', x, a, tol)
snapshot('silu_chain')

for dtype in (torch.float32, torch.float16):
    a = torch.randn(2, 3, 7, dtype=dtype)
    x = a.to('npu')
    for op, y, r in (
        ('view', x.view(6, 7), a.view(6, 7)),
        ('transpose', x.transpose(0, 2), a.transpose(0, 2)),
        ('unsqueeze', x.unsqueeze(1), a.unsqueeze(1)),
        ('expand', x.unsqueeze(0).expand(4, -1, -1, -1), a.unsqueeze(0).expand(4, -1, -1, -1)),
        ('as_strided', x.as_strided((2, 3), (21, 7), 1), a.as_strided((2, 3), (21, 7), 1)),
    ):
        alias(y, x)
        assert y.stride() == r.stride() and y.storage_offset() == r.storage_offset()
        compare(op, y, r, stride=list(y.stride()), offset=y.storage_offset())
    y = x.unsqueeze(0).expand(2, -1, -1, -1)
    compare('cat_expanded', torch.cat((y, y), -1), torch.cat((a.unsqueeze(0).expand(2, -1, -1, -1),) * 2, -1))

snapshot('verified')
(evidence / 'modules.txt').write_text('\n'.join(sorted(sys.modules)))
(evidence / 'process.txt').write_text(Path('/proc/self/status').read_text())
(evidence / 'environment.json').write_text(json.dumps(dict(
    TORCH_DEVICE_BACKEND_AUTOLOAD=os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],
    torch_version=torch.__version__, pid=os.getpid(), torch_npu='torch_npu' in sys.modules,
    descendants=descendants()), indent=2))
gc.collect()
ctypes.CDLL(None).malloc_trim(0)
snapshot('trimmed')
(evidence / 'dispatch.txt').write_text('\n\n'.join(torch._C._dispatch_dump_table('aten::' + op) for op in ('cat', 'split.Tensor', 'split_with_sizes', 'silu_', 'sub.Tensor', 'mul.Tensor', 'div.Tensor', 'as_strided', 'view', 'transpose.int', 'unsqueeze', 'expand')))
(evidence / 'results.json').write_text(json.dumps(dict(results=results, stages=stages, pid=os.getpid(), torch_npu=False, descendants=descendants()), indent=2))
print('ALL_PASS', len(results), flush=True)
