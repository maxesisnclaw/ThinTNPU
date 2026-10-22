import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import json
from pathlib import Path
import runpy
import sys
import torch
import thin_backend

torch.set_num_threads(1)
torch.manual_seed(2718)
evidence = Path('evidence-softmax')
evidence.mkdir(exist_ok=True)
results, stages = [], []

def descendants():
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
            return sorted(found - {os.getpid()})
        found |= new

def snapshot(stage):
    assert not any(k == 'torch_npu' or k.startswith('torch_npu.') for k in sys.modules)
    maps = Path('/proc/self/maps').read_text()
    for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'):
        assert token not in maps, token
    children = descendants()
    assert children == [], children
    smaps = Path('/proc/self/smaps_rollup').read_text()
    pss = int(next(x.split()[1] for x in smaps.splitlines() if x.startswith('Pss:'))) / 1024
    row = dict(stage=stage, pid=os.getpid(), pss_mib=pss, torch_npu=False, descendants=children)
    stages.append(row)
    (evidence / (stage + '-smaps_rollup.txt')).write_text(smaps)
    (evidence / 'maps.txt').write_text(maps)
    print(json.dumps(row), flush=True)

def compare(name, cpu, device, dim, half_to_float=False):
    assert device.device.type == 'npu'
    if half_to_float:
        actual = torch.ops.aten._softmax.default(device, dim, True)
        expected = torch.softmax(cpu.float(), dim)
    else:
        actual = torch.softmax(device, dim)
        expected = torch.softmax(cpu, dim)
    assert actual.device.type == 'npu'
    got = actual.cpu()
    assert got.dtype == expected.dtype and got.shape == expected.shape
    tol = 1e-3 if got.dtype == torch.float16 else 1e-5
    if torch.isfinite(expected).all():
        assert torch.isfinite(got).all()
        error = (got.float() - expected.float()).abs().max().item() if got.numel() else 0.0
    else:
        assert torch.equal(torch.isnan(got), torch.isnan(expected))
        assert torch.equal(torch.isposinf(got), torch.isposinf(expected))
        assert torch.equal(torch.isneginf(got), torch.isneginf(expected))
        finite = torch.isfinite(expected)
        error = (got.float()[finite] - expected.float()[finite]).abs().max().item() if finite.any() else 0.0
    assert error <= tol, (name, cpu.shape, dim, cpu.dtype, error, got, expected)
    row = dict(case=name, shape=list(cpu.shape), dim=dim, input_dtype=str(cpu.dtype), output_dtype=str(got.dtype),
               half_to_float=half_to_float, max_abs=error, atol=tol, nonfinite_match=True, output_device=str(actual.device))
    results.append(row)
    print(json.dumps(row), flush=True)

snapshot('loaded')
for dtype in (torch.float32, torch.float16):
    cpu = torch.randn(2, 16, 32, dtype=dtype)
    device = cpu.to('npu')
    for dim in (1, 2):
        compare('warm', cpu, device, dim)
snapshot('warm')
for dtype in (torch.float32, torch.float16):
    for shape, dims in (
        ((), (0, -1)), ((0,), (0,)), ((2, 0, 3), (0, 1, 2)),
        ((1,), (0,)), ((3,), (0,)), ((7,), (-1,)), ((100,), (0,)),
        ((3, 16), (-1, 0)), ((5, 1000), (1,)),
        ((2, 2049), (1,)), ((2, 4097), (-1,)), ((2, 65537), (-1,)),
        ((2, 3, 7), (0, 1, 2, -2)), ((2, 16, 17), (1,)),
        ((2, 1000, 17), (1,)), ((3, 7, 1), (1,)),
        ((1, 4, 16, 8400), (1, 2)), ((1, 16, 4, 8400), (1,)),
        ((2, 4, 16, 2100), (2,)),
    ):
        cpu = (torch.randn(shape) * 3).to(dtype)
        device = cpu.to('npu')
        for dim in dims:
            compare('random', cpu, device, dim)
    for shape, dim in (((3, 1000), -1), ((2, 16, 37), 1), ((2, 1000, 17), 1)):
        for shift in (-10000, 10000):
            cpu = (torch.randn(shape) * 10 + shift).to(dtype)
            compare('large_shift', cpu, cpu.to('npu'), dim)
        cpu = torch.zeros(shape, dtype=dtype)
        compare('uniform', cpu, cpu.to('npu'), dim)
    cpu = torch.randn(2, 3, 7, dtype=dtype)
    device = cpu.to('npu')
    for dim in (0, 1, -1):
        compare('transpose', cpu.transpose(0, 2), device.transpose(0, 2), dim)
    compare('offset_view', cpu[:, :, 1:6], device.as_strided((2, 3, 5), (21, 7, 1), 1), -1)
    compare('offset_view_inner', cpu[:, :, 1:6], device.as_strided((2, 3, 5), (21, 7, 1), 1), 1)
    compare('expanded', cpu.unsqueeze(0).expand(4, -1, -1, -1), device.unsqueeze(0).expand(4, -1, -1, -1), 0)
    for shape, dim in (((2, 16), -1), ((2, 16, 17), 1)):
        cpu = torch.randn(shape, dtype=dtype)
        device = cpu.to('npu')
        y = torch.softmax(device, dim)
        ref = torch.softmax(cpu, dim)
        z = torch.cat((torch.add(y, y), torch.ops.thin_backend.sigmoid(y)), dim)
        expected = torch.cat((torch.add(ref, ref), torch.sigmoid(ref)), dim)
        got = z.cpu()
        error = (got.float() - expected.float()).abs().max().item()
        tol = 1e-3 if dtype == torch.float16 else 1e-5
        assert error <= tol, ('coexist', error)
        results.append(dict(case='coexist_add_sigmoid_cat', shape=list(shape), dim=dim,
                            input_dtype=str(dtype), output_dtype=str(dtype), max_abs=error, atol=tol))
    for values in ([float('-inf'), 0, 1], [float('-inf')] * 3, [0, float('inf'), 1], [0, float('nan'), 1]):
        a = torch.tensor(values, dtype=dtype)
        for shape, dim in (((1, 3), 1), ((3, 1), 0), ((3, 17), 0)):
            cpu = a.reshape((1, 3) if dim else (3, 1)).expand(shape).contiguous()
            compare('nonfinite', cpu, cpu.to('npu'), dim)
    cpu = torch.tensor([[0, 0, float('-inf'), 0, 0],
                        [1, float('nan'), float('-inf'), float('inf'), float('-inf')],
                        [2, 2, float('-inf'), 1, 3]], dtype=dtype)
    compare('mixed_nonfinite_inner', cpu, cpu.to('npu'), 0)
    compare('mixed_nonfinite_last', cpu.t().contiguous(), cpu.t().contiguous().to('npu'), 1)
    snapshot('softmax_' + str(dtype).split('.')[-1])

for shape, dim in (((3, 1000), -1), ((2, 16, 17), 1)):
    cpu = torch.randn(shape, dtype=torch.float16)
    compare('half_to_float', cpu, cpu.to('npu'), dim, True)

snapshot('softmax_verified')
print('SOFTMAX_ALL_PASS', len(results), flush=True)
(evidence / 'dispatch.txt').write_text(torch._C._dispatch_dump_table('aten::_softmax'))
(evidence / 'environment.json').write_text(json.dumps(dict(
    TORCH_DEVICE_BACKEND_AUTOLOAD=os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'], torch_version=torch.__version__,
    pid=os.getpid(), torch_npu='torch_npu' in sys.modules, descendants=descendants()), indent=2))
(evidence / 'modules.txt').write_text('\n'.join(sorted(sys.modules)))
(evidence / 'process.txt').write_text(Path('/proc/self/status').read_text())
(evidence / 'threads.json').write_text(json.dumps([
    dict(tid=p.name, name=(p / 'comm').read_text().strip()) for p in Path('/proc/self/task').iterdir()
], indent=2))
(evidence / 'results.json').write_text(json.dumps(dict(results=results, stages=stages), indent=2))
runpy.run_path('verify_ops.py', run_name='__main__')
snapshot('all_verified')
(evidence / 'results.json').write_text(json.dumps(dict(results=results, stages=stages), indent=2))
