import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import json
from pathlib import Path
import sys

import torch
import thin_backend

torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libbmm_flow.so'))
torch.set_num_threads(1)
torch.manual_seed(894)
ops = torch.ops.thin_native_flow


def reference(a, b):
    result = torch.zeros(a.size(0), a.size(1), b.size(2), dtype=torch.float32)
    for k in range(a.size(2)):
        result += a[:, :, k].float().unsqueeze(2) * b[:, k, :].float().unsqueeze(1)
    return result.half()


def check(name, compute, expected, update):
    actual = compute().cpu().contiguous()
    assert torch.equal(actual.view(torch.int16), expected.contiguous().view(torch.int16)), name
    graph = torch.classes.thin_backend.RuntimeGraph()
    graph.begin()
    captured = compute()
    graph.end()
    graph.replay()
    assert torch.equal(captured.cpu().contiguous().view(torch.int16), actual.view(torch.int16)), name
    update()
    changed = compute().cpu().contiguous()
    graph.replay()
    assert torch.equal(captured.cpu().contiguous().view(torch.int16), changed.view(torch.int16)), name
    graph.close()
    print(json.dumps(dict(case=name, shape=list(actual.shape), sequential_reference_bit_equal=True,
                          eager_capture_bit_equal=True, changed_input_replay_bit_equal=True)), flush=True)


with torch.no_grad():
    for batch, m, k, n in ((2, 19, 33, 23), (1, 1, 600, 257), (3, 5, 37, 1031)):
        for transpose_a, transpose_b in ((False, False), (True, False), (False, True), (True, True)):
            ac = torch.randn(batch, k, m + 1).half() if transpose_a else torch.randn(batch, m, k + 1).half()
            bc = torch.randn(batch, n, k + 1).half() if transpose_b else torch.randn(batch, k, n + 1).half()
            ad, bd = ac.to('npu'), bc.to('npu')
            a = ad[:, :, 1:].transpose(1, 2) if transpose_a else ad[:, :, 1:]
            b = bd[:, :, 1:].transpose(1, 2) if transpose_b else bd[:, :, 1:]
            ar = ac[:, :, 1:].transpose(1, 2) if transpose_a else ac[:, :, 1:]
            br = bc[:, :, 1:].transpose(1, 2) if transpose_b else bc[:, :, 1:]
            expected = reference(ar, br)
            def update():
                ad.copy_((ac * 0.5).to('npu'))
                bd.copy_((bc * -0.25).to('npu'))
            check(f'strided-{batch}-{m}-{k}-{n}-{transpose_a}-{transpose_b}',
                  lambda: torch.bmm(a, b), expected, update)
    for heads, key, dim, tokens in ((4, 32, 64, 400), (2, 16, 32, 160),
                                    (1, 16, 16, 1), (3, 48, 80, 37)):
        source = torch.randn(heads, (2 * key + dim) // 16, tokens, 16).half()
        device = source.to('npu')
        q, k, v = device.split((key // 16, key // 16, dim // 16), 1)
        qr, kr, vr = source.split((key // 16, key // 16, dim // 16), 1)
        def logical(x):
            return x.permute(0, 1, 3, 2).reshape(heads, -1, tokens)
        expected = reference(logical(qr).transpose(1, 2), logical(kr))
        check(f'blocked-qk-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.layout_bmm(q, k, True, False, False), expected,
              lambda: device.copy_((source * 0.5).to('npu')))
        device.copy_(source.to('npu'))
        probability = torch.randn(heads, tokens, tokens).half()
        pd = probability.to('npu')
        expected = reference(logical(vr), probability.transpose(1, 2))
        def update():
            device.copy_((source * 0.5).to('npu'))
            pd.copy_((probability * 0.25).to('npu'))
        check(f'blocked-vp-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.layout_bmm(v, pd, False, True, False), expected, update)
        device.copy_(source.to('npu'))
        pd.copy_(probability.to('npu'))
        expected = expected.view(heads, dim // 16, 16, tokens).permute(0, 1, 3, 2).contiguous()
        check(f'blocked-output-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.layout_bmm(v, pd, False, True, True), expected, update)
    for heads, key, dim, tokens in ((4, 32, 64, 400), (4, 8, 16, 37),
                                    (2, 24, 48, 19), (8, 3, 6, 19)):
        group = 2 * key + dim
        logical = torch.randn(heads, group, tokens).half()
        source = logical.reshape(1, heads * group // 16, 16, tokens).permute(0, 1, 3, 2).contiguous()
        device = source.to('npu')
        qr, kr, vr = logical.split((key, key, dim), 1)
        expected = reference(qr.transpose(1, 2), kr)
        check(f'grouped-qk-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.layout_bmm(device, device, True, False, False, heads, 0, key, key, key),
              expected, lambda: device.copy_((source * 0.5).to('npu')))
        device.copy_(source.to('npu'))
        probability = torch.randn(heads, tokens, tokens).half()
        pd = probability.to('npu')
        expected = reference(vr, probability.transpose(1, 2))
        expected = expected.reshape(1, heads * dim // 16, 16, tokens).permute(0, 1, 3, 2).contiguous()
        check(f'grouped-vp-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.layout_bmm(device, pd, False, True, True, heads, 2 * key, dim),
              expected, lambda: device.copy_((source * 0.5).to('npu')))
        device.copy_(source.to('npu'))
        expected = vr.reshape(1, heads * dim // 16, 16, tokens).permute(0, 1, 3, 2).contiguous()
        check(f'grouped-select-{heads}-{key}-{dim}-{tokens}',
              lambda: ops.channel_select(device, heads, 2 * key, dim), expected,
              lambda: device.copy_((source * 0.5).to('npu')))
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(token in maps for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
print('BMM_LAYOUT_ALL_PASS', flush=True)
