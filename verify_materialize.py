import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import sys
from pathlib import Path
import torch
import thin_backend
from thin_backend.graph import CapturedForward

torch.set_num_threads(1)
torch.manual_seed(913)

def bits(t):
    return t.cpu().contiguous().view(torch.int16)

with torch.inference_mode():
    cases = 0
    for dtype in (torch.float16, torch.float32):
        for groups, channels, rows in ((1, 16, 16), (3, 16, 6400), (2, 32, 48),
                                       (4, 48, 400), (2, 7, 19), (1, 85, 96)):
            x = torch.randn(groups, channels, rows).to(dtype)
            npu = x.to('npu')
            expected = x.permute(0, 2, 1).contiguous()
            def forward(value):
                return value.permute(0, 2, 1).contiguous()
            assert torch.equal(bits(forward(npu)), bits(expected))
            graph = CapturedForward(forward, npu)
            assert torch.equal(bits(graph.replay()), bits(expected))
            changed = (x * .5).to('npu')
            assert torch.equal(bits(graph(changed)), bits(forward(changed)))
            graph.close()
            cases += 1
        for shape in ((3, 128, 16), (2, 19, 7), (1, 32, 85), (4, 48, 32)):
            x = torch.randn(shape).to(dtype)
            npu = x.to('npu')
            for view in (lambda t: t[..., 2:4], lambda t: t[..., 4:],
                         lambda t: t[:, ::2, ::2], lambda t: t[:, :1].expand(-1, 31, -1)):
                assert torch.equal(bits(view(npu).contiguous()), bits(view(x).contiguous()))
                cases += 1
    assert not any(k.startswith('torch_npu') for k in sys.modules)
    assert not any(k in Path('/proc/self/maps').read_text().lower()
                   for k in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
print('MATERIALIZE_ALL_PASS', cases, flush=True)

with torch.inference_mode():
    cases = 0
    for dtype in (torch.float16, torch.float32):
        for rows in (1, 7, 127, 128, 129, 6400):
            for widths in ((2, 2, 12), (1, 7, 24), (3, 5, 8), (6, 10)):
                sources = [torch.randn(rows, w).to(dtype) for w in widths]
                values = [x.to('npu') for x in sources]
                assert torch.equal(bits(torch.cat(values, -1)), bits(torch.cat(sources, -1)))
                graph = CapturedForward(lambda _: torch.cat(values, -1), values[0])
                assert torch.equal(bits(graph.replay()), bits(torch.cat(sources, -1)))
                graph.close()
                cases += 1
        for mode in ('sliced', 'broadcast', 'offset'):
            transform = {'sliced': lambda t: t[::3],
                         'broadcast': lambda t: t[:1].expand(273, -1),
                         'offset': lambda t: t[:, 1:]}[mode]
            widths = (2, 2, 12)
            sources = [torch.randn(273, w + (mode == 'offset')).to(dtype) for w in widths]
            values = [transform(x.to('npu')) for x in sources]
            sources = [transform(x) for x in sources]
            assert torch.equal(bits(torch.cat(values, -1)), bits(torch.cat(sources, -1)))
            cases += 1
print('STRIDED_CONCAT_ALL_PASS', cases, flush=True)

with torch.inference_mode():
    raw = torch.randint(-32768, 32768, (3, 32, 48), dtype=torch.int16).view(torch.float16)
    actual = raw.to('npu').transpose(1, 2).contiguous()
    assert torch.equal(bits(actual), bits(raw.transpose(1, 2).contiguous()))
    slices = [raw[..., :2].contiguous(), raw[..., 2:4].contiguous(), raw[..., 4:16].contiguous()]
    assert torch.equal(bits(torch.cat([x.to('npu') for x in slices], -1)), bits(torch.cat(slices, -1)))
print('MATERIALIZE_SPECIAL_BITS_PASS', flush=True)
