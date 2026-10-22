import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
import thin_backend
from thin_backend.graph import CapturedForward


def bits_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


cases = 0
with torch.inference_mode():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    for dtype in (torch.float16, torch.float32):
        for count in (1, 2, 3, 4, 6, 8, 9, 17):
            for batch in (1, 3):
                for plane in (20, 80):
                    cpu = [torch.randn(batch, 1 + i % 3, plane, plane, 16, dtype=dtype)
                           for i in range(count)]
                    special = torch.tensor([0., -0., float('nan'), float('inf'), -float('inf')], dtype=dtype)
                    for value in cpu:
                        value.flatten()[:special.numel()] = special
                    device = [value.to('npu') for value in cpu]
                    actual = torch.cat(device, 1)
                    bits_equal(actual.cpu(), torch.cat(cpu, 1))
                    cases += 1
        for rows in (8, 81, 129):
            cpu = [torch.randn(1, rows, length, dtype=dtype) for length in (6400, 1600, 400)]
            device = [value.to('npu') for value in cpu]
            bits_equal(torch.cat(device, 2).cpu(), torch.cat(cpu, 2))
            cases += 1
        for batch in (1, 3):
            cpu = torch.randn(batch, 12, 40, 40, 16, dtype=dtype)
            device = cpu.to('npu')
            for sizes in ((3, 3, 6), (1, 2, 3, 6)):
                bits_equal(torch.cat(device.split(sizes, 1), 1).cpu(), cpu)
                cases += 1
        for count in (2, 6, 9, 17):
            x = torch.randn(1, count * 2, 20, 20, 16, dtype=dtype)
            def concatenate(value):
                return torch.cat(value.split(2, 1), 1)
            graph = CapturedForward(concatenate, x.to('npu'))
            for scale in (1, -0.75, 0.5):
                changed = x * scale
                bits_equal(graph(changed.to('npu')).cpu(), changed)
                cases += 1
            graph.close()
print('CAT_PLAN_BITWISE_PASS', cases, flush=True)
