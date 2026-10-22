import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import thin_backend

torch.set_num_threads(1)
torch.manual_seed(42)
cases = 0

def compare(cpu, device, dim):
    global cases
    actual = torch.cat(device, dim)
    expected = torch.cat(cpu, dim)
    assert actual.device.type == "npu"
    assert actual.is_contiguous()
    bits = torch.int16 if actual.dtype == torch.float16 else torch.int32
    assert torch.equal(actual.cpu().view(bits), expected.view(bits)), (dim, [list(t.shape) for t in cpu])
    cases += 1

with torch.inference_mode():
    for dtype in (torch.float16, torch.float32):
        for dim in range(4):
            for outer in (1, 3):
                shape = [outer, 3, 32, 32]
                inputs = []
                for size in (1, 2, 5):
                    current = shape.copy()
                    current[dim] = size if dim != 3 else size * 32
                    inputs.append(torch.randn(current, dtype=dtype))
                compare(inputs, [t.to("npu") for t in inputs], dim)
        for length in (16, 8192, 16384, 16384 + 16, 131072 + 16):
            inputs = [torch.randn(3, length, dtype=dtype) for _ in range(3)]
            compare(inputs, [t.to("npu") for t in inputs], -1)
        for offset in (1, 16):
            source = torch.randn(3 * 64 * 64 + offset, dtype=dtype)
            device = source.to("npu")
            cpu_view = source[offset:].view(3, 64, 64)
            npu_view = device[offset:].view(3, 64, 64)
            compare([cpu_view] * 2, [npu_view] * 2, 1)
            compare([cpu_view.transpose(0, 2)] * 2, [npu_view.transpose(0, 2)] * 2, 1)
        source = torch.randn(3, 64, 32, dtype=dtype)
        device = source.to("npu")
        compare(source.split(32, 1), device.split(32, 1), 1)
        compare([source[:1].expand(3, -1, -1)] * 2,
                [device[:1].expand(3, -1, -1)] * 2, 1)
        empty = torch.empty(0, dtype=dtype)
        compare([empty, source, empty], [empty.to("npu"), device, empty.to("npu")], 1)
        compare([empty, empty], [empty.to("npu"), empty.to("npu")], 0)
        shaped_empty = torch.empty(3, 0, 32, dtype=dtype)
        compare([shaped_empty, source], [shaped_empty.to("npu"), device], 1)
        compare([source], [device], 1)
        values = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -1.0, 0.5], dtype=dtype).repeat(16)
        compare([values, values], [values.to("npu"), values.to("npu")], 0)
print("CAT_BITWISE_PASS", cases, flush=True)
