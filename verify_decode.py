import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import thin_backend

with torch.inference_mode():
    cases = {
        "arange": lambda d: torch.arange(80, device=d, dtype=torch.float32),
        "arange_step": lambda d: torch.arange(-2, 10, 0.5, device=d, dtype=torch.float32),
        "full": lambda d: torch.full((8400, 1), 8.0, device=d),
        "stack": lambda d: torch.stack((torch.arange(80, device=d, dtype=torch.float32), torch.arange(80, device=d, dtype=torch.float32)), dim=-1),
        "broadcast_add": lambda d: torch.arange(80, device=d, dtype=torch.float32).view(1, 80) + torch.full((2, 1), 0.5, device=d),
        "scalar_mul": lambda d: torch.arange(80, device=d, dtype=torch.float32) * 8,
        "scalar_div": lambda d: torch.arange(80, device=d, dtype=torch.float32) / 2,
    }
    for name, call in cases.items():
        actual = call("npu").cpu()
        expected = call("cpu")
        torch.testing.assert_close(actual, expected)
        print("PASS", name, "max_abs", (actual - expected).abs().max().item())
    x = torch.arange(12, device="npu", dtype=torch.float32).view(3, 4)
    torch.testing.assert_close(torch.stack(torch.unbind(x)).cpu(), torch.arange(12, dtype=torch.float32).view(3, 4))
    print("PASS unbind")
    assert torch.full((), 8.0, device="npu").item() == 8.0
    print("PASS _local_scalar_dense")
