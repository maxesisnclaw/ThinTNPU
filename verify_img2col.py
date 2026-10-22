import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import thin_backend


torch.set_num_threads(1)
rows = []
cases = [
    ("stem32", 3, 640, 640, 32, 3, 2, True),
    ("stem_tail", 3, 13, 640, 17, 3, 2, True),
    ("stem_no_bias", 3, 13, 640, 7, 3, 2, False),
    ("point64", 64, 80, 80, 64, 1, 1, True),
    ("point256", 128, 40, 40, 256, 1, 1, True),
    ("conv64", 64, 80, 80, 64, 3, 1, True),
    ("conv128", 128, 40, 40, 128, 3, 1, True),
    ("down128", 64, 80, 80, 128, 3, 2, True),
    ("down256", 128, 40, 40, 256, 3, 2, True),
    ("head512", 512, 20, 20, 512, 3, 1, True),
    ("cout17", 3, 64, 64, 17, 3, 2, True),
    ("cout31", 17, 20, 20, 31, 3, 1, True),
    ("point65", 19, 20, 20, 65, 1, 1, True),
    ("point_down", 64, 40, 40, 80, 1, 2, True),
    ("odd_down", 5, 13, 17, 23, 3, 2, True),
    ("odd_same", 3, 11, 13, 7, 3, 1, True),
    ("odd_point", 5, 11, 13, 9, 1, 1, False),
    ("dfl", 16, 4, 8400, 1, 1, 1, False),
    ("tiny", 3, 1, 1, 5, 3, 2, True),
]
with torch.inference_mode():
    for name, ci, h, width, co, k, stride, has_bias in cases:
        torch.manual_seed(42)
        x = torch.randn(1, ci, h, width).half()
        bound = 1 / math.sqrt(ci * k * k)
        weight = torch.empty(co, ci, k, k).uniform_(-bound, bound).half()
        bias = torch.empty(co).uniform_(-bound, bound).half() if has_bias else None
        expected_conv = F.conv2d(x.float(), weight.float(), None if bias is None else bias.float(), stride, k // 2)
        xd, wd = x.to("npu"), weight.to("npu")
        bd = None if bias is None else bias.to("npu")
        packed = torch.ops.thin_backend.conv2d_prepare(wd)
        for fused in (False, True):
            expected = F.silu(expected_conv) if fused else expected_conv
            for fp32 in (True, False):
                plain = torch.ops.thin_backend.conv2d_fused(xd, wd, bd, stride, k // 2, fp32, fused)
                prepared = torch.ops.thin_backend.conv2d_prepared(xd, wd, packed, bd, stride, k // 2, fp32, fused)
                actual = prepared.cpu()
                torch.testing.assert_close(actual, plain.cpu(), atol=0, rtol=0)
                torch.testing.assert_close(actual.float(), expected, atol=1e-4 if fp32 else 2e-3, rtol=0)
                error = (actual.float() - expected).abs().max().item()
                row = {"name": name, "fused": fused, "fp32_output": fp32,
                       "max_abs_cpu_fp32": error, "plain_prepared_max_abs": 0.0}
                rows.append(row)
                print(json.dumps(row), flush=True)
        parent = torch.cat((xd, xd), 1)
        offset = parent[:, ci:]
        got = torch.ops.thin_backend.conv2d_prepared(offset, wd, packed, bd, stride, k // 2)
        baseline = torch.ops.thin_backend.conv2d(xd, wd, bd, stride, k // 2)
        torch.testing.assert_close(got.cpu(), baseline.cpu(), atol=0, rtol=0)
        rows.append({"name": name, "offset_input": offset.storage_offset(), "max_abs": 0.0})
assert not any(name == "torch_npu" or name.startswith("torch_npu.") for name in sys.modules)
maps = Path("/proc/self/maps").read_text()
assert not any(token in maps.lower() for token in ("torch_npu", "aclnn", "libopcompiler", "libtbe"))
assert not any(path.read_text().strip() for path in Path("/proc/self/task").glob("*/children"))
evidence = Path("evidence-img2col")
evidence.mkdir(exist_ok=True)
(evidence / "results.json").write_text(json.dumps(rows, indent=2))
print("IMG2COL_ALL_PASS", len(rows), flush=True)
