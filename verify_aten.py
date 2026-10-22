import copy
import gc
import json
import os
from pathlib import Path
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
import torch
from torch import nn
from torch.nn import functional as F
import thin_backend

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence-aten"
EVIDENCE.mkdir(exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(42)
results = {"pid": os.getpid(), "torch": torch.__version__, "autoload": os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"],
           "library": str(ROOT / "thin_backend/libthin_backend.so"), "checks": [], "memory": [], "probes": {}}


def children(pid):
    path = Path(f"/proc/{pid}/task/{pid}/children")
    direct = path.read_text().split() if path.exists() else []
    return direct + [descendant for child in direct for descendant in children(child)]


def memory(stage):
    modules = sorted(sys.modules)
    assert not any(name == "torch_npu" or name.startswith("torch_npu.") for name in modules)
    maps = Path("/proc/self/maps").read_text()
    for forbidden in ("torch_npu", "aclnn", "libopcompiler", "libtbe"):
        assert forbidden not in maps.lower(), forbidden
    descendants = children(os.getpid())
    assert descendants == [], descendants
    smaps = Path("/proc/self/smaps_rollup").read_text()
    pss = int(next(line.split()[1] for line in smaps.splitlines() if line.startswith("Pss:"))) / 1024
    sample = {"stage": stage, "pss_mib": pss, "descendants": descendants, "torch_npu": False}
    results["memory"].append(sample)
    (EVIDENCE / f"{stage}-smaps_rollup.txt").write_text(smaps)
    (EVIDENCE / "maps.txt").write_text(maps)
    (EVIDENCE / "modules.txt").write_text("\n".join(modules))
    print("MEMORY", json.dumps(sample), flush=True)


def compare(name, actual, expected, atol=1e-3):
    assert str(actual.device) == "npu:0", (name, actual.device)
    got = actual.cpu()
    assert got.shape == expected.shape and got.dtype == expected.dtype, name
    error = (got - expected).abs().max().item() if got.numel() else 0
    torch.testing.assert_close(got, expected, atol=atol, rtol=0)
    row = {"name": name, "shape": list(actual.shape), "device": str(actual.device),
           "dtype": str(actual.dtype), "max_abs": error, "atol": atol}
    results["checks"].append(row)
    print("PASS", json.dumps(row), flush=True)


class Model(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 7, 3, padding=1, bias=True), nn.SiLU(inplace=inplace),
                                    nn.Conv2d(7, 17, 3, stride=2, padding=1, bias=True),
                                    nn.MaxPool2d(5, 1, 2), nn.Upsample(scale_factor=2, mode="nearest"), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


def e2e(inplace):
    model = Model(inplace).eval()
    reference = copy.deepcopy(model)
    x = torch.randn(1, 3, 15, 17)
    expected = []
    actual = []
    for layer in reference.layers:
        layer.register_forward_hook(lambda module, inputs, out: expected.append(out.clone()))
    for layer in model.layers:
        layer.register_forward_hook(lambda module, inputs, out: actual.append(out.clone()))
    model.to("npu")
    with torch.inference_mode() if inplace else torch.no_grad():
        ref = reference(x)
        assert all(str(p.device) == "npu:0" for p in model.parameters())
        output = model(x.to("npu"))
        for index, (got, want) in enumerate(zip(actual, expected)):
            compare(f"e2e-inplace{inplace}-layer{index}-{type(model.layers[index]).__name__}", got, want)
        compare(f"e2e-inplace{inplace}-output", output, ref)
    return model


memory("loaded")
e2e(False)
e2e(True)
memory("e2e")

with torch.inference_mode():
    host = torch.randn(1, 14, 15, 17)
    parent = host.to("npu")
    section = parent.split(7, 1)[1]
    assert section.storage_offset() > 0 and section.contiguous().storage_offset() > 0
    results["offset_case"] = {"offset": section.storage_offset(), "contiguous": section.is_contiguous(),
                              "contiguous_offset": section.contiguous().storage_offset()}
    w, b = torch.randn(9, 7, 1, 1) / 4, torch.randn(9) / 4
    compare("split-offset-conv", F.conv2d(section, w.to("npu"), b.to("npu")), F.conv2d(host[:, 7:], w, b))
    compare("strided-conv", F.conv2d(section.transpose(2, 3), w.to("npu"), b.to("npu")),
            F.conv2d(host[:, 7:].transpose(2, 3), w, b))
    compare("offset-sigmoid", section.sigmoid(), host[:, 7:].sigmoid())
    values, indices = torch.ops.aten.max_pool2d_with_indices.default(section, [5, 5], [1, 1], [2, 2], [1, 1], False)
    compare("pool-values", values, F.max_pool2d(host[:, 7:], 5, 1, 2))
    assert str(indices.device) == "npu:0" and indices.dtype == torch.int64 and torch.count_nonzero(indices.cpu()).item() == 0
    results["indices"] = {"device": str(indices.device), "dtype": str(indices.dtype), "all_zero": True, "semantics": "placeholder, not argmax"}
    compare("nearest-default", torch.ops.aten.upsample_nearest2d.default(section, [30, 34], 2.0, 2.0),
            F.interpolate(host[:, 7:], scale_factor=2, mode="nearest"), atol=0)
    compare("nearest-vec-size", torch.ops.aten.upsample_nearest2d.vec(section, [30, 34], None),
            F.interpolate(host[:, 7:], scale_factor=2, mode="nearest"), atol=0)
    compare("nearest-vec-scale", torch.ops.aten.upsample_nearest2d.vec(section, None, [2.0, 2.0]),
            F.interpolate(host[:, 7:], scale_factor=2, mode="nearest"), atol=0)

    a, b = torch.randn(1, 7, 5, 7), torch.rand(1, 7, 5, 7) + 1
    an, bn = a.to("npu"), b.to("npu")
    for name, op in (("add", torch.add), ("sub", torch.sub), ("mul", torch.mul), ("div", torch.div)):
        compare("inherited-" + name, op(an, bn), op(a, b), 1e-5)
    compare("inherited-softmax", torch.softmax(an, 1), torch.softmax(a, 1), 1e-5)
    compare("inherited-split-cat", torch.cat(an.split([3, 4], 1), 1), a, 0)
    compare("inherited-view", an.view(7, 35), a.view(7, 35), 0)
    compare("inherited-metadata", an.transpose(2, 3).unsqueeze(0).expand(2, 1, 7, 7, 5),
            a.transpose(2, 3).unsqueeze(0).expand(2, 1, 7, 7, 5), 0)
    compare("inherited-silu_", F.silu(an, inplace=True), F.silu(a), 1e-5)

    negatives = {
        "groups": lambda: F.conv2d(torch.randn(1, 4, 7, 7).to("npu"), torch.randn(4, 2, 3, 3).to("npu"), groups=2),
        "dilation": lambda: F.conv2d(torch.randn(1, 3, 7, 7).to("npu"), torch.randn(5, 3, 3, 3).to("npu"), dilation=2),
        "transposed": lambda: F.conv_transpose2d(torch.randn(1, 3, 7, 7).to("npu"), torch.randn(3, 5, 3, 3).to("npu")),
        "pool-k3": lambda: F.max_pool2d(an, 3, 1, 1),
        "nearest-3x": lambda: F.interpolate(an, scale_factor=3, mode="nearest"),
    }
    results["unsupported"] = {}
    for name, op in negatives.items():
        try:
            op()
        except RuntimeError as error:
            text = str(error)
            assert "thin aten" in text, text
            results["unsupported"][name] = text
            print("REJECT", name, text, flush=True)
        else:
            raise AssertionError("unsupported configuration accepted: " + name)

    probes = {
        "empty": lambda: torch.empty((2, 3), device="npu"),
        "arange": lambda: torch.arange(8, device="npu"),
        "expand": lambda: an[:, :1].expand(1, 4, 5, 7),
        "stack": lambda: torch.stack([an, an]),
        "transpose": lambda: an.transpose(2, 3),
        "unsqueeze": lambda: an.unsqueeze(0),
        "full": lambda: torch.full((2, 3), 2.0, device="npu"),
        "_local_scalar_dense": lambda: torch.tensor(1.0).to("npu").item(),
        "view": lambda: an.view(7, 35),
        "broadcast_add": lambda: an + torch.ones(1, 7, 1, 1).to("npu"),
        "scalar_mul": lambda: an * 2.0,
        "batch_norm": lambda: F.batch_norm(an, torch.zeros(7).to("npu"), torch.ones(7).to("npu"), training=False),
    }
    for name, op in probes.items():
        try:
            out = op()
            row = {"supported": True, "device": str(out.device) if isinstance(out, torch.Tensor) else "host scalar"}
        except Exception as error:
            row = {"supported": False, "error": str(error)}
        results["probes"][name] = row
        summary = {key: value.splitlines()[0] if key == "error" else value for key, value in row.items()}
        print("PROBE", name, json.dumps(summary), flush=True)

operators = ["aten::convolution", "aten::conv2d", "aten::max_pool2d", "aten::max_pool2d_with_indices",
             "aten::upsample_nearest2d", "aten::upsample_nearest2d.vec", "aten::sigmoid", "aten::silu", "aten::silu_",
             "aten::_softmax", "aten::cat", "aten::split.Tensor", "aten::split_with_sizes", "aten::add.Tensor",
             "aten::sub.Tensor", "aten::mul.Tensor", "aten::div.Tensor", "aten::view", "aten::as_strided",
             "aten::empty.memory_format", "aten::empty_strided", "aten::copy_", "aten::arange", "aten::stack",
             "aten::full", "aten::_local_scalar_dense", "aten::transpose.int", "aten::unsqueeze", "aten::expand"]
(EVIDENCE / "dispatch.txt").write_text("\n\n".join(op + "\n" + torch._C._dispatch_dump_table(op) for op in operators))
memory("complete")
gc.collect()
import ctypes
ctypes.CDLL(None).malloc_trim(0)
memory("trimmed")
(EVIDENCE / "results.json").write_text(json.dumps(results, indent=2))
print("ATEN_ALL_PASS", len(results["checks"]), "PID", os.getpid(), flush=True)
