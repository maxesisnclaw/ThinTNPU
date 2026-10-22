import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import gc
import sys
import threading
from pathlib import Path

import torch
import thin_backend
from thin_backend.graph import CapturedForward


def expression(x):
    return torch.sigmoid((x + x) * 0.5)


with torch.inference_mode():
    x = torch.linspace(-2, 2, 2048).to("npu")
    graph = CapturedForward(expression, x)
    expected = expression(x).cpu()
    for _ in range(5):
        gc.collect()
        assert torch.equal(graph.replay().cpu(), expected)
    other = torch.linspace(1, 3, 2048).to("npu")
    assert torch.equal(graph(other).cpu(), expression(other).cpu())
    try:
        graph(torch.empty(32, device="npu"))
    except ValueError:
        pass
    else:
        raise AssertionError("Changed shape accepted")
    errors = []
    def wrong_thread():
        for call in (graph.replay, lambda: graph(x)):
            try:
                call()
            except RuntimeError as error:
                errors.append(str(error))
    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 2 and all("creating thread/stream" in error for error in errors)
    second = CapturedForward(expression, x)
    retained = graph.outputs
    graph.close()
    graph.close()
    assert torch.equal(retained.cpu(), expression(other).cpu())
    assert torch.equal(second.replay().cpu(), expected)
    second.close()
    del graph, second
    gc.collect()
    assert torch.equal(expression(x).cpu(), expected)
    try:
        CapturedForward(lambda value: value.cpu(), x)
    except RuntimeError as error:
        assert "Device-to-host" in str(error)
    else:
        raise AssertionError("Device-to-host capture accepted")
    graph = CapturedForward(expression, x)
    assert torch.equal(graph.replay().cpu(), expected)
    retained = graph.outputs
    del graph
    gc.collect()
    assert torch.equal(retained.cpu(), expected)
    maps = Path("/proc/self/maps").read_text()
    assert not any(name.startswith("torch_npu") for name in sys.modules)
    assert not any(token in maps.lower() for token in ("torch_npu", "aclnn"))
print("GRAPH_ALL_PASS", flush=True)
