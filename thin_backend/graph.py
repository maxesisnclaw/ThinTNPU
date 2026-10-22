from threading import get_ident

import torch


class CapturedForward:
    """Fixed-shape inference with graph-owned workspace and borrowed output buffers.

    The callable, its parameters and captured tensors must not be mutated or moved.
    Calls reuse the returned output tensors; clone outputs to retain prior results.
    Create, replay and close on the same thread. No concurrent replay is supported.
    """

    def __init__(self, forward, example, warmup=3):
        if example.device.type != "npu" or example.requires_grad:
            raise ValueError("CapturedForward requires an inference-only NPU input")
        self._input_layout = example.layout
        self._input_stride = example.stride()
        self._thread = get_ident()
        self.forward = forward
        self.input = torch.empty_like(example)
        self.input.copy_(example)
        self.outputs = None
        self.graph = torch.classes.thin_backend.RuntimeGraph()
        with torch.inference_mode():
            for _ in range(warmup):
                forward(self.input)
            self.graph.begin()
            try:
                self.outputs = forward(self.input)
            finally:
                self.graph.end()
        self.graph.replay()

    def replay(self):
        """Enqueue one replay using the existing static input; return borrowed outputs."""
        self.graph.replay()
        return self.outputs

    def __call__(self, value):
        if get_ident() != self._thread:
            raise RuntimeError("RuntimeGraph must stay on its creating thread/stream")
        if (value.shape != self.input.shape or value.dtype != self.input.dtype
                or value.device != self.input.device or value.requires_grad
                or value.layout != self._input_layout or value.stride() != self._input_stride):
            raise ValueError("Graph input shape, dtype, device and layout/strides must match capture")
        self.input.copy_(value)
        return self.replay()

    def close(self):
        self.graph.close()
