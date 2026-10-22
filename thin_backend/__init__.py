import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

torch.ops.load_library(str(Path(__file__).with_name("libthin_backend.so")))
torch.utils.rename_privateuse1_backend("npu")
torch._register_device_module("npu", SimpleNamespace(
    is_available=lambda: True,
    device_count=lambda: 1,
    current_device=lambda: 0,
    _is_in_bad_fork=lambda: False,
    manual_seed_all=lambda seed: None,
    _lazy_init=lambda: None,
))
torch.utils.generate_methods_for_privateuse1_backend()
