from types import MethodType

import torch
from torch import nn


def prepared_forward(module, x):
    return torch.ops.thin_backend.conv2d_prepared(
        x, module.weight.detach(), module._thin_packed_weight,
        None if module.bias is None else module.bias.detach(),
        module.stride[0], module.padding[0], False, False,
    )


def fused_forward(module, x):
    conv = module.conv
    return torch.ops.thin_backend.conv2d_prepared(
        x, conv.weight.detach(), conv._thin_packed_weight,
        None if conv.bias is None else conv.bias.detach(),
        conv.stride[0], conv.padding[0], False, True,
    )


def prepare_model(model, fuse_silu=False):
    prepared = []
    fused = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.weight.device.type != "npu" or module.weight.dtype != torch.float16:
                raise ValueError("prepare_model requires an FP16 NPU model")
            if module.groups != 1 or module.dilation != (1, 1):
                raise ValueError("prepare_model supports only dense, undilated convolutions")
            kernel = module.kernel_size[0]
            if (kernel not in (1, 3) or module.kernel_size != (kernel, kernel)
                    or module.stride not in ((1, 1), (2, 2))
                    or module.padding != (kernel // 2, kernel // 2)
                    or module.padding_mode != "zeros"):
                raise ValueError("prepare_model requires square k1/p0 or k3/p1, stride 1/2, zero padding")
            module.register_buffer(
                "_thin_packed_weight",
                torch.ops.thin_backend.conv2d_prepare(module.weight.detach()),
                persistent=False,
            )
            module.forward = MethodType(prepared_forward, module)
            prepared.append(name)
    if fuse_silu:
        from ultralytics.nn.modules.conv import Conv
        for name, module in model.named_modules():
            if type(module) is Conv and isinstance(module.act, nn.SiLU):
                if not hasattr(module.conv, "_thin_packed_weight"):
                    raise ValueError("SiLU fusion requires prepared convolution weights")
                module.forward = MethodType(fused_forward, module)
                fused.append(name)
    return {"prepared_layers": prepared, "fused_silu_layers": fused}
