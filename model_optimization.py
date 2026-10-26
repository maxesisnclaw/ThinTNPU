from types import MethodType

import torch
from torch import nn


def is_conv_wrapper(module):
    return (type(module).__name__ in ("Conv", "DWConv") and isinstance(getattr(module, "conv", None), nn.Conv2d)
            and isinstance(getattr(module, "act", None), (nn.SiLU, nn.Identity)))


def depthwise_nchw(module, x, fused):
    batch, channels, height, width = x.shape
    blocked = x.contiguous().view(batch, channels // 16, 16, height, width).permute(0, 1, 3, 4, 2).contiguous()
    result = torch.ops.thin_native_flow.depthwise(
        blocked, module._thin_packed_weight, module.bias, module.kernel_size[0], module.stride[0], fused)
    return result.permute(0, 1, 4, 2, 3).contiguous().view(batch, channels, result.size(2), result.size(3))


def prepared_forward(module, x):
    if module.groups > 1:
        return depthwise_nchw(module, x, False)
    if module.kernel_size == (6, 6):
        return torch.ops.thin_native_flow.conv(
            torch.ops.thin_native_flow.pack_input(x), module._thin_packed_weight,
            module._thin_native_bias, module.out_channels, 6, module.stride[0], False, True)
    return torch.ops.thin_backend.conv2d_prepared(
        x, module.weight.detach(), module._thin_packed_weight,
        None if module.bias is None else module.bias.detach(),
        module.stride[0], module.padding[0], False, False,
    )


def fused_forward(module, x):
    conv = module.conv
    fused = isinstance(module.act, nn.SiLU)
    if conv.groups > 1:
        return depthwise_nchw(conv, x, fused)
    if conv.kernel_size == (6, 6):
        return torch.ops.thin_native_flow.conv(
            torch.ops.thin_native_flow.pack_input(x), conv._thin_packed_weight,
            conv._thin_native_bias, conv.out_channels, 6, conv.stride[0], fused, True)
    return torch.ops.thin_backend.conv2d_prepared(
        x, conv.weight.detach(), conv._thin_packed_weight,
        None if conv.bias is None else conv.bias.detach(),
        conv.stride[0], conv.padding[0], False, fused,
    )


def prepare_model(model, fuse_silu=False):
    prepared = []
    fused = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.weight.device.type != "npu" or module.weight.dtype != torch.float16:
                raise ValueError("prepare_model requires an FP16 NPU model")
            depthwise = module.groups == module.in_channels == module.out_channels and module.groups > 1
            if (module.groups != 1 and not depthwise) or module.dilation != (1, 1):
                raise ValueError("prepare_model supports dense or multiplier-one depthwise undilated convolutions")
            kernel = module.kernel_size[0]
            if (kernel not in (1, 3, 6) or module.kernel_size != (kernel, kernel)
                    or module.stride not in ((1, 1), (2, 2))
                    or module.padding != ((kernel - 1) // 2,) * 2
                    or module.padding_mode != "zeros"):
                raise ValueError("prepare_model requires k1/p0, k3/p1 or k6/p2, stride 1/2, zero padding")
            if depthwise:
                if module.in_channels % 16 or kernel != 3 or module.stride != (1, 1):
                    raise ValueError("depthwise requires aligned channels and k3/s1/p1")
                packed = module.weight.detach().cpu().view(module.in_channels // 16, 16, kernel, kernel).permute(0, 2, 3, 1).contiguous().to(module.weight.device)
            else:
                packed = torch.ops.thin_backend.conv2d_prepare(module.weight.detach())
            if kernel == 6:
                bias = module.bias.detach().cpu() if module.bias is not None else None
                if bias is not None:
                    bias = torch.nn.functional.pad(bias, (0, (-module.out_channels) % 32)).to(module.weight.device)
                module.register_buffer("_thin_native_bias", bias, persistent=False)
            module.register_buffer(
                "_thin_packed_weight",
                packed,
                persistent=False,
            )
            module.forward = MethodType(prepared_forward, module)
            prepared.append(name)
    if fuse_silu:
        for name, module in model.named_modules():
            if is_conv_wrapper(module):
                if not hasattr(module.conv, "_thin_packed_weight"):
                    raise ValueError("SiLU fusion requires prepared convolution weights")
                module.forward = MethodType(fused_forward, module)
                if isinstance(module.act, nn.SiLU):
                    fused.append(name)
    return {"prepared_layers": prepared, "fused_silu_layers": fused}
