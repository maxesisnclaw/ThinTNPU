import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
from types import MethodType

import torch
from torch import nn
import thin_backend  # registers the PrivateUse1 backend: empty/copy_/cat/split/add on 5D

ops = torch.ops.thin_native_flow


def load_native_flow(conv_lib=None, ops_lib=None):
    """Register thin_native_flow.{conv,stem} and {maxpool,upsample}.

    Defaults load libconv_flow.so and libflow_ops.so from the thin_backend directory next
    to this file; override with explicit paths or the THIN_CONV_FLOW_LIB / THIN_FLOW_OPS_LIB
    environment variables.
    """
    here = Path(__file__).parent / 'thin_backend'
    torch.ops.load_library(str(conv_lib or os.environ.get('THIN_CONV_FLOW_LIB') or here / 'libconv_flow.so'))
    torch.ops.load_library(str(ops_lib or os.environ.get('THIN_FLOW_OPS_LIB') or here / 'libflow_ops.so'))


def pad_bias(bias, co):
    """Return a bias padded to length ceil(co/32)*32, or None when co already fits.

    thin_native_flow.conv reads a bias of that length; every intermediate co here is a
    multiple of 32 and only the class terminals (co=17 -> 32) need padding. The padded copy
    is built on CPU and moved to the device so no device-side zero_/fill_ is required.
    """
    target = (co + 31) // 32 * 32
    if target == bias.numel():
        return None
    padded = torch.zeros(target, dtype=bias.dtype)
    padded[:co] = bias.detach().to('cpu')
    return padded.to(bias.device)


def stem_forward(module, x):
    conv = module.conv
    return ops.stem(x, conv.weight.detach(), conv.bias)


def conv_forward(module, x):
    conv = module.conv
    bias = getattr(conv, '_thin_native_bias', conv.bias)
    return ops.conv(x, conv._thin_packed_weight, bias,
                    conv.out_channels, conv.kernel_size[0], conv.stride[0], True, False)


def terminal_forward(module, x):
    bias = getattr(module, '_thin_native_bias', module.bias)
    return ops.conv(x, module._thin_packed_weight, bias,
                    module.out_channels, module.kernel_size[0], module.stride[0], False, True)


def maxpool_forward(module, x):
    return ops.maxpool(x, module.kernel_size)


def upsample_forward(module, x):
    return ops.upsample(x)


class WholeBlocked:
    """Whole-YOLOv8s blocked NC1HWC0 execution flow with a single frozen-state guard.

    Replaces only leaf operators (stem, every SiLU convolution, the Detect terminal 1x1
    convolutions, SPPF maxpool, neck upsample). Container forwards (C2f, Bottleneck, SPPF,
    Detect, Concat, Sequential, DetectionModel) keep their original implementation: chunk
    (dim=1), cat (dim=1) and the residual add run directly on physical 5D tensors, and the
    Detect decode (DFL, box decode, sigmoid) runs on the NCHW the terminal convolutions emit.

    The model must be fused (bn folded into the convolutions) and prepacked with
    prepare_model(model), which registers the _thin_packed_weight buffers reused here.
    """

    def __init__(self, model):
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import DFL
        from ultralytics.nn.modules.head import Detect

        root = model.model if hasattr(model, 'model') else model
        if type(root[0]) is not Conv or type(root[-1]) is not Detect:
            raise ValueError('Expected a fused YOLOv8 DetectionModel sequence ending in Detect')
        detect = root[-1]
        if detect.nl != 3 or detect.reg_max != 16 or detect.export or detect.end2end:
            raise ValueError('Expected a three-level DFL(16) non-export Detect')

        stem = root[0]
        if (stem.conv.in_channels != 3 or stem.conv.kernel_size != (3, 3) or stem.conv.stride != (2, 2)
                or type(stem.act) is not nn.SiLU):
            raise ValueError('Expected a 3-channel k3 s2 SiLU stem at model.0')

        wrapped = set(m.conv for m in model.modules() if type(m) is Conv)
        terminals = set()
        for branch in (*detect.cv2, *detect.cv3):
            tail = branch[-1]
            if type(tail) is not nn.Conv2d or tail.kernel_size != (1, 1):
                raise ValueError('Expected a 1x1 nn.Conv2d tail on each Detect branch')
            terminals.add(tail)
        dfl_conv = detect.dfl.conv if type(detect.dfl) is DFL else None

        convs, maxpools, upsamples = [], [], []
        for module in model.modules():
            if module is stem:
                continue
            if type(module) is Conv:
                conv = module.conv
                if (conv.groups != 1 or conv.dilation != (1, 1) or conv.padding_mode != 'zeros'
                        or conv.kernel_size[0] not in (1, 3) or conv.kernel_size[0] != conv.kernel_size[1]
                        or conv.padding != (conv.kernel_size[0] // 2,) * 2
                        or conv.stride not in ((1, 1), (2, 2)) or conv.bias is None
                        or type(module.act) is not nn.SiLU):
                    raise ValueError('Expected dense k1/p0 or k3/p1 stride-1/2 SiLU convolutions with bias')
                convs.append(module)
            elif isinstance(module, nn.Conv2d):
                if module in wrapped or module is dfl_conv:
                    continue
                if module not in terminals:
                    raise ValueError('Unexpected bare nn.Conv2d outside the Detect terminals/DFL')
            elif isinstance(module, nn.MaxPool2d):
                if module.stride not in (1, (1, 1)) or module.kernel_size != 5 or module.padding != 2:
                    raise ValueError('Expected SPPF k5 s1 p2 maxpool')
                maxpools.append(module)
            elif isinstance(module, nn.Upsample):
                if float(module.scale_factor) != 2.0 or module.mode != 'nearest':
                    raise ValueError('Expected 2x nearest upsample')
                upsamples.append(module)

        for module in model.modules():
            for tensor in (*module._parameters.values(), *module._buffers.values()):
                if tensor is None:
                    continue
                if tensor.requires_grad or tensor.dtype != torch.float16 or tensor.device.type != 'npu':
                    raise ValueError('Prepare an FP16 NPU inference model under no_grad before constructing the pilot')

        for leaf in convs:
            if not hasattr(leaf.conv, '_thin_packed_weight'):
                raise ValueError('Run prepare_model(model) to prepack convolution weights before WholeBlocked')
            padded = pad_bias(leaf.conv.bias, leaf.conv.out_channels)
            if padded is not None:
                leaf.conv.register_buffer('_thin_native_bias', padded, persistent=False)
        for tail in terminals:
            if not hasattr(tail, '_thin_packed_weight'):
                raise ValueError('Run prepare_model(model) to prepack convolution weights before WholeBlocked')
            padded = pad_bias(tail.bias, tail.out_channels)
            if padded is not None:
                tail.register_buffer('_thin_native_bias', padded, persistent=False)

        self.model = model
        self.detect = detect
        self.replacements = {
            stem: (stem.forward, MethodType(stem_forward, stem)),
            **{m: (m.forward, MethodType(conv_forward, m)) for m in convs},
            **{m: (m.forward, MethodType(terminal_forward, m)) for m in terminals},
            **{m: (m.forward, MethodType(maxpool_forward, m)) for m in maxpools},
            **{m: (m.forward, MethodType(upsample_forward, m)) for m in upsamples},
        }
        from region_contract import RegionContract
        self.contract = RegionContract((model,), ignored_forward=tuple(self.replacements))

    def validate(self):
        for module, (original, entry) in self.replacements.items():
            if module.forward not in (original, entry):
                raise ValueError('A replaced operator forward was changed after pilot construction')
        self.contract.validate()

    def enable(self):
        self.validate()
        for module, (_, entry) in self.replacements.items():
            module.forward = entry

    def disable(self):
        for module, (original, _) in self.replacements.items():
            module.forward = original

    def run(self, x):
        self.validate()
        if (x.dim() != 4 or x.size(0) != 1 or x.size(1) != 3 or
                x.size(2) <= 0 or x.size(3) <= 0 or x.size(2) % 32 or x.size(3) % 32 or x.dtype != torch.float16 or x.device.type != 'npu'
                or not x.is_contiguous() or x.storage_offset() != 0 or x.requires_grad):
            raise ValueError('Expected allocation-base contiguous FP16 NPU 1x3xHxW input with positive /32 H/W')
        return self.model(x)
