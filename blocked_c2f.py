import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
from pathlib import Path
from types import MethodType
import weakref
import torch
import thin_backend
from model_optimization import fused_forward, prepared_forward

torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libthin_blocked.so'))


def blocked_conv(module, x):
    conv = module.conv
    return torch.ops.thin_blocked.conv(x, conv._thin_packed_weight, conv.bias,
                                      conv.out_channels, conv.kernel_size[0], conv.stride[0], False, True)


def blocked_forward(module, x):
    y = list(blocked_conv(module.cv1, torch.ops.thin_blocked.pack(x)).chunk(2, 1))
    for bottleneck in module.m:
        value = blocked_conv(bottleneck.cv2, blocked_conv(bottleneck.cv1, y[-1]))
        y.append(y[-1] + value)
    return torch.ops.thin_blocked.unpack(blocked_conv(module.cv2, torch.cat(y, 1)))


class BlockedC2f:
    def __init__(self, module):
        from ultralytics.nn.modules.block import C2f, Bottleneck
        from ultralytics.nn.modules.conv import Conv
        if type(module) is not C2f or module.c != 128 or len(module.m) != 2:
            raise ValueError('Expected model.6 C2f with two 128-channel bottlenecks')
        if getattr(module.forward, '__func__', None) is not C2f.forward:
            raise ValueError('Expected the original C2f.forward implementation')
        if any(type(m) is not Bottleneck or getattr(m.forward, '__func__', None) is not Bottleneck.forward
               or not m.add for m in module.m):
            raise ValueError('Expected original residual Bottleneck.forward implementations')
        self.module = module
        self.original = module.forward
        owner = weakref.ref(self)
        def entry(mod, x):
            pilot = owner()
            if pilot is None:
                raise RuntimeError('BlockedC2f owner must remain alive')
            return pilot.region(x)
        self.entry = MethodType(entry, module)
        self.convs = (module.cv1, module.cv2, *(c for m in module.m for c in (m.cv1, m.cv2)))
        self.expected = ((256,256,1), (512,256,1), *((128,128,3),) * 4)
        for item, (ci, co, kernel) in zip(self.convs, self.expected):
            if type(item) is not Conv or type(item.conv) is not torch.nn.Conv2d or type(item.act) is not torch.nn.SiLU:
                raise ValueError('Expected Conv/Conv2d/SiLU without custom subclasses')
            conv = item.conv
            if (conv.in_channels != ci or conv.out_channels != co or tuple(conv.weight.shape) != (co,ci,kernel,kernel) or
                    conv.kernel_size != (kernel,kernel) or conv.padding != (kernel//2,kernel//2) or
                    conv.padding_mode != 'zeros' or conv.stride != (1,1) or conv.dilation != (1,1) or conv.groups != 1 or
                    conv.bias is None or tuple(conv.bias.shape) != (co,) or
                    getattr(item.forward,'__func__',None) is not fused_forward or
                    getattr(conv.forward,'__func__',None) is not prepared_forward):
                raise ValueError('Expected exact model.6 fused/prepared convolution configuration')
            if not hasattr(conv, '_thin_packed_weight'):
                raise ValueError('Prepare the fused model before constructing the pilot')
            if any(not tensor.is_contiguous() or tensor.storage_offset() != 0 for tensor in (conv.weight,conv.bias,conv._thin_packed_weight)):
                raise ValueError('Expected allocation-base contiguous parameters/prepack')
            if tuple(conv._thin_packed_weight.shape) != (ci//16,kernel*kernel,co,16):
                raise ValueError('Prepared shape does not match the supported convolution')
            expected = conv.weight.detach().cpu().reshape(co,ci//16,16,kernel*kernel).permute(1,3,0,2).contiguous()
            if not torch.equal(conv._thin_packed_weight.cpu(), expected):
                raise ValueError('Prepared weights do not match the current OIHW parameters')
        from region_contract import RegionContract
        self.contract = RegionContract((module,), ignored_forward=(module,))

    def validate(self):
        if self.module.forward not in (self.original, self.entry):
            raise ValueError('C2f forward was changed after pilot construction')
        self.contract.validate()

    def enable(self):
        self.validate()
        self.module.forward = self.entry

    def disable(self):
        self.module.forward = self.original

    def region(self, x):
        self.validate()
        if (tuple(x.shape) != (1,256,40,40) or x.dtype != torch.float16 or
                x.device.type != 'npu' or not x.is_contiguous() or x.storage_offset() != 0 or x.requires_grad):
            raise ValueError('Expected allocation-base contiguous FP16 NPU 1x256x40x40 input')
        return blocked_forward(self.module, x)
