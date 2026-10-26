import torch
from _region_guard import Guard


class RegionContract:
    attributes = ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'padding_mode',
                  'dilation', 'groups', 'add', 'c', 'inplace', 'scale_factor', 'mode',
                  'align_corners', 'size', 'ceil_mode', 'return_indices',
                  'num_heads', 'head_dim', 'key_dim', 'scale', 'shortcut',
                  'nl', 'na', 'no', 'nc', 'export', 'reg_max', 'end2end', '_thin_strides')

    def __init__(self, roots, ignored_forward=()):
        modules = tuple(m for root in roots for m in root.modules())
        for module in modules:
            for tensor in (*module._parameters.values(), *module._buffers.values()):
                if tensor is None:
                    continue
                try:
                    tensor._version
                except RuntimeError as error:
                    raise ValueError('Prepare versioned parameters under no_grad(), not inference_mode()') from error
                if tensor.requires_grad or tensor.dtype != torch.float16 or tensor.device.type != 'npu':
                    raise ValueError('Expected fixed FP16 NPU inference parameters')
        self.guard = Guard(modules, set(ignored_forward), self.attributes)

    def validate(self):
        self.guard.validate()
