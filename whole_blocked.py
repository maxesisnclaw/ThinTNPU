import os
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
from types import MethodType

import torch
from torch import nn
from model_optimization import is_conv_wrapper
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
    torch.ops.load_library(str(here / 'libbmm_flow.so'))
    torch.ops.load_library(str(here / 'libdepthwise_flow.so'))


def pad_bias(bias, co):
    """Pad bias to the native epilogue's 32-channel read granularity on CPU."""
    target = (co + 31) // 32 * 32
    if target == bias.numel():
        return None
    padded = torch.zeros(target, dtype=bias.dtype)
    padded[:co] = bias.detach().to('cpu')
    return padded.to(bias.device)


def stem_forward(module, x):
    conv = module.conv
    if conv.kernel_size == (3, 3) and conv.out_channels == 32:
        return ops.stem(x, conv.weight.detach(), conv.bias)
    return ops.conv(ops.pack_input(x), conv._thin_packed_weight,
                    getattr(conv, "_thin_native_bias", conv.bias),
                    conv.out_channels, conv.kernel_size[0], conv.stride[0], True, False)


def conv_forward(module, x):
    conv = module.conv
    bias = getattr(conv, '_thin_native_bias', conv.bias)
    fused = isinstance(module.act, nn.SiLU)
    if conv.groups > 1:
        return ops.depthwise(x, conv._thin_packed_weight, conv.bias, conv.kernel_size[0], conv.stride[0], fused)
    return ops.conv(x, conv._thin_packed_weight, bias,
                    conv.out_channels, conv.kernel_size[0], conv.stride[0], fused, False)


def terminal_forward(module, x):
    bias = getattr(module, '_thin_native_bias', module.bias)
    return ops.conv(x, module._thin_packed_weight, bias,
                    module.out_channels, module.kernel_size[0], module.stride[0], False, True)


def maxpool_forward(module, x):
    return ops.maxpool(x, module.kernel_size)


def upsample_forward(module, x):
    return ops.upsample(x)


def anchor_detect_forward(module, features):
    decoded, raw = [], []
    for level, (terminal, feature) in enumerate(zip(module.m, features)):
        head = terminal(feature)
        batch, _, height, width = head.shape
        head = head.view(batch, module.na, module.no, height, width).permute(0, 1, 3, 4, 2).contiguous()
        raw.append(head)
        key = (level, height, width)
        if key not in module._thin_grid_cache:
            y, x = torch.meshgrid(torch.arange(height, dtype=torch.float16),
                                  torch.arange(width, dtype=torch.float16), indexing='ij')
            module._thin_grid_cache[key] = (torch.stack((x, y), -1) - 0.5).to(head.device)
        probability = head.sigmoid()
        xy = (probability[..., :2] * 2 + module._thin_grid_cache[key]) * module._thin_strides[level]
        doubled = probability[..., 2:4] * 2
        wh = (doubled * doubled) * module._thin_pixel_anchors[level].view(1, module.na, 1, 1, 2)
        decoded.append(torch.cat((xy, wh, probability[..., 4:]), -1).view(batch, -1, module.no))
    return torch.cat(decoded, 1), raw


def attention_forward(module, x):
    batch, _, height, width, _ = x.shape
    tokens = height * width
    heads = batch * module.num_heads
    qkv = module.qkv(x).view(batch, -1, tokens, 16)
    if module._thin_attention_prescale:
        query = ops.channel_select(qkv, heads, 0, module.key_dim) * module.scale
        scores = ops.layout_bmm(query, qkv, True, False, False, heads,
                                0, module.key_dim, module.key_dim, module.key_dim)
    else:
        scores = ops.layout_bmm(qkv, qkv, True, False, False, heads,
                                0, module.key_dim, module.key_dim, module.key_dim) * module.scale
    probability = scores.softmax(dim=-1)
    mixed = ops.layout_bmm(qkv, probability, False, True, True, heads,
                           2 * module.key_dim, module.head_dim)
    channels = module.num_heads * module.head_dim
    mixed = mixed.view(batch, channels // 16, height, width, 16)
    position = module.pe(ops.channel_select(qkv, heads, 2 * module.key_dim, module.head_dim)
                         .view(batch, channels // 16, height, width, 16))
    return module.proj(mixed + position)


def c2psa_forward(module, x):
    a, b = module.cv1(x).split((module.c // 16, module.c // 16), dim=1)
    return module.cv2(torch.cat((a, module.m(b)), dim=1))


def anchorfree_predictions(module, features):
    box_head, cls_head = ((module.one2one_cv2, module.one2one_cv3) if module.end2end
                          else (module.cv2, module.cv3))
    batch = features[0].size(0)
    boxes = torch.cat([head(x).view(batch, 4 * module.reg_max, -1)
                       for head, x in zip(box_head, features)], -1)
    scores = torch.cat([head(x).view(batch, module.nc, -1)
                        for head, x in zip(cls_head, features)], -1)
    key = tuple((x.size(2), x.size(3)) for x in features)
    if key not in module._thin_grid_cache:
        points, strides = [], []
        for (height, width), stride in zip(key, module._thin_strides):
            y, x = torch.meshgrid(torch.arange(height, dtype=torch.float16) + 0.5,
                                  torch.arange(width, dtype=torch.float16) + 0.5, indexing='ij')
            points.append(torch.stack((x, y), 0).reshape(2, -1))
            strides.append(torch.full((1, height * width), stride, dtype=torch.float16))
        module._thin_grid_cache[key] = (torch.cat(points, -1).to(boxes.device),
                                        torch.cat(strides, -1).to(boxes.device))
    anchors, strides = module._thin_grid_cache[key]
    distance = module.dfl(boxes)
    left_top, right_bottom = distance.split((2, 2), dim=1)
    lower, upper = anchors.unsqueeze(0) - left_top, anchors.unsqueeze(0) + right_bottom
    if not module.end2end and not module.xyxy:
        decoded_boxes = torch.cat(((lower + upper) / 2, upper - lower), 1) * strides
    else:
        decoded_boxes = torch.cat((lower, upper), 1) * strides
    decoded = torch.cat((decoded_boxes, scores.sigmoid()), 1)
    raw = {'boxes': boxes, 'scores': scores, 'feats': features}
    return decoded, raw


def anchorfree_detect_forward(module, features):
    decoded, raw = anchorfree_predictions(module, features)
    if module.end2end:
        output = ops.detection_select(decoded.transpose(1, 2), module.max_det, module.agnostic_nms)
        raw = {'one2many': {}, 'one2one': raw}
    else:
        output = decoded
    return output, raw


class WholeBlocked:
    """Fused YOLO blocked flow with direct-distance, DFL or anchor/objectness Detect.

    Intermediate channels use NC1HWC0; Detect terminals restore NCHW. Each captured
    graph fixes its input shape, while eager execution and fresh captures accept /32 H/W.
    attention_scale_before_matmul selects the frontend's FP16 rounding order; it must
    match the original Attention forward rather than an algebraic rearrangement.
    """

    def __init__(self, model, attention_scale_before_matmul=False):
        from ultralytics.nn.modules.block import Attention, C2PSA, DFL
        from ultralytics.nn.modules.head import Detect

        root = model.model if hasattr(model, 'model') else model
        detect = root[-1]
        anchor_based = (type(detect).__name__ == 'Detect' and hasattr(detect, 'na')
                        and hasattr(detect, 'anchors') and hasattr(detect, 'm'))
        if not is_conv_wrapper(root[0]) or not (type(detect) is Detect or anchor_based):
            raise ValueError('Expected a fused YOLO sequence ending in a supported Detect')
        if detect.nl != 3 or detect.export:
            raise ValueError('Expected a three-level non-export Detect')
        if anchor_based:
            if (detect.no != detect.nc + 5 or detect.na <= 0 or len(detect.m) != detect.nl
                    or tuple(detect.anchors.shape) != (detect.nl, detect.na, 2)):
                raise ValueError('Expected classic anchor/objectness Detect parameters')
            strides = detect.stride.detach().cpu().float()
            anchors = detect.anchors.detach().cpu().float()
            if not (torch.isfinite(strides).all() and (strides > 0).all()
                    and torch.isfinite(anchors).all() and (anchors > 0).all()):
                raise ValueError('Expected positive finite checkpoint anchors and strides')
            detect._thin_strides = tuple(strides.tolist())
            detect._thin_grid_cache = {}
            detect.register_buffer('_thin_pixel_anchors',
                                   (anchors * strides[:, None, None]).half().to(detect.anchors.device),
                                   persistent=False)
        elif detect.reg_max not in (1, 16):
            raise ValueError('Expected direct-distance or DFL(16) Detect')
        dictionary_head = not anchor_based and hasattr(detect, 'forward_head')
        if not anchor_based:
            if detect.end2end and not dictionary_head:
                raise ValueError('End-to-end Detect requires dictionary head semantics')
            detect._thin_strides = tuple(detect.stride.detach().cpu().float().tolist())
            detect._thin_grid_cache = {}
            if detect.end2end:
                torch.ops.load_library(str(Path(__file__).parent / 'thin_backend' / 'libselection_flow.so'))

        stem = root[0]
        if (not isinstance(stem.act, nn.SiLU) or stem.conv.in_channels != 3 or stem.conv.kernel_size not in ((3, 3), (6, 6))
                or stem.conv.stride != (2, 2) or stem.conv.out_channels % 16
                or stem.conv.padding != ((stem.conv.kernel_size[0] - 1) // 2,) * 2
                or stem.conv.groups != 1 or stem.conv.dilation != (1, 1)
                or stem.conv.padding_mode != 'zeros' or stem.conv.bias is None):
            raise ValueError('Expected an aligned dense 3-channel k3/k6 s2 SiLU stem')

        wrapped = set(m.conv for m in model.modules() if is_conv_wrapper(m))
        terminals = set()
        if anchor_based:
            tails = list(detect.m)
        else:
            branches = ((detect.one2one_cv2, detect.one2one_cv3) if detect.end2end
                        else (detect.cv2, detect.cv3))
            if any(branch is None or len(branch) != detect.nl for branch in branches):
                raise ValueError('Selected inference branch must contain all Detect levels')
            tails = [branch[-1] for family in branches for branch in family]
        for tail in tails:
            if (type(tail) is not nn.Conv2d or tail.kernel_size != (1, 1)
                    or tail.stride != (1, 1) or tail.groups != 1 or tail.bias is None
                    or (anchor_based and tail.out_channels != detect.na * detect.no)):
                raise ValueError('Expected 1x1 nn.Conv2d Detect terminals')
            terminals.add(tail)
        dfl_conv = None if anchor_based else (detect.dfl.conv if type(detect.dfl) is DFL else None)

        convs, maxpools, upsamples, attentions, channel_splits = [], [], [], [], []
        for module in model.modules():
            if module is stem:
                continue
            if is_conv_wrapper(module):
                conv = module.conv
                depthwise = conv.groups == conv.in_channels == conv.out_channels and conv.groups > 1
                if ((conv.groups != 1 and not depthwise) or conv.dilation != (1, 1) or conv.padding_mode != 'zeros'
                        or conv.kernel_size[0] not in (1, 3) or conv.kernel_size[0] != conv.kernel_size[1]
                        or conv.padding != (conv.kernel_size[0] // 2,) * 2
                        or conv.stride not in ((1, 1), (2, 2)) or conv.bias is None
                        or not isinstance(module.act, (nn.SiLU, nn.Identity))
                        or (depthwise and (conv.kernel_size != (3, 3) or conv.stride != (1, 1)))):
                    raise ValueError('Expected dense k1/k3 or depthwise k3/s1, SiLU/Identity convolutions with bias')
                convs.append(module)
            elif isinstance(module, Attention):
                if module.head_dim <= 0 or module.key_dim <= 0 or module.num_heads <= 0:
                    raise ValueError('Attention dimensions must be positive')
                attentions.append(module)
            elif isinstance(module, C2PSA):
                if module.c % 16:
                    raise ValueError('Blocked channel split requires C0-aligned hidden channels')
                channel_splits.append(module)
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

        for leaf in [stem, *convs]:
            if not hasattr(leaf.conv, '_thin_packed_weight'):
                raise ValueError('Run prepare_model(model) to prepack convolution weights before WholeBlocked')
            padded = None if leaf.conv.groups > 1 else pad_bias(leaf.conv.bias, leaf.conv.out_channels)
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
        for module in attentions:
            module._thin_attention_prescale = attention_scale_before_matmul
        self.replacements = {
            stem: (stem.forward, MethodType(stem_forward, stem)),
            **{m: (m.forward, MethodType(conv_forward, m)) for m in convs},
            **{m: (m.forward, MethodType(terminal_forward, m)) for m in terminals},
            **{m: (m.forward, MethodType(attention_forward, m)) for m in attentions},
            **{m: (m.forward, MethodType(c2psa_forward, m)) for m in channel_splits},
            **{m: (m.forward, MethodType(maxpool_forward, m)) for m in maxpools},
            **{m: (m.forward, MethodType(upsample_forward, m)) for m in upsamples},
            **({detect: (detect.forward, MethodType(anchor_detect_forward, detect))} if anchor_based else {}),
            **({detect: (detect.forward, MethodType(anchorfree_detect_forward, detect))} if dictionary_head else {}),
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
