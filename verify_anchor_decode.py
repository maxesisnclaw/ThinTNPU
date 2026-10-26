import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from torch import nn
import thin_backend
from whole_blocked import anchor_detect_forward


def make_head(na, nc):
    anchors = torch.arange(1, 3 * na * 2 + 1, dtype=torch.float16).reshape(3, na, 2) * 3
    return SimpleNamespace(na=na, no=nc + 5, m=[nn.Identity() for _ in range(3)],
                           _thin_pixel_anchors=anchors.to('npu'),
                           _thin_strides=(8., 16., 32.), _thin_grid_cache={})


def expected(head, features):
    scales = []
    for level, feature in enumerate(features):
        h, w = feature.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        xy = (torch.stack((x, y), -1).half() + .5) * head._thin_strides[level]
        xy = xy[None, None].expand(1, head.na, h, w, 2)
        wh = head._thin_pixel_anchors.cpu()[level].view(1, head.na, 1, 1, 2).expand_as(xy)
        scores = torch.full((1, head.na, h, w, head.no - 4), .5, dtype=torch.float16)
        scales.append(torch.cat((xy, wh, scores), -1).reshape(1, -1, head.no))
    return torch.cat(scales, 1)


with torch.inference_mode():
    for na, nc in ((3, 1), (2, 7), (3, 80)):
        head = make_head(na, nc)
        features = [torch.zeros(1, na * head.no, h, w).half().to('npu')
                    for h, w in ((8, 12), (4, 6), (2, 3))]
        actual, raw = anchor_detect_forward(head, features)
        reference = expected(head, features)
        assert torch.equal(actual.cpu().view(torch.int16), reference.view(torch.int16))
        assert all(not x.cpu().count_nonzero() for x in raw)
        print('ANCHOR_DECODE_BITS_PASS', na, nc, list(actual.shape), flush=True)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(token in maps for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
print('ANCHOR_DECODE_ALL_PASS', flush=True)
