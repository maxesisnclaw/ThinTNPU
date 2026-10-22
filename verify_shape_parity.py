import argparse
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
from pathlib import Path
import sys
p = argparse.ArgumentParser()
p.add_argument('--backend-root', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--reference', type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.backend_root.resolve()))
import torch
from ultralytics import YOLO
import thin_backend
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
load_native_flow()
torch.set_num_threads(1)
shapes = [(640,640),(1280,1280),(1280,704),(640,1920),(320,320),(352,608),(32,32),(32,3200),(3200,32),(320,1280),(1280,320),(1024,1024)]
def leaves(v):
    if isinstance(v, torch.Tensor): return [v.cpu()]
    return [t for x in v for t in leaves(x)]
a.output.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    model = YOLO(str(a.model)).model.fuse().eval().half().to('npu')
    prepare_model(model, fuse_silu=True)
    pilot = WholeBlocked(model)
    pilot.enable()
fixture = torch.load(a.fixture, map_location='cpu', weights_only=False)['x']
with torch.inference_mode():
    for h,w in shapes:
        torch.manual_seed(4321+h+w)
        images = [torch.nn.functional.interpolate(fixture.float(), size=(h,w), mode='bilinear', align_corners=False).half(), torch.randn(1,3,h,w).half()]
        for index,image in enumerate(images):
            filename = f'{h}x{w}-{index}.pt'
            actual = leaves(model(image.to('npu')))
            if a.reference:
                expected = torch.load(a.reference/filename, weights_only=False)
                assert len(actual) == len(expected)
                assert all(x.shape == y.shape and x.dtype == y.dtype and torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)) for x,y in zip(actual,expected)), filename
            else:
                torch.save(actual, a.output/filename)
        print('SHAPE_RAW_BIT_PASS', h, w, 'inputs', len(images), flush=True)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(token in maps for token in ('torch_npu','aclnn','libopcompiler','libtbe'))
    (a.output/'maps.txt').write_text(maps)
print('SHAPE_PARITY_ALL_PASS', len(shapes), flush=True)
