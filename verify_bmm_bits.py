import argparse
import os
from pathlib import Path
import runpy

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('save', 'compare'))
parser.add_argument('directory', type=Path)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
original_cpu = torch.Tensor.cpu
outputs = []


def capture_cpu(tensor, *a, **kw):
    result = original_cpu(tensor, *a, **kw)
    outputs.append(result.detach().clone().contiguous())
    return result


for test in ('verify_bmm_flow.py', 'verify_bmm_layout.py'):
    outputs.clear()
    torch.Tensor.cpu = capture_cpu
    try:
        runpy.run_path(str(Path(__file__).parent / test), run_name='__main__')
    finally:
        torch.Tensor.cpu = original_cpu
    target = args.directory / (Path(test).stem + '.pt')
    if args.mode == 'save':
        torch.save(outputs, target)
    else:
        expected = torch.load(target, weights_only=True)
        assert len(outputs) == len(expected), test
        for index, (actual, reference) in enumerate(zip(outputs, expected)):
            assert actual.shape == reference.shape and actual.dtype == reference.dtype, (test, index)
            assert torch.equal(actual.reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8)), (test, index)
    print(f'{test}: {len(outputs)} outputs {args.mode} bitwise PASS', flush=True)
