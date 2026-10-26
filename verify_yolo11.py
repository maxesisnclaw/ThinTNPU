import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import thin_backend
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
from blocked_graph import FrozenGraph

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--shape-edges', action='store_true')
args = parser.parse_args()
load_native_flow()
torch.set_num_threads(1)
fixture = torch.load(args.fixture, weights_only=False)


def leaves(value):
    if isinstance(value, torch.Tensor):
        return [value.cpu().contiguous()]
    return [tensor for item in value for tensor in leaves(item)]


def errors(actual, reference):
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    difference = (actual.float() - reference.float()).abs()
    rounded = reference.half().contiguous()
    return dict(shape=list(actual.shape), max_abs=difference.max().item(), mean_abs=difference.mean().item(),
                bit_equal_rounded_reference=torch.equal(actual.contiguous().view(torch.int16), rounded.view(torch.int16)))


def attention_outputs(model, image):
    from ultralytics.nn.modules.block import Attention, C2PSA, PSABlock
    captured, handles = {}, []

    def hook(name):
        def record(module, inputs, output):
            captured[name] = output
        return record

    for module in model.modules():
        entries = []
        if isinstance(module, C2PSA):
            entries = [('c2psa.cv1', module.cv1), ('c2psa.cv2', module.cv2)]
        elif isinstance(module, PSABlock):
            entries = [('psa.attn', module.attn), ('psa.ffn', module.ffn)]
        elif isinstance(module, Attention):
            entries = [('attn.qkv', module.qkv), ('attn.proj', module.proj), ('attn.pe', module.pe)]
        handles.extend(leaf.register_forward_hook(hook(name)) for name, leaf in entries)
    try:
        model(image)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name, tensor in captured.items():
        tensor = tensor.cpu()
        if tensor.dim() == 5:
            batch, c1, height, width, c0 = tensor.shape
            tensor = tensor.permute(0, 1, 4, 2, 3).contiguous().view(batch, c1*c0, height, width)
        result[name] = tensor
    return result


def assert_bits(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert torch.equal(left.view(torch.int16), right.view(torch.int16))


with torch.no_grad():
    model = torch.load(args.model, weights_only=False)['model'].eval().requires_grad_(False).half().to('npu')
    preparation = prepare_model(model, fuse_silu=True)
    pilot = WholeBlocked(model)
    pilot.enable()
print('MODEL_PREPARED', len(preparation['prepared_layers']), flush=True)
results = dict(cases={}, shape_edges=[])
with torch.inference_mode():
    for name, case in fixture['cases'].items():
        image = case['image_fp16'].to('npu')
        start = time.perf_counter()
        output = pilot.run(image)
        eager = leaves(output)
        elapsed = time.perf_counter() - start
        print('EAGER_DONE', name, elapsed, flush=True)
        attention = attention_outputs(model, image)
        pilot.validate()
        raw = output[1]
        box = torch.cat([head.view(1, pilot.detect.no, -1) for head in raw], dim=2)[:, :4 * pilot.detect.reg_max]
        dfl = pilot.detect.dfl(box).cpu()
        dfl_reference = case['dfl'][0]['output']
        graph = FrozenGraph(pilot, model, image)
        assert_bits(leaves(graph.outputs), eager)
        changed = (case['image_fp16'] * 0.5).to('npu')
        changed_eager = leaves(pilot.run(changed))
        assert_bits(leaves(graph(changed)), changed_eager)
        assert_bits(leaves(graph(image)), eager)
        graph.close()
        row = dict(input_shape=list(image.shape), eager_seconds=elapsed,
                   eager_capture_bit_equal=True, changed_input_replay_bit_equal=True,
                   restored_input_replay_bit_equal=True,
                   attention={key: errors(value, case['c2psa'][key]['out']) for key, value in attention.items()},
                   coordinates=errors(eager[0][:, :4], case['decoded'][:, :4]),
                   classes=errors(eager[0][:, 4:], case['decoded'][:, 4:]),
                   dfl=errors(dfl, dfl_reference),
                   raw_heads=[errors(actual, expected) for actual, expected in zip(eager[1:], case['raw_heads'])])
        half_reference = case['fp16_params_ref']
        row['vs_fp16_params_ref'] = dict(
            attention={key: errors(value, half_reference['c2psa'][key]['out']) for key, value in attention.items()},
            coordinates=errors(eager[0][:, :4], half_reference['decoded'][:, :4]),
            classes=errors(eager[0][:, 4:], half_reference['decoded'][:, 4:]),
            dfl=errors(dfl, half_reference['dfl'][0]['output']),
            raw_heads=[errors(actual, expected) for actual, expected in zip(eager[1:], half_reference['raw_heads'])])
        results['cases'][name] = row
        print(json.dumps(row), flush=True)
    if args.shape_edges:
        for height, width in ((32,32), (32,3200), (3200,32), (352,608), (128,224)):
            torch.manual_seed(height + width)
            image = torch.rand(1,3,height,width).half().to('npu')
            eager = leaves(pilot.run(image))
            graph = FrozenGraph(pilot, model, image)
            assert_bits(leaves(graph.outputs), eager)
            assert_bits(leaves(graph.replay()), eager)
            graph.close()
            results['shape_edges'].append(dict(shape=[height,width], eager_capture_bit_equal=True))
            print('SHAPE_EDGE_BITS_PASS', height, width, flush=True)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(token in maps for token in ('torch_npu','aclnn','libopcompiler','libtbe'))
    results['no_torch_npu_aclnn_tbe'] = True
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=2))
print('YOLO11_CAPTURE_REPLAY_PASS', flush=True)
