import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import ctypes
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
import thin_backend
from model_optimization import prepare_model
from thin_backend.graph import CapturedForward
from blocked_graph import FrozenGraph
from whole_blocked import WholeBlocked, load_native_flow

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path('evidence-whole-blocked/measurement'))
parser.add_argument('--iterations', type=int, default=101)
parser.add_argument('--rounds', type=int, default=5)
parser.add_argument('--verify-only', action='store_true')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
load_native_flow()
acl = ctypes.CDLL('libacl_rt.so')


def sync():
    code = acl.aclrtSynchronizeDevice()
    if code:
        raise RuntimeError('aclrtSynchronizeDevice: %d' % code)


def leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in leaves(item)]


def copied(value):
    return [tensor.cpu() for tensor in leaves(value)]


def equal(actual, expected):
    actual = copied(actual)
    assert len(actual) == len(expected)
    rows = []
    for got, ref in zip(actual, expected):
        assert got.shape == ref.shape and got.dtype == ref.dtype
        integer = {torch.float16: torch.int16, torch.float32: torch.int32}[got.dtype]
        bitwise = torch.equal(got.view(integer), ref.view(integer))
        difference = (got.float() - ref.float()).abs()
        row = dict(shape=list(got.shape), bitwise_equal=bitwise,
                   max_abs=difference.max().item(), mean_abs=difference.mean().item())
        assert bitwise, row
        rows.append(row)
    return rows


def timed(call):
    for _ in range(10):
        call()
        sync()
    samples = []
    for _ in range(args.iterations):
        sync()
        start = time.perf_counter_ns()
        call()
        sync()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return dict(median_us=statistics.median(samples), samples_us=samples)


with torch.no_grad():
    model = YOLO(str(args.model)).model.fuse().eval().half().to('npu')
    preparation = prepare_model(model, fuse_silu=True)
    pilot = WholeBlocked(model)
reference = torch.load(args.reference, map_location='cpu', weights_only=False)

with torch.inference_mode():
    image = reference['x'].half().to('npu')
    baseline = copied(model(image))
    pilot.enable()
    whole_eager = equal(pilot.run(image), baseline)
    pilot.disable()
    assert equal(model(image), baseline)

    result = dict(model=str(args.model), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  input_shape=list(image.shape),
                  scope='complete Detect forward including DFL/box decode; NMS excluded',
                  precision='FP16 parameters/features; fused FP32 bias and SiLU before FP16 cast',
                  layout='NC1HWC0 from stem output through the neck; Detect terminals emit NCHW',
                  measurement=dict(iterations=args.iterations, rounds=args.rounds, warmup=10,
                                   input_copy='NPU D2D', entry_and_exit_conversions_included=True),
                  preparation=preparation, whole_eager=whole_eager,
                  captured_checks=[], timings=[], fresh_captures=[])

    expected = copied(model(image))
    baseline_graph = CapturedForward(model, image)
    sync()
    pilot.enable()
    blocked_graph = FrozenGraph(pilot, model, image)
    sync()
    pilot.disable()
    equal(blocked_graph.outputs, expected)

    for scale in (0.5, 1.5, -0.75):
        other = (reference['x'] * scale).half().to('npu')
        expected = copied(model(other))
        baseline_graph(other)
        sync()
        baseline_checks = equal(baseline_graph.outputs, expected)
        blocked_graph(other)
        sync()
        blocked_checks = equal(blocked_graph.outputs, expected)
        result['captured_checks'].append(dict(scale=scale, baseline=baseline_checks, blocked=blocked_checks))

    baseline_graph(image)
    blocked_graph(image)
    sync()
    if not args.verify_only:
        for index in range(args.rounds):
            row = dict(round=index)
            order = ('baseline', 'blocked') if index % 2 == 0 else ('blocked', 'baseline')
            row['order'] = order
            for name in order:
                pilot.enable() if name == 'blocked' else pilot.disable()
                graph = blocked_graph if name == 'blocked' else baseline_graph
                row[name] = dict(eager=timed(lambda: model(image)), replay=timed(graph.replay),
                                 copy_and_replay=timed(lambda: graph(image)))
            result['timings'].append(row)
            print(index, {name: {mode: round(data['median_us'], 3) for mode, data in row[name].items()}
                          for name in order}, flush=True)

    baseline_graph.close()
    blocked_graph.close()
    del baseline_graph, blocked_graph
    gc.collect()
    pilot.disable()

    for _ in range(3):
        expected = copied(model(image))
        pilot.enable()
        graph = FrozenGraph(pilot, model, image)
        sync()
        equal(graph.outputs, expected)
        other = (reference['x'] * 0.5).half().to('npu')
        pilot.disable()
        expected = copied(model(other))
        graph(other)
        sync()
        checked = equal(graph.outputs, expected)
        graph.close()
        del graph
        gc.collect()
        result['fresh_captures'].append(checked)

    assert equal(model(image), baseline)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    assert not any(name in maps for name in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    print('WHOLE_BLOCKED_PASS', flush=True)
