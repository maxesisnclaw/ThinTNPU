import os

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def measure(model, image, *, backend, output, warmup=30, samples=101, metadata=None):
    """Measure one fused CPU nn.Module in a fresh Thin or native process."""
    import torch

    if backend not in ('thin', 'native'):
        raise ValueError('backend must be thin or native')
    if warmup < 0 or samples < 1:
        raise ValueError('warmup must be nonnegative and samples must be positive')
    if list(image.shape) != [1, 3, 640, 640]:
        raise ValueError('This comparison requires a 1x3x640x640 CPU fixture')
    if image.device.type != 'cpu':
        raise ValueError('Supply the same CPU fixture to each fresh process')
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image = image.half().contiguous()
    result = dict(metadata or {})
    result.update(backend=backend, input_shape=list(image.shape), dtype=str(image.dtype),
                  warmup=warmup, samples=samples, torch=torch.__version__,
                  timing_scope='synchronized graph replay; excludes input copy and preprocessing')
    graph = None

    def leaves(value):
        if isinstance(value, torch.Tensor):
            return [value]
        return [tensor for item in value for tensor in leaves(item)]

    def cpu(value):
        return [tensor.cpu().contiguous() for tensor in leaves(value)]

    def equality(actual, expected):
        actual = cpu(actual)
        assert len(actual) == len(expected)
        rows = []
        for left, right in zip(actual, expected):
            assert left.shape == right.shape and left.dtype == right.dtype
            same = torch.equal(left.view(torch.int16), right.view(torch.int16))
            finite = bool(left.isfinite().all())
            rows.append(dict(shape=list(left.shape), bit_equal=same, finite=finite))
            assert same and finite, rows
        return rows

    try:
        if backend == 'native':
            import torch_npu
            torch.npu.set_device(0)
            sync = torch.npu.synchronize
            result['torch_npu'] = torch_npu.__version__
        else:
            import thin_backend
            acl = ctypes.CDLL('libacl_rt.so')

            def sync():
                code = acl.aclrtSynchronizeDevice()
                if code:
                    raise RuntimeError(f'ACL synchronization failed: {code}')

            result['torch_npu'] = None
        with torch.no_grad():
            model = model.eval().requires_grad_(False).half().to('npu')
            if backend == 'thin':
                from model_optimization import prepare_model
                from whole_blocked import WholeBlocked, load_native_flow
                from blocked_graph import FrozenGraph
                load_native_flow()
                prepare_model(model, fuse_silu=True)
                blocked = WholeBlocked(model)
                blocked.enable()
            image_npu = image.to('npu')
            sync()
            start = time.perf_counter_ns()
            eager = model(image_npu)
            sync()
            result['first_eager_ms_not_benchmark'] = (time.perf_counter_ns() - start) / 1e6
            expected = cpu(eager)
            torch.save(expected, output / 'eager.pt')
            if backend == 'native':
                stream = torch.npu.Stream()
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    for _ in range(10):
                        model(image_npu)
                sync()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    captured = model(image_npu)
                sync()
            else:
                graph = FrozenGraph(blocked, model, image_npu)
                captured = graph.outputs
            replay = graph.replay
            replay()
            sync()
            result['capture_eager_equality'] = equality(captured, expected)
            for _ in range(warmup):
                replay()
                sync()
            times = []
            for _ in range(samples):
                sync()
                start = time.perf_counter_ns()
                replay()
                sync()
                times.append((time.perf_counter_ns() - start) / 1e6)
            result['samples_ms'] = times
            result['median_ms'] = statistics.median(times)
            torch.save(cpu(captured), output / 'capture.pt')
            if backend == 'thin':
                maps = Path('/proc/self/maps').read_text().lower()
                assert not any(name.startswith('torch_npu') for name in sys.modules)
                assert not any(token in maps for token in ('torch_npu', 'aclnn', 'libtbe', 'libopcompiler'))
                result['no_torch_npu_aclnn_tbe'] = True
            result['success'] = True
    except Exception as error:
        result['success'] = False
        result['error_type'] = type(error).__name__
        raise
    finally:
        try:
            if graph is not None:
                if backend == 'thin':
                    graph.close()
                else:
                    graph.reset()
        finally:
            (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='One fresh-process YOLO family graph-replay measurement')
    parser.add_argument('--family', choices=('v5m', 'v8s', 'v11s'), required=True)
    parser.add_argument('--backend', choices=('thin', 'native'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-format', choices=('fused-module', 'ultralytics'), required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--samples', type=int, default=101)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.fixture.is_file():
        raise FileNotFoundError(args.fixture)
    if args.family == 'v5m' and args.model_format != 'fused-module':
        parser.error('Classic v5 requires an upstream-constructed fused-module artifact')

    import torch
    torch.set_num_threads(1)
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=False)
    image = fixture['x'] if args.family == 'v8s' else fixture['cases']['640x640']['image_fp16']
    if args.model_format == 'ultralytics':
        from ultralytics import YOLO
        model = YOLO(str(args.model)).model.fuse()
    else:
        model = torch.load(args.model, map_location='cpu', weights_only=False)['model']
    if not isinstance(model, torch.nn.Module):
        raise TypeError('The supplied model must be an nn.Module')
    measure(model, image, backend=args.backend, output=args.output,
            warmup=args.warmup, samples=args.samples,
            metadata=dict(family=args.family, model_format=args.model_format,
                          model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                          fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
