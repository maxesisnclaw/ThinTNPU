import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
import gc
import hashlib
import inspect
import json
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

started = time.perf_counter()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--variant', choices=['thin-eager', 'thin-capture', 'thin-blocked', 'thin-whole', 'native-eager', 'native-capture'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--iterations', type=int, default=101)
parser.add_argument('--warmup', type=int, default=30)
parser.add_argument('--profile-iterations', type=int, default=0)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
peak = dict(pss_KiB=0, rss_KiB=0)
observed_children = set()
stop = threading.Event()


def children():
    result = set()
    for path in Path('/proc/self/task').glob('*/children'):
        try:
            result.update(int(pid) for pid in path.read_text().split())
        except FileNotFoundError:
            pass
    return result


def child_details():
    rows = []
    for pid in sorted(children()):
        try:
            proc = Path('/proc') / str(pid)
            row = dict(pid=pid, name=(proc/'comm').read_text().strip())
            fields = dict(line.split(':',1) for line in (proc/'smaps_rollup').read_text().splitlines() if ':' in line)
            row['pss_KiB'] = int(fields['Pss'].split()[0])
            rows.append(row)
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            rows.append(dict(pid=pid, snapshot_unavailable=True))
    return rows


def resident():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if ':' in line)
    return dict(pss_KiB=int(fields['Pss'].split()[0]), rss_KiB=int(fields['Rss'].split()[0]), child_pids=sorted(children()))


def sample():
    while not stop.is_set():
        current = resident()
        for key in peak:
            peak[key] = max(peak[key], current[key])
        observed_children.update(current['child_pids'])
        stop.wait(0.02)


monitor = threading.Thread(target=sample, daemon=True)
monitor.start()
stages = dict(process_start=resident())
import torch
from ultralytics import YOLO
native = args.variant.startswith('native')
if native:
    import torch_npu
    torch.npu.set_device(0)
    sync = torch.npu.synchronize
else:
    import thin_backend
    import ctypes
    acl = ctypes.CDLL('libacl_rt.so')
    def sync():
        code = acl.aclrtSynchronizeDevice()
        if code:
            raise RuntimeError(code)

torch.set_num_threads(1)
stages['imports'] = dict(elapsed_ms=(time.perf_counter() - started) * 1000, **resident())


def copied(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    return type(value)(copied(item) for item in value)


def leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in leaves(item)]


def compare(actual, expected):
    first, second = leaves(copied(actual)), leaves(expected)
    assert len(first) == len(second)
    rows = []
    for got, ref in zip(first, second):
        assert got.shape == ref.shape
        bits = {torch.float16: torch.int16, torch.float32: torch.int32}[got.dtype]
        difference = (got.float() - ref.float()).abs()
        rows.append(dict(shape=list(got.shape), bitwise=torch.equal(got.view(bits), ref.view(bits)),
                         max_abs=difference.max().item(), mean_abs=difference.mean().item()))
    return rows


def timed(call):
    for _ in range(args.warmup):
        call()
        sync()
    times = []
    for _ in range(args.iterations):
        sync()
        start = time.perf_counter_ns()
        call()
        sync()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return dict(median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times), samples_ms=times)


with torch.no_grad():
    model_path = args.model
    net = YOLO(str(model_path)).model.fuse().eval().half().to('npu')
    if not native:
        from model_optimization import prepare_model
        prepare_model(net, fuse_silu=True)
    pilot = None
    if args.variant == 'thin-blocked':
        from blocked_c2f import BlockedC2f
        pilot = BlockedC2f(net.model[6])
        pilot.enable()
    elif args.variant == 'thin-whole':
        from whole_blocked import WholeBlocked, load_native_flow
        load_native_flow()
        pilot = WholeBlocked(net)
        pilot.enable()
reference = torch.load(args.reference, map_location='cpu', weights_only=False)
result = dict(variant=args.variant, native=native, scope='complete Detect forward including DFL and box decode; NMS excluded',
              input='same cpu_ref x quantized to FP16, N=1 640x640; no added resize or padding',
              torch=torch.__version__, torch_npu=getattr(sys.modules.get('torch_npu'), '__version__', None),
              model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), pid=os.getpid(),
              warmup=args.warmup, iterations=args.iterations, stages=stages, cold={}, correctness=[], timing={})
with torch.inference_mode():
    x = reference['x'].half().to('npu')
    sync()
    stages['model_prepared'] = dict(elapsed_ms=(time.perf_counter() - started) * 1000, **resident())
    start = time.perf_counter()
    eager = net(x)
    sync()
    result['cold']['first_forward_ms'] = (time.perf_counter() - start) * 1000
    eager_cpu = copied(eager)
    assert all(bool(t.isfinite().all()) for t in leaves(eager_cpu))
    stages['first_forward'] = dict(elapsed_ms=(time.perf_counter() - started) * 1000, **resident())
    torch.save(eager_cpu, args.output / 'eager-output.pt')
    static = x
    graph = None
    captured = False
    output = eager
    if args.variant.endswith('capture') or args.variant in ('thin-blocked', 'thin-whole'):
        start = time.perf_counter()
        try:
            if native:
                result['native_graph_api'] = dict(NPUGraph=repr(torch.npu.NPUGraph), graph_signature=str(inspect.signature(torch.npu.graph)))
                static = torch.empty_like(x)
                static.copy_(x)
                capture_stream = torch.npu.Stream()
                capture_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(capture_stream):
                    for _ in range(10):
                        net(static)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=capture_stream):
                    output = net(static)
                torch.npu.synchronize()
                def replay():
                    graph.replay()
                    return output
                def copy_replay(value=x):
                    static.copy_(value)
                    graph.replay()
                    return output
            else:
                from thin_backend.graph import CapturedForward
                if pilot:
                    from blocked_graph import FrozenGraph
                    graph = FrozenGraph(pilot, net, x)
                else:
                    graph = CapturedForward(net, x)
                output = graph.outputs
                replay = graph.replay
                copy_replay = lambda value=x: graph(value)
            captured = True
            replay()
            sync()
            result['capture_equality'] = compare(output, eager_cpu)
            assert all(row['bitwise'] for row in result['capture_equality'])
        except Exception as error:
            result['capture_error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            if graph is not None:
                try:
                    graph.reset() if native else graph.close()
                except Exception as cleanup:
                    result['capture_error']['cleanup_error'] = str(cleanup)
            graph = None
        result['cold']['capture_setup_ms'] = (time.perf_counter() - start) * 1000
    if args.variant in ('thin-eager', 'native-eager'):
        static = torch.empty_like(x)
        static.copy_(x)
        def replay():
            return net(x)
        def copy_replay(value=x):
            static.copy_(value)
            return net(static)
    supported = args.variant in ('thin-eager', 'native-eager') or captured
    result['capture_supported_and_executed'] = captured
    result['measurement_available'] = supported
    if supported:
        for _ in range(args.warmup):
            replay()
            sync()
        stages['steady_before_measurement'] = resident()
        stop.set()
        monitor.join()
        result['memory_sampling_scope'] = '20ms sampler during startup/warmup only; stopped before all timed samples; steady/close snapshots outside timing'
        result['timing']['forward'] = timed(replay)
        result['timing']['d2d_copy_and_forward'] = timed(copy_replay)
        stages['steady_after_measurement'] = resident()
        if args.profile_iterations:
            for _ in range(args.profile_iterations):
                replay()
                sync()
        for scale in (() if args.profile_iterations else (0.5, 1.5, -0.75)):
            other = (reference['x'] * scale).half().to('npu')
            expected = copied(net(other))
            sync()
            actual = copy_replay(other)
            sync()
            checks = compare(actual, expected)
            assert all(row['bitwise'] for row in checks), checks
            result['correctness'].append(dict(scale=scale, self_eager=checks))
            torch.save(copied(actual), args.output / f'output-scale-{scale}.pt')
        if not args.profile_iterations:
            copy_replay(x)
            sync()
            torch.save(copied(replay()), args.output / 'output.pt')
    if graph is not None:
        sync()
        graph.reset() if native else graph.close()
        del graph
        gc.collect()
    stages['graphs_closed'] = resident()
    if native:
        result['native_settings'] = dict(conv_allow_hf32=torch.npu.conv.allow_hf32, matmul_allow_hf32=torch.npu.matmul.allow_hf32, input_and_weights='FP16; HF32 flags do not change FP16 operand dtype')
        result['device_allocator_kind'] = 'torch_npu caching allocator; allocated/reserved are not driver total HBM'
        result['device_allocator'] = dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved(),
                                          max_allocated=torch.npu.max_memory_allocated(), max_reserved=torch.npu.max_memory_reserved())
    maps = Path('/proc/self/maps').read_text()
    forbidden = [line for line in maps.splitlines() if any(token in line.lower() for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))]
    if not native:
        assert not forbidden
        assert not any(name.startswith('torch_npu') for name in sys.modules)
        assert not children()
        assert not observed_children
    result['torch_npu_in_modules'] = any(name.startswith('torch_npu') for name in sys.modules)
    result['children_at_exit'] = sorted(children())
    result['child_details_at_exit'] = child_details()
    stop.set()
    monitor.join()
    result['observed_child_pids'] = sorted(observed_children)
    result['peak_sampled'] = peak
    result['cold']['total_process_body_ms'] = (time.perf_counter() - started) * 1000
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    (args.output / 'maps.txt').write_text(maps)
    print(json.dumps({key: value for key, value in result.items() if key not in ('correctness', 'capture_equality')}, indent=2), flush=True)
