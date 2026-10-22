import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse, ctypes, gc, json, sys, time, threading
from pathlib import Path
import torch
from ultralytics import YOLO
import thin_backend
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow
load_native_flow()
from blocked_graph import FrozenGraph
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--shapes', nargs='+', default=['640x640', '1280x1280', '1280x704', '640x1920', '320x320', '352x608', '32x32', '32x3200', '3200x32', '320x1280', '1280x320', '1024x1024'])
parser.add_argument('--output', type=Path, default=Path('evidence-shape-generalize'))
parser.add_argument('--main-output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
acl = ctypes.CDLL('libacl_rt.so')

def sync():
    code = acl.aclrtSynchronizeDevice()
    if code:
        raise RuntimeError(f'aclrtSynchronizeDevice: {code}')

def leaves(v):
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, (list, tuple)):
        return sum((leaves(x) for x in v), [])
    return []

def cpu(v):
    return [x.detach().cpu() for x in leaves(v)]

def bits(v):
    out = []
    for x in cpu(v):
        dtype = {torch.float16: torch.int16, torch.float32: torch.int32}.get(x.dtype)
        out.append(x.view(dtype) if dtype else x)
    return out

def compare_raw(a, b):
    aa, bb = (bits(a), bits(b))
    assert len(aa) == len(bb) and all((x.shape == y.shape for x, y in zip(aa, bb)))
    return all((torch.equal(x, y) for x, y in zip(aa, bb)))

def compare_cpu(a, b):
    aa, bb = (cpu(a), cpu(b))
    rows = []
    assert len(aa) == len(bb)
    for x, y in zip(aa, bb):
        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        d = (x.float() - y.float()).abs()
        rows.append(dict(shape=list(x.shape), max_abs=float(d.max()) if d.numel() else 0.0, mean_abs=float(d.mean()) if d.numel() else 0.0, dtype=str(x.dtype)))
    return rows

def timed(fn):
    sync()
    t = time.perf_counter_ns()
    v = fn()
    sync()
    return (v, (time.perf_counter_ns() - t) / 1000000.0)
cpu_model = YOLO(str(args.model)).model.fuse().eval().float()
npu_model = YOLO(str(args.model)).model.fuse().eval().half().to('npu')
prep = prepare_model(npu_model, fuse_silu=True)
pilot = WholeBlocked(npu_model)
all_rows = []
fixture = torch.load(args.reference, map_location='cpu', weights_only=False)['x'].float()
try:
    mem = ctypes.CDLL(None).c2f_memory_snapshot
    mem.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    reset_peak = ctypes.CDLL(None).c2f_memory_reset_peak
except AttributeError:
    mem = reset_peak = None

def memory():
    values = (ctypes.c_uint64 * 5)()
    if mem:
        mem(values)
    pss = int(next((line.split()[1] for line in Path('/proc/self/smaps_rollup').read_text().splitlines() if line.startswith('Pss:')))) * 1024
    return dict(pss_bytes=pss, device_live_requested=values[0], device_peak_live_requested=values[1], device_total_requested=values[2])

class MemorySample:

    def __enter__(self):
        self.stop = threading.Event()
        self.peak = 0

        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, memory()['pss_bytes'])
                self.stop.wait(0.01)
        self.thread = threading.Thread(target=sample)
        self.thread.start()
        if reset_peak:
            reset_peak()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.result = memory()
        self.result['pss_sampled_peak_bytes'] = max(self.peak, self.result['pss_bytes'])
with torch.inference_mode():
    for spec in args.shapes:
        h, w = map(int, spec.lower().split('x'))
        assert h % 32 == 0 and w % 32 == 0
        torch.manual_seed(1234 + h + w)
        x = torch.nn.functional.interpolate(fixture, size=(h, w), mode='bilinear', align_corners=False)
        ref, t_cpu = timed(lambda: cpu_model(x))
        image = x.half().to('npu')
        pilot.disable()
        pilot.enable()
        blocked, t_cold = timed(lambda: pilot.run(image))
        blocked, t_block = timed(lambda: pilot.run(image))
        block_err = compare_cpu(blocked, ref)
        decoded = blocked[0] if isinstance(blocked, (list, tuple)) else blocked
        cpu_decoded = ref[0] if isinstance(ref, (list, tuple)) else ref
        assert decoded.shape[-1] == sum((h // stride * (w // stride) for stride in (8, 16, 32)))
        for head, stride in zip(leaves(blocked)[1:], (8, 16, 32)):
            assert tuple(head.shape[-2:]) == (h // stride, w // stride)
        coordinate_difference = (decoded[:, :4].cpu().float() - cpu_decoded[:, :4].float()).abs()
        coordinate_error = float(coordinate_difference.max())
        coordinate_mean = float(coordinate_difference.mean())
        probability_error = float((decoded[:, 4:].cpu().float() - cpu_decoded[:, 4:].float()).abs().max())
        coordinate_scale = max(max(h, w), float(cpu_decoded[:, :4].abs().max()))
        assert coordinate_error / coordinate_scale <= 0.002, (spec, coordinate_error, coordinate_scale)
        assert probability_error <= 0.01, (spec, probability_error)
        main_bitwise = None
        if args.main_output and (h, w) == (640, 640):
            baseline = torch.load(args.main_output, weights_only=False)
            main_bitwise = compare_raw(blocked, baseline)
            assert main_bitwise, '640 differs from main'
        with MemorySample() as allocation_sample:
            graph = FrozenGraph(pilot, pilot.run, image)
            sync()
        captured = graph.outputs
        sync()
        cap_err = compare_cpu(captured, ref)
        assert compare_raw(blocked, captured), (spec, 'blocked eager/capture differs')
        torch.manual_seed(4321 + h + w)
        other = torch.randn(1, 3, h, w).half().to('npu')
        pilot.enable()
        blocked2, t_block2 = timed(lambda: npu_model(other))
        graph(other)
        sync()
        captured2 = graph.outputs
        _, replay_ms = timed(graph.replay)
        _, copy_replay_ms = timed(lambda: graph(other))
        try:
            graph(torch.empty((1, 3, h, w + 32), device='npu', dtype=torch.float16))
        except ValueError:
            pass
        else:
            raise AssertionError('graph accepted a changed shape')
        assert compare_raw(blocked2, captured2), (spec, 'second blocked eager/capture differs')
        if (h, w) == (640, 640):
            wrong = (torch.empty(image.shape, dtype=torch.float32, device='npu'), image.cpu(), image.transpose(2, 3), image.expand(2, 3, h, w))
            for value in wrong:
                try:
                    graph(value)
                except ValueError:
                    pass
                else:
                    raise AssertionError('graph accepted changed dtype/device/layout/batch')
            padding = npu_model.model[0].conv.padding
            npu_model.model[0].conv.padding = (0, 0)
            try:
                graph.replay()
            except ValueError:
                pass
            else:
                raise AssertionError('graph accepted changed padding')
            finally:
                npu_model.model[0].conv.padding = padding
            for shape in ((2, 3, 640, 640), (1, 3, 641, 640), (1, 3, 0, 640)):
                try:
                    pilot.run(torch.empty(shape, dtype=torch.float16, device='npu'))
                except ValueError:
                    pass
                else:
                    raise AssertionError('whole flow accepted unsupported input')
            del wrong, value
        graph.close()
        del graph
        row = dict(shape=[1, 3, h, w], main_bitwise=main_bitwise, coordinate_max_abs=coordinate_error, coordinate_normalized=coordinate_error / coordinate_scale, coordinate_scale=coordinate_scale, coordinate_mean_normalized=coordinate_mean / coordinate_scale, probability_max_abs=probability_error, memory=allocation_sample.result, capture_replay_wall_ms=replay_ms, copy_replay_wall_ms=copy_replay_ms, cpu_wall_ms=t_cpu, thin_blocked_eager_wall_ms=t_block, thin_blocked_second_wall_ms=t_block2, blocked_cpu_error=block_err, capture_cpu_error=cap_err, raw_bit_blocked_capture=True)
        all_rows.append(row)
        (args.output / 'results.json').write_text(json.dumps(all_rows, indent=2))
        print(json.dumps(row), flush=True)
        del x, image, ref, blocked, blocked2, captured, captured2
        gc.collect()
    maps = Path('/proc/self/maps').read_text().lower()
    (args.output / 'maps.txt').write_text(maps)
    (args.output / 'modules.txt').write_text('\n'.join(sorted(sys.modules)))
    assert not any((k.startswith('torch_npu') for k in sys.modules))
    assert not any((k in maps for k in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe')))
(args.output / 'results.json').write_text(json.dumps(dict(model=str(args.model), prep=prep, rows=all_rows, torch_npu=False), indent=2) + '\n')
print('SHAPE_GENERALIZE_PASS', flush=True)
