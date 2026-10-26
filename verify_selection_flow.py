import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import json
from pathlib import Path
import random
import sys
import torch

TRACE_NAMES = ('output', 'class_max', 'stage1_indices', 'gathered_scores',
               'top_values', 'stage2_indices', 'anchor_indices', 'labels')
TILE = 128


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def gather_bits(x, dim, indices):
    # CPU half gather quiets signaling NaNs; the device's record movement must not.
    return (x.view(torch.int16).gather(dim, indices).view(torch.float16)
            if x.dtype == torch.float16 else x.gather(dim, indices))


def reference(x, max_det, agnostic=False, stable=True):
    b, anchors, channels = x.shape
    nc, k = channels - 4, min(max_det, anchors)
    scores = x[..., 4:]
    maximum, labels = scores.max(-1)
    def select(t):
        return (torch.argsort(t, dim=1, descending=True, stable=True)[:, :k] if stable
                else t.topk(k, dim=1).indices)
    first = select(maximum)
    gather = gather_bits if stable else lambda x, dim, indices: x.gather(dim, indices)
    if agnostic:
        top, label, anchor = gather(maximum, 1, first), labels.gather(1, first), first
        gathered, second = x.new_empty((b, 0, nc)), torch.empty((b, 0), dtype=torch.long)
    else:
        gathered = gather(scores, 1, first[..., None].expand(-1, -1, nc))
        flat = gathered.reshape(b, k * nc)
        second = select(flat)
        top, label, anchor = gather(flat, 1, second), second % nc, first.gather(1, second // nc)
    boxes = gather(x[..., :4], 1, anchor[..., None].expand(-1, -1, 4))
    output = torch.cat((boxes.float(), top[..., None].float(), label[..., None].float()), -1)
    return dict(zip(TRACE_NAMES, (output, maximum, first, gathered, top, second, anchor, label)))


def validate_trace(x, max_det, agnostic, actual):
    expected = reference(x, max_det, agnostic)
    for name in TRACE_NAMES:
        assert bits_equal(actual[name], expected[name]), ('stable-stage-mismatch', name, x.shape)
    anchor, label = actual['anchor_indices'], actual['labels']
    scores = gather_bits(x[..., 4:], 1, anchor[..., None].expand(-1, -1, x.shape[-1] - 4))
    picked = gather_bits(scores, 2, label[..., None]).squeeze(-1)
    assert bits_equal(picked, actual['top_values']), 'selected tuple absent from input'
    assert all(len(set(row.tolist())) == row.numel() for row in actual['stage1_indices'])
    official = reference(x, max_det, agnostic, stable=False)
    finite = torch.isfinite(actual['top_values']) & torch.isfinite(official['top_values'])
    assert torch.equal(actual['top_values'][finite], official['top_values'][finite])
    return dict(official_output_bit_equal=bits_equal(actual['output'], official['output']),
        official_output_differing_rows=int((actual['output'].view(torch.int32)
                                            != official['output'].view(torch.int32)).any(-1).sum()),
        official_stage1_differing_indices=int((actual['stage1_indices'] != official['stage1_indices']).sum()),
        official_final_anchor_differing_indices=int((anchor != official['anchor_indices']).sum()),
        official_label_differences=int((label != official['labels']).sum()),
        official_top_score_bit_equal=bits_equal(actual['top_values'], official['top_values']),
        stable_trace_bit_equal=True, selected_tuples_valid=True)


def score_key(h):
    magnitude = h & 0x7fff
    if magnitude > 0x7c00:
        return 0
    if magnitude == 0:
        return 0x7fff
    return h if h & 0x8000 else (~(h ^ 0x8000)) & 0xffff


def float_bits(h):
    sign, exponent, mantissa = (h & 0x8000) << 16, (h >> 10) & 31, h & 1023
    if exponent == 31:
        return sign | 0x7f800000 | (mantissa << 13) | (0x00400000 if mantissa else 0)
    if exponent:
        return sign | ((exponent + 112) << 23) | (mantissa << 13)
    if not mantissa:
        return sign
    exponent = 113
    while not mantissa & 1024:
        mantissa <<= 1
        exponent -= 1
    return sign | (exponent << 23) | ((mantissa & 1023) << 13)


def sort_tile(records):
    size = 2
    while size <= TILE:
        stride = size // 2
        while stride:
            for i in range(TILE):
                j = i ^ stride
                if j > i and ((records[j] < records[i]) if i & size == 0 else (records[i] < records[j])):
                    records[i], records[j] = records[j], records[i]
            stride //= 2
        size *= 2
    return records


def merge_partition(source, start, run):
    pitch = len(source)
    group = start // (2 * run) * (2 * run)
    ln, rn = min(pitch - group, run), max(0, min(pitch - group - run, run))
    diagonal = start - group
    lo, hi = max(diagonal - rn, 0), min(diagonal, ln)
    while lo <= hi:
        li = lo + (hi - lo) // 2
        ri = diagonal - li
        if li > 0 and ri < rn and source[group + ln + ri] < source[group + li - 1]:
            hi = li - 1
        elif ri > 0 and li < ln and source[group + li] < source[group + ln + ri - 1]:
            lo = li + 1
        else:
            break
    left = source[group + li:group + min(ln, li + TILE)]
    right = source[group + ln + ri:group + ln + min(rn, ri + TILE)]
    l = r = 0
    output = []
    for _ in range(TILE):
        if r == len(right) or (l < len(left) and left[l] < right[r]):
            output.append(left[l])
            l += 1
        else:
            output.append(right[r])
            r += 1
    return output


def cpu_algorithm_checks():
    half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    keys = [score_key(i) for i in range(65536)]
    assert sorted(range(65536), key=lambda i: (keys[i], i)) == torch.argsort(
        half, descending=True, stable=True).tolist()
    assert [float_bits(i) for i in range(65536)] == [i & 0xffffffff for i in half.float().view(torch.int32).tolist()]
    rng = random.Random(149)
    shapes = (1, 2, 127, 128, 129, 255, 256, 257, 384, 513, 1025, 3360)
    for n in shapes:
        pitch = (n + TILE - 1) // TILE * TILE
        original = [(score_key(rng.randrange(65536)), i) for i in range(n)]
        original += [(65536, i) for i in range(n, pitch)]
        records = sum((sort_tile(original[s:s + TILE].copy()) for s in range(0, pitch, TILE)), [])
        run = TILE
        while run < pitch:
            records = sum((merge_partition(records, s, run) for s in range(0, pitch, TILE)), [])
            run *= 2
        assert records == sorted(original), ('merge partition', n)
    print(json.dumps(dict(cpu_algorithm_checks=True, half_patterns=65536, merge_shapes=shapes)), flush=True)


def predictions(batch, anchors, classes, unique=False):
    boxes = torch.randn(batch, anchors, 4).half()
    scores = (torch.stack([torch.randperm(anchors * classes).reshape(anchors, classes)
                          for _ in range(batch)]).half() / 128 if unique
              else torch.randn(batch, anchors, classes).sigmoid().half())
    return torch.cat((boxes, scores), -1)


def fixtures():
    for agnostic in (False, True):
        for layout in ('contiguous', 'transposed', 'gapped', 'expanded'):
            yield f'unique-{layout}-{agnostic}', predictions(2, 17, 7, True), 11, agnostic, layout, True
        tied = torch.cat((torch.arange(164).reshape(1, 41, 4).half(), torch.ones(1, 41, 5).half()), -1)
        yield f'all-tied-{agnostic}', tied, 13, agnostic, 'transposed', False
        for anchors in (1, 127, 128, 129, 257):
            yield f'boundary-{anchors}-{agnostic}', predictions(3, anchors, 3), anchors, agnostic, 'contiguous', False
        yield f'large-class-{agnostic}', predictions(1, 3, 1031), 2, agnostic, 'contiguous', False
        yield f'zero-k-{agnostic}', predictions(2, 17, 3), 0, agnostic, 'transposed', False
        yield f'zero-anchors-{agnostic}', torch.empty(2, 0, 9).half(), 7, agnostic, 'contiguous', False
        yield f'zero-batch-{agnostic}', torch.empty(0, 17, 9).half(), 7, agnostic, 'contiguous', False
        yield f'min-k-{agnostic}', predictions(2, 21, 80), 300, agnostic, 'transposed', False
    duplicate = torch.tensor([[[1, 2, 3, 4, .9, .8, .01], [5, 6, 7, 8, .7, .6, .02],
                               [9, 10, 11, 12, .5, .4, .03]]], dtype=torch.float16)
    yield 'duplicate-anchor', duplicate, 2, False, 'gapped', True
    special = torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 2**-24, -2**-24], dtype=torch.float16)
    x = special.repeat(51)[:3 * 13 * 9].reshape(3, 13, 9)
    yield 'special', x, 9, False, 'gapped', False
    yield 'special-agnostic', x, 9, True, 'transposed', False
    for anchors in (3360, 8400):
        yield f'detection-{anchors}', predictions(1, anchors, 80), 300, False, 'transposed', False
    for k in (511, 512, 513):
        tied = torch.cat((torch.randn(1, 1025, 4).half(), torch.ones(1, 1025, 80).half()), -1)
        yield f'prefix-boundary-{k}', tied, k, False, 'transposed', False
    for nc in (1, 15, 16, 17, 31, 80, 257):
        yield f'class-block-{nc}', predictions(2, 144, nc), 129, False, 'transposed', False
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    x = torch.cat((torch.zeros(1, 128, 4).half(), patterns.reshape(1, 128, 512)), -1)
    yield 'all-half-class-blocks', x, 128, False, 'transposed', False
    x = torch.cat((torch.zeros(1, 65536, 4).half(), patterns.reshape(1, 65536, 1)), -1)
    yield 'all-half-ordered', x, 65536, True, 'transposed', False


def backing_and_view(x, layout):
    b, a, c = x.shape
    if layout == 'contiguous':
        return x.contiguous(), lambda base: base
    if layout == 'transposed':
        return x.transpose(1, 2).contiguous(), lambda base: base.transpose(1, 2)
    if layout == 'expanded':
        return x[:1].contiguous(), lambda base: base.expand(b, -1, -1)
    base = torch.zeros(b + 1, a * 2 + 2, c * 2 + 3, dtype=x.dtype)
    view = lambda backing: backing[1:, 1:1 + 2 * a:2, 1:1 + 2 * c:2]
    view(base).copy_(x)
    return base, view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--case', help='fixture name substring')
    parser.add_argument('--library', type=Path, help='selection implementation shared library')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(149)
    cpu_algorithm_checks()
    if not args.cpu_only:
        import thin_backend
        torch.ops.load_library(str(args.library or Path(__file__).parent / 'thin_backend/libselection_flow.so'))
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    cases = 0
    with torch.inference_mode():
        for name, x, max_det, agnostic, layout, unique in fixtures():
            if args.case and args.case not in name:
                continue
            base, view = backing_and_view(x, layout)
            x = view(base)
            row = dict(case=name, shape=list(x.shape), stride=list(x.stride()), max_det=max_det,
                       agnostic=agnostic, cpu_only=args.cpu_only)
            actual = reference(x, max_det, agnostic)
            if not args.cpu_only:
                print(json.dumps(dict(starting=name)), flush=True)
                device_base = base.to('npu')
                device_input = view(device_base)
                eager_device = torch.ops.thin_native_flow.detection_select(device_input, max_det, agnostic)
                assert eager_device.dtype == torch.float32 and eager_device.is_contiguous()
                eager = eager_device.cpu()
                trace = torch.ops.thin_native_flow.detection_select_trace(device_input, max_det, agnostic)
                actual = {key: value.cpu() for key, value in zip(TRACE_NAMES, trace)}
                assert bits_equal(eager, actual['output']), 'trace changes output'
                if eager.numel():
                    graph = torch.classes.thin_backend.RuntimeGraph()
                    graph.begin()
                    captured = torch.ops.thin_native_flow.detection_select(device_input, max_det, agnostic)
                    graph.end()
                    graph.replay()
                    assert bits_equal(eager, captured.cpu()), 'eager/capture differs'
                    changed = -base
                    device_base.copy_(changed.to('npu'))
                    changed_eager = torch.ops.thin_native_flow.detection_select(device_input, max_det, agnostic).cpu()
                    assert bits_equal(changed_eager, reference(view(changed), max_det, agnostic)['output'])
                    graph.replay()
                    assert bits_equal(changed_eager, captured.cpu()), 'changed-input replay differs'
                    graph.close()
                    row.update(eager_capture_bit_equal=True, changed_input_replay_bit_equal=True)
            row.update(validate_trace(x, max_det, agnostic, actual))
            if unique:
                assert row['official_output_bit_equal'], ('unique scores differ from official', name)
            if args.save_dir:
                torch.save(dict(decoded=x, max_det=max_det, agnostic=agnostic, actual=actual,
                                official=reference(x, max_det, agnostic, stable=False)), args.save_dir / f'{name}.pt')
            print(json.dumps(row), flush=True)
            cases += 1
    assert not any(name.startswith('torch_npu') for name in sys.modules)
    print(json.dumps(dict(selection_checks_passed=True, cpu_only=args.cpu_only, cases=cases)), flush=True)


if __name__ == '__main__':
    main()
