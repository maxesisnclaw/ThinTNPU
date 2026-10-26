import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import json
from pathlib import Path
import sys

import torch
from model_optimization import prepare_model
from whole_blocked import WholeBlocked, load_native_flow, anchorfree_predictions
from blocked_graph import FrozenGraph


def tensor_leaves(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix: value.cpu().contiguous()}
    entries = value.items() if isinstance(value, dict) else enumerate(value)
    return {name: tensor for key, item in entries
            for name, tensor in tensor_leaves(item, prefix + '/' + str(key)).items()}


def assert_bits(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        left, right = actual[name], expected[name]
        assert left.shape == right.shape and left.dtype == right.dtype, name
        assert torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), name


def nchw(tensor):
    tensor = tensor.cpu()
    if tensor.dim() == 5:
        batch, channels, height, width, lanes = tensor.shape
        return tensor.permute(0, 1, 4, 2, 3).contiguous().view(batch, channels * lanes, height, width)
    return tensor.contiguous()


def error(actual, reference):
    actual, reference = actual.cpu().contiguous(), reference.cpu().contiguous()
    assert actual.shape == reference.shape, (actual.shape, reference.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    difference = (actual.float() - reference.float()).abs()
    return dict(shape=list(actual.shape), dtype=str(actual.dtype), max_abs=difference.max().item(),
                mean_abs=difference.mean().item(), equal_count=(actual == reference).sum().item(),
                count=actual.numel(), bit_equal=(actual.dtype == reference.dtype and
                    torch.equal(actual.reshape(-1).view(torch.uint8), reference.reshape(-1).view(torch.uint8))))


def selection_stages(decoded, max_det, nc):
    boxes, scores = decoded.transpose(1, 2).split((4, nc), -1)
    values, labels = scores.max(-1)
    k = min(max_det, scores.size(1))
    top_values, top_index = values.topk(k, dim=1)
    gathered = scores.gather(1, top_index[..., None].expand(-1, -1, nc))
    values2, index2 = gathered.flatten(1).topk(k, dim=1)
    anchor = top_index.gather(1, index2 // nc)
    selected = boxes.gather(1, anchor[..., None].expand(-1, -1, 4))
    return dict(class_max_values=values, class_max_indices=labels,
                topk1_values=top_values, topk1_indices=top_index, gathered_scores=gathered,
                topk2_values=values2, topk2_indices=index2, candidate_indices=index2 // nc,
                labels_int64=index2 % nc, anchor_indices=anchor, selected_boxes=selected,
                selected_scores=values2[..., None], labels_float32=(index2 % nc)[..., None].float(),
                final=torch.cat((selected, values2[..., None], (index2 % nc)[..., None].float()), -1))


def no_vendor_backend():
    assert not any(name.split('.')[0] == 'torch_npu' for name in sys.modules)
    maps = Path('/proc/self/maps').read_text().lower()
    assert not any(token in maps for token in ('torch_npu', 'aclnn', 'libopcompiler', 'libtbe'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    load_native_flow()
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=False)
    with torch.no_grad():
        model = torch.load(args.model, map_location='cpu', weights_only=False)['model']
        model = model.eval().requires_grad_(False).half().to('npu')
        preparation = prepare_model(model, fuse_silu=True)
        pilot = WholeBlocked(model, attention_scale_before_matmul=True)
        pilot.enable()
    result = dict(cases={}, prepared_convolutions=len(preparation['prepared_layers']))
    no_vendor_backend()
    with torch.inference_mode():
        for name, case in fixture['cases'].items():
            if args.case and name not in args.case:
                continue
            reference = case['fp16_params_ref']
            image = case['image_fp16'].to('npu')
            pilot.validate()
            recorded, layers, handles = {}, {}, []

            def record(target, key):
                def hook(module, inputs, output):
                    target[key] = output
                return hook

            for key, module in model.named_modules():
                if key in reference['intermediates']:
                    handles.append(module.register_forward_hook(record(recorded, key)))
            for module in list(model.model)[:-1]:
                handles.append(module.register_forward_hook(record(layers, str(module.i))))
            try:
                output = model(image)
            finally:
                for handle in handles:
                    handle.remove()
            pilot.validate()
            eager = tensor_leaves(output)
            intermediates = {key: nchw(value) for key, value in recorded.items()}
            layer_values = {key: nchw(value) for key, value in layers.items()}
            decoded_device, raw = anchorfree_predictions(pilot.detect, output[1]['one2one']['feats'])
            decoded = decoded_device.cpu()
            stages = selection_stages(decoded, pilot.detect.max_det, pilot.detect.nc)
            official_same_input = pilot.detect.postprocess(decoded.transpose(1, 2))
            assert torch.equal(stages['final'], official_same_input)
            trace_names = ('final', 'class_max_values', 'topk1_indices', 'gathered_scores',
                           'topk2_values', 'topk2_indices', 'anchor_indices', 'labels_int64')
            trace = dict(zip(trace_names, (tensor.cpu() for tensor in
                torch.ops.thin_native_flow.detection_select_trace(
                    decoded_device.transpose(1, 2), pilot.detect.max_det, pilot.detect.agnostic_nms))))
            assert_bits(tensor_leaves(trace['final']), tensor_leaves(output[0]))
            selected_pairs = set(zip(trace['anchor_indices'].flatten().tolist(), trace['labels_int64'].flatten().tolist()))
            reference_pairs = set(zip(reference['selection']['anchor_indices'].flatten().tolist(),
                                      reference['selection']['labels_int64'].flatten().tolist()))
            row = dict(input_shape=list(image.shape),
                       raw_boxes=error(raw['boxes'], reference['raw_boxes']),
                       raw_scores=error(raw['scores'], reference['raw_scores']),
                       decoded_boxes=error(decoded[:, :4], reference['decoded_dbox']),
                       decoded_scores=error(decoded[:, 4:], reference['decoded_cls']),
                       final=error(output[0], reference['final']),
                       selection_vs_official_same_input=error(output[0], official_same_input),
                       device_selection_vs_reference={key: error(value, reference['selection'][key])
                                                      for key, value in trace.items()},
                       selected_anchor_class_overlap=len(selected_pairs & reference_pairs),
                       selected_anchor_class_count=len(selected_pairs),
                       selection_cpu_diagnostic_vs_reference={key: error(value, reference['selection'][key])
                                                             for key, value in stages.items()},
                       intermediates={key: error(value, reference['intermediates'][key]['out'])
                                      for key, value in intermediates.items()},
                       layers={key: error(value, reference['layers'][key]) for key, value in layer_values.items()})
            graph = FrozenGraph(pilot, model, image)
            assert_bits(tensor_leaves(graph.outputs), eager)
            assert_bits(tensor_leaves(graph.replay()), eager)
            changed = (case['image_fp16'] * 0.5).to('npu')
            changed_eager = tensor_leaves(pilot.run(changed))
            assert_bits(tensor_leaves(graph(changed)), changed_eager)
            assert_bits(tensor_leaves(graph(image)), eager)
            graph.close()
            row.update(eager_capture_bit_equal=True, replay_bit_equal=True,
                       changed_input_replay_bit_equal=True, restored_input_replay_bit_equal=True)
            no_vendor_backend()
            result['cases'][name] = row
            args.output.write_text(json.dumps(result, indent=2))
            print('YOLO26_CAPTURE_REPLAY_BITS_PASS', name, json.dumps(dict(
                raw_boxes=row['raw_boxes'], decoded_boxes=row['decoded_boxes'],
                decoded_scores=row['decoded_scores'], final=row['final'],
                selection_vs_official_same_input=row['selection_vs_official_same_input'])), flush=True)
            torch.save(dict(full_output=eager, intermediates=intermediates, layers=layer_values,
                            image_fp16=case['image_fp16'], decoded=decoded, final=output[0].cpu(),
                            stages_device=trace, stages_cpu_diagnostic=stages,
                            raw_boxes=raw['boxes'].cpu(), raw_scores=raw['scores'].cpu()),
                       args.output.parent / ('device-' + name + '.pt'))
    result['no_torch_npu_aclnn_tbe'] = True
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
