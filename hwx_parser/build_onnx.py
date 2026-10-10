"""Offline graph smoke test, not Apple execution or FPV15 output equivalence.

Requires every face-branch tensor. Never fills missing weights with zeros.
Uses the existing ONNX builder with plain float32 arithmetic; no ANE emulation.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from espresso_converter import OnnxGraphBuilder


def face_layers(layers):
    producers = {t.strip(): i for i, layer in enumerate(layers)
                 for t in layer['top'].split(',')}
    needed, pending = set(), ['flatten_output']
    while pending:
        index = producers.get(pending.pop())
        if index is None or index in needed:
            continue
        needed.add(index)
        pending.extend(t.strip() for t in layers[index]['bottom'].split(','))
    return [layers[i] for i in sorted(needed)]


def build(net, archive):
    layers = face_layers(net['layers'])
    blobs = {}
    for layer in layers:
        kind = layer['type']
        if kind == 'convolution':
            name = layer['name']
            shape = (layer['C'], layer['K'] // layer['n_groups'], layer['Ny'], layer['Nx'])
            weight = archive[name + '.weight']
            assert weight.shape == shape and np.isfinite(weight).all(), name
            blobs[layer['blob_weights_f16']] = weight.astype(np.float32)
            if layer['has_biases']:
                bias = archive[name + '.bias']
                assert bias.shape == (shape[0],) and np.isfinite(bias).all(), name
                blobs[layer['blob_biases']] = bias.astype(np.float32)
            assert not layer.get('has_batch_norm') and not layer.get('fused_tanh')
        elif kind == 'pool':
            assert layer.get('is_global') == 1 and layer.get('avg_or_max') == 0
        elif kind == 'activation':
            assert layer['mode'] == 3 and layer.get('beta', 0) == 0
        elif kind == 'elementwise':
            assert layer['operation'] in (0, 1, 2)
        else:
            raise ValueError('Unsupported face operation: ' + kind)
    parser = SimpleNamespace(layers=layers, read_blob=lambda index, **_: blobs[index].reshape(-1))
    builder = OnnxGraphBuilder(parser, ['flatten_output'], fp16_sim=False)
    builder.build()
    graph = helper.make_graph(builder.nodes, 'recovered_compiled_face_candidate',
                              [helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 128, 128])],
                              [helper.make_tensor_value_info('flatten_output', TensorProto.FLOAT, [1, 128, 1, 1])],
                              builder.initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model, len(layers), sum(l['type'] == 'convolution' for l in layers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--net', type=Path, required=True, help='Copied .espresso.net; adjacent copied .shape is required')
    parser.add_argument('--onnx', type=Path, required=True, help='New output path')
    parser.add_argument('--shape', type=Path, required=True, help='Explicit copied .espresso.shape')
    args = parser.parse_args()
    source = args.net
    assert not args.onnx or not args.onnx.exists(), 'Do not overwrite an existing model'
    with np.load(args.weights, allow_pickle=False) as weights:
        model, count, weighted = build(json.loads(source.read_bytes()), weights)
    assert (count, weighted) == (308, 175)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    known = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
             for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    declared = json.loads(args.shape.read_bytes())['layer_shapes']
    compared = [(name, known[name], [shape[k] for k in ('n', 'k', 'h', 'w')])
                for name, shape in declared.items() if name in known]
    assert len(compared) == 308 and all(got == expected for _, got, expected in compared)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(20261002)
    inputs = {'zeros': np.zeros((1, 3, 128, 128), np.float32),
              'ones': np.ones((1, 3, 128, 128), np.float32),
              'random_0_1': rng.random((1, 3, 128, 128), dtype=np.float32)}
    outcomes = {}
    for name, data in inputs.items():
        out = session.run(None, {'data': data})[0]
        assert out.shape == (1, 128, 1, 1) and np.isfinite(out).all(), name
        outcomes[name] = dict(shape=list(out.shape), l2=float(np.linalg.norm(out)),
                              minimum=float(out.min()), maximum=float(out.max()),
                              sha256=hashlib.sha256(out.tobytes()).hexdigest())
    assert outcomes['zeros']['sha256'] != outcomes['random_0_1']['sha256'], 'Input-independent output'
    if args.onnx:
        onnx.save_model(model, args.onnx)
    print(json.dumps(dict(graph_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          arrays_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                          layers=count, weighted_layers=weighted, matching_intermediate_shapes=len(compared),
                          arithmetic='host_float32',
                          apple_output_equivalence_verified=False, results=outcomes), indent=2))


if __name__ == '__main__':
    main()
