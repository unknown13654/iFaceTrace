"""Prove complete output-filter recovery in known-control H18 HWX files."""
import json
import sys
from pathlib import Path

import numpy as np

from espresso_inference import EspressoParser
from inspect_hwx import inspect


def recover(weights, biases, data):
    locations = []
    for row, bias in zip(weights, biases):
        needle = row.tobytes()
        if len(needle) < 16 or len(set(needle)) < 8:
            return None
        found = data.find(needle)
        if found < 2 or data[found - 2:found] != bias.tobytes():
            return None
        locations.append(found)
    reconstructed = np.stack([np.frombuffer(data[p:p + weights.shape[1] * 2], dtype='<f2') for p in locations])
    assert reconstructed.tobytes() == weights.tobytes()
    return locations


def main():
    test_w = np.arange(16, dtype='<f2').reshape(1, -1)
    test_b = np.array([1.25], dtype='<f2')
    assert recover(test_w, test_b, test_b.tobytes() + test_w.tobytes()) == [2]
    reports = []
    for path in sorted(Path(sys.argv[1]).rglob('*.H18.espresso.hwx')):
        weights = path.with_name(path.name.replace('.H18.espresso.hwx', '.espresso.weights'))
        net = weights.with_suffix('.net')
        if not weights.exists() or not net.exists():
            continue
        parser = EspressoParser(net, weights)
        data = path.read_bytes()
        try:
            sections = inspect(path)
        except (AssertionError, StopIteration) as exc:
            sections = {'groups': [], 'inspection_error': type(exc).__name__}
        layers = []
        for layer in parser.layers:
            key = 'blob_weights_f16' if 'blob_weights_f16' in layer else 'blob_weights'
            if layer['type'] != 'convolution' or key not in layer or 'blob_biases' not in layer:
                continue
            o, n, _ = parser.blob_map[layer[key]]
            source = np.frombuffer(parser.weights_raw[o:o+n], dtype='<f2' if key.endswith('f16') else '<f4').astype('<f2')
            if len(source) != layer['C'] * layer['K'] * layer['Nx'] * layer['Ny'] // layer['n_groups']:
                continue
            o, n, _ = parser.blob_map[layer['blob_biases']]
            biases = np.frombuffer(parser.weights_raw[o:o+n], dtype='<f4').astype('<f2')
            if len(biases) != layer['C']:
                continue
            matrix = source.reshape(layer['C'], -1)
            locations = recover(matrix, biases, data)
            if locations is None:
                continue
            groups = sorted({g['name'] for g in sections['groups'] for p in locations if g['start'] <= p < g['start'] + g['span']})
            layers.append(dict(name=layer['name'], channels=layer['C'], input_channels=layer['K'], kernel=[layer['Ny'], layer['Nx']], groups=layer['n_groups'], weights_bytes=source.nbytes, bias_bytes=biases.nbytes, coefficient_offsets=locations, symbol_groups=groups, exact_all_weights_and_fp16_biases=True))
        reports.append(dict(model=path.name, recovered_layers=layers, inspection_error=sections.get('inspection_error')))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
