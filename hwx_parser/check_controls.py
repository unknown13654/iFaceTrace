"""Run the unchanged dense-lane checker on hash-bound copied control resources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-dir', type=Path, required=True, help='Copied six-model resource directory')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report; never overwrite')
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name('INPUTS.json').read_text(encoding='utf-8'))['controls']
    for name, entry in pins.items():
        path = args.controls_dir / name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            parser.error(name + ' input SHA-256 mismatch')
    if args.output.exists():
        parser.error('Never overwrite an existing report')
    command = [sys.executable, '-B', str(Path(__file__).with_name('verify_lane_layout.py')), str(args.controls_dir)]
    completed = subprocess.run(command, check=True, capture_output=True)
    models = json.loads(completed.stdout)
    selected = [model for model in models if model.get('matches')]
    layers = [dict(model=model['model'], **layer) for model in selected for layer in model['matches']]
    weights = sum(layer['weight_values'] for layer in layers)
    biases = sum(layer['bias_values'] for layer in layers)
    compared_bytes = 2 * (weights + biases)
    assert (len(selected), len(layers), weights, biases, compared_bytes) == (6, 148, 3052688, 11915, 6129206)
    assert all(layer['exact_bits'] for layer in layers)
    result = {'models': models, 'layers': layers, 'selected_models': len(selected), 'selected_layers': len(layers),
              'weight_values': weights, 'bias_values': biases, 'matching_fp16_bytes': compared_bytes,
              'scope': 'Selected full convolution tensors; FP32 sources rounded to FP16; not native ANE execution',
              'native_execution': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ('selected_models', 'selected_layers', 'weight_values', 'bias_values', 'matching_fp16_bytes', 'native_execution')}, indent=2))


if __name__ == '__main__':
    main()
