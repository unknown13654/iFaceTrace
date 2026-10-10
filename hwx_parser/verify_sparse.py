"""Offline H18 sparse reconstruction: full known-control tensors, then face candidate."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
from inspect_hwx import inspect, dense_tasks, kernel_dma
from espresso_inference import EspressoParser
from sparse_decode import decode_sparse, self_check
from npz_io import write_npz


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_model(path, selected=None):
    report = inspect(path)
    tasks = dense_tasks(path, report['sections'])
    dma = kernel_dma(report, tasks if selected is None else [t for t in tasks if t['tid'] in selected])
    return path.read_bytes(), {t['tid']: t for t in tasks}, {t['tid']: t for t in dma['active_tasks']}


def decode(data, task, layer, tile, storage_tile, lane_tiles=None):
    refs = task['refs']
    return decode_sparse(data, [r['fileoff'] for r in refs], [r['fileoff']+r['size'] for r in refs],
                         layer['C'], layer['K'], layer['Ny'], layer['Nx'], tile,
                         depthwise=layer['n_groups'] == layer['K'], spatial='parity2',
                         storage_tile=storage_tile, lane_tiles=lane_tiles)


def compare(actual, expected):
    actual_bits, expected_bits = actual.view('<u2'), expected.view('<u2')
    different = actual_bits != expected_bits
    assert np.all(actual_bits[different] == 0), 'Unexpected nonzero recovered coefficient difference'
    assert np.all((expected_bits[different] & 0x7c00) == 0), 'Difference outside signed-zero/subnormal class'
    return dict(elements=actual.size, exact_bits=int(np.count_nonzero(~different)),
                changed_signed_zero=int(np.count_nonzero(different & ((expected_bits & 0x7fff) == 0))),
                flushed_subnormal=int(np.count_nonzero(different & ((expected_bits & 0x7fff) != 0))))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-dir', type=Path, required=True)
    parser.add_argument('--hwx', type=Path, required=True)
    parser.add_argument('--net', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    self_check()
    base = args.controls_dir
    path = base / 'anodv4_drop6_fp16.H18.espresso.hwx'
    data, tasks, dma = read_model(path)
    parser = EspressoParser(base / 'anodv4_drop6_fp16.espresso.net', base / 'anodv4_drop6_fp16.espresso.weights')
    layers = {l['name']: l for l in parser.layers}
    controls = []
    # Known source tensors choose the unique full-tensor correspondence. No prefix matching.
    for tid, name, tile, storage_tile, lane_tiles in [
            (0, 'Conv_1', 2, 2, [2]*8+[1]*8),
            (3, 'Conv_7', 1, 2, None),
            (38, 'Conv_36', 1, 2, None),
            (73, 'Conv_95', 1, 2, None)]:
        layer = layers[name]
        regs = tasks[tid]['registers']
        assert tile == 1 << (regs[0x40] & 7)
        assert storage_tile == max(tile, 2 << ((regs[0x3c] >> 2) & 3))
        w, b, coverage = decode(data, dma[tid], layer, tile, storage_tile, lane_tiles)
        offset, size, _ = parser.blob_map[layer['blob_weights_f16']]
        expected = np.frombuffer(parser.weights_raw[offset:offset+size], '<f2').reshape(w.shape)
        offset, size, _ = parser.blob_map[layer['blob_biases']]
        bias = np.frombuffer(parser.weights_raw[offset:offset+size], '<f4').astype('<f2')
        controls.append(dict(tid=tid, layer=name, shape=list(w.shape), weight=compare(w, expected),
                             bias=compare(b, bias), coverage=coverage, tile=tile, storage_tile=storage_tile,
                             lane_tiles=lane_tiles, weight_sha256=sha(w.tobytes()), bias_sha256=sha(b.tobytes())))
    extra_path = base / 'solo_landmarks_s9min6ugm8_opt.H18.espresso.hwx'
    extra_data, extra_tasks, extra_dma = read_model(extra_path, {16, 18})
    extra_parser = EspressoParser(base / 'solo_landmarks_s9min6ugm8_opt.espresso.net', base / 'solo_landmarks_s9min6ugm8_opt.espresso.weights')
    for tid, suffix in [(16, 'block8/conv3x3/Conv2D'), (18, 'block9/conv3x3/Conv2D')]:
        layer = next(l for l in extra_parser.layers if l['name'].endswith(suffix))
        tile = layer['C'] // 16
        regs = extra_tasks[tid]['registers']
        assert tile == 1 << (regs[0x40] & 7)
        assert max(tile, 2 << ((regs[0x3c] >> 2) & 3)) == 4
        w, b, coverage = decode(extra_data, extra_dma[tid], layer, tile, 4)
        offset, size, _ = extra_parser.blob_map[layer['quantization_lut_weights_blob']]
        indices = np.frombuffer(extra_parser.weights_raw[offset:offset+size], 'u1').reshape(layer['C'], -1)
        offset, size, _ = extra_parser.blob_map[layer['quantization_ranges_blob']]
        ranges = np.frombuffer(extra_parser.weights_raw[offset:offset+size], '<f4')
        assert len(ranges) == layer['C'] * 2
        lo, hi = np.split(ranges, 2)
        scale = (hi-lo) / np.float32(255)
        expected = (lo[:, None] + indices * scale[:, None]).astype('<f2').reshape(w.shape)
        offset, size, _ = extra_parser.blob_map[layer['blob_biases']]
        bias = np.frombuffer(extra_parser.weights_raw[offset:offset+size], '<f4').astype('<f2')
        controls.append(dict(tid=tid, layer=layer['name'], model=extra_path.name, shape=list(w.shape),
                             weight=compare(w, expected), bias=compare(b, bias), coverage=coverage,
                             tile=tile, storage_tile=4, quantized_source='FP32 scale=(hi-lo)/255; FP32 lo+index*scale; cast FP16',
                             weight_sha256=sha(w.tobytes()), bias_sha256=sha(b.tobytes())))
    target = args.hwx
    data, tasks, dma = read_model(target)
    net = json.loads(args.net.read_text())['layers']
    layers = {l['name']: l for l in net}
    mapping = json.loads(args.mapping.read_text())['face_mapping']
    by_tid = {m['tid']: m['layer'] for m in mapping}
    reports, tensors = [], {}
    for tid, task in dma.items():
        if not task['sparse']:
            continue
        assert not task['palettized']
        name = by_tid.get(tid)
        # Auxiliary-head mapping is not inferred from dimensions. Export task coordinates separately.
        layer = layers[name] if name else dict(C=task['output_channels'], K=task['input_channels'],
                                               Ny=3, Nx=3, n_groups=task['input_channels'])
        regs = tasks[tid]['registers']
        assert regs[0x4904] & 16, 'Bias prefix must be enabled'
        assert (regs[0x4904] >> 16) & 3 == 1, 'Nonlinear LUT prefix not supported in this sparse fixture'
        tile = 1 << (regs[0x40] & 7)
        small_source = (regs[0x3c] >> 2) & 3
        assert small_source in (0, 1)
        storage_tile = max(tile, 2 << small_source)
        assert regs[0x28] & 0xfff == 0xc3
        w, b, coverage = decode(data, task, layer, tile, storage_tile)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        key = name or f'task_{tid}'
        tensors[key+'.weight'] = w
        tensors[key+'.bias'] = b
        reports.append(dict(tid=tid, layer=name, shape=list(w.shape), weights=w.size, tile=tile,
                            storage_tile=storage_tile, spatial='parity2', has_bias=True,
                            small_source=small_source, ne_config=regs[0x40], mac_config=regs[0x3c],
                            weight_sha256=sha(w.tobytes()), bias_sha256=sha(b.tobytes()), coverage=coverage,
                            bit_identical_roundtrip=True, independent_portable_target_comparison=False))
    assert len(reports) == 6 and sum(r['layer'] is not None for r in reports) == 4
    out = dict(control_hwx=str(path), control_hwx_sha256=sha(path.read_bytes()),
               control_weights_sha256=sha(parser.weights_raw), controls=controls,
               extra_control_hwx=str(extra_path), extra_control_hwx_sha256=sha(extra_data),
               extra_control_weights_sha256=sha(extra_parser.weights_raw),
               target_hwx=str(target), target_hwx_sha256=sha(data), target_sparse=reports,
               note='Compiled effective tensors; omitted source signed-zero/subnormal values are unrecoverable.')
    (args.output_dir / 'sparse_results.json').write_text(json.dumps(out, indent=2)+'\n')
    write_npz(args.output_dir / 'sparse_tensors.npz', tensors)
    print(json.dumps(dict(control_layers=len(controls), control_weights=sum(c['weight']['elements'] for c in controls),
                          control_weight_exact=sum(c['weight']['exact_bits'] for c in controls),
                          target_layers=len(reports), target_face_layers=sum(r['layer'] is not None for r in reports),
                          target_weights=sum(r['weights'] for r in reports)), indent=2))


if __name__ == '__main__':
    main()
