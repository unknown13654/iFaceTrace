"""Known-control proof of H18 dense lane layout, without coefficient-byte search."""
import json
import sys
from pathlib import Path

import numpy as np

from espresso_inference import EspressoParser
from inspect_hwx import inspect, dense_tasks


def decode_dense(data, lane_offsets, outputs, input_elements, tile, has_bias=True, prefix_bytes=0):
    """Each lane: optional LUT prefix, then bias/weight tiles in source order.

    The caller must verify register modes, mapped shape, lane lengths and prefix.
    This routine does not infer sparse/quantized formats or compiler fusion.
    """
    lanes = len(lane_offsets)
    assert lanes > 0 and tile > 0 and input_elements > 0 and prefix_bytes >= 0
    assert outputs > 0 and outputs % (lanes * tile) == 0
    lane_bytes = prefix_bytes + outputs // lanes * (input_elements + int(has_bias)) * 2
    assert all(0 <= start and start + lane_bytes <= len(data) for start in lane_offsets)
    weights = np.empty((outputs, input_elements), dtype='<f2')
    biases = np.zeros(outputs, dtype='<f2')
    for lane, start in enumerate(lane_offsets):
        pos = start + prefix_bytes
        for block in range(outputs // (lanes * tile)):
            first = (block * lanes + lane) * tile
            if has_bias:
                biases[first:first+tile] = np.frombuffer(data[pos:pos+tile*2], dtype='<f2')
                pos += tile * 2
            length = input_elements * tile * 2
            weights[first:first+tile] = np.frombuffer(data[pos:pos+length], dtype='<f2').reshape(input_elements, tile).T
            pos += length
    return weights, biases


def main():
    data = np.array([2, 3, 10, 11, 20, 21], dtype='<f2').tobytes()
    w, b = decode_dense(data, [0], 2, 2, 2)
    assert w.tolist() == [[10, 20], [11, 21]] and b.tolist() == [2, 3]
    pw, pb = decode_dense(bytes(128) + data, [0], 2, 2, 2, prefix_bytes=128)
    assert pw.tobytes() == w.tobytes() and pb.tobytes() == b.tobytes()
    report = []
    for hwx in sorted(Path(sys.argv[1]).rglob('*.H18.espresso.hwx')):
        wp = hwx.with_name(hwx.name.replace('.H18.espresso.hwx', '.espresso.weights'))
        if not wp.exists():
            continue
        try:
            meta = inspect(hwx)
        except (AssertionError, StopIteration):
            report.append(dict(model=hwx.name, skipped='existing inspect parser rejects symbols'))
            continue
        parser = EspressoParser(wp.with_suffix('.net'), wp)
        raw = hwx.read_bytes()
        kernel = next(s for s in meta['sections'] if s['name'] == '__kern_0')
        task_by_start = {}
        for task in dense_tasks(hwx, meta['sections']):
            regs = task['registers']
            if regs.get(0x5520, 0) & 1:
                start = kernel['fileoff'] + regs.get(0x5510, 0) + regs.get(0x5560, 0)
                task_by_start[start] = task
        found = []
        checked = 0
        for layer in parser.layers:
            key = 'blob_weights_f16' if 'blob_weights_f16' in layer else 'blob_weights'
            if layer['type'] != 'convolution' or key not in layer:
                continue
            c = layer['C']
            n = layer['K'] * layer['Nx'] * layer['Ny'] // layer['n_groups']
            offset, size, _ = parser.blob_map[layer[key]]
            expected_w = np.frombuffer(parser.weights_raw[offset:offset+size], dtype='<f2' if key.endswith('f16') else '<f4').astype('<f2')
            if expected_w.size != c*n:
                continue
            has_bias = 'blob_biases' in layer
            if has_bias:
                offset, size, _ = parser.blob_map[layer['blob_biases']]
                expected_b = np.frombuffer(parser.weights_raw[offset:offset+size], dtype='<f4').astype('<f2')
            else:
                expected_b = np.zeros(c, dtype='<f2')
            if expected_b.size != c:
                continue
            checked += 1
            expected_w = expected_w.reshape(c, n)
            for group in meta['groups']:
                symbols = [s for s in meta['kernel_symbols'] if s['name'].split('_ne_')[0] == group['name']]
                starts = sorted(set(s['fileoff'] for s in symbols))
                lanes = len(starts)
                # Duplicate symbol addresses alias a lane; they do not add filters.
                if lanes > 16 or c % lanes:
                    continue
                task = task_by_start.get(starts[0])
                nl_mode = ((task['registers'].get(0x4904, 0) >> 16) & 3) if task else 0
                prefix = 128 if nl_mode == 2 else 0
                if prefix and len({raw[s:s+prefix] for s in starts}) != 1:
                    continue
                per_lane = prefix + c // lanes * (n + int(has_bias)) * 2
                stride = (per_lane + 63) // 64 * 64
                if starts != [starts[0] + i*stride for i in range(lanes)] or group['span'] != lanes*stride:
                    continue
                for tile in (1, 2, 4, 8, 16, 32):
                    if c % (lanes*tile):
                        continue
                    if has_bias and raw[starts[0]+prefix:starts[0]+prefix+tile*2] != expected_b[:tile].tobytes():
                        continue
                    w, b = decode_dense(raw, starts, c, n, tile, has_bias, prefix)
                    if w.tobytes() != expected_w.tobytes() or b.tobytes() != expected_b.tobytes():
                        continue
                    ne_config = task['registers'].get(0x40) if task else None
                    if ne_config is not None:
                        assert tile == 1 << (ne_config & 7), (layer['name'], tile, ne_config)
                    found.append(dict(layer=layer['name'], group=group['name'], first_lane=starts[0], lane_stride=stride, lanes=lanes,
                                      output_channels=c, input_elements=n, groups=layer['n_groups'], tile=tile,
                                      weight_values=c*n, bias_values=c if has_bias else 0, has_bias=has_bias, exact_bits=True,
                                      prefix_bytes=prefix, nonlinear_mode=nl_mode,
                                      ne_config=ne_config, explicit_tile_agrees=ne_config is not None,
                                      consumed_per_lane=per_lane, padding_per_lane=stride-per_lane,
                                      padding_nonzero_bytes=sum(sum(v != 0 for v in raw[s+per_lane:s+stride]) for s in starts)))
        report.append(dict(model=hwx.name, eligible_layers=checked, matches=found))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
