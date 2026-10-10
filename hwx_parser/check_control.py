"""Read-only bounded weight-sequence search; stdout is the evidence report."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from espresso_inference import EspressoParser


def matches(data, payload):
    # ponytail: sample <= 17 windows per blob; exhaustive layout recovery is separate.
    hits = []
    if len(data) < 32:
        return hits
    starts = sorted({int(v) // 2 * 2 for v in np.linspace(0, len(data) - 32, 17)})
    for start in starts:
        needle = data[start:start + 32]
        if len(set(needle)) < 12:
            continue
        pos = payload.find(needle)
        if pos >= 0:
            length = 32
            while start + length < len(data) and pos + length < len(payload) and data[start + length] == payload[pos + length]:
                length += 1
            hits.append({'source_offset': start, 'hwx_offset': pos, 'bytes': length, 'hex32': needle.hex()})
    return hits


def main():
    assert matches(bytes(range(64)), b'prefix' + bytes(range(64)))[0]['hwx_offset'] == 6
    assert not matches(bytes(64), bytes(128))
    root = Path(sys.argv[1])
    results = []
    for hwx in sorted(root.rglob('*.H18.espresso.hwx')):
        weights = hwx.with_name(hwx.name.replace('.H18.espresso.hwx', '.espresso.weights'))
        net = weights.with_suffix('.net')
        if not weights.exists() or not net.exists():
            continue
        parser = EspressoParser(net, weights)
        raw = hwx.read_bytes()
        assert max(o + n for o, n, _ in parser.blob_map.values()) == len(parser.weights_raw)
        entry = {'hwx': str(hwx), 'weights': str(weights), 'hwx_sha256': hashlib.sha256(raw).hexdigest(), 'weights_sha256': hashlib.sha256(parser.weights_raw).hexdigest(), 'blobs': len(parser.blob_map), 'probes': 0, 'hits': []}
        for layer in parser.layers:
            for key, index in layer.items():
                if key not in ('blob_weights_f16', 'blob_weights', 'blob_biases'):
                    continue
                offset, size, _ = parser.blob_map[index]
                data = parser.weights_raw[offset:offset + size]
                variants = [('raw', data)]
                if key != 'blob_weights_f16' and size % 4 == 0:
                    with np.errstate(over='ignore', invalid='ignore'):
                        variants.append(('float32_to_float16', np.frombuffer(data, dtype='<f4').astype('<f2').tobytes()))
                for transform, candidate in variants:
                    entry['probes'] += 1
                    found = matches(candidate, raw)
                    if found:
                        entry['hits'].append({'layer': layer['name'], 'field': key, 'blob': index, 'blob_file_offset': offset, 'blob_bytes': size, 'transform': transform, 'matches': found})
        results.append(entry)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
