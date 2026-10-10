"""Static-derived H18 sparse streams. No Apple binary execution."""
import json
import sys
from pathlib import Path

import numpy as np


def decode_stream(data, start, limit, bitmap_bytes=2):
    """Return FP16 bits and end; LE24 length excludes its own three bytes."""
    assert 0 <= start <= start + 3 <= limit <= len(data)
    length = int.from_bytes(data[start:start+3], 'little')
    pos, end = start + 3, start + 3 + length
    assert end <= limit
    values = []
    while pos < end:
        assert pos + bitmap_bytes <= end
        mask = int.from_bytes(data[pos:pos+bitmap_bytes], 'little')
        pos += bitmap_bytes
        for bit in range(bitmap_bytes * 8):
            if mask >> bit & 1:
                assert pos + 2 <= end
                values.append(int.from_bytes(data[pos:pos+2], 'little'))
                pos += 2
            else:
                values.append(0)
    assert pos == end
    return np.array(values, dtype='<u2'), end


def encode_stream(values, bitmap_bytes=2):
    """Canonical inverse for retained nonzero FP16 bits; prove exact round trips."""
    payload = bytearray()
    width = bitmap_bytes * 8
    for start in range(0, len(values), width):
        block = values[start:start+width]
        mask = sum(1 << bit for bit, value in enumerate(block) if value)
        payload += mask.to_bytes(bitmap_bytes, 'little')
        for value in block:
            if value:
                payload += int(value).to_bytes(2, 'little')
    assert len(payload) < 1 << 24
    return len(payload).to_bytes(3, 'little') + payload


def decode_sparse(data, starts, ends, outputs, inputs, height, width, tile,
                  depthwise=False, spatial='plain', has_bias=True, storage_tile=None,
                  lane_tiles=None):
    """Return OIHW weights, bias and byte coverage. Each tile has one stream."""
    lanes = len(starts)
    lane_tiles = [tile] * lanes if lane_tiles is None else lane_tiles
    assert len(ends) == len(lane_tiles) == lanes and outputs % sum(lane_tiles) == 0
    assert spatial in ('plain', 'parity2')
    storage_tile = tile if storage_tile is None else storage_tile
    assert storage_tile >= max(lane_tiles)
    channels = 1 if depthwise else inputs
    ph = (height + 1) // 2 * 2 if spatial == 'parity2' else height
    pw = (width + 1) // 2 * 2 if spatial == 'parity2' else width
    count = channels * ph * pw * storage_tile
    weights = np.empty((outputs, channels, height, width), dtype='<u2')
    biases = np.zeros(outputs, dtype='<u2')
    coverage = []
    for lane, (start, limit) in enumerate(zip(starts, ends)):
        pos = start
        records = []
        real_tile = lane_tiles[lane]
        for block in range(outputs // sum(lane_tiles)):
            first = block * sum(lane_tiles) + sum(lane_tiles[:lane])
            begin = pos
            if has_bias:
                assert pos + real_tile * 2 <= limit
                biases[first:first+real_tile] = np.frombuffer(data[pos:pos+real_tile*2], '<u2')
                pos += real_tile * 2
            stream_start = pos
            flat, pos = decode_stream(data, pos, limit)
            assert count <= len(flat) < count + 16, (len(flat), count)
            assert not np.any(flat[count:]), 'Nonzero terminal bitmap padding'
            packed = flat[:count].reshape(channels, ph * pw, storage_tile)
            assert not np.any(packed[:, :, real_tile:]), 'Nonzero output-channel padding'
            used = set()
            for y in range(height):
                for x in range(width):
                    index = y * width + x if spatial == 'plain' else ((y % 2) * 2 + x % 2) * (ph // 2) * (pw // 2) + (y // 2) * (pw // 2) + x // 2
                    used.add(index)
                    weights[first:first+real_tile, :, y, x] = packed[:, index, :real_tile].T
            unused = sorted(set(range(ph * pw)) - used)
            assert not unused or not np.any(packed[:, unused, :]), 'Nonzero spatial padding'
            inverse = np.zeros_like(packed)
            for y in range(height):
                for x in range(width):
                    index = y * width + x if spatial == 'plain' else ((y % 2) * 2 + x % 2) * (ph // 2) * (pw // 2) + (y // 2) * (pw // 2) + x // 2
                    inverse[:, index, :real_tile] = weights[first:first+real_tile, :, y, x].T
            assert encode_stream(inverse.ravel()) == data[stream_start:pos], 'Sparse tensor byte round trip failed'
            records.append(dict(start=begin, end=pos, coefficient_slots=len(flat), spatial_padding_slots=channels*len(unused)*storage_tile, output_padding_slots=channels*height*width*(storage_tile-real_tile), exact_reencode=True))
        assert pos <= limit
        padding = data[pos:limit]
        assert not any(padding), 'Nonzero lane tail'
        coverage.append(dict(lane=lane, start=start, end=limit, consumed=pos-start, padding=len(padding), records=records))
    return weights.view('<f2'), biases.view('<f2'), coverage


def self_check():
    raw = bytes.fromhex('0400000500003c0040')
    try:
        decode_stream(raw, 0, len(raw))
    except AssertionError:
        pass
    else:
        raise AssertionError('Accepted truncated stream')
    raw = bytes.fromhex('0600000500003c0040')
    result, end = decode_stream(raw, 0, len(raw))
    assert end == 9 and result[:3].tolist() == [0x3c00, 0, 0x4000]


if __name__ == '__main__':
    self_check()
    print('sparse stream self-check PASS')
