"""Rebuild the statically identified high-precision sigmoid table and check HWX."""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from inspect_hwx import inspect, dense_tasks, kernel_dma


def vm_bytes(data, address, length):
    offset = 32
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        command, size = struct.unpack_from('<II', data, offset)
        if command == 25:
            vm, _, fileoff, filesize = struct.unpack_from('<4Q', data, offset + 24)
            if vm <= address and address + length <= vm + filesize:
                return data[fileoff + address - vm:fileoff + address - vm + length]
        offset += size
    raise ValueError(hex(address))


def sigmoid_lut(compiler_data):
    # GetSigmoidHighPrecisionLut 0x1a8cff990..0x1a8cffa7c.
    table = bytearray(86)
    table[:8] = vm_bytes(compiler_data, 0x1A904EDC8, 8)
    for i in range(33):
        x = np.float32(-i / 4)
        y = np.float32(np.float32(1) / np.float32(np.exp(x) + np.float32(1)))
        struct.pack_into('<e', table, 8 + 2*i, np.float32(y - np.float32(.5)))
    last = np.float32(struct.unpack_from('<e', table, 0x48)[0])
    slope = np.float16(np.float32(np.float32(.5) - last) * np.float32(.5))
    # Fused multiply-add here is exact for these powers-of-two operands.
    intercept = np.float16(np.float32(np.float32(slope) * np.float32(-8) + last))
    struct.pack_into('<eeHH', table, 0x4E, slope, intercept, 2, 0x60)
    return bytes(table) + bytes(128 - len(table))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--hwx', type=Path, required=True)
    parser.add_argument('--ipsw', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compiler, path = args.compiler, args.hwx
    binary = compiler.read_bytes()
    expected = sigmoid_lut(binary)
    raw = path.read_bytes()
    meta = inspect(path)
    tasks = dense_tasks(path, meta['sections'])
    refs = {t['tid']: t['refs'] for t in kernel_dma(meta, tasks)['active_tasks']}
    checked = []
    for task in tasks:
        mode = (task['registers'].get(0x4904, 0) >> 16) & 3
        if mode != 2 or task['tid'] not in refs:
            continue
        for ref in refs[task['tid']]:
            assert raw[ref['fileoff']:ref['fileoff']+128] == expected, (task['tid'], ref)
        checked.append(dict(tid=task['tid'], lane_count=len(refs[task['tid']]),
                            prefix_bytes=128, serialized_lut_bytes=86, alignment_zero_bytes=42,
                            byte_equal_to_static_algorithm=True))
    assert checked
    evidence = {}
    for label, address, count in [('sigmoid_high_precision', 0x1A8CFF990, 60),
                                  ('activation_serialize', 0x1A8CFE9DC, 39),
                                  ('kernel_activation_prefix', 0x1A8E7E7F4, 52)]:
        evidence[label] = subprocess.check_output([str(args.ipsw), 'macho', 'disass', str(compiler),
                                                  '-a', hex(address), '-c', str(count), '-q'], text=True)
    result = dict(compiler_sha256=hashlib.sha256(binary).hexdigest(),
                  model_sha256=hashlib.sha256(raw).hexdigest(),
                  table_hex=expected.hex(), checked=checked, disassembly=evidence,
                  scope='Static bytes and compiler algorithm; no Apple binary execution')
    output = args.output
    assert not output.exists(), 'Never overwrite an existing result'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(layers=len(checked), lanes=sum(x['lane_count'] for x in checked),
                         exact=True, output=str(output))))


if __name__ == '__main__':
    main()
