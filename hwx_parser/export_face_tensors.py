"""Export compiled FP16 coefficients; never fill unresolved layers with zeros.

The graph-to-task assignment is conditional on the documented one-to-one model.
An export is not proof that this candidate is Photos' native FPV15 generator.
"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from verify_lane_layout import decode_dense
from verify_sigmoid_prefix import sigmoid_lut
from sparse_decode import decode_sparse, self_check as sparse_self_check
from inspect_hwx import inspect, dense_tasks, kernel_dma
from freeze_inputs import copy_input
from npz_io import write_npz


def digest(data):
    return hashlib.sha256(data).hexdigest()


def default_evidence(binary, ipsw):
    """Read the static table. Application to H18 remains a stated hypothesis."""
    raw = binary.read_bytes()
    address, size = 0x1A8FF3730, 0x890
    magic, _, _, _, ncmd, _, _, _ = struct.unpack_from('<8I', raw)
    assert magic == 0xFEEDFACF
    pos = 32
    for _ in range(ncmd):
        command, length = struct.unpack_from('<II', raw, pos)
        if command == 0x19:
            vmaddr, vmsize, fileoff, filesize = struct.unpack_from('<4Q', raw, pos + 24)
            if vmaddr <= address and address + size <= vmaddr + filesize:
                offset = fileoff + address - vmaddr
                table = raw[offset:offset+size]
                break
        pos += length
    else:
        raise AssertionError('Default table is not file backed')
    pairs = list(struct.iter_unpack('<II', table))
    assert len(pairs) == 274 and len(dict(pairs)) == 274
    values = dict(pairs)
    assert values[0x4900] == 0x82 and values[0xC] == values[0x1C] == 16
    records = {}
    for name, start, count in [('table_initializer', 0x1A7E4A750, 15),
                               ('initializer_branch', 0x1A7E4A350, 7)]:
        records[name] = subprocess.check_output(
            [str(ipsw), 'macho', 'disass', str(binary), '-a', hex(start), '-c', str(count), '-q'], text=True)
    return dict(binary_sha256=digest(raw), table_address=hex(address), table_bytes=size,
                table_sha256=digest(table), registers={hex(k): v for k, v in pairs},
                code=records, application='Static subtype table; H18 selection chain not yet independently proved')


def prepare(output, inputs):
    """Copy the explicit, hash-checked inputs into a new output directory."""
    receipt = output / 'input_receipt.json'
    assert not receipt.exists()
    records = [copy_input(source, output / 'inputs' / name) for name, source in inputs.items()]
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(records, stream, indent=2)


def build(base, ipsw):
    stem = 'facerec_fp4.0_fa1.4_md2_2_fp16'
    net_path, hwx_path = base / (stem + '.espresso.net'), base / (stem + '_dev_op.H18.espresso.hwx')
    net_bytes, raw = net_path.read_bytes(), hwx_path.read_bytes()
    net = json.loads(net_bytes)['layers']
    layers = {l['name']: l for l in net}
    mapping_bytes = (base / 'face_mapping.json').read_bytes()
    mapping = json.loads(mapping_bytes)
    assert mapping['status'] == 'sat' and mapping['alternate_face_assignment'] == 'unsat'
    assert len(mapping['face_mapping']) == mapping['face_weight_layers'] == 175
    meta = inspect(hwx_path)
    tasks = dense_tasks(hwx_path, meta['sections'])
    by_tid = {t['tid']: t for t in tasks}
    dma = kernel_dma(meta, tasks)['active_tasks']
    producer = {top: l['name'] for l in net for top in l['top'].split(',')}
    ancestors = set()
    def include(name):
        if name in ancestors:
            return
        ancestors.add(name)
        for bottom in layers[name]['bottom'].split(','):
            if bottom in producer:
                include(producer[bottom])
    include(producer['flatten_output'])
    weighted = {n for n in ancestors if 'blob_weights_f16' in layers[n]}
    assert weighted == {m['layer'] for m in mapping['face_mapping']}
    assert len({m['tid'] for m in mapping['face_mapping']}) == 175
    arrays, records, unresolved = {}, [], []
    sparse_self_check()
    sparse_report_path = base / 'sparse_results.json'
    sparse_report = json.loads(sparse_report_path.read_text())
    assert sparse_report['target_hwx_sha256'] == digest(raw)
    sparse_specs = {x['layer']: x for x in sparse_report['target_sparse'] if x['layer']}
    compiler = base / 'ANECompiler'
    expected_lut = sigmoid_lut(compiler.read_bytes())
    defaults = default_evidence(compiler, ipsw)
    for m in mapping['face_mapping']:
        name = m['layer']
        l, d = layers[name], dma[m['task_position']]
        assert d['tid'] == m['tid']
        regs = by_tid[d['tid']]['registers']
        refs = sorted(d['refs'], key=lambda x: x['lane'])
        assert [r['lane'] for r in refs] == list(range(len(refs)))
        c, k, h, w, groups = l['C'], l['K'], l['Ny'], l['Nx'], l['n_groups']
        assert k % groups == 0
        has_bias = bool(l['has_biases'])
        assert has_bias == ('blob_biases' in l)
        assert 0x4904 in regs and bool(regs[0x4904] & 16) == has_bias, 'Graph/compiled bias flag differs'
        count = k // groups * h * w
        starts, ends = [r['fileoff'] for r in refs], [r['fileoff']+r['size'] for r in refs]
        shape = [c, k // groups, h, w]
        record = dict(**m, weight_shape=shape, has_bias=has_bias,
                      blob_weights_f16=l['blob_weights_f16'], blob_biases=l.get('blob_biases'),
                      kernel_config_written=d['kernel_config_written'],
                      kernel_config_explicit=d['kernel_config'] if d['kernel_config_written'] else None,
                      sparse_explicit=d['sparse'], mac_config=regs[0x4904],
                      graph_compiled_bias_agreement=True, refs=refs)
        records.append(record)
        try:
            if d['sparse']:
                spec = sparse_specs[name]
                assert spec['tid'] == d['tid'] and spec['shape'] == shape
                assert spec['has_bias'] == has_bias and d['kernel_config_written']
                tile = 1 << (regs[0x40] & 7)
                small_source = (regs[0x3c] >> 2) & 3
                assert small_source in (0, 1)
                storage_tile = max(tile, 2 << small_source)
                assert tile == spec['tile'] and storage_tile == spec['storage_tile']
                assert (regs[0x4904] >> 16) & 3 == 1
                weight, bias, coverage = decode_sparse(
                    raw, starts, ends, c, k, h, w, tile, depthwise=groups == k == c,
                    spatial=spec['spatial'], has_bias=has_bias, storage_tile=storage_tile)
                assert digest(weight.tobytes()) == spec['weight_sha256']
                assert digest(bias.tobytes()) == spec['bias_sha256']
                assert coverage == spec['coverage']
                record.update(format='sparse_fp16', tile=tile, storage_tile=storage_tile,
                              spatial=spec['spatial'], small_source=small_source, coverage=coverage)
            else:
                nl_mode = (regs.get(0x4904, 0) >> 16) & 3
                prefix = 128 if nl_mode == 2 else 0
                assert c % len(starts) == 0
                ne_config = regs.get(0x40, defaults['registers']['0x40'])
                tile = 1 << (ne_config & 7)
                assert c % (len(starts) * tile) == 0
                consumed = prefix + c // len(starts) * (count + int(has_bias)) * 2
                assert all(consumed <= end-start for start,end in zip(starts,ends)), 'Insufficient coefficient buffer'
                assert all(not any(raw[start+consumed:end]) for start,end in zip(starts,ends)), 'Nonzero unexplained lane tail'
                if prefix:
                    assert all(raw[start:start+prefix] == expected_lut for start in starts), 'Static sigmoid LUT differs'
                weight, bias = decode_dense(raw, starts, c, count, tile, has_bias, prefix)
                for lane, start in enumerate(starts):
                    encoded = bytearray(expected_lut if prefix else b'')
                    for block in range(c // (len(starts) * tile)):
                        first = (block * len(starts) + lane) * tile
                        if has_bias:
                            encoded.extend(bias[first:first+tile].tobytes())
                        encoded.extend(weight[first:first+tile].T.copy().tobytes())
                    assert len(encoded) == consumed
                    assert encoded == raw[start:start+consumed], 'Dense inverse encoding differs'
                record.update(format='dense_fp16', tile=tile, nonlinear_mode=nl_mode,
                              ne_config=ne_config, ne_config_written=0x40 in regs,
                              prefix_bytes=prefix, consumed_per_lane=consumed,
                              padding_per_lane=[end-start-consumed for start,end in zip(starts,ends)])
            weight = weight.reshape(shape)
            assert np.isfinite(weight).all() and np.isfinite(bias).all(), 'Nonfinite decoded coefficient'
            arrays[name + '.weight'] = weight
            if has_bias:
                arrays[name + '.bias'] = bias
            record.update(status='decoded_candidate',
                          decoded_payload_roundtrip_equal=True,
                          weight_sha256=digest(weight.tobytes()),
                          bias_sha256=digest(bias.tobytes()) if has_bias else None)
        except (AssertionError, ValueError) as exc:
            record.update(status='unresolved', reason=str(exc))
            unresolved.append(name)
    report = dict(scope='Static candidate reconstruction, not native Photos execution parity',
                  source_net_sha256=digest(net_bytes), source_hwx_sha256=digest(raw),
                  mapping_sha256=digest(mapping_bytes), mapping_assumption=mapping['assumption'],
                  face_layers=len(ancestors), face_weight_layers=len(weighted),
                  decoded_layers=len(weighted)-len(unresolved), unresolved=unresolved,
                  complete=not unresolved, coefficient_precision='Compiled FP16, original FP32 biases not recovered',
                  decoder_basis='Known-control lane layout applied to candidate; target source weights unavailable',
                  source_checks={p.name:digest(p.read_bytes()) for p in
                      [Path(__file__), HERE/'verify_lane_layout.py',
                       base/'lane_layout.json',
                       HERE/'verify_sigmoid_prefix.py',
                       HERE/'sparse_decode.py', sparse_report_path]},
                  sparse_precision_limit='Omitted source signed-zero or subnormal bits cannot be recovered',
                  omitted_input_channels=[layers[m['layer']]['K'] for m in mapping['face_mapping']
                                          if 12 not in by_tid[m['tid']]['registers']],
                  omitted_output_channels=[layers[m['layer']]['C'] for m in mapping['face_mapping']
                                           if 28 not in by_tid[m['tid']]['registers']],
                  default_evidence=defaults, layers=records)
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('net', 'shape', 'hwx', 'compiler', 'mapping', 'lane-layout', 'sparse-results', 'sigmoid-prefix'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--ipsw', required=True, help='Explicit ipsw static-reader executable or command name')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output directory; never overwrite')
    args = parser.parse_args()
    supplied = {name: getattr(args, name.replace('-', '_')) for name in
                ('net', 'shape', 'hwx', 'compiler', 'mapping', 'lane-layout', 'sparse-results', 'sigmoid-prefix')}
    recipe = json.loads(HERE.joinpath('INPUTS.json').read_text(encoding='utf-8'))
    for name, path in supplied.items():
        if digest(path.read_bytes()) != recipe['restore'][name]['sha256']:
            parser.error(name + ' input SHA-256 mismatch')
    inputs = {recipe['restore'][name]['filename']: path for name, path in supplied.items()}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    prepare(output_dir, inputs)
    arrays, report = build(output_dir / 'inputs', args.ipsw)
    assert report['complete'], 'Unresolved coefficient layers; no zero substitution'
    output = output_dir / 'face_tensors.npz'
    write_npz(output, arrays)
    report['array_file'] = output.name
    report['array_file_sha256'] = digest(output.read_bytes())
    report['input_receipt_sha256'] = digest((output_dir / 'input_receipt.json').read_bytes())
    with (output_dir / 'face_tensor_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('decoded_layers', 'complete', 'array_file_sha256')}, indent=2))


if __name__ == '__main__':
    main()
