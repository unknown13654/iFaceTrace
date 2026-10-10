"""Read-only HWX section/symbol audit. Sizes between symbols are upper bounds."""
import collections
import hashlib
import json
import pathlib
import struct
import sys


def kernel_file_offset(section, address):
    relative = address - section['address']
    assert 0 <= relative <= section['size'], (section, address)
    return None if relative == section['size'] else section['fileoff'] + relative


def inspect(path):
    data = path.read_bytes()
    magic, _, _, _, count, command_size, _, _ = struct.unpack_from('<8I', data)
    assert magic == 0xBEEFFACE
    sections = []
    symtab = None
    offset = 32
    for _ in range(count):
        command, size = struct.unpack_from('<II', data, offset)
        assert size >= 8 and offset + size <= len(data)
        if command == 0x19:
            segment_fileoff, segment_filesize = struct.unpack_from('<QQ', data, offset + 40)
            nsects = struct.unpack_from('<I', data, offset + 64)[0]
            assert 72 + nsects * 80 == size
            for index in range(nsects):
                start = offset + 72 + index * 80
                name = data[start:start + 16].split(b'\0')[0].decode()
                address, length, fileoff = struct.unpack_from('<QQI', data, start + 32)
                backed = bool(segment_filesize)
                if backed:
                    assert segment_fileoff <= fileoff <= fileoff + length <= segment_fileoff + segment_filesize <= len(data)
                sections.append(dict(name=name, address=address, size=length, fileoff=fileoff,
                                     header_offset=start, file_backed=backed))
        elif command == 2:
            symtab = struct.unpack_from('<4I', data, offset + 8)
        offset += size
    assert offset == 32 + command_size and symtab
    symoff, nsyms, stroff, strsize = symtab
    assert symoff + 16 * nsyms <= len(data) and stroff + strsize <= len(data)
    groups = collections.defaultdict(list)
    for index in range(nsyms):
        string, kind, section, _, address = struct.unpack_from('<IBBHQ', data, symoff + 16 * index)
        assert string < strsize
        end = data.index(b'\0', stroff + string, stroff + strsize)
        name = data[stroff + string:end].decode()
        if section and sections[section - 1]['name'] == '__kern_0':
            target = sections[section - 1]
            fileoff = kernel_file_offset(target, address)
            if fileoff is None:
                continue  # An end marker is not a buffer.
            groups[name.split('_ne_')[0]].append(dict(name=name, address=address, fileoff=fileoff))
    ordered = sorted(groups.items(), key=lambda pair: min(s['fileoff'] for s in pair[1]))
    kernel = next(s for s in sections if s['name'] == '__kern_0')
    report = []
    for i, (name, symbols) in enumerate(ordered):
        symbols.sort(key=lambda s: s['fileoff'])
        lo = symbols[0]['fileoff']
        hi = min(s['fileoff'] for s in ordered[i + 1][1]) if i + 1 < len(ordered) else kernel['fileoff'] + kernel['size']
        report.append(dict(name=name, start=lo, span=hi-lo, lanes=len(symbols),
                           strides=sorted(set(b['fileoff']-a['fileoff'] for a,b in zip(symbols,symbols[1:]))),
                           sha256=hashlib.sha256(data[lo:hi]).hexdigest(), first32=data[lo:lo+32].hex()))
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), sections=sections,
                symbols=nsyms, groups=report, kernel_symbols=[s for _, entries in ordered for s in entries])


def graph(path):
    result = []
    for layer in json.loads(path.read_text())['layers']:
        if 'blob_weights_f16' not in layer:
            continue
        if layer['type'] == 'convolution':
            elements = layer['C'] * layer['K'] * layer['Nx'] * layer['Ny'] // layer['n_groups']
            outputs = layer['C']
        elif layer['type'] == 'inner_product':
            elements = layer['nB'] * layer['nC']
            outputs = layer['nC']
        else:
            raise ValueError(layer)
        result.append(dict(name=layer['name'], type=layer['type'], blob=layer['blob_weights_f16'],
                           fp16_weight_bytes=elements*2, output_channels=outputs))
    return result


def dense_tasks(path, sections):
    """Decode v11+ burst/scatter syntax; retain raw register addresses."""
    data = path.read_bytes()
    text = next(s for s in sections if s['name'] == '__text')
    payload = data[text['fileoff']:text['fileoff'] + text['size']]
    offset = 0
    tasks = []
    while offset + 40 <= len(payload):
        tid, words = struct.unpack_from('<HH', payload, offset)
        words &= 0x7ff
        if not words:
            offset += 16
            continue
        assert tid <= 0x1000 and words >= 9 and offset + words*4 <= len(payload)
        values = struct.unpack_from(f'<{words}I', payload, offset)
        pos = 9
        registers = {}
        while pos < words:
            header = values[pos]
            pos += 1
            base = (header & 0x7fff) * 4
            if header & 0x80000000:
                mask = (header >> 15) & 0xffff
                addresses = [base] + [base + 4*(bit+1) for bit in range(16) if mask & (1 << bit)]
            else:
                addresses = [base + 4*j for j in range(((header >> 15) & 0x3f)+1)]
            assert pos + len(addresses) <= words, (offset, pos, words, header)
            for address in addresses:
                registers[address] = values[pos]
                pos += 1
        tasks.append(dict(tid=tid, fileoff=text['fileoff']+offset, registers=registers))
        offset += (words*4 + 15) & ~15
    return tasks


def kernel_dma(report, tasks):
    kernel = next(s for s in report['sections'] if s['name'] == '__kern_0')
    symbols = collections.defaultdict(list)
    for symbol in report['kernel_symbols']:
        symbols[symbol['fileoff']].append(symbol['name'])
    buffers = {}
    active_tasks = []
    for task in tasks:
        regs = task['registers']
        refs = []
        for lane in range(16):
            if not regs.get(0x5520 + 4*lane, 0) & 1:
                continue
            relative = regs.get(0x5510, 0) + regs.get(0x5560 + 4*lane, 0)
            length = regs.get(0x55a0 + 4*lane, 0)
            assert length > 0 and 0 <= relative < relative + length <= kernel['size']
            fileoff = kernel['fileoff'] + relative
            assert fileoff in symbols, (task['tid'], lane, fileoff)
            refs.append(dict(lane=lane, fileoff=fileoff, size=length, symbols=symbols[fileoff]))
            buffers[fileoff, length] = True
        if refs:
            cfg = regs.get(0x4900, 0)
            active_tasks.append(dict(tid=task['tid'], fileoff=task['fileoff'], input_channels=regs.get(12),
                                     output_channels=regs.get(28), kernel_config=cfg,
                                     kernel_config_written=0x4900 in regs,
                                     sparse=bool(cfg & 0x100), palettized=bool(cfg & 4), refs=refs))
    end = kernel['fileoff']
    covered = 0
    overlaps = 0
    for start, length in sorted(buffers):
        overlaps += start < end
        covered += max(0, start + length - max(start, end))
        end = max(end, start + length)
    return dict(active_tasks=active_tasks, unique_buffers=len(buffers), covered=covered,
                overlaps=overlaps, kernel_size=kernel['size'], complete=covered == kernel['size'])


if __name__ == '__main__':
    check_section = dict(address=100, size=10, fileoff=20)
    assert kernel_file_offset(check_section, 109) == 29
    assert kernel_file_offset(check_section, 110) is None
    try:
        kernel_file_offset(check_section, 111)
    except AssertionError:
        pass
    else:
        raise AssertionError('out-of-section symbol was accepted')
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hwx', type=pathlib.Path, required=True)
    parser.add_argument('--net', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = inspect(args.hwx)
    result['tasks'] = dense_tasks(args.hwx, result['sections'])
    result['kernel_dma'] = kernel_dma(result, result['tasks'])
    result['graph_fp16_layers'] = graph(args.net)
    print(json.dumps(result, indent=2))
