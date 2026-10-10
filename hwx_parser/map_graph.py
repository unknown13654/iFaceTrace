"""Constrain HWX coefficient tasks by explicit geometry and graph dependencies.

This is a candidate mapper, not tensor-order proof. Missing registers are unknown.
"""
import json
from pathlib import Path
import inspect_hwx


def candidates(net_path, shape_path, hwx_path):
    net = json.loads(net_path.read_text())['layers']
    shapes = json.loads(shape_path.read_text())['layer_shapes']
    weighted = {i for i, layer in enumerate(net) if 'blob_weights_f16' in layer}
    producers = {top: i for i, layer in enumerate(net) for top in layer['top'].split(',')}
    dependencies = {}
    def ancestors(index):
        if index in dependencies:
            return dependencies[index]
        found = set()
        for bottom in net[index]['bottom'].split(','):
            parent = producers.get(bottom)
            if parent is not None:
                found |= {parent} if parent in weighted else ancestors(parent)
        dependencies[index] = found
        return found
    for index in range(len(net)):
        ancestors(index)
    path = hwx_path
    report = inspect_hwx.inspect(path)
    tasks = inspect_hwx.dense_tasks(path, report['sections'])
    tasks = [t for t in tasks if any(t['registers'].get(0x5520+4*l, 0)&1 for l in range(16))]

    def matches(index, task):
        layer = net[index]
        regs = task['registers']
        cin = layer.get('K', layer.get('nB'))
        cout = layer.get('C', layer.get('nC'))
        expected = {12: cin, 28: cout}
        for name, addresses in [(layer['bottom'], (4,8)), (layer['top'], (20,24))]:
            shape = shapes.get(name)
            if shape:
                expected.update(zip(addresses, (shape['w'], shape['h'])))
        if any(address in regs and regs[address] != value for address, value in expected.items()):
            return False
        if 40 in regs and layer['type'] == 'convolution':
            cfg = regs[40]
            if (cfg & 63, (cfg >> 6) & 63) != (layer['Nx'], layer['Ny']):
                return False
        return True

    # Missing registers stay unknown; do not invent device reset values.
    consumed = set()
    mapping = []
    for task in tasks:
        eligible = [i for i in sorted(weighted-consumed)
                    if dependencies[i] <= consumed and matches(i, task)]
        mapping.append(dict(tid=task['tid'], candidates=[net[i]['name'] for i in eligible],
                            net_indices=eligible))
        if len(eligible) != 1:
            break
        consumed.add(eligible[0])
    solutions = []
    visits = 0
    def search(position, used, selected):
        nonlocal visits
        visits += 1
        if visits > 100000 or len(solutions) >= 2:
            return
        if position == len(tasks):
            solutions.append([net[i]['name'] for i in selected])
            return
        for index in sorted(weighted-used):
            if dependencies[index] <= used and matches(index, tasks[position]):
                search(position+1, used | {index}, selected + [index])
    search(0, set(), [])
    # z3 is already installed. Solve all assignments, including auxiliary-head permutations.
    import z3
    positions = {index: z3.Int(f'layer_{index}') for index in sorted(weighted)}
    solver = z3.Solver()
    solver.set(timeout=60000)
    solver.add(z3.Distinct(list(positions.values())))
    for index, variable in positions.items():
        domain = [p for p,t in enumerate(tasks) if matches(index,t)]
        solver.add(z3.Or([variable == p for p in domain]))
        for parent in dependencies[index]:
            solver.add(positions[parent] < variable)
    status = solver.check()
    sat_report = dict(status=str(status), assumption='one task per weighted layer; graph precedence preserved')
    if status == z3.sat:
        model = solver.model()
        assignment = {i: model.eval(v).as_long() for i,v in positions.items()}
        face = set()
        def include(index):
            if index in face:
                return
            if index in weighted:
                face.add(index)
            for bottom in net[index]['bottom'].split(','):
                if bottom in producers:
                    include(producers[bottom])
        include(producers['flatten_output'])
        solver.add(z3.Or([positions[i] != assignment[i] for i in face]))
        alternate = solver.check()
        sat_report.update(face_weight_layers=len(face), alternate_face_assignment=str(alternate),
                          face_mapping=[dict(layer=net[i]['name'], task_position=assignment[i],
                                             tid=tasks[assignment[i]]['tid']) for i in sorted(face)])
        if alternate == z3.unknown:
            sat_report['reason'] = solver.reason_unknown()
    return dict(unique_prefix=len(consumed), task_count=len(tasks), mapping=mapping,
                search_visits=visits, solution_limit=2, search_limit=100000, solutions=solutions,
                constraint_solver=sat_report)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--net', type=Path, required=True)
    parser.add_argument('--shape', type=Path, required=True)
    parser.add_argument('--hwx', type=Path, required=True)
    args = parser.parse_args()
    result = candidates(args.net, args.shape, args.hwx)
    assert result['task_count'] == 250
    assert result['mapping'][0]['candidates'] == ['Conv_0']
    print(json.dumps(result, indent=2))
