def infill_op_dict(ops, ops_bar, op_dict):
    op_dict[((),)] = op_dict[()]
    op_dict[((),())] = op_dict[()]

    for op in ops:
        op_dict[(op, ())] = op_dict[(op,)]
        op_dict[((), op)] = op_dict[(op,)]
    for op in ops_bar:
        op_dict[(op, ())] = op_dict[(op,)]
        op_dict[((), op)] = op_dict[(op,)]

    to_add = {}
    for ops in op_dict:
        if type(ops) is not tuple or len(ops) != 2: continue

        opposite_pairing = (ops[1], ops[0])
        if opposite_pairing in op_dict: continue
        to_add[opposite_pairing] = op_dict[ops]
    op_dict.update(to_add)