def infill_op_dict(ops, ops_bar, op_dict):
    op_dict[((),)] = op_dict[()]
    op_dict[((),())] = op_dict[()]

    for op in ops:
        op_dict[(op, ())] = op_dict[(op,)]
        op_dict[((), op)] = op_dict[(op,)]
    for op in ops_bar:
        op_dict[(op, ())] = op_dict[(op,)]
        op_dict[((), op)] = op_dict[(op,)]