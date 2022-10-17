def str_to_list(state_str):
    return list(map(float, state_str.split(',')))


def list_to_str(state_list):
    res_list = []
    for tmp in state_list:
        obj_str = ','.join([str(_) for _ in tmp])
        res_list.append(obj_str)
    return res_list
