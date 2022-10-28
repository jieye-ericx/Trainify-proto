import bisect
import copy
import time
from queue import Queue

import numpy as np
from rtree import index
from trainify.utils import str_to_list, list_to_str


def near_bound(current, target, standard):
    dim = len(current)
    half_dim = int(dim / 2)
    counter = 0
    record_dim = None
    for i in range(half_dim):
        if abs(current[i] - target[i]) > standard[i] or abs(current[i + half_dim] - target[i + half_dim]) > \
                standard[i]:
            counter += 1
            record_dim = i
    if counter <= 0 or contain(current, target):
        return True
    elif counter == 1:
        if current[record_dim] - target[record_dim + half_dim] <= standard[record_dim] or target[record_dim] - \
                current[
                    record_dim + half_dim] <= standard[record_dim]:
            return True
        elif (target[record_dim] < current[record_dim] < target[record_dim + half_dim]) or (
                current[record_dim] < target[record_dim] < current[record_dim + half_dim]):
            return True
        else:
            return False
    else:
        return False


def combine_bound_list(bound_list, std):
    relation = []
    length = len(bound_list)
    compacted_bound_list = []
    for i in range(length):
        relation.append([False] * length)
    for i in range(length):
        for j in range(length):
            if i != j and near_bound(bound_list[i], bound_list[j], std):
                relation[i][j] = True
                relation[j][i] = True
    # 先构造邻接链表，之后根据表格合并bound
    flag_list = [False for i in range(length)]
    for i in range(length):
        if flag_list[i]:
            continue
        bfs = Queue()
        bfs_set = set()
        bfs.put(i)
        tmp = bound_list[i]
        while not bfs.empty():
            index = bfs.get()
            flag_list[index] = True
            tmp = combine(tmp, bound_list[index])
            bfs_set.add(index)
            near_list = get_near_bound_list(relation, index, bound_list)
            for near_index in near_list:
                if near_index not in bfs_set:
                    bfs.put(near_index)
        compacted_bound_list.append(tmp)

    return compacted_bound_list


# 如果bound的范围大于diameter，则进行二分
def split(bound_list, diameter):
    assert len(bound_list) > 0
    dim = len(bound_list[0])
    half_dim = int(dim / 2)
    assert half_dim == len(diameter)
    res_list = []
    for bound in bound_list:
        split_list = [bound]
        for i in range(half_dim):
            if bound[i + half_dim] - bound[i] > diameter[i]:
                tmp_list = []
                for b in split_list:
                    b1 = copy.deepcopy(b)
                    b2 = copy.deepcopy(b)
                    mid = (bound[i] + bound[i + half_dim]) / 2
                    b1[i] = mid
                    b2[i + half_dim] = mid
                    tmp_list.append(b1)
                    tmp_list.append(b2)
                split_list = copy.deepcopy(tmp_list)
        res_list = res_list + split_list
    return res_list


def contain(current, target):
    dim = len(current)
    half_dim = int(dim / 2)
    cur_in_tar = True
    tar_in_cur = True
    for i in range(half_dim):
        if not (current[i] >= target[i] and current[i + half_dim] <= target[i + half_dim]):
            cur_in_tar = False
        if not (target[i] >= current[i] and target[i + half_dim] <= current[i + half_dim]):
            tar_in_cur = False
    return cur_in_tar or tar_in_cur


# 获取与指定index的bound能合并的bound list
def get_near_bound_list(relation, current_index, bound_list):
    res_list = []
    for i in range(len(bound_list)):
        if i != bound_list and relation[current_index][i]:
            res_list.append(i)
    return res_list


def combine(current, target):
    res = []
    dim = len(current)
    half_dim = int(dim / 2)
    for i in range(dim):
        if i < half_dim:
            res.append(min(current[i], target[i]))
        else:
            res.append(max(current[i], target[i]))
    return res


def max_min_clip(bound, state):
    dim = len(bound)
    half_dim = int(dim / 2)
    s = copy.deepcopy(state)
    for i in range(half_dim):
        s[i] = max(s[i], bound[i])
        s[i + half_dim] = min(s[i + half_dim], bound[i + half_dim])
    return s


# 用于生成不带rtree的divide_tool
def initiate_divide_tool(state_space, initial_intervals):
    divide_point = []
    dp_sets = []
    for i in range(len(state_space[0])):
        lb = state_space[0][i]
        ub = state_space[1][i]
        tmp = [lb]
        dp_set = set()
        while lb < ub:
            lb = round(lb + initial_intervals[i], 10)
            lb = min(lb, state_space[1][i])
            tmp.append(lb)
            dp_set.add(lb)
        divide_point.append(tmp)
        dp_sets.append(dp_set)
    return DivideTool(divide_point, dp_sets)


def initiate_adaptive_divide_tool(state_space, initial_intervals, key_dim, file_name):
    # divide_point = []
    # dp_sets = []
    # for i in range(len(state_space[0])):
    #     lb = state_space[0][i]
    #     ub = state_space[1][i]
    #     tmp = [lb]
    #     dp_set = set()
    #     dp_set.add(lb)
    #     while lb < ub:
    #         lb = round(lb + initial_intervals[i], 10)
    #         tmp.append(lb)
    #         dp_set.add(lb)
    #     divide_point.append(tmp)
    #     dp_sets.append(dp_set)
    small_initial_interval = [division(num) for num in initial_intervals]
    fine_grained_divide_tool = initiate_divide_tool(state_space, small_initial_interval)
    normal_grained_divide_tool = initiate_divide_tool(state_space, initial_intervals)
    adp = initiate_divide_tool_rtree(state_space, initial_intervals, key_dim, file_name)
    adp.fine_grained_divide_tool = fine_grained_divide_tool
    adp.fine_grained_interval = small_initial_interval
    adp.normal_grained_divide_tool = normal_grained_divide_tool
    if adp.rtree is None:
        return adp, 0
    return adp, adp.rtree.get_size()


# 用于生成带有rtree的divide_tool
def initiate_divide_tool_rtree(state_space, initial_intervals, key_dim, file_name):
    divide_point = []
    dp_sets = []
    # for i in range(len(state_space[0])):
    #     lb = state_space[0][i]
    #     ub = state_space[1][i]
    #     tmp = [lb]
    #     dp_set = set()
    #     dp_set.add(lb)
    #     while lb < ub:
    #         lb = round(lb + initial_intervals[i], 10)
    #         tmp.append(lb)
    #         dp_set.add(lb)
    #     dp_sets.append(dp_set)

    key_state_space = [[], []]
    key_initial_intevals = []
    for i in range(len(state_space[0])):
        if key_dim is not None:
            if i in key_dim:
                key_state_space[0].append(state_space[0][i])
                key_state_space[1].append(state_space[1][i])
                key_initial_intevals.append(initial_intervals[i])
                continue
        lb = state_space[0][i]
        ub = state_space[1][i]
        tmp = [lb]
        dp_set = set()
        dp_set.add(lb)
        while lb < ub:
            lb = round(lb + initial_intervals[i], 10)
            tmp.append(lb)
            dp_set.add(lb)
        dp_sets.append(dp_set)
        divide_point.append(tmp)
    rtree = None
    if key_dim is not None:
        p = index.Property()
        p.dimension = len(key_dim)
        # rtree = index.Index(file_name, properties=p)
        # if rtree.get_size() == 0:
        rtree = index.Index(file_name, divide(key_state_space, key_initial_intevals), properties=p)
        print('DivideTool rtree状态数量', rtree.get_size())
    else:
        key_dim = []
    divide_tool = DivideTool(divide_point, dp_sets)
    divide_tool.key_dim = key_dim
    divide_tool.rtree = rtree
    return divide_tool


# 用于yield rtree构造中的状态的上下界
def divide(state_space, intervals):
    lb = state_space[0]
    ub = state_space[1]
    np_initial_intervals = np.array(intervals)
    id = 1
    while True:
        flag = True
        for j in range(len(lb)):
            if lb[j] >= ub[j]:
                flag = False
                break
        if flag:
            upper = np.array(lb) + np_initial_intervals
            for i in range(len(upper)):
                upper[i] = round(upper[i], 10)
            # print(str(lb) + str(upper))
            # print(id)
            obj_str = ','.join([str(_) for _ in lb]) + ',' + ','.join([str(_) for _ in upper])
            # print(obj_str)
            yield id, tuple(lb) + tuple(upper.tolist()), obj_str
            # rtree.insert(id, tuple(lb) + tuple(upper.tolist()),
            #              obj=obj_str)
            id += 1
            if id % 1000 == 0:
                print(id)
        i, lb = get_next_lb(lb, state_space, intervals)
        # print(lb)
        if lb is None:
            break


# 给定一个当前的下界lb，以及状态范围和划分粒度，返回下一个lb
def get_next_lb(lb, state_space, intervals):
    tmp_lb = copy.deepcopy(lb)
    flag = False
    for i in range(len(tmp_lb)):
        if tmp_lb[i] < state_space[1][i]:
            flag = True
            tmp_lb[i] = round(intervals[i] + tmp_lb[i], 10)
            if i >= 1:
                j = i - 1
                while j >= 0:
                    tmp_lb[j] = state_space[0][j]
                    j -= 1
            # if i >= 2:
            #     tmp_lb[1] = state_space.lb[1]
            # if i >= 3:
            #     tmp_lb[2] = state_space.lb[2]
            return i, tmp_lb
    if not flag:
        return -1, None


# 将一个bound的各个维度进行二分
def bound_bisect(bound):
    dim = int(len(bound) / 2)
    bounds = []
    interval = []
    state_space = [[], []]
    for i in range(dim):
        interval.append(round((bound[i + dim] - bound[i]) / 2, 10))
        state_space[0].append(bound[i])
        state_space[1].append(bound[i + dim])
    lb = bound[0:dim]

    count = 0
    # print(bound)
    while lb is not None:
        upper = np.array(lb) + np.array(interval)
        flag = True
        for i in range(len(upper)):
            upper[i] = round(upper[i], 10)
            if upper[i] > bound[i + dim]:
                flag = False
                break
        if flag:
            bounds.append(lb + upper.tolist())
        # print(lb)
        id1, lb = get_next_lb(lb, state_space, interval)
        count += 1
    return bounds


def division(num):
    granularity = 10
    return round(num / granularity, 10)


class DivideTool:
    def __init__(self, divide_point, dp_sets):
        self.divide_point = divide_point
        self.dp_sets = dp_sets
        self.key_dim = []
        self.rtree = None
        self.fine_grained_divide_tool = None
        self.fine_grained_interval = None
        self.normal_grained_divide_tool = None
        self.restore_dp = copy.deepcopy(divide_point)
        self.restore_dp_sets = copy.deepcopy(dp_sets)
        # self.part_state1 = None
        # self.part_state2 = None

    # 将给定的状态范围，分成普通的状态空间和关键状态空间
    def part_state(self, bound):
        dim = len(bound)
        half_dim = int(dim / 2)
        state_space = list(range(dim - 2 * len(self.key_dim)))
        key_state_space = list(range(2 * len(self.key_dim)))
        half_dim1 = int(len(state_space) / 2)
        half_dim2 = int(len(key_state_space) / 2)
        id1 = 0
        id2 = 0
        for i in range(half_dim):
            if i in self.key_dim:
                key_state_space[id2] = bound[i]
                key_state_space[id2 + half_dim2] = bound[i + half_dim]
                id2 += 1
            else:
                state_space[id1] = bound[i]
                state_space[id1 + half_dim1] = bound[i + half_dim]
                id1 += 1
        return state_space, key_state_space

    # 给定范围查询
    def intersection(self, bound):
        dim = len(bound)
        half_dim = int(dim / 2)
        state_space, key_state_space = self.part_state(bound)
        half_dim1 = int(len(state_space) / 2)
        half_dim2 = int(len(key_state_space) / 2)

        # 获取各个维度的划分点
        t0 = time.time()
        region = self.get_abstract_region(state_space)
        t1 = time.time()
        state_list = []
        abstract_state = list(range(len(state_space)))
        self.point_to_state(region, 0, abstract_state, state_list)
        t2 = time.time()
        # print((t2 - t1) / (t1 - t0))
        length1 = len(state_list)

        res_list = []
        # 关键维度上的查询
        if self.rtree is not None:
            rtree_state_list = list(self.rtree.intersection(key_state_space, objects=True))
            length2 = len(rtree_state_list)

            final_state_list = []
            # for i in range(len(state_space)):

            for i in range(length1):
                part_state1 = state_list[i]
                for j in range(length2):
                    ik1 = 0
                    ik2 = 0
                    final_abs = list(range(dim))
                    part_state2 = rtree_state_list[j]
                    for k in range(half_dim):
                        if k in self.key_dim:
                            final_abs[k] = part_state2.bbox[ik2]
                            final_abs[k + half_dim] = part_state2.bbox[ik2 + half_dim2]
                            ik2 += 1
                        else:
                            final_abs[k] = part_state1[ik1]
                            final_abs[k + half_dim] = part_state1[ik1 + half_dim1]
                            ik1 += 1
                    final_state_list.append(final_abs)
            res_list = list_to_str(final_state_list)
        else:
            res_list = list_to_str(state_list)
        return res_list

    # 给定一个范围，返回在相应维度上的分界点
    def get_abstract_region(self, s):
        # s = s.tolist()
        dim = int(len(s) / 2)
        tmp = []

        # 范围查询
        for i in range(dim):
            tmp2 = []
            pos_left = bisect.bisect_left(self.divide_point[i], s[i])
            pos_right = bisect.bisect(self.divide_point[i], s[i + dim])
            if s[i] != self.divide_point[i][pos_left]:
                pos_left -= 1
            while pos_left <= pos_right and pos_left < len(self.divide_point[i]):
                try:
                    tmp2.append(self.divide_point[i][pos_left])
                except:
                    print('debug')
                pos_left += 1
            tmp.append(tmp2)
            # tmp[i] = divide_point[i][pos_left - 1]
            # tmp[i + dim] = divide_point[i][pos_right]
        return tmp

    # 具体状态到抽象状态，即查询具体状态
    def get_abstract_state(self, s):
        dim = len(s)
        # s需要是list
        # s = s.tolist()
        if type(s) is np.ndarray:
            s = s.tolist()
        bound = s + s
        # s1, s2 = self.part_state(bound)
        tt = self.intersection(bound)
        if len(tt) == 0:
            print('bug')
        return tt[0]
        # tmp = []
        # # 点查询
        # if dim == len(self.divide_point):
        #     tmp = list(range(2 * dim))
        #     for i in range(dim):
        #         pos = bisect.bisect(self.divide_point[i], s[i])
        #         tmp[i] = self.divide_point[i][pos - 1]
        #         tmp[i + dim] = self.divide_point[i][pos]
        #     obj_str = ','.join([str(_) for _ in tmp])
        #     return obj_str
        # return None

    # 将划分边界点，转化为抽象状态list
    def point_to_state(self, divide_point, current_dim, abstract_state, state_list):
        dim = len(divide_point)
        # abstract_state = list(range(2 * dim))

        if current_dim < dim:
            if len(divide_point[current_dim]) == 1:
                a_s = copy.deepcopy(abstract_state)
                a_s[current_dim] = divide_point[current_dim][0]
                a_s[current_dim + dim] = divide_point[current_dim][0]
                self.point_to_state(divide_point, current_dim + 1, a_s, state_list)
            else:
                for i in range(len(divide_point[current_dim]) - 1):
                    a_s = copy.deepcopy(abstract_state)
                    a_s[current_dim] = divide_point[current_dim][i]
                    try:
                        a_s[current_dim + dim] = divide_point[current_dim][i + 1]
                    except:
                        print('ssss')
                    self.point_to_state(divide_point, current_dim + 1, a_s, state_list)
        else:
            state_list.append(abstract_state)

    def rtree_refinement(self, violate_states, start_id):
        print(self.rtree.bounds)
        all_states = list(self.rtree.intersection(self.rtree.bounds, objects=True))
        print('所有状态数量', len(all_states))
        key_dim = len(self.key_dim)
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        set_vio = set(violate_states)
        for state in all_states:
            state_str = state.object
            state_str = ','.join([str(_) for _ in state.bbox])
            if state_str in set_vio:
                cnt1 += 1
                if cnt1 % 1000 == 0:
                    print('cnt1', cnt1)
                refine_bounds = bound_bisect(str_to_list(state_str))
                for b in refine_bounds:
                    obj_str = ','.join([str(_) for _ in b[0:key_dim]]) + ',' + ','.join(
                        [str(_) for _ in b[key_dim:2 * key_dim]])
                    # print(obj_str)
                    # rtree.insert(start_id, tuple(b), obj=obj_str)
                    # print(start_id)
                    cnt3 += 1
                    yield start_id, tuple(b), obj_str
                    start_id += 1
            else:
                cnt2 += 1
                yield state.id, tuple(state.bbox), state_str
        print('---------', cnt1, cnt2, cnt3)
        return start_id

    def adaptive_partition(self, abstract_state, bound, max_id):
        # bound_state_list = self.intersection(bound)
        if isinstance(abstract_state, str):
            abstract_state = str_to_list(abstract_state)
        old_key_states = [tuple(abstract_state)]
        # new_key_states = [tuple(abstract_state)]
        new_key_states = []
        dim = len(bound)
        assert dim == len(abstract_state)
        half_dim = int(dim / 2)
        key_dim_num = 0

        n, k = self.part_state(abstract_state)
        a_obj = []
        if len(k) != 0:
            a_obj = [val for val in list(self.rtree.intersection(tuple(k), objects=True)) if val.bbox == k]
        deal_key_dim = len(a_obj) != 0

        for i in range(half_dim):
            if i in self.key_dim:
                key_dim_num += 1
            # 抽象状态的下界低于了bound
            if abstract_state[i] < round(bound[i] - self.fine_grained_interval[i], 10):
                # 关键维度，需要在rtree里处理
                if i in self.key_dim and deal_key_dim:
                    new_key_states = self.get_new_key_states(i, half_dim, bound[i], old_key_states)
                    old_key_states = copy.deepcopy(new_key_states)
                    continue
                else:
                    self.add_divide_line(i - key_dim_num, bound[i])
            # 抽象状态的上界大于了bound
            if abstract_state[i + half_dim] > round(bound[i + half_dim] + self.fine_grained_interval[i], 10):
                # 关键维度，需要在rtree里处理
                if i in self.key_dim and deal_key_dim:
                    new_key_states = self.get_new_key_states(i, half_dim, bound[i + half_dim],
                                                             old_key_states, False)
                    old_key_states = copy.deepcopy(new_key_states)
                else:
                    self.add_divide_line(i - key_dim_num, bound[i + half_dim], False)

        if deal_key_dim and len(new_key_states) != 0:
            self.rtree.delete(a_obj[0].id, tuple(k))
            for new_ks in new_key_states:
                a, b = self.part_state(new_ks)
                obj_str = ','.join([str(_) for _ in b])
                self.rtree.insert(max_id + 1, tuple(b), obj_str)
                max_id += 1
        return max_id

    # 在非关键维度上增加分界线
    def add_divide_line(self, dim, value, low=True):
        if low:
            p1 = bisect.bisect_left(self.fine_grained_divide_tool.divide_point[dim], value)
            assert p1 > 0
            # 查找分割线
            element = self.fine_grained_divide_tool.divide_point[dim][p1 - 1]
            if element not in self.dp_sets[dim]:
                bisect.insort(self.divide_point[dim], element)
                self.dp_sets[dim].add(element)
            # if self.fine_grained_divide_tool.divide_point[dim][p1 - 1] not in self.divide_point[dim]:
            #     bisect.insort(self.divide_point[dim], self.fine_grained_divide_tool.divide_point[dim][p1 - 1])
            # bisect.insort(self.divide_point[dim], self.fine_grained_divide_tool.divide_point[dim][p1 - 1])
        else:
            p2 = bisect.bisect(self.fine_grained_divide_tool.divide_point[dim], value)
            element = self.fine_grained_divide_tool.divide_point[dim][p2]
            if element not in self.dp_sets[dim]:
                bisect.insort(self.divide_point[dim], element)
                self.dp_sets[dim].add(element)
            # if self.fine_grained_divide_tool.divide_point[dim][p2] not in self.divide_point[dim]:
            #     bisect.insort(self.divide_point[dim], self.fine_grained_divide_tool.divide_point[dim][p2])
            # bisect.insort(self.divide_point[dim], self.fine_grained_divide_tool.divide_point[dim][p2])

    # 对抽象状态进行rtree维度上的划分
    def get_new_key_states(self, dim, half_dim, value, old_key_states, low=True):
        new_key_states = []
        if low:
            p1 = bisect.bisect_left(self.fine_grained_divide_tool.divide_point[dim], value)
            assert p1 > 0
            element = self.fine_grained_divide_tool.divide_point[dim][p1 - 1]
        else:
            p2 = bisect.bisect(self.fine_grained_divide_tool.divide_point[dim], value)
            element = self.fine_grained_divide_tool.divide_point[dim][p2]
        for old_key_state in old_key_states:
            new_key_state1 = list(old_key_state)
            new_key_state2 = list(old_key_state)

            new_key_state1[dim + half_dim] = element
            new_key_state2[dim] = element

            new_key_states.append(tuple(new_key_state1))
            new_key_states.append(tuple(new_key_state2))
        return new_key_states

    def restore(self):
        self.divide_point = copy.deepcopy(self.restore_dp)
        self.dp_sets = copy.deepcopy(self.restore_dp_sets)


class AdaptiveDivideTool(DivideTool):
    def __init__(self, divide_point, dp_sets, fine_grained_divide_tool, fine_grained_interval, key_dim=None,
                 file_name=None):
        super().__init__(divide_point, dp_sets)
        self.fine_grained_divide_tool = fine_grained_divide_tool
        self.fine_grained_interval = fine_grained_interval
        self.key_dim = key_dim


if __name__ == "__main__":
    print('test')
    # dt = DivideTool([[-1, 0, 1], [-0.5, 0, 0.5]])
    # abstract_state = list(range(4))
    # state_list = []
    # dt.point_to_state([[-1, 0], [-0.5, 0, 0.5]], 0, abstract_state, state_list)
    # res = dt.intersection([0, -0.3, 0, 0.1])
    # print('eeee')

    # sp = split([[0, 0, 1, 1], [0, 0, 1.5, 1.5], [0, 0, 0.5, 2]], [1, 1])
    #
    # initial_interval = [0.1, 0.01, 0.1, 0.2]
    # lb = [-1, 0, -1, -2]
    # ub = [-0.5, 2, -0.5, 2]
    # adt = initiate_adaptive_divide_tool([lb, ub], initial_interval, [0, 1, 2, 3], 'adt')
    # max_id = adt.adaptive_partition([-0.9, 0.01, -0.9, -1.8, -0.8, 0.02, -0.8, -1.6],
    #                                 [-0.82, 0.015, -1, -2, -0.7, 0.03, -0.87, -1.64], 1200)
    # res = adt.intersection([-0.82, 0.015, -1, -2, -0.7, 0.03, -0.87, -1.61])
    #
    # small_initial_interval = [division(num) for num in initial_interval]
    #
    # dt = initiate_divide_tool_rtree([lb, ub], initial_interval, [1, 2], 'divide_tool_test1')
    #
    # # res = dt.intersection([-0.7, 0.2, -0.9, 0, -0.6, 0.3, -0.6, 9])
    # res2 = dt.get_abstract_state([-0.6, 1, -0.8, 1])
    # vio_states = ['1.01,-0.8,0.2,1.02,-0.7,0.4']
    #
    # p = index.Property()
    # p.dimension = len(dt.key_dim)
    # dt.rtree = index.Index('divide_tool_test1', dt.rtree_refinement(vio_states, dt.rtree.get_size()), properties=p)
    # print('size:', dt.rtree.get_size())
    # res3 = dt.intersection([-0.65, 1.01, -0.8, 0.2, -0.65, 1.02, -0.7, 0.4])
    # print('ssss')

    # state_space = [[0, -0.5], [100, 0.5]]
    # divide_tool = initiate_divide_tool(state_space, [0.1,0.01])
    # print(divide_tool.divide_point)
    # print(divide_tool.dp_sets)
