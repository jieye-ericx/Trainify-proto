import copy
import multiprocessing
import time

import torch
from scipy.optimize import minimize, Bounds
import math
import numpy as np

from trainify.abstract.divide_tool import str_to_list, max_min_clip, combine_bound_list, near_bound, \
    initiate_divide_tool
import re

from trainify.agent import Actor as ddpgActor
from interval import interval, inf, imath


def create_function(str, max):
    labels = [
        ['sin', 'math.sin'],
        ['cos', 'math.cos'],
        ['tan', 'math.tan'],
        ['tanh', 'math.tanh']
    ]
    for a in labels:
        reg = re.compile(re.escape(a[0]), re.IGNORECASE)
        str = reg.sub(a[1], str)

    def cal(x):
        if max:
            return -eval(str)
        else:
            return eval(str)

    return cal


def calculate_reachable_sets(env, initial_bound, time_step, identifier=None):
    if isinstance(initial_bound, str):
        initial_bound = str_to_list(initial_bound)
    assert len(initial_bound) == 2 * len(env.cd)
    bound_list = [initial_bound]
    env.timer_reset()
    st = time.time()
    t = 0
    while True:
        t0 = time.time()
        bound_list = env.get_next_bound_list(bound_list)
        min_x1 = math.inf
        max_x1 = -math.inf
        min_x2 = math.inf
        max_x2 = -math.inf
        for bound in bound_list:
            min_x1 = min(bound[0], min_x1)
            max_x1 = max(bound[0 + len(env.cd)], max_x1)
            min_x2 = min(bound[1], min_x2)
            max_x2 = max(bound[1 + len(env.cd)], max_x2)

        t1 = time.time()
        if identifier is None:
            print(t, '：', len(bound_list), min_x1, max_x1, min_x2, max_x2, t1 - t0)
        t += 1
        if t == time_step:
            if identifier is not None:
                print(identifier, '：', 'min_x1,', min_x1, 'max_x1,', max_x1, 'min_x2,', min_x2, 'max_x2,', max_x2)
            break
    et = time.time()
    print('Overall Time', et - st)
    print('seg', env.time_seg, 'over-app', env.time_op, 'agg', env.time_agg)
    return True


def err_call_back(err):
    print('error---', err)


def parallel_cal(env, initial_bound, set_partition_gran, time_step, process_num=4):
    st = time.time()
    half_dim = len(env.cd)
    assert len(initial_bound) == 2 * len(env.cd)
    assert len(set_partition_gran) == len(env.cd)
    sr = [initial_bound[0:half_dim], initial_bound[half_dim:]]
    sr_dt = initiate_divide_tool(sr, set_partition_gran)
    bounds = sr_dt.intersection(initial_bound)
    print('Number of sub-tasks', len(bounds))
    results = []
    pool = multiprocessing.Pool(processes=process_num)
    cnt = 1
    for bound in bounds:
        results.append(
            pool.apply_async(calculate_reachable_sets, args=(env, bound, time_step, cnt), error_callback=err_call_back))
        cnt += 1
    pool.close()
    pool.join()
    # for res in results:
    #     print(res)
    et = time.time()
    print('Time of Parallel Calculation:', et - st)


class ReachEnv():
    def __init__(self, name, env_config={}, divide_tool=None, network=None):
        self.name = name
        self.xnames = ['x1', 'x2']
        # self.d = ["x[1] + x[2]*0.1", "x[2]+(x[0]*x[2]*x[2]-x[1])*0.1"]
        self.cd = ["x[0] + x[1]*0.1", "x[1]+(action*x[1]*x[1]-x[0])*0.1"]
        self.state_space = [[-2, -2], [2, 2]]
        self.verify_func = []
        # self.create_verify_func()
        self.divide_tool = divide_tool
        self.network = network
        self.standard = [0.0001, 0.0001]
        self.tau = 0.1
        self.state_dim = len(self.cd)

        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def timer_reset(self):
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    # def create_verify_func(self):
    #     labels = [
    #         ['sin', 'math.sin'],
    #         ['cos', 'math.cos'],
    #         ['tan', 'math.tan'],
    #         ['tanh', 'math.tanh']
    #     ]
    #
    #     def from_function(str, max):
    #         def cal(x):
    #             if max:
    #                 return -eval(str)
    #             else:
    #                 return eval(str)
    #
    #         return cal
    #
    #     for i, str in enumerate(self.d):
    #         for a in labels:
    #             reg = re.compile(re.escape(a[0]), re.IGNORECASE)
    #             self.d[i] = reg.sub(a[1], self.d[i])
    #
    #         setattr(self, self.xnames[i] + '_maximum',
    #                 from_function(self.d[i], True))
    #         setattr(self, self.xnames[i] + '_minimum',
    #                 from_function(self.d[i], False))
    #
    #         self.verify_func.append(from_function(self.d[i], False))
    #         self.verify_func.append(from_function(self.d[i], True))

    def get_next_bound_list(self, bound_list):
        res_list = []
        cnt = 0
        for bound in bound_list:
            next_bound_list, counter = self.get_next_states(bound)
            cnt += counter
            for next_bound in next_bound_list:
                res_list.append(next_bound)
        t1 = time.time()
        u = combine_bound_list(res_list, self.standard)
        t2 = time.time()
        self.time_agg = self.time_agg + t2 - t1
        # print(cnt)
        return u

    # Method must be implemented by users
    def get_next_states(self, current):
        if isinstance(current, str):
            current = str_to_list(current)
        current = copy.deepcopy(current)
        t0 = time.time()
        target_list = self.divide_tool.intersection(current)
        # print(len(target_list))
        tl = []
        for ele in target_list:
            s = str_to_list(ele)
            s = max_min_clip(current, s)
            tl.append(s)
        t1 = time.time()
        self.time_seg = self.time_seg + t1 - t0
        # over-approximation
        b_list = []
        for t in tl:
            b = self.gn(t)
            b_list.append(b)
        t2 = time.time()
        self.time_op = self.time_op + t2 - t1
        # aggregation
        res = combine_bound_list(b_list, self.standard)
        t3 = time.time()
        self.time_agg = self.time_agg + t3 - t2
        return res, len(b_list)

    def gn(self, current):
        cur_list = self.divide_tool.intersection(current)
        original = None
        dim = len(current)
        half_dim = int(dim / 2)
        for cu in cur_list:
            cu = str_to_list(cu)
            flag = True
            for i in range(half_dim):
                if cu[i] > current[i] or cu[i + half_dim] < current[i + half_dim]:
                    flag = False
                    break
            if flag:
                original = cu
        assert (original is not None)

        s0 = torch.tensor(original, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()
        offset = 0
        scala = 1
        self.action = (action[0] - offset) * scala
        next_bounds = self.execute_action(current)

        # action 通过约束传递求解很慢
        # mid_point = [self.action]
        # lb = []
        # ub = []
        # for i in range(half_dim):
        #     mid_point.append((current[i] + current[i + half_dim]) / 2)
        #     lb.append(current[i])
        #     ub.append(current[i + half_dim])
        # bound = Bounds(lb, ub)
        # cons_list = []
        # action_cons = {'type': 'eq', 'fun': lambda x: x[0] - self.action}
        # cons_list.append(action_cons)
        #
        # # 循环内的lambda表达式，坑。。在lambda函数中添加默认参数循环索引
        # for j in range(half_dim):
        #     state_lb_cons = {'type': 'ineq', 'fun': lambda x, j=j: x[j + 1] - lb[j]}
        #     cons_list.append(state_lb_cons)
        #     state_ub_cons = {'type': 'ineq', 'fun': lambda x, j=j: ub[j] - x[j + 1]}
        #     cons_list.append(state_ub_cons)
        #
        # next_bounds = [0 for i in range(dim)]

        # for i in range(half_dim):
        #     l = minimize(self.verify_func[2 * i], x0=mid_point, method='SLSQP', constraints=cons_list)
        #     l = self.verify_func[2 * i](l.x)
        #     r = minimize(self.verify_func[2 * i + 1], x0=mid_point, method='SLSQP', constraints=cons_list)
        #     r = -self.verify_func[2 * i + 1](r.x)
        #     # TODO state_bound
        #     l = np.clip(l, -2, 2)
        #     r = np.clip(r, -2, 2)
        #     next_bounds[i] = l
        #     next_bounds[i + half_dim] = r

        # 使用区间运算
        # interval_bound = []
        # for j in range(half_dim):
        #     ele = interval[lb[j], ub[j]]
        #     interval_bound.append(ele)
        #
        # next_bounds = [0 for i in range(dim)]
        # for i in range(half_dim):
        #     fun_str = self.cd[i]
        #     fun_str = fun_str.replace('action', str(self.action))
        #     lb_func = create_function(fun_str, False)
        #     next_interval_bound = lb_func(interval_bound)
        #     l = next_interval_bound[0][0]
        #     r = next_interval_bound[0][1]
        #     # TODO state_bound
        #     l = np.clip(l, -2, 2)
        #     r = np.clip(r, -2, 2)
        #     next_bounds[i] = l
        #     next_bounds[i + half_dim] = r

        return next_bounds

    def execute_action(self, current):
        dim = len(current)
        half_dim = int(dim / 2)
        # 将action直接通过字符串替换成常数，避免添加action的约束
        mid_point = []
        lb = []
        ub = []
        for i in range(half_dim):
            mid_point.append((current[i] + current[i + half_dim]) / 2)
            lb.append(current[i])
            ub.append(current[i + half_dim])
        bound = Bounds(lb, ub)
        cons_list = []

        # 循环内的lambda表达式，坑。。在lambda函数中添加默认参数循环索引
        for j in range(half_dim):
            state_lb_cons = {'type': 'ineq', 'fun': lambda x, j=j: x[j] - lb[j]}
            cons_list.append(state_lb_cons)
            state_ub_cons = {'type': 'ineq', 'fun': lambda x, j=j: ub[j] - x[j]}
            cons_list.append(state_ub_cons)

        next_bounds = [0 for i in range(dim)]
        for i in range(half_dim):
            fun_str = self.cd[i]
            fun_str = fun_str.replace('action', str(self.action))
            lb_func = create_function(fun_str, False)
            ub_func = create_function(fun_str, True)
            l = minimize(lb_func, x0=mid_point, method='COBYLA', constraints=cons_list)
            l = lb_func(l.x)
            r = minimize(ub_func, x0=mid_point, method='COBYLA', constraints=cons_list)
            r = -ub_func(r.x)
            l = np.clip(l, self.state_space[0][i], self.state_space[1][i])
            r = np.clip(r, self.state_space[0][i], self.state_space[1][i])
            next_bounds[i] = l
            next_bounds[i + half_dim] = r
        return next_bounds


if __name__ == "__main__":
    # env = ReachEnv("b1_env")
    # # l = minimize(env.verify_func[0], x0=[-0.02639258, 0.895, 0.535], method='SLSQP', constraints=cons)
    # gefunc = env.x2_minimum([0.96370608, 0.9, 0.54])
    # res = env.verify_func[2]([-0.96370608, 0.9, 0.54])
    # x0 = interval[0.89, 0.9]
    # x1 = interval[0.53, 0.54]
    # x = [x0, x1]
    # test_func = create_function("x[1]+(-1.9954697*x[1]*x[1]-x[0])*0.1", False)
    # tt = test_func(x)
    # self.network.load_state_dict(torch.load(pt_file0))

    # B1
    # pt_file = "b1_abs-actor_[0.05, 0.05]_2_20.pt"
    # network = ddpgActor(4, 20, 1)
    # network.load_state_dict(torch.load(pt_file))
    # state_space = [[-2.5, -2.5], [2.5, 2.5]]
    # initial_intervals = [0.05, 0.05]
    # divide_tool = initiate_divide_tool(state_space, initial_intervals)
    # b1 = ReachEnv("b1_env", divide_tool=divide_tool, network=network)
    # r = [0.8, 0.5, 0.9, 0.6]
    # # r = [0.89, 0.53, 0.9, 0.54]
    # calculate_reachable_sets(b1, r, 60)
    # parallel_cal(b1, r, [0.05, 0.1], 60, process_num=2)

    # B4
    pt_file = "b4_abs-actor_[0.5, 0.5, 0.5]_2_20.pt"
    network = ddpgActor(6, 20, 1)
    network.load_state_dict(torch.load(pt_file))
    state_space = [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]]
    initial_intervals = [0.5, 0.5, 0.5]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    b4 = ReachEnv("b4_env", divide_tool=divide_tool, network=network)
    b4.state_space = state_space
    b4.xnames = ['x1', 'x2', 'x3']
    b4.cd = ['x[0] + (-x[0] + x[1] - x[2]) * 0.05', 'x[1] + (-x[0] * (x[2] + 1) - x[1]) * 0.05',
             'x[2] + (-x[0] + action) * 0.05']
    b4.standard = [0.001, 0.0001, 0.0001]
    r = [0.25, 0.08, 0.25, 0.27, 0.1, 0.27]
    calculate_reachable_sets(b4, r, 15)
    parallel_cal(b4, r, [0.01, 0.01, 0.02], 15, process_num=2)
