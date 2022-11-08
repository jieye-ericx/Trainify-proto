import math
import time
from queue import Queue

from trainify.validator.pyModelChecking import *
from trainify.validator.pyModelChecking.CTL import Parser, modelcheck


class FMValidator:
    def __init__(self, configurator, net, logger):

        self.initial_states = configurator.initial_state
        self.initial_states_region = configurator.initial_state_region
        self.proposition_list = configurator.proposition_list
        self.limited_count = configurator.limited_count
        self.limited_depth = configurator.limited_depth
        self.atomic_propositions = configurator.atomic_propositions
        self.formula = configurator.formula
        # 具体状态转为抽象状态
        self.get_abstract_state = configurator.get_abstract_state
        self.get_abstract_state_label = configurator.get_abstract_state_label
        self.get_abstract_state_hash = configurator.get_abstract_state_hash
        self.get_next_states = configurator.get_next_states
        self.is_done = configurator.is_done
        self.get_low_dim_state = configurator.get_low_dim_state
        # 抽象初始状态列表
        self.abstract_initial_states = []

        self.network = net
        self.divide_tool = configurator.divide_tool
        # 临时状态

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # debug
        self.visited_list = []
        self.logger = logger

    def get_initial_state(self):
        self.abstract_initial_states = []
        if self.initial_states_region is not None:
            initial_states = self.divide_tool.intersection(self.initial_states_region)
            for s in initial_states:
                self.abstract_initial_states.append(s)
        else:
            self.abstract_initial_states.append(self.get_abstract_state(self.initial_states))
        self.logger.info('number of initial states:' + str(len(self.abstract_initial_states)))

    def create_kripke_ctl(self):
        label_dict = {}
        all_des_set = set()
        abstract_state_count = 0
        sta_cnt = 0
        self.get_initial_state()
        bfs = Queue()
        # 初始状态添加标签
        for abstract_state in self.abstract_initial_states:
            low_dim = self.get_low_dim_state(abstract_state)
            all_des_set.add(abstract_state)
            label_dict[low_dim] = self.get_abstract_state_label(low_dim, 0)
            bfs.put(abstract_state)
            abstract_state_count += 1
            sta_cnt += 1
            # self.logger.info("初始抽象状态"+ str(abstract_state))

        edges = []
        edge_set = set()
        tmp = Queue()
        old_abs_cnt = 1

        for i in range(self.limited_depth):
            bad_label_cnt = 0
            bad_state = False
            t0 = time.time()
            tt = 0
            if bfs.empty():
                break
            while not bfs.empty():
                # 取出同一层的所有节点
                node = bfs.get()
                tmp.put(node)
            # 计算该层所有节点的后继节点
            # mem_used = float(psutil.virtual_memory().used) / 1073741824
            while not tmp.empty():
                current = tmp.get()
                cur_low_dim = self.get_low_dim_state(current)
                t1 = time.time()
                des = self.get_next_states(current)
                t2 = time.time()
                tt = tt + t2 - t1
                for j in range(len(des)):
                    des_low_dim = self.get_low_dim_state(des[j])
                    if not (des[j] in all_des_set):
                        all_des_set.add(des[j])
                        bfs.put(des[j])
                        sta_cnt += 1
                    index = label_dict.get(des_low_dim, -1)
                    if not ((cur_low_dim, des_low_dim) in edge_set):
                        edges.append((cur_low_dim, des_low_dim))
                        edge_set.add((cur_low_dim, des_low_dim))
                    if index == -1:
                        s_label = self.get_abstract_state_label(des_low_dim, bad_label_cnt)
                        label_dict[des_low_dim] = s_label
                        # if len(s_label) == 0:
                        #     if bad_label_cnt <= 1:
                        #         self.logger.info('current state-----------'+ str(current))
                        #     bad_state = True
                        #     bad_label_cnt += 1
                        abstract_state_count += 1
            t3 = time.time()
            self.logger.info(str(abstract_state_count) + str(sta_cnt) + 'depth' + str(i) + str(t3 - t0) + ' ' + str(
                tt) + 'increase' +
                             str(abstract_state_count - old_abs_cnt))
            # if bad_state:
            #     return None
            old_abs_cnt = abstract_state_count
            # self.logger.info(str(abstract_state_count)+ 'depth'+str( i)+str( t3 - t0)+ ' '+ str(tt)+ '--'+ str(mlp)+ '---'+str( rmp))

        self.logger.info("final_state_count: " + str(abstract_state_count))
        for i in range(len(self.visited_list) - 1):
            if not (self.visited_list[i], self.visited_list[i + 1]) in edges:
                self.logger.error('error')
        k = Kripke(R=edges, L=label_dict)
        self.logger.info(str(k.labels()))

        return k

    def ctl_model_check(self, k):
        parser = Parser()
        f = parser(self.formula)
        qualified_states = modelcheck(k, f)
        s_n = len(self.abstract_initial_states)
        self.logger.info(str(s_n) + self.formula)
        # 验证qualified_states是否包含所有初始状态
        flag = True
        sat = 0
        unsat = 0
        for i in range(len(self.abstract_initial_states)):
            # 如果存在初始状态不满足该性质，则可以判定系统不满足该性质
            low_dim_state = self.get_low_dim_state(self.abstract_initial_states[i])
            if low_dim_state not in qualified_states:
                flag = False
                unsat += 1
            else:
                sat += 1

        if flag:
            self.logger.info("CTL formula satisfied:" + str(self.formula) + str(sat) + str(unsat))
            return True, qualified_states

        else:
            self.logger.info("CTL formula not satisfied:" + str(self.formula) + str(sat) + str(unsat))
            if unsat < s_n:
                return True, qualified_states
            else:
                return False, qualified_states
