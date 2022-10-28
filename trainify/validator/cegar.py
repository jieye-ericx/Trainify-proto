import time

import numpy as np
from rtree import index

from trainify.validator import FMValidator

record_num = 1


def cegar(rtree_name, agent, divide_tool, train_func, verify_env, config):
    max_iteration = config['max_iteration']

    print('Verify cegar验证 开始时间' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    # 此时agent应该已经训练好了
    # train_model(agent)
    # agent.load()
    # evaluate(agent)
    tr = time.time()
    v = FMValidator(verify_env, agent.network)
    k = v.create_kripke_ctl()
    t1 = time.time()
    if k is None:
        violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
        res2 = True
        print('Verify cegar number of counterexamples：', len(violated_states))
    else:
        # v.formula = 'not(A(G(safe)))'
        res2, violated_states = v.ctl_model_check(k)
        print('Verify cegar number of counterexamples：', len(violated_states))
    t2 = time.time()
    print('Verify cegar train time:', tr - t0, 'construct kripke structure:', t1 - tr, 'model checking:', t2 - t1)
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    start_id = divide_tool.rtree.get_size()
    p = index.Property()
    p.dimension = len(divide_tool.key_dim)
    iteration_time = 1
    while res2 and iteration_time <= max_iteration:
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        t0 = time.time()
        divide_tool.rtree = index.Index(rtree_name, divide_tool.rtree_refinement(violated_states, start_id),
                                        properties=p)
        print('Verify cegar number of states after refinement:', divide_tool.rtree.get_size())
        trr = time.time()
        start_id += divide_tool.rtree.get_size()
        agent.divide_tool = divide_tool
        # agent.load()
        agent.reset()
        train_func()

        agent.load_model()
        # evaluate(agent)
        tr = time.time()
        verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        # pd = PendulumEnv(divide_tool, agent.actor)
        v = FMValidator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('Verify cegar number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            print('Verify cegar number of counterexamples：', len(violated_states))
        print('Verify cegar iteration :', iteration_time, res2, len(violated_states))
        t2 = time.time()
        print('Verify cegar refine:', trr - t0, 'train:', tr - trr, 'construct kripke structure:', t1 - tr,
              'model checking:',
              t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        iteration_time += 1
    while res2:
        t0 = time.time()
        # agent.load()
        agent.reset()
        train_func()

        agent.load_model()
        tr = time.time()
        # verify_env.divide_tool = divide_tool
        verify_env.network = agent.network
        v = FMValidator(verify_env, agent.network)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            print('Verify cegar number of counterexamples：', len(violated_states))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            print('Verify cegar number of counterexamples：', len(violated_states))
        t2 = time.time()
        print('Verify cegar train:', tr - t0, 'construct kripke structure:', t1 - tr, 'model checking:', t2 - t1)
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    print('Verify 验证结束 ', '构建kripke所花时间:', t1 - tr, '模型检查时间:', t2 - t1)
