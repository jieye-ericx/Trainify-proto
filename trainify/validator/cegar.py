import time

import numpy as np
from rtree import index

from trainify.validator import FMValidator

record_num = 1


def cegar(rtree_name, agent, divide_tool, train_func, verify_env, config, logger):
    max_iteration = config['max_iteration']

    logger.info('Verify cegar验证 开始时间' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    # 此时agent应该已经训练好了
    # train_model(agent)
    # agent.load()
    # evaluate(agent)
    tr = time.time()
    v = FMValidator(verify_env, agent.network, logger)
    k = v.create_kripke_ctl()
    t1 = time.time()
    if k is None:
        violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
        res2 = True
        logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
    else:
        # v.formula = 'not(A(G(safe)))'
        res2, violated_states = v.ctl_model_check(k)
        logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
    t2 = time.time()
    logger.info('Verify cegar train time:' + str(tr - t0) + 'construct kripke structure:' + str(
        t1 - tr) + 'model checking:' + str(t2 - t1))
    logger.info(time.strftime("%Y-%m-%d-%H_%M_%S" + str(time.localtime())))
    start_id = divide_tool.rtree.get_size()
    p = index.Property()
    p.dimension = len(divide_tool.key_dim)
    iteration_time = 1
    while res2 and iteration_time <= max_iteration:
        logger.info(time.strftime("%Y-%m-%d-%H_%M_%S" + str(time.localtime())))
        t0 = time.time()
        divide_tool.rtree = index.Index(rtree_name, divide_tool.rtree_refinement(violated_states, start_id),
                                        properties=p)
        logger.info('Verify cegar number of states after refinement:' + str(divide_tool.rtree.get_size()))
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
        v = FMValidator(verify_env, agent.network, logger)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
        logger.info('Verify cegar iteration :' + str(iteration_time) + str(res2) + str(len(violated_states)))
        t2 = time.time()
        logger.info('Verify cegar refine:' + str(trr - t0) + str('train:') + str(
            tr - trr) + 'construct kripke structure:' + str(t1 - tr) + 'model checking:' + str(t2 - t1))
        logger.info(time.strftime("%Y-%m-%d-%H_%M_%S" + str(time.localtime())))
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
        v = FMValidator(verify_env, agent.network, logger)
        k = v.create_kripke_ctl()
        t1 = time.time()
        if k is None:
            violated_states = list(divide_tool.rtree.intersection(divide_tool.rtree.bounds, objects='raw'))
            res2 = True
            logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
        else:
            # v.formula = 'not(A(G(safe)))'
            res2, violated_states = v.ctl_model_check(k)
            logger.info('Verify cegar number of counterexamples：' + str(len(violated_states)))
        t2 = time.time()
        logger.info('Verify cegar train:' + str(tr - t0) +
                    'construct kripke structure:' + str(t1 - tr) + str('model checking:') + str(t2 - t1))
        logger.info(time.strftime("%Y-%m-%d-%H_%M_%S" + str(time.localtime())))

    logger.info('Verify 验证结束 ' + '构建kripke所花时间:' + str(t1 - tr) + str('模型检查时间:') + str(t2 - t1))
