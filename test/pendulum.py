import numpy as np

# 获取文件所在的当前路径
from env.pendulum import Pendulum
from abstract.divide_tool import initiate_divide_tool_rtree
from utils import str_to_list
from agents.pendulum_agent import Agent
from evaluate import evaluate_pendulum


if __name__ == "__main__":


    env = Pendulum()
    env.reset()
    # params = {
    #     'envs': env,
    #     'gamma': 0.99,
    #     'actor_lr': 0.0001,
    #     'critic_lr': 0.001,
    #     'tau': 0.02,
    #     'capacity': 20000,
    #     'batch_size': 32,
    # }
    state_space = [[-1.5, -1.5, -10], [1.5, 1.5, 10]]
    initial_intervals = [0.16, 0.16, 0.01]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], 'pendulum_abstraction1')
    agent = Agent(divide_tool, env)

    reward_list = []

    for episode in range(2000):
        s0 = env.reset()
        episode_reward = 0
        ab_s = agent.divide_tool.get_abstract_state(s0)
        for step in range(500):
            # envs.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            next_abs = agent.divide_tool.get_abstract_state(s1)
            agent.put(str_to_list(ab_s), a0, r1, str_to_list(next_abs))

            episode_reward += r1
            s0 = s1
            ab_s = next_abs
            agent.learn()
        if episode % 5 == 4:
            agent.save()
        reward_list.append(episode_reward)
        print(episode, ': ', episode_reward)
        if episode >= 10 and np.min(reward_list[-3:]) > -3:
            #     min_reward = evaluate(agent)
            #     if min_reward > -30:
            agent.save()
            break

    agent.load()
    evaluate_pendulum(agent,env)
