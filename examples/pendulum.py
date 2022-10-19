import numpy as np

# 获取文件所在的当前路径
from core.utils import str_to_list
from core.env import Pendulum
from core.agent import DDPGAgent
from core import Trainify
from evaluate.evaluate_pendulum import evaluate_pendulum

if __name__ == "__main__":
    env_config = {
        "dim": 3,
        "states_name": ['cos', 'sin', 'thdot'],
        "state_space": [[-1.5, -1.5, -10], [1.5, 1.5, 10]],
        "abs_initial_intervals": [0.16, 0.16, 0.01],
        "state_key_dim": [0, 1],
        "dynamics": ["sin(x[1])**4+2+x[2]+sin(x[2])", "x[0]+x[2]**3+x[1]", "x[1]*x[0]"]
    }

    agent_config = {
        'gamma': 0.99,
        'actor_lr': 0.0001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    t = Trainify(
        env_config=env_config,
        env_class=Pendulum,
        agent_config=agent_config,
        agent_class=DDPGAgent,
        verify=True,
        experiment_name="test_ddpg_pendulum",
        log_dir='tdp_log'
    )
    # reward_list = []
    # for episode in range(2000):
    #     episode_reward = 0
    #     s = t.env.reset()
    #     abs = t.divide_tool.get_abstract_state(s)
    #     for step in range(500):
    #         # env.render()
    #         a = t.agent.act(abs)
    #         s_next, r1, done, _ = t.env.step(a)
    #         abs_next = t.divide_tool.get_abstract_state(s_next)
    #         t.agent.put(str_to_list(abs), a, r1, str_to_list(abs_next))
    #         episode_reward += r1
    #         t.agent.learn()
    #         s = s_next
    #         abs = abs_next
    #     if episode % 5 == 4:
    #         t.save_model(['actor', 'critic', 'actor_target', 'critic_target'])
    #     reward_list.append(episode_reward)
    #     print(episode, ': ', episode_reward)
    #
    #     if episode >= 10 and np.min(reward_list[-3:]) > -3:
    #         #     min_reward = evaluate(agent)
    #         #     if min_reward > -30:
    #         t.save_model(['actor', 'critic', 'actor_target', 'critic_target'])
    #         break
    # t.load_model(['actor', 'critic', 'actor_target', 'critic_target'])
    # evaluate_pendulum(t.agent, t.env)
