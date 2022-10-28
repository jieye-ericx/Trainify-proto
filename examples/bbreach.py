import os

import numpy as np

# 获取文件所在的当前路径
from trainify.utils import str_to_list
from trainify.env import Pendulum
from trainify.agent import DDPGAgent
from trainify import Trainify
from trainify.env.BBReach.b4 import B4Env
from evaluate.evaluate_pendulum import evaluate_pendulum

if __name__ == "__main__":
    env_config = {
        "dim": 3,
        "states_name": ['x1', 'x2', 'x3'],
        "state_space": [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]],
        "abs_initial_intervals": [0.5, 0.5, 0.5],
        "state_key_dim": [0, 1],
        "dynamics": ['x[0] + (-x[0] + x[1] - x[2]) * 0.02', 'x[1] + (-x[0] * (x[2] + 1) - x[1]) * 0.02',
                     'x[2] + (-x[0] + action) * 0.02'],
    }

    agent_config = {
        'gamma': 0.99,
        'actor_lr': 0.0001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
        'hidden_size': 256,
        'models_need_save': ['actor', 'critic', 'actor_target', 'critic_target'],
        "modules": [
            {
                "type": "linear",
                "in_features": 6,
                "out_features": 20,
                "mean": 0,
                "std": 0.1,
                "bias_zero": True,
                "activation": "tanh",
            },
            {
                "type": "linear",
                "in_features": 20,
                "out_features": 20,
                "mean": 0,
                "std": 0.1,
                "bias_zero": True,
                "activation": "tanh",
            },
            {
                "type": "linear",
                "in_features": 20,
                "out_features": 1,
                "mean": 0,
                "std": 0.1,
                "bias_zero": True,
                "activation": 'tanh',
            },
        ]
    }

    verify_config = {
        "distance_threshold": [0.001, 0.0001, 0.0001],
        "initial_set": [0.25, 0.08, 0.25, 0.27, 0.1, 0.27],
        "max_step": 35,
        "initial_set_partition": [0.01, 0.01, 0.02]
    }

    train_config = {
        'step_num': 500,
        'episode_num': 2000,
        'reward_threshold': 470
    }

    t = Trainify(
        env_config=env_config,
        env_class=B4Env,
        agent_config=agent_config,
        agent_class=DDPGAgent,
        verify=False,
        verify_config=verify_config,
        experiment_name="test_b4",
    )
    t.train_agent(train_config)
    t.do_BBreach()
