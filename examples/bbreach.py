from trainify.agent import DDPGAgent
from trainify import Trainify
from trainify.env.BBReach.b4 import B4Env

# 环境相关配置
env_config = {
    "dim": 3,  # 环境状态维度
    "states_name": ['x1', 'x2', 'x3'],  # 环境状态的名称数组
    "state_space": [[-2.5, -2.5, -2.5], [2.5, 2.5, 2.5]],  # 环境状态的取值范围，和名称一一对应
    "abs_initial_intervals": [0.5, 0.5, 0.5],  # 抽象环境状态时各状态的抽象粒度，和名称一一对应
    "state_key_dim": [0, 1],  # 关键状态，类似于索引，选择比较重要的状态在名称数组中的下标
    "dynamics": ['x[0] + (-x[0] + x[1] - x[2]) * 0.02', 'x[1] + (-x[0] * (x[2] + 1) - x[1]) * 0.02',
                 'x[2] + (-x[0] + action) * 0.02'],  # 环境的dynamics
}
# 模型相关参数
agent_config = {
    # 网络相关参数
    'gamma': 0.99,
    'actor_lr': 0.0001,
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000,
    'batch_size': 32,
    'hidden_size': 256,
    # 需要被数据记录模块所保存的网络的名称，本例中与DDPGAgent Class中的变量名一一对应
    'models_need_save': ['actor', 'critic', 'actor_target', 'critic_target'],
    # 模型隐藏层的配置，根据modules数组，自动调用torch api生成对应的actor网络
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
# 验证相关配置
verify_config = {
    "distance_threshold": [0.001, 0.0001, 0.0001],
    "initial_set": [0.25, 0.08, 0.25, 0.27, 0.1, 0.27],
    "max_step": 35,
    "initial_set_partition": [0.01, 0.01, 0.02]
}
# 训练时配置
train_config = {
    'step_num': 500,  # 最大步数
    'episode_num': 2000,  # 最大episode数
    'reward_threshold': 470  # 期望获得的奖励目标
}

# 将上面的配置项传入构造函数
t = Trainify(
    env_config=env_config,
    env_class=B4Env,
    agent_config=agent_config,
    agent_class=DDPGAgent,
    verify=False,
    verify_config=verify_config,
    experiment_name="test_b4",
)
# 调用训练api，开始训练
t.train_agent(train_config)
# 调用可达集计算api，计算可达集
t.do_BBreach()
