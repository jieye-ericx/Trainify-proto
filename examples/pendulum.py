# 引入pendulum环境
from trainify.env import Pendulum
# 引入连续控制所需要的DDPG算法模型
from trainify.agent import DDPGAgent
# 引入原型工具主类
from trainify import Trainify

# 环境相关配置
env_config = {
    "dim": 3,  # 环境状态维度
    "states_name": ['cos', 'sin', 'thdot'],  # 环境状态的名称数组
    "state_space": [[-1.5, -1.5, -10], [1.5, 1.5, 10]],  # 环境状态的取值范围，和名称一一对应
    "abs_initial_intervals": [0.16, 0.16, 0.01],  # 抽象环境状态时各状态的抽象粒度，和名称一一对应
    "state_key_dim": [0, 1],  # 关键状态，类似于索引，选择比较重要的状态在名称数组中的下标
    "dynamics": ["sin(x[1])**4+2+x[2]+sin(x[2])", "x[0]+x[2]**3+x[1]", "x[1]*x[0]"]  # 环境的dynamics
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
            "out_features": 20,
            "mean": 0,
            "std": 0.1,
            "bias_zero": True,
            "activation": 'tanh',
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
    'max_iteration': 5  # 最大验证迭代次数
}

# 训练时配置
train_config = {
    'step_num': 500,  # 最大步数
    'episode_num': 2000,  # 最大episode数
    'reward_threshold': -3,  # 期望获得的奖励目标
}

# 创建查看episode和奖励的回调函数
# def on_episode_end_callback(e, r):
#     print('当前epidode是 ', e, ' 对应的奖励是 ', r)


# 将上面的配置项传入构造函数
t = Trainify(
    env_config=env_config,
    env_class=Pendulum,
    agent_config=agent_config,
    agent_class=DDPGAgent,
    verify_config=verify_config,
    experiment_name="test_ddpg_pendulum",
    verify=True,
    # on_episode_end=on_episode_end_callback
)
# 调用训练api，开始训练
t.train_agent(config=train_config)

# 调用验证api，开始cegar验证
t.verify_cegar()
