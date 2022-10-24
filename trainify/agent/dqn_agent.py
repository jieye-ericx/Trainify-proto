import bisect
import copy
import os
import random
import sys
from collections import deque, namedtuple
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rtree import index
import time
import pandas as pd
from trainify.utils.str_list import str_to_list

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

rindex = '0'
# 获取文件所在的当前路径
script_path = os.path.split(os.path.realpath(__file__))[0]
# 生成需要保存的文件名称以及路径
pt_file = os.path.join(script_path, "dqn" + rindex + ".pt")

class Model(nn.Module):  # 神经网络的类
    def __init__(self, hidden_size, output_size):  # 网络结构及其初始化
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.zero_()

    def forward(self, obs):  # 隐藏层的激活函数
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        action_val = self.fc4(obs)
        return action_val
        # 返回的是最终的每个action的数值

class Agent:
    def __init__(self, divide_tool, output_size,
                 hidden_size = 64,
                 memory_capacity = 10000,
                 learning_rate = 1e3,
                 gamma = 0.99,
                 e_greed = 0.1,
                 e_greed_dec = 1e-4,
                 e_greed_min = 0.01,
                 weight_decay = 0,
                 batch_size = 256,
                 warmup_size = 5120,
                 model_sync_count = 8,
                 use_dbqn = False):
        self.model = Model(hidden_size, output_size)  # 当前网络
        self.network = Model(hidden_size, output_size)
        self.target_model = Model(hidden_size, output_size)  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.memory = deque(maxlen=memory_capacity)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数
        self.e_greed = e_greed
        self.update_count = 0  # 更新次数
        self.noisy = [0, 0, 0, 0]
        self.divide_tool = divide_tool
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memory_capacity = memory_capacity
        self.learn_rate = learning_rate
        self.weight_decay = weight_decay
        self.e_greed_origin = e_greed
        self.e_greed_dec = e_greed_dec
        self.e_greed_min = e_greed_min
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.gamma = gamma
        self.use_dbqn = use_dbqn
        self.model_sync_count = model_sync_count

    def reset(self):
        self.model = Model()  # 当前网络
        self.network = Model()
        self.target_model = Model()  # 目标网络
        self.target_model.load_state_dict(self.model.state_dict())  # 当前网络对目标网络赋值
        self.memory = deque(maxlen=self.memory_capacity)  # 创建双向队列，保存经验回放
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learn_rate, weight_decay=self.weight_decay
        )  # 模型的各个参数
        self.loss_func = nn.MSELoss()  # 定义损失函数
        self.e_greed = self.e_greed_origin
        self.update_count = 0
        self.noisy = [0, 0, 0, 0]

    def update_egreed(self):  # 更新贪心参数e,该函数有问题？？？？
        self.e_greed = max(self.e_greed_min, self.e_greed - self.e_greed_dec)

    def predict(self, obs):  # 选取最大的action
        abs = str_to_list(self.divide_tool.get_abstract_state(obs))
        q_val = self.model(torch.FloatTensor(abs)).detach().numpy()
        q_max = np.max(q_val)
        choice_list = np.where(q_val == q_max)[0]
        return choice_list[0]

    def sample(self, obs):  # 决定是随机选，还是选最大
        if np.random.rand() < self.e_greed:
            return np.random.choice(self.output_size)
        return self.predict(obs)

    def store_transition(self, trans):  # 放入经验回放盒
        self.memory.append(trans)

    def learn(self):  # 该函数实现了：采样，计算损失，反向传播，更新参数
        assert self.warmup_size >= self.batch_size
        if len(self.memory) < self.warmup_size:
            return
        # 从经验回放盒中进行采样
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*(zip(*batch)))
        s0 = torch.FloatTensor(batch.state)
        a0 = torch.LongTensor(batch.action).unsqueeze(1)
        r1 = torch.FloatTensor(batch.reward)
        s1 = torch.FloatTensor(batch.next_state)
        d1 = torch.LongTensor(batch.done)

        q_pred = self.model(s0).gather(1, a0).squeeze()
        with torch.no_grad():
            if self.use_dbqn:
                acts = self.model(s1).max(1)[1].unsqueeze(1)
                q_target = self.target_model(s1).gather(1, acts).squeeze(1)
            else:
                q_target = self.target_model(s1).max(1)[0]

            q_target = r1 + self.gamma * (1 - d1) * q_target
        loss = self.loss_func(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.model_sync_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save(self, sava_path = pt_file):  # 保存网络的参数数据
        torch.save(self.model.state_dict(), sava_path)
        # print(pt_file + " saved.")

    def load(self, load_path = pt_file):  # 加载网络的参数数据
        self.model.load_state_dict(torch.load(load_path))
        self.network.load_state_dict(torch.load(load_path))
        self.target_model.load_state_dict(self.model.state_dict())
        print(load_path + " loaded.")

    def add_noisy(self, obs):

        obs[0] += random.gauss(0, self.noisy[0])
        obs[1] += random.gauss(0, self.noisy[1])
        obs[2] += random.gauss(0, self.noisy[2])
        obs[3] += random.gauss(0, self.noisy[3])
        return obs

    def update_noisy(self):
        # noisy = [0.005, 0.01, 0.0005, 0.01]
        step_length = [0.005, 0.005, 0.005, 0.005]
        for i in range(len(step_length)):
            self.noisy[i] += step_length[i]




def clip(state):
    # low = state_space[0][0]
    # high = state_space[1][0]
    state_space1 = [[-4.79, -9.99, -0.419, -9.99], [4.79, 9.99, 0.419, 9.99]]
    state[0] = np.clip(state[0], state_space1[0][0], state_space1[1][0])
    state[1] = np.clip(state[1], state_space1[0][1], state_space1[1][1])
    state[2] = np.clip(state[2], state_space1[0][2], state_space1[1][2])
    state[3] = np.clip(state[3], state_space1[0][3], state_space1[1][3])
    return state

