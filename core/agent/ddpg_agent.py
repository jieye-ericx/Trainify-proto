import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from core.utils import str_to_list

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)
        self.linear1.bias.data.zero_()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)
        self.linear2.bias.data.zero_()
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.linear3.weight.data.normal_(0, 0.1)
        self.linear3.bias.data.zero_()

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPGAgent(object):
    # def __init__(self, divide_tool, env):
    def __init__(self, env, config):
        self.originEnv = env
        self.env = env
        self.gamma = config['gamma']
        self.actor_lr = config['actor_lr']
        self.critic_lr = config['critic_lr']
        self.tau = config['tau']
        self.capacity = config['capacity']
        self.batch_size = config['batch_size']
        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]

        # self.divide_tool = divide_tool
        self.actor = Actor(s_dim, 128, a_dim)
        self.network = Actor(s_dim, 128, a_dim)
        self.actor_target = Actor(s_dim, 128, a_dim)
        self.critic = Critic(s_dim + a_dim, 128, a_dim)
        self.critic_target = Critic(s_dim + a_dim, 128, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        script_path = '/Users/ericx/PycharmProjects/Trainify-proto/data/test_ddpg_pendulum_20221017_18_20_05'
        self.pt_file0 = os.path.join(script_path, "pendulum-actor.pt")
        self.pt_file1 = os.path.join(script_path, "pendulum-critic.pt")
        self.pt_file2 = os.path.join(script_path, "pendulum-actor-target.pt")
        self.pt_file3 = os.path.join(script_path, "pendulum-critic-target.pt")

    def reset(self):
        self.env = self.originEnv
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32

        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim)
        self.network = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim + a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim + a_dim, 256, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def save(self):  # 保存网络的参数数据
        torch.save(self.actor.state_dict(), self.pt_file0)
        torch.save(self.critic.state_dict(), self.pt_file1)
        torch.save(self.actor_target.state_dict(), self.pt_file2)
        torch.save(self.critic_target.state_dict(), self.pt_file3)
        # print(pt_file + " saved.")

    # 加载网络的参数数据
    def load(self):
        self.actor.load_state_dict(torch.load(self.pt_file0))
        self.network.load_state_dict(torch.load(self.pt_file0))
        self.critic.load_state_dict(torch.load(self.pt_file1))
        self.actor_target.load_state_dict(torch.load(self.pt_file2))
        self.critic_target.load_state_dict(torch.load(self.pt_file3))
        print(self.pt_file3 + " loaded.")

    def act(self, s0):
        abs = str_to_list(s0)
        # abs = str_to_list(self.divide_tool.get_abstract_state(s0))
        # s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        s0 = torch.tensor(abs, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(np.array(s0), dtype=torch.float)
        a0 = torch.tensor(np.array(a0), dtype=torch.float)
        r1 = torch.tensor(np.array(r1), dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(np.array(s1), dtype=torch.float)

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)