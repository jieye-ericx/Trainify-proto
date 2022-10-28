import os
import time
import sys
import torch
from tensorboardX import SummaryWriter
import subprocess

ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DATA_PATH = os.path.join(ROOT_PROJECT_PATH, "data")


class Recorder:
    def __init__(self,
                 experiment_name,
                 data_dir_name,
                 log_path
                 ):
        print('Recorder init')
        self.experiment_name = experiment_name
        self.data_dir_name = data_dir_name
        if log_path is None:
            self.data_path = os.path.join(ROOT_DATA_PATH, data_dir_name)
        else:
            self.data_path = log_path
        # print(self.data_path)
        self.data_path_tensorboard = self.data_path + '/tensorboard'
        self._create_dir(self.data_path)
        self.reward = {}
        self.temp_name = ''

    def _create_dir(self, path):
        """
        创建tensorboard结果文件的储存文件夹
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        print('Recorder 数据存储文件夹创建完成：' + path)
        return path

    def create_data_result(self, title):
        self.temp_name = title
        self.reward.update({title: {'name': title, 'reward_list': []}})

    def get_data_path(self):
        return self.data_path

    def add_reward(self, reward, title=''):
        if title == '':
            title = self.temp_name
        self.reward[title]['reward_list'].append(reward)

    def get_reward_list(self, title=''):
        if title == '':
            title = self.temp_name
        return self.reward[title]['reward_list']

    def get_reward(self, title=''):
        if title == '':
            title = self.temp_name
        return self.reward[title]

    def writeAll2TensorBoard(self, title=''):
        """
        :param title: str,数据标题
        :return:
        """
        if title == '':
            title = self.temp_name

        writer = SummaryWriter(log_dir=self.data_path_tensorboard)
        for title in self.reward.keys():
            for i in range(len(self.reward[title]['reward_list'])):
                writer.add_scalar(tag=title, scalar_value=self.reward[title]['reward_list'][i], global_step=i)
        writer.close()

    def openTensorBoard(self):
        """
        :param path: 当前工程路径
        :param directory: tensorboard文件的存储目录名称
        :return:
        """
        cmds = [ROOT_PROJECT_PATH.split("\\")[0], "cd " + ROOT_PROJECT_PATH, "tensorboard --logdir=data"]
        p = subprocess.Popen('cmd.exe', stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for cmd in cmds:
            p.stdin.write((cmd + "\n").encode(encoding='UTF-8', errors='strict'))
        p.stdin.close()
        print(p.stdout.read())

    def create_save_model(self, agent, agent_config):
        def save_model():
            for name in agent_config['models_need_save']:
                if hasattr(agent, name):
                    torch.save(agent.__dict__[name].state_dict(), self.data_path + '/' + name + '.pt')
                    print('Trainify 模型 ' + name + ' 保存完毕')

        return save_model

    def create_load_model(self, agent, agent_config):
        def load_model():
            for name in agent_config['models_need_save']:
                if hasattr(agent, name):
                    agent.__dict__[name].load_state_dict(torch.load(self.data_path + '/' + name + '.pt'))
                    print('Trainify 模型 ' + name + ' 加载完毕')

        return load_model


if __name__ == '__main__':
    a = Recorder('ttt', 'ttt_results')
    a.create_data_result("test1")
    for i in range(100):
        a.add_reward("test1", i * 6)
    a.create_data_result("test2")
    for i in range(100):
        a.add_reward("test2", i * 10)
    a.writeAll2TensorBoard()
    a.openTensorBoard()
