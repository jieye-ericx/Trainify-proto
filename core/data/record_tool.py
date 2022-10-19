import os
import time
import sys
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import subprocess


ROOT_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DATA_PATH = os.path.join(ROOT_PROJECT_PATH, "data")


class Recorder:
    def __init__(self,
                 experiment_name,
                 data_dir_name='',
                 ):
        print('Recorder init')
        self.data_path = os.path.join(ROOT_DATA_PATH, data_dir_name) + time.strftime("_%Y%m%d_%H_%M_%S",
                                                                                     time.localtime())
        self._create_dir(self.data_path)
        self.experiment_name = experiment_name
        self.title_reward = {}
        print(self.data_path)

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
        self.title_reward[title] = []

    def get_data_path(self):
        return self.data_path

    def add_reward(self, title, reward):
        self.title_reward[title].append(reward)

    def writeAll2TensorBoard(self):
        """
        :param title: str,数据标题
        :return:
        """
        writer = SummaryWriter(log_dir=self.data_path)
        for title in self.title_reward.keys():
            for i in range(len(self.title_reward[title])):
                writer.add_scalar(tag=title, scalar_value=self.title_reward[title][i], global_step=i)
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

