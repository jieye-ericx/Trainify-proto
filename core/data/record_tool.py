import os
import time
import torch

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

    def _create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print('Recorder 本次实验数据存储文件夹创建完成：' + path)

    def get_data_path(self):
        return self.data_path

    def create_save_model(self, agent):
        def save_model(model_names):
            for name in model_names:
                torch.save(agent.__dict__[name].state_dict(), self.data_path + '/' + name + '.pt')
                print('Trainify 模型 ' + name + ' 保存完毕')

        return save_model

    def create_load_model(self, agent):
        def load_model(model_names):
            for name in model_names:
                agent.__dict__[name].load_state_dict(torch.load(self.data_path + '/' + name + '.pt'))
                print('Trainify 模型 ' + name + ' 加载完毕')

        return load_model


if __name__ == '__main__':
    a = Recorder('ttt', 'ttt_results')
    # print(PROJECT_ROOT_PATH)
    # script_path = get_data_save_dir('pendulum')
    # pt_file0 = os.path.join(script_path, "pendulum-actor.pt")
    # print(pt_file0)
