import os
import time
import sys

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
        print(self.data_path)

    def _create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        print('Recorder 数据存储文件夹创建完成：' + path)
        return path

    def get_data_path(self):
        return self.data_path


if __name__ == '__main__':
    a = Recorder('ttt', 'ttt_results')
    # print(PROJECT_ROOT_PATH)
    # script_path = get_data_save_dir('pendulum')
    # pt_file0 = os.path.join(script_path, "pendulum-actor.pt")
    # print(pt_file0)
