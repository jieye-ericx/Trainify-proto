import os
import time
import sys

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data")

def get_data_save_dir(task_name):
    time_now = time.strftime("_%Y%m%d_%H_%M_%S", time.localtime())
    dir_path=os.path.join(DATA_PATH,task_name+time_now)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('数据存储文件夹创建完成：' + dir_path)
    return dir_path

if __name__ == "__main__":

    script_path = get_data_save_dir('pendulum')
    pt_file0 = os.path.join(script_path, "pendulum-actor.pt")
    print(pt_file0)
