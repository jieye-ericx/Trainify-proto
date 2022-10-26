from torch.utils.tensorboard import SummaryWriter
import subprocess
import os
import threading

def writeTensorBoard(directory, title, data):
    """
    :param directory: str,tensorboard文件的存储目录
    :param title: str,数据标题
    :param data: 一维列表,要写到tensorboard中的数据
    :return:
    """
    writer = SummaryWriter(log_dir= directory)
    for i in range(len(data)):
        writer.add_scalar(tag= title, scalar_value= data[i], global_step=i)
    writer.close()

def _openTensorBoard(path, directory):
    """
    :param path: 当前工程路径
    :param directory: tensorboard文件的存储目录名称
    :return:
    """
    if os.name == "nt":
        # windows
        cmds = [path.split("\\")[0], "cd " + path, "tensorboard --logdir=" + directory]
        p = subprocess.Popen('cmd.exe', stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for cmd in cmds:
            p.stdin.write((cmd + "\n").encode(encoding='UTF-8', errors='strict'))
        p.stdin.close()
        print(p.stdout.read())
    elif os.name == "posix":
        # linux
        cmds = ["cd " + path, "tensorboard --logdir=" + directory]
        p = subprocess.Popen('shell', stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for cmd in cmds:
            p.stdin.write((cmd + "\n").encode(encoding='UTF-8', errors='strict'))
        p.stdin.close()
        print(p.stdout.read())
    else:
        # mac
        pass

def openTensorBoard(path, directory):
    t1 = threading.Thread(target=_openTensorBoard, args=(path,directory,))
    t1.start()

