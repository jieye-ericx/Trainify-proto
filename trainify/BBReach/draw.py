import math

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator


def draw_box(box_list, recorder=None, parallel=False):
    l = math.inf
    r = -math.inf
    b = math.inf
    u = -math.inf
    for rec in box_list:
        l = min(l, rec[0])
        b = min(b, rec[1])
        r = max(r, rec[2])
        u = max(u, rec[3])
    fig, ax = plt.subplots()
    width = r - l
    height = u - b
    # 画布范围
    # ax.plot([-0.2, 0.3], [-0.05, 0.1], color='white', alpha=0)
    ax.plot([l - width / 10, r + width / 10], [b - height / 10, u + height / 10], color='white', alpha=0)
    # 绘制box list
    for rec in box_list:
        ax.add_patch(
            patches.Rectangle(
                (rec[0], rec[1]),  # (x,y)
                rec[2] - rec[0],  # width
                rec[3] - rec[1],  # height
                fill=None,
                edgecolor='red'
            )
        )
    if recorder is not None:
        if parallel:
            plt.savefig(recorder.get_data_path() + '/env_parallel_BBReach.png', dpi=200)
        else:
            plt.savefig(recorder.get_data_path() + '/env_BBReach.png', dpi=200)
    plt.show(aspect='auto')


# 数组长度补齐
def box_pad(arr, dim, max_len):
    max = 10000
    min = -10000
    res = []
    pad_list = []
    half_dim = int(dim / 2)
    l = []
    for i in range(dim):
        if i < half_dim:
            l.append(max)
        else:
            l.append(min)
    pad_list.append(l)
    for item in arr:
        item_len = item.shape[0]
        while item_len < max_len:
            item = np.append(item, pad_list, axis=0)
            item_len += 1
        res.append(item)
    return np.array(res)
