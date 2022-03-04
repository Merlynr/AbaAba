# Gosper 曲线
from envir.L_system import replacement, interpretation

grammer = {
    "start": "A",
    "rules": {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"},
}

geometry = {
    "rotate": 60,
    "actions": {"+": "left", "-": "right", "A": "forward", "B": "forward"}
}

import matplotlib.pyplot as plt
import sys
import time

str = replacement(grammer["start"], grammer["rules"], 3)
x, y = interpretation(str, geometry["actions"], geometry["rotate"], 30, 0, 0)

pos = {
    'x': 0.00,
    'y': 0.00
}


def showTrack(pos):
    print(pos)


while True:
    plt.plot(x, y)
    price = input("price:[eg, 10,20]\n")
    a = price.split(',')
    if a[0] == 'end' and a[1] == 'end':
        break;

    plt.plot(0, 0, marker='o', color='b')
    m = [0, 2.220446049250313e-16]
    n = [0, 0.9999999999999999]
    pos['x'] = 1
    pos['y'] = 2
    showTrack(pos)
    plt.plot(m, n, color='r')
    plt.show()

# 进行数据绑定
# 同层划分LOD
# 加入条件判断，是否可以进行下一步
# 进行移动，模糊定位
# 记录移动轨迹，记录状态
