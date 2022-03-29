import matplotlib.pyplot as plt
import sys
import time
import pandas as pd

data = pd.read_csv('gosper.csv')
pos = {
    'x': 0.00,
    'y': 0.00
}
x=data['x']
y=data['y']
el = data['element']
LOD = data['LOD']

def showTrack(pos):
    print(pos)


# class Gosper_Show:
#     def __init__(self, data):
#         '''
#         定义环境元素
#         '''
#         self.x = data['x']
#         self.y = data['y']
#
#         print(self.x[1:3])



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


#运行轨迹是同层进行搜索

# 进行数据绑定
# 同层划分LOD
# 加入条件判断，是否可以进行下一步
# 进行移动，模糊定位
# 记录移动轨迹，记录状态
# 优化最少的步骤达到目的

# **
# LOD表示层级之间跳转
# page表示页面跳转，可以实现不同层次之间跳跃

# if __name__ == "__main__":
#     print('main')
#     Gosper_Show(data)
