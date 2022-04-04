# %% enivro
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point, MultiPoint
import sys
import time
import pandas as pd
import matplotlib as mpl
import numpy as np

mpl.style.use('classic')
# %% datasource
data = pd.read_csv('gosper.csv')
pos = {
    'x': 0.00,
    'y': 0.00
}
x = data['x']
y = data['y']
el = data['element']
LOD = data['LOD']
color = data['color'].values.tolist()

# 要生成德劳内三角网和泰森多边形的点文件
data_point = pd.read_csv('gosper.csv', usecols=['x', 'y' ])
# all points
points = data_point.values.tolist()
point_length = len(points)
#%%
#取凸包，确保所有点都能取到一个多边形
# radius单位为度，0.00001约等于1米
radius = 1.0
convexbuffer = MultiPoint(points).convex_hull.buffer(radius)
points.extend(convexbuffer.exterior.coords)
array = np.array(points)
# %%
# 沃罗诺伊图
vor = Voronoi(array,furthest_site=False, incremental=True, qhull_options=None)
voronoi_plot_2d(vor, show_vertices=False,show_points=False)
# 泰森多边形的顶点
vertices = vor.vertices
# 泰森多边形的面，-1代表无穷索引
regions = vor.regions
for i,region in enumerate(vor.regions):
    # print(np.where(vor.point_region==i))
    # if -1 in region:
    #     print(i)
    if not -1 in region:
        if i<len(color):
            polygon = [vor.vertices[i] for i in region]
        # print(*polygon)
            plt.fill(*zip(*polygon), alpha=0.5,color=color[i])
            print(color[i])
plt.scatter(x, y, color=color)
plt.plot(x, y)
plt.show()

# def showTrack(pos):
#     print(pos)

# class Gosper_Show:
#     def __init__(self, data):
#         '''
#         定义环境元素
#         '''
#         self.x = data['x']
#         self.y = data['y']
#
#         print(self.x[1:3])


# while True:
#     plt.plot(x, y)
#
#     price = input("price:[eg, 10,20]\n")
#     a = price.split(',')
#     if a[0] == 'end' and a[1] == 'end':
#         break;
#
#     plt.plot(0, 0, marker='o', color='b')
#     m = [0, 2.220446049250313e-16]
#     n = [0, 0.9999999999999999]
#     pos['x'] = 1
#     pos['y'] = 2
#     showTrack(pos)
#     plt.plot(m, n, color='r')
#     plt.show()

# 运行轨迹是同层进行搜索

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
