import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point, MultiPoint
import sys
import time
import pandas as pd
import matplotlib as mpl
import numpy as np


class GridMapEnv:
    def __init__(self, file):
        self.file = file
        self.figure = plt.figure()

    # read data from csv
    def readData(self):
        data = pd.read_csv(self.file)
        # x = data['x']
        # y = data['y']
        # el = data['element']
        # LOD = data['LOD']
        # color = data['color']
        # 要生成德劳内三角网和泰森多边形的点文件
        data_point = pd.read_csv(self.file, usecols=['x', 'y'])
        # all points
        points = data_point.values.tolist()
        return data, points

    # draw Voronoi
    def createVoronoi(self, points):
        # 取凸包，确保所有点都能取到一个多边形
        # radius单位为度，0.00001约等于1米
        radius = 1.0
        convexbuffer = MultiPoint(points).convex_hull.buffer(radius)
        points.extend(convexbuffer.exterior.coords)
        # 沃罗诺伊图
        vor = Voronoi(points, furthest_site=False, incremental=True, qhull_options=None)
        voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        # 泰森多边形的顶点
        vertices = vor.vertices
        # 泰森多边形的面，-1代表无穷索引
        regions = vor.regions
        # 画出框
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
        # 图形上色，未完成
        for i, region in enumerate(regions):
            # c_i = color[vor.point_region[i]]
            # if -1 in region:
            # c_i = vor.point_region[i]
            polygon = [vor.vertices[n] for n in region]
            plt.fill(*zip(*polygon), color="#ffffff")

    # draw line and points
    def drawGridMap(self, x, y, color):
        plt.scatter(x, y, color=color, s=320, marker='8')
        plt.plot(x, y)
        # plt.axis([-18, 8, -2, 22])
        plt.axis([-10, 4, 1, 14])
        # plt.xlim([-9,0]), plt.ylim([3,12])
        plt.savefig('voro.png')
        plt.show()

    # demo
    def createBaseEnv(self):
        # datasource
        data = pd.read_csv(self.file)
        x = data['x']
        y = data['y']
        el = data['element']
        LOD = data['LOD']
        color = data['color']
        # 要生成德劳内三角网和泰森多边形的点文件
        data_point = pd.read_csv(self.file, usecols=['x', 'y'])
        # all points
        points = data_point.values.tolist()
        point_length = len(points)
        # 取凸包，确保所有点都能取到一个多边形
        # radius单位为度，0.00001约等于1米
        radius = 1.0
        convexbuffer = MultiPoint(points).convex_hull.buffer(radius)
        points.extend(convexbuffer.exterior.coords)
        # 沃罗诺伊图
        vor = Voronoi(points, furthest_site=False, incremental=True, qhull_options=None)
        voronoi_plot_2d(vor, show_vertices=False, show_points=False)
        # 泰森多边形的顶点
        vertices = vor.vertices
        # 泰森多边形的面，-1代表无穷索引
        regions = vor.regions
        # 画出框
        for simplex in vor.ridge_vertices:
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                plt.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], 'k-')
        # 图形上色，未完成
        for i, region in enumerate(regions):
            # c_i = color[vor.point_region[i]]
            # if -1 in region:
            # c_i = vor.point_region[i]
            polygon = [vor.vertices[n] for n in region]
        # if i<len(color):
        #     plt.fill(*zip(*polygon), color="#000000")

        plt.scatter(x, y, color=color, s=150, marker='8')
        plt.plot(x, y)
        plt.axis([-18, 8, -2, 22])
        # plt.xlim([-9,0]), plt.ylim([3,12])
        plt.savefig('voro.png')
        plt.show()

    # init don
    def bornToDes(self, bornObj, destinationObj):
        plt.plot(bornObj.x, bornObj.y, marker=bornObj.s, color=bornObj.c)
        plt.plot(destinationObj.x, destinationObj.y, marker=destinationObj.s, color=destinationObj.c)

    # TODO Agent move,六个方向，步长也需要重新定义，
    def action(self, direction, step):

        return

    # TODO save trace
    def saveTrace(self):
        return

    # TODO 沿着边界移动且不超出边界
    def stopInSide(self):
        return

    # TODO 沿着边界移动
    def moveBySide(self):
        return

    # TODO 试图越过边界
    def acrossSide(self):
        return
