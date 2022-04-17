import csv

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, LineString, Point, MultiPoint
import sys
import time
import pandas as pd
import matplotlib as mpl
import numpy as np
import copy
import math


class MathTools:
    def __init__(self):
        r = 30 / 180.0 * math.pi
        self.cos = math.cos(r)
        self.sin = math.sin(r)
# 用于保存每一步
class TracePart:
    def __init__(self, s, e):
        self.start_point = s
        self.end_point = e
        self._items = []

class GridMapEnv:
    def __init__(self, file):
        self.file = file
        self.figure = plt.figure()
        self.trace = []
        self.mathTools = MathTools()
        self.trace_x = []
        self.trace_y = []
        # 图片保存
        self.filePath = './output/test'

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
        # plt.plot(x, y)
        # plt.axis([-18, 8, -2, 22])
        plt.axis([-11, 4, 0.5, 14.5])
        # plt.xlim([-9,0]), plt.ylim([3,12])
        plt.savefig('./output/voro.png')
        # plt.show()

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
        # plt.show()

    # init don
    def bornToDes(self, bornObj, destinationObj):
        plt.plot(bornObj.x, bornObj.y, marker=bornObj.s, color=bornObj.c)
        plt.plot(destinationObj.x, destinationObj.y, marker=destinationObj.s, color=destinationObj.c)

    # DONE Agent move,六个方向，步长也需要重新定义，
    def action(self, point, direction, step):
        dires = {
            10: self.toUP,
            0: self.toDOWN,
            110: self.toLEFTUP,
            1: self.toRIGHTDOWN,
            100: self.toLEFTDOWN,
            11: self.toRIGHTUP
        }
        act = dires.get(direction)
        if act:
            return act(point, step)

    def toUP(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y += round(1 * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    def toDOWN(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y -= round(1 * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    def toLEFTUP(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y += round(self.mathTools.sin * step, 2)
        point.x -= round(self.mathTools.cos * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    def toRIGHTDOWN(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y -= round(self.mathTools.sin * step, 2)
        point.x += round(self.mathTools.cos * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    def toLEFTDOWN(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y -= round(self.mathTools.sin * step, 2)
        point.x -= round(self.mathTools.cos * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    def toRIGHTUP(self, point, step):
        tp = TracePart({}, {})
        tp.start_point = copy.copy(point)
        point.y += round(self.mathTools.sin * step, 2)
        point.x += round(self.mathTools.cos * step, 2)
        tp.end_point = copy.copy(point)
        self.trace.append(tp)
        return point

    # DONE save trace
    def saveTrace(self):
        # with open('./datasets/trace.txt', "w") as f:  # 设置文件对象
        #     for i in self.trace:  # 对于双层列表中的数据
        #         i = '(' + str(round(i.start_point.x,2)) + ',' + str(round(i.start_point.y,2)) + ')' + ' \t '+ '(' + str(
        #             round(i.end_point.x,2)) + ',' + str(round(i.end_point.y,2)) + ')\n'  # 将其中每一个列表规范化成字符串
        #         f.writelines(i)  # 写入文件

        with open(r"./datasets/trace.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["origin_x", "origin_y", "end_x", "end_y"])
            for i in self.trace:
                writer.writerows([[round(i.start_point.x, 2), round(i.start_point.y, 2), round(i.end_point.x, 2),
                                   round(i.end_point.y, 2)]])

    # TODO show trace
    def showTrace(self):
        # clear data
        self.trace_x = []
        self.trace_y = []
        # trace_data = np.loadtxt('./datasets/trace.txt')
        # trace  = pd.DataFrame(trace_data).to_csv('./datasets/trace.csv',index=False)
        # data = pd.read_table('./datasets/trace.txt',sep='\t',header=None)
        # list_0=data[0][0]
        # print(list_0)
        # plt.plot([line.start_point.x,line.start_point.y], [line.end_point.x,line.end_point.y], color='r',linestyle=':')
        trace = pd.read_csv("./datasets/trace.csv")
        self.trace_x.extend(trace['end_x'].values.tolist())
        self.trace_x.insert(0, trace['origin_x'][0])
        self.trace_y.extend(trace['end_y'])
        self.trace_y.insert(0, trace['origin_y'][0])
        # 轨迹
        print(self.trace_x)
        plt.plot(self.trace_x, self.trace_y, color='r')

    from matplotlib import animation
    def show(self):
        plt.savefig(self.filePath)
        plt.show()

    # TODO 试图越过边界
    def acrossSide(self):
        return

    # TODO 试图违法规则
    def againstRules(self):
        return

    #TODO 是否