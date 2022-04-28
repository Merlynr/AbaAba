import csv
import os
import shutil
from turtle import st

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
from gym import logger, spaces
from gym.spaces.box import Box
# from gym.spaces.multi_binary import MultiBinary
import tensorflow as tf

from sklearn import preprocessing

# 点class
from tensorflow.python.ops.numpy_ops import np_random


class Points:
    def __init__(self, x=0.0, y=0.0, style='8', color='#ffffff', ele="@", LOD="0"):
        self.x = x
        self.y = y
        self.s = style
        self.c = color
        self.element = ele
        self.LOD = LOD

    def __repr__(self):
        return f'Point({self.x!r}, {self.y!r}, {self.s!r}, {self.c!r}, {self.element!r},{self.LOD!r})'


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
    def __init__(self):
        # 始发点，目的点，中转点
        self.born = Points()
        self.dest = Points()
        self.transit = Points()

        super(GridMapEnv, self).__init__()
        self.file = './envir/gosper.csv'
        # self.figure = plt.figure()
        self.trace = []
        self.mathTools = MathTools()
        # 轨迹
        self.trace_x = []
        self.trace_y = []
        # 图片保存
        self.filePath = './output/test'
        # 边界
        self.border_x = [-10, 3]
        self.border_y = [1, 14]
        # 惩罚# 奖罚,奖励>同级跳转》越级》非父子》跨页后非向下跳转》边界》层级limit
        self.punish = [50, 20, -1000, -1000, -1000, -1000, 3]
        self.data = {}
        # 跨页flag
        self.isToPageThenStep = 0
        # 动作空间，状态特征空间，动作空间3*7,根据当前环境决定
        self.action_space = spaces.Discrete(21)
        # 状态空间的构建,由已完成的 TODO 四个位置+场景名称+当前元素名称
        low = np.array([
            self.border_x[0],
            self.border_y[0], self.border_x[0],
            self.border_y[0], self.border_x[0],
            self.border_y[0], self.border_x[0],
            self.border_y[0]
        ], dtype=np.float32)
        high = np.array([
            self.border_x[1],
            self.border_y[1], self.border_x[1],
            self.border_y[1], self.border_x[1],
            self.border_y[1], self.border_x[1],
            self.border_y[1]
        ], dtype=np.float32)

        self.observation_space = Box(low, high)

        self.createVoronoi()
        self.bornToDes()
        self.drawGridMap()

    # read data from csv
    def readData(self):
        self.data = pd.read_csv(self.file)
        # x = data['x']
        # y = data['y']
        # el = data['element']
        # LOD = data['LOD']
        # color = data['color']
        # 要生成德劳内三角网和泰森多边形的点文件
        data_point = pd.read_csv(self.file, usecols=['x', 'y'])
        # all points
        points = data_point.values.tolist()
        return self.data, points

    # draw Voronoi
    def createVoronoi(self):
        # 动态
        plt.ion()
        # 取凸包，确保所有点都能取到一个多边形
        # radius单位为度，0.00001约等于1米
        (data, points) = self.readData()
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
    def drawGridMap(self):
        (data, points) = self.readData()
        plt.scatter(data['x'].tolist(), data['y'].tolist(), color=data['color'].tolist(), s=320, marker='8')
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
    def bornToDes(self):
        self.born = Points(1.73,13.00,'8','r')
        self.dest = Points(-8.66,5.00, '8', 'r')
        self.transit = self.born
        plt.plot(self.born.x, self.born.y, marker=self.born.s, color=self.born.c)
        plt.plot(self.dest.x, self.dest.y, marker=self.dest.s, color=self.dest.c)

    # DONE Agent move,六个方向，步长也需要重新定义，TODO 返回当前结点，奖惩，状态，done，_
    def step(self, direction):
        dires = {
            0: (self.toUP, 1),
            1: (self.toDOWN, 1),
            2: (self.toLEFTUP, 1),
            3: (self.toRIGHTDOWN, 1),
            4: (self.toLEFTDOWN, 1),
            5: (self.toRIGHTUP, 1),
            6: (self.toPage, 0),  # 可能存在单页面，无法跳转，原地并惩罚
            7: (self.toUP, 2),
            8: (self.toDOWN, 2),
            9: (self.toLEFTUP, 2),
            10: (self.toRIGHTDOWN, 2),
            11: (self.toLEFTDOWN, 2),
            12: (self.toRIGHTUP, 2),
            13: (self.toPage, 1),  # 可能存在单页面，无法跳转，原地并惩罚
            14: (self.toUP, 3),
            15: (self.toDOWN, 3),
            16: (self.toLEFTUP, 3),
            17: (self.toRIGHTDOWN, 3),
            18: (self.toLEFTDOWN, 3),
            19: (self.toRIGHTUP, 3),
            20: (self.toPage, 2),  # 可能存在单页面，无法跳转，原地并惩罚
        }
        act = dires.get(direction)
        if act:
            print('action==>', act[0].__name__, act[1])
            return act[0](act[1])

    # 存在边界惩罚则按边界惩罚计算
    def toUP(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y += round(1 * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            # self.acrossSide(point)
            # self.againstRules(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    def toDOWN(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y -= round(1 * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    def toLEFTUP(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y += round(self.mathTools.sin * step, 2)
        self.transit.x -= round(self.mathTools.cos * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    def toRIGHTDOWN(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y -= round(self.mathTools.sin * step, 2)
        self.transit.x += round(self.mathTools.cos * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    def toLEFTDOWN(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y -= round(self.mathTools.sin * step, 2)
        self.transit.x -= round(self.mathTools.cos * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    def toRIGHTUP(self, step):
        self.transit = self.getPointInfo(self.transit)
        tp = TracePart({}, {})
        tp.start_point = copy.copy(self.transit)
        self.transit.y += round(self.mathTools.sin * step, 2)
        self.transit.x += round(self.mathTools.cos * step, 2)
        if self.getPointInfo(self.transit) == 'taoyi':
            return self.createState(), -1000, False, self.transit
        else:
            self.transit = self.getPointInfo(self.transit)
            tp.end_point = copy.copy(self.transit)
            self.trace.append(tp)
            reward, done = self.acrossSide(self.transit, tp)
            self.saveTrace()
            if self.transit == self.getPointInfo(self.dest):
                return self.createState(), 100, False, self.transit
            return self.createState(), reward, done, self.transit

    # 页面之间跳转，只能向下
    def toPage(self, pageNum):
        self.transit = self.getPointInfo(self.transit)
        pageNums = self.getPageDom(self.transit)
        if (pageNum + 1) >= len(pageNums):
            # 负奖励 无效跨页
            print("无效页面跨页，负奖励")
            return self.createState(), -1000, False, self.transit
        else:
            tp = TracePart({}, {})
            tp.start_point = copy.copy(self.transit)
            if self.getPointInfo(Point(pageNums.iloc[pageNum].values[1], pageNums.iloc[pageNum].values[2])) == 'taoyi':
                return self.createState(), -1000, False, self.transit
            else:
                self.transit = self.getPointInfo(
                    Point(pageNums.iloc[pageNum].values[1], pageNums.iloc[pageNum].values[2]))
                tp.end_point = copy.copy(self.transit)
                self.trace.append(tp)
                reward, done = self.acrossSide(self.transit, tp)
                self.isToPageThenStep = 1
                self.saveTrace()
                if self.transit == self.getPointInfo(self.dest):
                    return self.createState(), 100, False, self.transit
                return self.createState(), reward, done, self.transit

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
                # norm_x, norm_y = self.normalization(i.end_point.x, i.end_point.y)
                writer.writerows([[round(i.start_point.x, 2), round(i.start_point.y, 2), round(i.end_point.x, 2),
                                   round(i.end_point.y, 2)]])
        f.close()

    # DONE show trace
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
        if not os.path.getsize("./datasets/trace.csv"):
            return
        trace = pd.read_csv("./datasets/trace.csv")
        self.trace_x.extend(trace['end_x'].values.tolist())
        self.trace_x.insert(0, trace['origin_x'][0])
        self.trace_y.extend(trace['end_y'])
        self.trace_y.insert(0, trace['origin_y'][0])
        trace.drop(index=trace.index)
        trace.drop(index=trace.index)
        # 轨迹
        plt.plot(self.trace_x, self.trace_y, color='r')
        plt.pause(0.5)

    from matplotlib import animation
    def show(self):
        # plt.draw()
        # time.sleep(0.1)
        # plt.pause(0.2)
        plt.savefig(self.filePath)
        # plt.clf()
        plt.show()

    # DONE 试图越过边界 DONE 优化函数
    def acrossSide(self, point, tp):
        now_point = self.getPointInfo(point)
        if (now_point.x >= self.border_x[0] and now_point.x < self.border_x[1] and now_point.y >= self.border_y[
            0] and now_point.y <
                self.border_y[1]):
            if now_point.LOD == '-1':
                '''
                    无效数据，惩罚
                '''
                print("边界惩罚")
                return self.punish[5], False
            # 贱内
            else:
                return self.againstRules(tp)
        else:
            '''
                无效数据，惩罚
            '''
            print("边界惩罚")
            return self.punish[5], False

    # DONE 试图违法规则
    def againstRules(self, tp):
        s = tp.start_point.LOD.split(':')
        e = tp.end_point.LOD.split(':')
        # isLoger 判断层级
        isLoger = True if len(e) >= len(s) else False
        # 层数设置
        # LOD,相差为0即为同级，比较上一级
        LOD_l = abs(len(s) - len(e))
        # 非同级
        if LOD_l < self.punish[6]:
            # 向下
            if isLoger:
                # 同级
                if LOD_l == 0:
                    # 同脉
                    if e[:-1] == s[:-1]:
                        print("同级跳转，奖")
                        return self.punish[1], True
                    # 页面跳转错误
                    if self.isToPageThenStep == 1:
                        print("页面之后向同级层跳转 ERROR")
                        return self.punish[4], False
                else:
                    if e[:-LOD_l] == s:
                        self.isToPageThenStep = 0
                        print("跨级奖励")
                        return self.punish[0], True
                    else:
                        print("非同脉惩罚")
                        return self.punish[3], False
            # 向上
            else:
                # 页面跳转错误
                if self.isToPageThenStep == 1:
                    print("页面之后向上一层跳转 ERROR")
                    return self.punish[4], False
                # 向上
                if s[:-LOD_l] == e:
                    print("奖励")
                    return self.punish[0], True
                else:
                    print("非父子惩罚")
                    return self.punish[3], False
        # 页面跳转错误
        if self.isToPageThenStep == 1:
            print("页面之后同级元素进行跳转 ERROR")
            return self.punish[4], False
        # 越级惩罚
        else:
            print("越级惩罚")
            return self.punish[2], False

    # Done 根据坐标获取元素信息
    def getPointInfo(self, point):
        useful_data_0 = self.data[
            (self.data['x'] == point.x) & (self.data['y'] == point.y)]
        useful_data_1 = self.data[(self.data['x'] == (round((point.x - 0.01), 2))) & (self.data['y'] == point.y)]
        useful_data_2 = self.data[(self.data['x'] == (round((point.x + 0.01), 2))) & (self.data['y'] == point.y)]
        # print(useful_data_0, useful_data_1, useful_data_2)

        if not useful_data_0.empty:
            return Points(useful_data_0['x'].values[0], useful_data_0['y'].values[0], '',
                          useful_data_0['color'].values[0], useful_data_0['element'].values[0],
                          useful_data_0['LOD'].values[0])
        if not useful_data_1.empty:
            return Points(useful_data_1['x'].values[0], useful_data_1['y'].values[0], '',
                          useful_data_1['color'].values[0], useful_data_1['element'].values[0],
                          useful_data_1['LOD'].values[0])
        if not useful_data_2.empty:
            return Points(useful_data_2['x'].values[0], useful_data_2['y'].values[0], '',
                          useful_data_2['color'].values[0], useful_data_2['element'].values[0],
                          useful_data_2['LOD'].values[0])
        else:
            print("useful_data", useful_data_0, useful_data_1, useful_data_2)
            print("跳出数据集")
            return "taoyi"

    # DONE 查询page相关结点
    def getPageDom(self, point):
        pageDom = self.data[(self.data['element'] == point.element)]
        return pd.DataFrame(pageDom.drop(pageDom[(pageDom['x'] == point.x) & (pageDom['y'] == point.y)].index))

    def createState(self):
        obs = []
        trace_file = "./datasets/trace.csv"
        if not os.path.getsize(trace_file):
            a = np.zeros((6,))
            obs = np.hstack((a, ([self.born.x, self.born.y])))
        else:
            _trace = pd.read_csv(trace_file, usecols=['end_x', 'end_y'])
            if len(_trace) >= 4:
                obs = [
                    _trace.iloc[-4].values[0], _trace.iloc[-4].values[1],
                    _trace.iloc[-3].values[0], _trace.iloc[-3].values[1],
                    _trace.iloc[-2].values[0], _trace.iloc[-2].values[1],
                    _trace.iloc[-1].values[0], _trace.iloc[-1].values[1]
                ]
                obs = np.array(obs)

            if len(_trace) > 0 and len(_trace) < 4:
                a = np.zeros((8 - len(_trace) * 2))
                b = np.array((_trace['end_x'].values.tolist(), _trace['end_y'].values.tolist()), float)
                obs = np.hstack((a, b.reshape(-1)))

            if len(_trace) == 0:
                a = np.zeros((6,))
                obs = np.hstack((a, ([self.born.x, self.born.y])))

        self.observation_space = obs
        return obs

    # TODO 结点条件加入

    # 归一化
    def normalization(self, x, y):
        rang_x = self.border_x[1] - self.border_x[0]
        rang_y = self.border_y[1] - self.border_y[0]
        return (x - self.border_x[0]) / rang_x, (y - self.border_y[0]) / rang_y

    def reset(self):
        # 清空轨迹
        self.transit = self.born
        file_path = './datasets/trace.csv'
        self.trace = []
        self.data = {}
        # 跨页flag
        self.isToPageThenStep = 0
        plt.close()
        if os.path.isfile(file_path):
            os.remove(file_path)
            open(file_path, 'w').close()
        # f=open('', 'r+').close()
        return self.createState()

# TODO 目的地中止，
# TODO linux读取模型
