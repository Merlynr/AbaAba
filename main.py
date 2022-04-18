import time

from envir.GridMapEnv import GridMapEnv, Points

if __name__ == '__main__':
    # 设置边界
    border_x = [-10, 3]
    border_y = [1, 14]
    # 惩罚
    punish = []

    born = Points(1.73, 13, '8', 'r')
    dest = Points(-5.20, 6.00, '8', 'r')
    gridMap = GridMapEnv('./envir/gosper.csv',border_x,border_y,punish)
    (data, points) = gridMap.readData()
    gridMap.createVoronoi(points)
    gridMap.bornToDes(born, dest)
    gridMap.drawGridMap(data['x'], data['y'], data['color'])
    # 运动
    point = gridMap.action(born, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 100, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 1, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 11, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 10, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 100, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 110, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 100, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(point, 0, 1)
    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()