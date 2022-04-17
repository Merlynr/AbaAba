import time

from envir.GridMapEnv import GridMapEnv


class Point:
    def __init__(self, x, y, style, color):
        self.x = x
        self.y = y
        self.s = style
        self.c = color

    def __repr__(self):
        return f'Point({self.x!r}, {self.y!r}, {self.s!r}, {self.c!r})'


if __name__ == '__main__':
    born = Point(1.73, 13, '8', 'r')
    dest = Point(-5.20, 6.00, '8', 'r')
    gridMap = GridMapEnv('./envir/gosper.csv')
    (data, points) = gridMap.readData()
    gridMap.createVoronoi(points)
    gridMap.bornToDes(born, dest)
    gridMap.drawGridMap(data['x'], data['y'], data['color'])
    # 运动
    point = gridMap.action(born, 0, 1)
    point = gridMap.action(point, 100, 1)
    point = gridMap.action(point, 0, 1)
    point = gridMap.action(point, 1, 1)
    point = gridMap.action(point, 10, 1)
    point = gridMap.action(point, 11, 1)
    point = gridMap.action(born, 0, 1)
    point = gridMap.action(point, 100, 1)
    point = gridMap.action(point, 110, 1)
    point = gridMap.action(point, 100, 1)
    point = gridMap.action(point, 100, 1)

    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()

    point = gridMap.action(born, 0, 1)

    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()
    time.sleep(1)
    point = gridMap.action(born, 0, 1)

    gridMap.saveTrace()
    gridMap.showTrace()
    gridMap.show()
