from envir.GridMapEnv import GridMapEnv


class Point:
    def __init__(self, x, y, style, color):
        self.x = x
        self.y = y
        self.s = style
        self.c = color

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getStyle(self):
        return self.s

    def getColor(self):
        return self.c


if __name__ == '__main__':
    born = Point(1.73,13,'8','r')
    dest = Point(-5.20,6.00,'8','r')
    gridMap = GridMapEnv('./envir/gosper.csv')
    (data, poins) = gridMap.readData()
    gridMap.createVoronoi(poins)
    gridMap.bornToDes(born,dest)
    gridMap.drawGridMap(data['x'], data['y'], data['color'])


