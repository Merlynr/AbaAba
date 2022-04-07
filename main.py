from envir.GridMapEnv import GridMapEnv


def show_env():
    gridMap = GridMapEnv('./envir/gosper.csv')
    (data, poins) = gridMap.readData()
    gridMap.createVoronoi(poins)
    gridMap.drawGridMap(data['x'], data['y'], data['color'])

if __name__ == '__main__':
    show_env()
    print()
