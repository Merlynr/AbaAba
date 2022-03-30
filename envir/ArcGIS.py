# %% envir
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import polygonize
import pandas as pd
# %% data
# 要生成德劳内三角网和泰森多边形的点文件
data_point = pd.read_csv('gosper.csv',usecols=['x', 'y'])
# all points
points = data_point.values.tolist()
# 记录点个数，德劳内三角形用
pointlength = len(points)

# %%
# 取凸包，确保所有点都能获取一个多边形
# radius单位为度，0.00001约等于1米
radius = 1.0
convexbuffer = MultiPoint(points).convex_hull.buffer(radius)
points.extend(convexbuffer.exterior.coords)
array = np.array(points)

# %%
# 沃罗诺伊图
vor = Voronoi(array,furthest_site=False, incremental=True, qhull_options=None)
# 泰森多边形的顶点
vertices = vor.vertices
fvertices = open('voronoi_vertices.txt','a',encoding='utf-8')
for index,v in enumerate(vertices):
    fvertices.write(str(index)+'\t'+'POINT('+str(v[0])+' '+str(v[1])+')'+'\n')
fvertices.close()
# 泰森多边形的面，-1代表无穷索引
regions = vor.regions
fregions = open('voronoi_regions.txt','a',encoding='utf-8')
for index,r in enumerate(regions):
    if len(r) == 0:
        continue
    if -1 in r:
        continue
    angulars = []
    for id in r:
        angulars.append(vertices[id])
    angulars.append(vertices[r[0]])
    polygon = Polygon(angulars)
    fregions.write(str(index)+'\t'+str(polygon.wkt)+'\n')
fregions.close()
# 德劳内三角形的边，用原始的点数量
vorOriginal = Voronoi(array[0:pointlength],furthest_site=False, incremental=True, qhull_options=None)
ridge_points = vorOriginal.ridge_points
polylines = []
for ridge in ridge_points:
    polyline = LineString([points[ridge[0]],points[ridge[1]]])
    polylines.append(polyline)
# 德劳内三角形构面
delaunays = polygonize(polylines)
fdelaunay = open(r'voronoi_delaunays.txt','a',encoding='utf-8')
for index,p in enumerate(delaunays):
    fdelaunay.write(str(index)+'\t'+str(p.wkt)+'\n')
fdelaunay.close()
# 泰森多边形的边的索引数组，-1代表无穷索引
ridge_vertices = vor.ridge_vertices
# 打印到控制台上看看结果
print('vertices')
print(vertices)
print('region')
print(regions)
print('ridge_points')
print(ridge_points)
print('ridge_vertices')
print(ridge_vertices)
# 打开界面，看看效果
fig = voronoi_plot_2d(vor)
plt.show()