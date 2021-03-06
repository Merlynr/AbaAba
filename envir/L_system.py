def replacement(str, rules, order):
    for i in range(order):
        dst = ""
        for s in str:
            if s in rules:
                dst += rules[s]
            else:
                dst += s
        str = dst
    return str


def interpretation(str, actions, rotate, angle, x, y):
    import math

    point_x = [x, ]
    point_y = [y, ]

    for s in str:
        if s not in actions:
            continue
        if actions[s] == "left":
            angle -= rotate
        elif actions[s] == "right":
            angle += rotate
        elif actions[s] == "forward":
            r = angle / 180.0 * math.pi
            point_x.append(point_x[-1] + math.cos(r))
            point_y.append(point_y[-1] + math.sin(r))
    return point_x, point_y


grammer = {
    "start": "A",
    "rules": {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"},
}

geometry = {
    "rotate": 60,
    "actions": {"+": "left", "-": "right", "A": "forward", "B": "forward"}
}
import pandas as pd

str = replacement(grammer["start"], grammer["rules"], 3)
x, y = interpretation(str, geometry["actions"], geometry["rotate"], 30, 0, 0)
domData = pd.read_csv('dom.csv', usecols=['element', 'LOD', 'color'])
'''
添加默认值，使得两个list长度一样
'''
LOD = domData['LOD'].values.tolist()
element = domData['element'].values.tolist()
color = domData['color'].values.tolist()
max_len = max(len(x), len(element))
difference_len = max_len - len(element)

num = 75
lod = (['-1']*num)
lod.extend(LOD)
lod.extend(['-1'] * (difference_len-num))
el = (['@']*num)
el.extend(element)
el.extend(['@'] * (difference_len-num))
co = (['#000000']*num)
co.extend(color)
co.extend(['#000000'] * (difference_len-num))
# LOD.extend(['-1'] * difference_len)
# element.extend('@' * difference_len)
# color.extend(['#000000'] * difference_len)
dataframe = pd.DataFrame({'x': ['{:.2f}'.format(i) for i in x], 'y': ['{:.2f}'.format(i) for i in y], 'element': el, 'LOD': lod,'color':co})
dataframe.to_csv("gosper.csv", sep=',')
