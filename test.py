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
    coordinates = [[point_x, point_y], ]

    stack = []
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
        elif actions[s] == "push":
            stack.append((angle, point_x[-1], point_y[-1]))
        elif actions[s] == "pop":
            angle, _x, _y = stack[-1]
            stack = stack[:-1]

            point_x = [_x, ]
            point_y = [_y, ]
            coordinates.append([point_x, point_y])

    return coordinates

# Gosper 曲线
grammer = {
  "start": "A",
  "rules": {"A": "A-B--B+A++AA+B-", "B": "+A-BB--B-A++A+B"},
}

geometry = {
  "rotate": 60,
  "actions": {"+": "left", "-": "right", "A": "forward", "B": "forward"}
}

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 4)
for i in range(4):
    str = replacement(grammer["start"], grammer["rules"], i + 1)
    x, y = interpretation(str, geometry["actions"], geometry["rotate"], 30, 0, 0)

    sub = ax[i]
    sub.axis("off")
    sub.set_aspect("equal")
    sub.plot(x, y)

plt.show()