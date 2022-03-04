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

    point_x = [x,]
    point_y = [y,]

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