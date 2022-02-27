# https://www.tspweb.com/key/%E7%A9%BA%E9%97%B4%E5%A1%AB%E5%85%85%E6%9B%B2%E7%BA%BF.html
def _gosper_A(direction, order):

    if order == 0:
        step(direction)
        return

    _gosper_A(direction, order - 1)

    direction -= 1
    _gosper_B(direction, order - 1)

    direction -= 2
    _gosper_B(direction, order - 1)

    direction += 1
    _gosper_A(direction, order - 1)

    direction += 2
    _gosper_A(direction, order - 1)
    _gosper_A(direction, order - 1)

    direction += 1
    _gosper_B(direction, order - 1)

def _gosper_B(direction, order):
    if order == 0:
        step(direction)
        return

    direction += 1
    _gosper_A(direction, order - 1)

    direction -= 1
    _gosper_B(direction, order - 1)
    _gosper_B(direction, order - 1)

    direction -= 2
    _gosper_B(direction, order - 1)

    direction -= 1
    _gosper_A(direction, order - 1)

    direction += 2
    _gosper_A(direction, order - 1)

    direction += 1
    _gosper_B(direction, order - 1)

cos30 = 3.0 ** 0.5 / 2.0
sin30 = 0.5
nexts = {0: (cos30, sin30), 1: (0, 1), 2: (-cos30, sin30), 3: (-cos30, -sin30), 4: (0, -1), 5: (cos30, -sin30)}
def step(direction):
    next = nexts[direction % 6]

    global x, y
    x.append(x[-1] + next[0])
    y.append(y[-1] + next[1])

def gosper(order):
    global x, y
    x = [0,]
    y = [0,]

    _gosper_A(0, order)

    return (x, y)

import matplotlib.pyplot as plt
x, y = gosper(5);
plt.plot(x, y);
plt.show()