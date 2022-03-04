from __future__ import print_function

# hexagonal tiling walk directions
directions = [
    {'x': +1, 'y': -1, 'z': 0},
    {'x': +1, 'y': 0, 'z': -1},
    {'x': 0, 'y': +1, 'z': -1},
    {'x': -1, 'y': +1, 'z': 0},
    {'x': -1, 'y': 0, 'z': +1},
    {'x': 0, 'y': -1, 'z': +1}
]

# write a new hexagon
from shapely.geometry.polygon import Polygon


def write_hex(c, file):
    # conversion from hex coordinates to rect
    x = int(2 * (c['x'] + c['z'] / 2.0))
    y = 2 * c['z']

    hex = Polygon(((x, y + 2), (x + 1, y + 1), (x + 1, y), (x, y - 1), (x - 1, y), (x - 1, y + 1)))

    print(hex.wkt, file=file)


# start the walk from the origin cell, facing east
position = {'x': 0, 'y': 0, 'z': 0}
dir_i = 0


# print hexagons according to the given sequence and the global status
def emit(sequence, file):
    global dir_i, position

    for char in sequence:
        if char == '+':
            dir_i = (dir_i + 1) % len(directions)
        elif char == '-':
            dir_i = dir_i - 1
            if dir_i == -1:
                dir_i = 5
        elif char == 'F':
            dir = directions[dir_i]
            position = {'x': position['x'] + dir['x'], 'y': position['y'] + dir['y'], 'z': position['z'] + dir['z']}
            write_hex(position, file)


# emit characters executing a Lindenmayer system given an axiom, a number of steps and rules
def fractalize(input, steps, rules, file):
    if steps == 0:
        emit(input, file)
        return

    for char in input:
        output = ''

        if char in rules:
            output += rules[char]
        else:
            output += char

        fractalize(output, steps - 1, rules, file)


# determine the order of the fractal
ORDER = 6

# bufsize is 1: line-by-line
with open('hexes.wkt.csv', 'w', 1) as output:
    # write the origin
    write_hex(position, output)

    fractalize(
        file=output,
        input='A',
        steps=ORDER,
        rules={
            'A': 'A+BF++BF-FA--FAFA-BF+',
            'B': '-FA+BFBF++BF+FA--FA-B'
        })
