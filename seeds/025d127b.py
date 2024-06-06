from common import *

import numpy as np
from typing import *

black, blue, red, green, yellow, grey, pink, orange, teal, maroon = range(10)

# concepts:
# objects

# description:
# In the input you will see a set of objects, each consisting of a horizontal top/bottom and diagonal left/right edges (but that structure is not important)
# To make the output shift right each pixel in the object *except* when there are no other pixels down and to the right

def main(input_grid: np.ndarray) -> np.ndarray:
    # find the connected components, which are monochromatic objects
    objects = find_connected_components(input_grid, background=black, connectivity=8, monochromatic=True)

    output_grid = np.zeros_like(input_grid)

    for obj in objects:
        transformed_object = np.zeros_like(obj)

        for x in range(obj.shape[0]):
            for y in range(obj.shape[1]):
                if obj[x, y] != black:
                    # check that there are other colored pixels down and to the right
                    down_and_to_the_right = obj[x+1:, y+1:]
                    if np.any(down_and_to_the_right != black):
                        transformed_object[x+1, y] = obj[x, y]
                    else:
                        transformed_object[x, y] = obj[x, y]

        blit(output_grid, transformed_object, 0, 0, transparent=black)

    return output_grid


def generate_input() -> np.ndarray:
    n, m = np.random.randint(10, 30), np.random.randint(10, 30)
    grid = np.zeros((n, m), dtype=int)

    n_objects = np.random.randint(1, 3)

    for _ in range(n_objects):
        color = np.random.randint(1, 10)

        bar_width = np.random.randint(3, n//2)
        side_height = np.random.randint(3, m - bar_width)

        width, height = bar_width+side_height, side_height
        obj = np.zeros((width, height), dtype=int)

        # make the horizontal top edge
        obj[:bar_width+1, 0] = color
        # make the horizontal bottom edge
        obj[-bar_width:, -1] = color
        # make the diagonal left edge
        for i in range(side_height):
            obj[i, i] = color
        # make the diagonal right edge
        for i in range(side_height-1):
            obj[bar_width+i+1, i] = color

        # place the object randomly on the grid, assuming we can find a spot
        try:
            x, y = random_free_location_for_object(grid, obj, background=black)
        except:
            continue

        blit(grid, obj, x, y, transparent=black)

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)