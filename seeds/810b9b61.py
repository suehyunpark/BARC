from common import *

import numpy as np
from typing import *

# concepts:
# objects, topology

# description:
# In the input grid, you will see various blue objects. Some are "hollow" and contain a fully-enclosed region, while others do not have a middle that is separate from outside the object, and fully enclosed.
# To create the output grid, copy the input grid. Then, change the color of all "hollow" shapes to be green.

def main(input_grid):
    objects = find_connected_components(input_grid, connectivity=4)
    output_grid = input_grid.copy()
    for object in objects:
        if is_hollow(object):
            object[object != Color.BLACK] = Color.GREEN
        blit(output_grid, object, background=Color.BLACK)

    return output_grid

def is_hollow(object):
    # to check if hollow, we can use flood fill:
    # - place the object in a larger black array
    # - apply flood fill starting "outside" the object
    # - if there are black squares after flood filling, the object is hollow
    
    # Another way is to use topology primitives:
    # interior_mask = object_interior(object)
    # boundary_mask = object_boundary(object)
    # inside_but_not_on_edge = interior_mask & ~boundary_mask
    # hollow = np.any(inside_but_not_on_edge)
    
    padded = np.pad(object, pad_width=1)
    flood_fill(padded, 0, 0, color=Color.GREEN, connectivity=4)
    return np.any(padded == Color.BLACK)


def generate_input():
    n = np.random.randint(10, 28)
    input_grid = np.full((n, n), Color.BLACK)
    # create a bunch of random objects. all objects are either (1) hollow, in which case they are the border of a rectangle of some size, or (2) not hollow, in which case they are a subset of a border of a rectangle of some size.
    # make sure we place at least one hollow and nonhollow object. then add random objects until somewhat full.

    def random_hollow_object():
        n, m = np.random.randint(3, 7), np.random.randint(3, 7)
        obj = np.full((n, m), Color.BLUE)
        obj[1:n-1, 1:m-1] = Color.BLACK
        return obj

    def random_nonhollow_object():
        obj = random_hollow_object()
        # remove a random number of dots from it
        size = np.count_nonzero(obj)
        new_size = np.random.randint(1, size)
        xs, ys = np.where(obj != Color.BLACK)
        for i in range(size - new_size):
            obj[xs[i], ys[i]] = Color.BLACK

        return obj

    try:
        # add one hollow and one nonhollow object, then add random objects until somewhat full.
        obj = random_hollow_object()
        x, y = random_free_location_for_object(input_grid, obj, padding=1)
        blit(input_grid, obj, x, y)

        obj = random_nonhollow_object()
        x, y = random_free_location_for_object(input_grid, obj, padding=1)
        blit(input_grid, obj, x, y)
    except ValueError:
        return generate_input()

    while True:
        obj = random_hollow_object() if np.random.rand() < 0.5 else random_nonhollow_object()
        try:
            x, y = random_free_location_for_object(input_grid, obj, padding=1)
            blit(input_grid, obj, x, y)
        except ValueError:
            return input_grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
