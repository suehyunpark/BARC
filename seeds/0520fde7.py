import numpy as np
from typing import *
from common import *

# concepts:
# boolean logical operations, bitmasks with separator

# description:
# In the input you will see two blue bitmasks separated by a grey vertical bar
# To make the output, color teal the red that are set in both bitmasks (logical AND)

def main(input_grid: np.ndarray) -> np.ndarray:
    # Find the grey vertical bar. Vertical means constant X
    for x_bar in range(input_grid.shape[0]):
        if np.all(input_grid[x_bar, :] == Color.GREY):
            break

    left_mask = input_grid[:x_bar, :]
    right_mask = input_grid[x_bar+1:, :]

    output_grid = np.zeros_like(left_mask)
    output_grid[(left_mask == Color.BLUE) & (right_mask == Color.BLUE)] = Color.RED
    
    return output_grid


def generate_input() -> np.ndarray:
    # create a pair of equally sized maroon bitmasks
    width, height = np.random.randint(2, 10), np.random.randint(2, 10)

    grid1 = np.zeros((width, height), dtype=int)
    grid2 = np.zeros((width, height), dtype=int)

    for x in range(width):
        for y in range(height):
            grid1[x, y] = np.random.choice([Color.BLUE, Color.BLACK])
            grid2[x, y] = np.random.choice([Color.BLUE, Color.BLACK])
    
    # create a blue vertical bar
    bar = np.zeros((1, height), dtype=int)
    bar[0, :] = Color.GREY

    grid = np.concatenate((grid1, bar, grid2), axis=0)

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
