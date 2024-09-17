from common import *

import numpy as np
from typing import *

# concepts:
# complete shape, object detection

# description:
# In the input you will see an imcomplete teal ractangle
# To make the output grid, you should use the red color to complete the rectangle.

def main(input_grid):
    # Find the bounding box of the incomplete rectangle
    x, y, x_len, y_len = bounding_box(grid=input_grid)
    rectangle = input_grid[x:x + x_len, y:y + y_len]

    # Find out the missing parts of the rectangle and complete it with red color
    rectangle = np.where(rectangle == Color.BLACK, Color.RED, rectangle)
    output_grid = np.copy(input_grid)
    output_grid = blit_sprite(grid=output_grid, sprite=rectangle, x=x, y=y)

    return output_grid

def generate_input():
    # Generate a grid with a size of n x m
    n, m = np.random.randint(7, 15), np.random.randint(7, 15)
    grid = np.zeros((n, m), dtype=int)

    # Randomly generate a rectangle with a size of x_len x y_len, not too big nor too small
    x_len = np.random.randint(n // 2, n - 2)
    y_len = np.random.randint(m // 2, m - 2)

    # Randomly generate a rectangle with a size of x_len x y_len that is incomplete
    rectangle = random_sprite(n=x_len, m=y_len, color_palette=[Color.TEAL], density=0.2)

    # Draw half of the border of the rectangle
    # Randomly choose a position to draw the border
    line_pos = random.choice([[0, 0], [0, y_len - 1], [x_len - 1, 0], [x_len - 1, y_len - 1]])
    if line_pos[0] == 0 and line_pos[1] == 0:
        direction_horizontal = (1, 0)
        direction_vertical = (0, 1)
    elif line_pos[0] == 0 and line_pos[1] == y_len - 1:
        direction_horizontal = (1, 0)
        direction_vertical = (0, -1)
    elif line_pos[0] == x_len - 1 and line_pos[1] == 0:
        direction_horizontal = (-1, 0)
        direction_vertical = (0, 1)
    else:
        direction_horizontal = (-1, 0)
        direction_vertical = (0, -1)

    # Draw half of the border of the rectangle
    rectangle = draw_line(grid=rectangle, x=line_pos[0], y=line_pos[1], direction=direction_horizontal, length=x_len, color=Color.TEAL)
    rectangle = draw_line(grid=rectangle, x=line_pos[0], y=line_pos[1], direction=direction_vertical, length=y_len, color=Color.TEAL)

    # Randomly choose a position to draw the rectangle
    x, y = random_free_location_for_sprite(grid=grid, sprite=rectangle, border_size=1)
    grid = blit_sprite(grid=grid, sprite=rectangle, x=x, y=y)

    return grid


# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)