from common import *

import numpy as np
from typing import *

# concepts:
# boundary detection, object extraction

# description:
# In the input you will see several teal pixels and two vertical parallel gray lines with four teal pixels indicates the boundary of the output grid.
# To make the output grid, you should extract the part of grid that is bounded by the two vertical parallel gray lines and four teal pixels in each corner.

def main(input_grid):
    # Detect the vertical parallel gray lines.
    vertical_lines = detect_objects(grid=input_grid, colors=[Color.GRAY], monochromatic=True, connectivity=4)
    pos_list = []
    for vertical_line in vertical_lines:
        pos_x, pos_y, length_v, height_v = bounding_box(grid=vertical_line)
        pos_list.append((pos_x, pos_y, length_v, height_v))
    
    # Get the left upper position and width, length of the extract part.
    pos_list.sort(key=lambda x: x[0])
    x1, y1 = pos_list[0][0], pos_list[0][1] - 1
    x2, y2 = pos_list[1][0], pos_list[1][1] + pos_list[1][3]

    # Extract the bounded part of the grid.
    output_grid = input_grid[x1:x2 + 1, y1:y2 + 1]
    return output_grid

def generate_input():
    # Generate the background grid with size of n x m.
    n, m = np.random.randint(9, 15), np.random.randint(9, 15)
    grid = np.zeros((n, m), dtype=int)

    # Randomly scatter color pixels on the grid.
    def random_scatter_point_on_grid(grid, color, density):
        n, m = grid.shape
        colored = 0
        # Randomly scatter density of color pixels on the grid.
        while colored < density * n * m:
            x = np.random.randint(0, n)
            y = np.random.randint(0, m)
            if grid[x, y] == Color.BLACK:
                grid[x, y] = color
                colored += 1
        return grid
    
    # Generate random teal pixels on the grid.
    grid = random_scatter_point_on_grid(grid=grid, color=Color.TEAL, density=0.2)

    # Randomly get the width, length and position of the extract part.
    width, length = np.random.randint(4, n - 1), np.random.randint(4, m - 1)
    x, y = np.random.randint(0, n - width), np.random.randint(0, m - length)

    # Draw two vertical parallel gray lines with four teal pixels in each corner to indicate the boundary of the extract part.
    draw_line(grid=grid, x=x, y=y, color=Color.GRAY, direction=(0, 1), length=length)
    draw_line(grid=grid, x=x + width, y=y, color=Color.GRAY, direction=(0, 1), length=length)

    grid[x, y] = Color.TEAL
    grid[x, y + length] = Color.TEAL
    grid[x + width, y] = Color.TEAL
    grid[x + width, y + length] = Color.TEAL

    return grid

# ============= remove below this point for prompting =============

if __name__ == '__main__':
    visualize(generate_input, main)
