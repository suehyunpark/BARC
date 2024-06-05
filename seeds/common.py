"""Common library for ARC"""
import numpy as np
import random

black, blue, red, green, yellow, grey, pink, orange, teal, maroon = range(10)


def flood_fill(grid, x, y, color, background=black):
    """
    Fill the connected region that contains the point (x, y) with the specified color.
    """

    assert color != background, "Color and background must be different."
    
    if grid[x, y] != background:
        return
    
    # must be equal to the background, therefore we only recurse if that point is the background
    grid[x, y] = color

    # flood fill in all directions
    if x > 0:
        flood_fill(grid, x - 1, y, color, background)
    if x < grid.shape[0] - 1:
        flood_fill(grid, x + 1, y, color, background)
    if y > 0:
        flood_fill(grid, x, y - 1, color, background)
    if y < grid.shape[1] - 1:
        flood_fill(grid, x, y + 1, color, background)

def draw_line(grid, x, y, length, color, direction):
    """
    Draws a line of the specified length in the specified direction starting at (x, y).
    Direction should be a vector with elements -1, 0, or 1.
    If length is None, then the line will continue until it hits the edge of the grid.

    Example:
    draw_line(grid, 0, 0, length=3, color=blue, direction=(1, 1)) will draw a diagonal line of blue pixels from (0, 0) to (2, 2).
    """
    
    if length is None:
        length = max(grid.shape)*2
    
    for i in range(length):
        new_x = x + i * direction[0]
        new_y = y + i * direction[1]
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            grid[new_x, new_y] = color

    return grid

def blit(grid, sprite, x, y, transparent=None):
    """
    Copies the sprite into the grid at the specified location. Modifies the grid in place.
    """

    new_grid = grid

    for i in range(sprite.shape[0]):
        for j in range(sprite.shape[1]):
            if transparent is None or sprite[i, j] != transparent:
                # check that it is inbounds
                if 0 <= x + i < grid.shape[0] and 0 <= y + j < grid.shape[1]:
                    new_grid[x + i, y + j] = sprite[i, j]

    return new_grid

def collision(_=None, object1=None, object2=None, x1=0, y1=0, x2=0, y2=0, background=black):
    """
    Check if object1 and object2 collide when object1 is at (x1, y1) and object2 is at (x2, y2).
    """
    n1, m1 = object1.shape
    n2, m2 = object2.shape

    dx = x2 - x1
    dy = y2 - y1

    for x in range(n1):
        for y in range(m1):
            if object1[x, y] != background:
                new_x = x - dx
                new_y = y - dy
                if 0 <= new_x < n2 and 0 <= new_y < m2 and object2[new_x, new_y] != background:
                    return True
    
    return False


def random_free_location_for_object(grid, sprite, background=black):
    """Find a random free location for the sprite in the grid."""

    n, m = grid.shape
    dim1, dim2 = sprite.shape
    possible_locations = [(x,y) for x in range(0, n - dim1) for y in range(0, m - dim2)]

    non_background_grid = np.sum(grid != background)
    non_background_sprite = np.sum(sprite != background)
    target_non_background = non_background_grid + non_background_sprite

    # prune possible locations by making sure there is no overlap with non-background pixels if we were to put the sprite there
    pruned_locations = []
    for x, y in possible_locations:
        # try blitting the sprite and see if the resulting non-background pixels is the expected value
        new_grid = grid.copy()
        new_grid[x:x+dim1, y:y+dim2] = np.maximum(new_grid[x:x+dim1, y:y+dim2], sprite)
        if np.sum(new_grid != background) == target_non_background:
            pruned_locations.append((x, y))

    if len(pruned_locations) == 0:
        raise ValueError("No free location for sprite found.")
    
    return random.choice(pruned_locations)


def show_colored_grid(grid):
    """Not used by the language model, used by the rest of the code for debugging"""

    color_names = ['black', 'blue', 'red', 'green', 'yellow', 'grey', 'pink', 'orange', 'teal', 'maroon']
    color_8bit = {"black": 0, "blue": 4, "red": 1, "green": 2, "yellow": 3, "grey": 7, "pink": 13, "orange": 202, "teal": 6, "maroon": 196}

    for row in grid:
        for cell in row:
            color_code = color_8bit[color_names[cell]]
            print(f"\033[38;5;{color_code}m{cell}\033[0m", end="")
        print()

    
def visualize(input_generator, transform, n_examples=5):
    """Not used by the language model. For us to help with debugging"""
        
    for index in range(n_examples):
        input_grid = input_generator()
        print("Input:")
        show_colored_grid(input_grid)

        output_grid = transform(input_grid)
        print("Output:")
        show_colored_grid(output_grid)

        if index < n_examples-1:
            print("\n\n---------------------\n\n")





# ------------------- API for generating Sprites (in progress) -------------------
def apply_symmetry(sprite, symmetry_type):
    """Apply the specified symmetry within the bounds of the sprite."""
    n, m = sprite.shape
    if symmetry_type == 'horizontal':
        for y in range(m):
            for x in range(n // 2):
                sprite[x, y] = sprite[n - 1 - x, y] = sprite[x, y] or sprite[n - 1 - x, y]
    elif symmetry_type == 'vertical':
        for x in range(n):
            for y in range(m // 2):
                sprite[x, y] = sprite[x, m - 1 - y] = sprite[x, y] or sprite[x, m - 1 - y]
    return sprite

def apply_diagonal_symmetry(sprite):
    """Apply diagonal symmetry within the bounds of the sprite. Assumes square sprite."""
    n, m = sprite.shape
    if n != m:
        raise ValueError("Diagonal symmetry requires a square sprite.")
    for x in range(n):
        for y in range(x+1, m):
            sprite[x, y] = sprite[y, x] = sprite[x, y] or sprite[y, x]
    return sprite

def is_contiguous(sprite):
    """Check if a sprite is contiguous"""
    from scipy.ndimage import label
    labeled, n_objects = label(sprite)
    return n_objects == 1
    

def generate_sprite(n, m, symmetry_type, fill_percentage=0.5, max_colors=9, color_palate=None):
    # pick random colors, number of colors follows a geometric distribution truncated at 9
    if color_palate is None:
        n_colors = 1
        while n_colors < max_colors and random.random() < 0.3:
            n_colors += 1
        color_palate = random.sample(range(1, 10), n_colors)
    else:
        n_colors = len(color_palate)
    
    grid = np.zeros((n, m))
    if symmetry_type == "not_symmetric":
        x, y = random.randint(0, n-1), random.randint(0, m-1)
    elif symmetry_type == 'horizontal':
        x, y = random.randint(0, n-1), m//2
    elif symmetry_type == 'vertical':
        x, y = n//2, random.randint(0, m-1)
    elif symmetry_type == 'diagonal':
        # coin flip for which diagonal orientation
        diagonal_orientation = random.choice([True, False])
        x = random.randint(0, n-1)
        y = x if diagonal_orientation else n - 1 - x
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    color_index = 0
    while np.sum(grid>0) < fill_percentage * n * m:
        grid[x, y] = color_palate[color_index]
        if random.random() < 0.33:
            color_index = random.choice(range(n_colors))
        dx, dy = random.choice(moves)
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < m:
            x, y = new_x, new_y

    #return grid

    if symmetry_type == 'horizontal':
        grid = apply_symmetry(grid, 'horizontal')
    elif symmetry_type == 'vertical':
        grid = apply_symmetry(grid, 'vertical')
    elif symmetry_type == 'diagonal':        
        # diagonal symmetry goes both ways, flip a coin to decide which way
        if diagonal_orientation:
            grid = np.flipud(grid)
            grid = apply_diagonal_symmetry(grid)
            grid = np.flipud(grid)
        else:
            grid = apply_diagonal_symmetry(grid)

    return grid

def random_sprite(n, m, symmetry = None, color_palette = None):
    """
    Generate a sprite (an object), represented as a numpy array.

    n, m: dimensions of the sprite. If these are lists, then a random value will be chosen from the list.
    symmetry: optional type of symmetry to apply to the sprite. Can be 'horizontal', 'vertical', 'diagonal', 'not_symmetric'. If None, a random symmetry type will be chosen.
    color_palette: optional list of colors to use in the sprite. If None, a random color palette will be chosen.


    """
    # Decide on symmetry type before generating the sprites
    symmetry_types = ['horizontal', 'vertical', 'diagonal', "not_symmetric"]
    symmetry = symmetry or random.choice(symmetry_types)

    # Decide on dimensions
    if isinstance(n, list):
        n = random.choice(n)
    if isinstance(m, list):
        m = random.choice(m)    

    while True:
        sprite = generate_sprite(n, m, symmetry_type=symmetry, color_palate=color_palette)
        assert is_contiguous(sprite), "Generated sprite is not contiguous."
        # check that the sprite has pixels that are flushed with the border
        if np.sum(sprite[0, :]) > 0 and np.sum(sprite[-1, :]) > 0 and np.sum(sprite[:, 0]) > 0 and np.sum(sprite[:, -1]) > 0:
            return sprite