"""Common library for ARC"""

import numpy as np
import random


class Color:
    """
    Enum for colors

    Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON

    Use Color.ALL_COLORS for `set` of all possible colors
    Use Color.NOT_BLACK for `set` of all colors except black

    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing on them.
    (The exception is Color.BLACK, which is 0)
    """

    # The above comments were lies to trick the language model into not treating the colours like ints
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def flood_fill(grid, x, y, color, connectivity=4):
    """
    Fill the connected region that contains the point (x, y) with the specified color.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. 8-way counts diagonals as connected, 4-way only counts cardinal directions as connected.
    """

    old_color = grid[x, y]

    assert connectivity in [4, 8], "flood_fill: Connectivity must be 4 or 8."

    _flood_fill(grid, x, y, color, old_color, connectivity)


def _flood_fill(grid, x, y, color, old_color, connectivity):
    """
    internal function not used by LLM
    """
    if grid[x, y] != old_color or grid[x, y] == color:
        return

    grid[x, y] = color

    # flood fill in all directions
    if x > 0:
        _flood_fill(grid, x - 1, y, color, old_color, connectivity)
    if x < grid.shape[0] - 1:
        _flood_fill(grid, x + 1, y, color, old_color, connectivity)
    if y > 0:
        _flood_fill(grid, x, y - 1, color, old_color, connectivity)
    if y < grid.shape[1] - 1:
        _flood_fill(grid, x, y + 1, color, old_color, connectivity)

    if connectivity == 4:
        return

    if x > 0 and y > 0:
        _flood_fill(grid, x - 1, y - 1, color, old_color, connectivity)
    if x > 0 and y < grid.shape[1] - 1:
        _flood_fill(grid, x - 1, y + 1, color, old_color, connectivity)
    if x < grid.shape[0] - 1 and y > 0:
        _flood_fill(grid, x + 1, y - 1, color, old_color, connectivity)
    if x < grid.shape[0] - 1 and y < grid.shape[1] - 1:
        _flood_fill(grid, x + 1, y + 1, color, old_color, connectivity)


def draw_line(grid, x, y, length, color, direction, stop_at_color=[]):
    """
    Draws a line of the specified length in the specified direction starting at (x, y).
    Direction should be a vector with elements -1, 0, or 1.
    If length is None, then the line will continue until it hits the edge of the grid.

    stop_at_color: optional list of colors that the line should stop at. If the line hits a pixel of one of these colors, it will stop.

    Example:
    draw_line(grid, 0, 0, length=3, color=blue, direction=(1, 1)) will draw a diagonal line of blue pixels from (0, 0) to (2, 2).
    """

    if length is None:
        length = max(grid.shape) * 2

    for i in range(length):
        new_x = x + i * direction[0]
        new_y = y + i * direction[1]
        if 0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1]:
            if grid[new_x, new_y] in stop_at_color:
                break
            grid[new_x, new_y] = color

    return grid


def find_connected_components(
    grid, background=Color.BLACK, connectivity=4, monochromatic=True
):
    """
    Find the connected components in the grid. Returns a list of connected components, where each connected component is a numpy array.

    connectivity: 4 or 8, for 4-way or 8-way connectivity.
    monochromatic: if True, each connected component is assumed to have only one color. If False, each connected component can include multiple colors.
    """

    from scipy.ndimage import label

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    if (
        not monochromatic
    ):  # if we allow multiple colors in a connected component, we can ignore color except for whether it's the background
        labeled, n_objects = label(grid != background, structure)
        connected_components = []
        for i in range(n_objects):
            connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
            connected_components.append(connected_component)

        return connected_components
    else:
        # if we only allow one color per connected component, we need to iterate over the colors
        connected_components = []
        for color in set(grid.flatten()) - {background}:
            labeled, n_objects = label(grid == color, structure)
            for i in range(n_objects):
                connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
                connected_components.append(connected_component)
        return connected_components


def blit(grid, sprite, x=0, y=0, background=None):
    """
    Copies the sprite into the grid at the specified location. Modifies the grid in place.

    background: color treated as transparent. If specified, only copies the non-background pixels of the sprite.
    """

    new_grid = grid

    for i in range(sprite.shape[0]):
        for j in range(sprite.shape[1]):
            if background is None or sprite[i, j] != background:
                # check that it is inbounds
                if 0 <= x + i < grid.shape[0] and 0 <= y + j < grid.shape[1]:
                    new_grid[x + i, y + j] = sprite[i, j]

    return new_grid


def bounding_box(grid, background=Color.BLACK):
    """
    Find the bounding box of the non-background pixels in the grid.
    Returns a tuple (x, y, width, height) of the bounding box.
    """
    n, m = grid.shape
    x_min, x_max = n, -1
    y_min, y_max = m, -1

    for x in range(n):
        for y in range(m):
            if grid[x, y] != background:
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def crop(grid, background=Color.BLACK):
    """
    Crop the grid to the smallest bounding box that contains all non-background pixels.
    """
    x, y, w, h = bounding_box(grid, background)
    return grid[x : x + w, y : y + h]


def translate(grid, x, y, background=Color.BLACK):
    """
    Translate the grid by the vector (x, y). Fills in the new pixels with the background color.

    Example usage:
    red_object = input_grid[input_grid==Color.RED]
    shifted_red_object = translate(red_object, x=1, y=1)
    """
    n, m = grid.shape
    new_grid = np.zeros((n, m), dtype=grid.dtype)
    new_grid[:, :] = background
    for i in range(n):
        for j in range(m):
            new_x, new_y = i + x, j + y
            if 0 <= new_x < n and 0 <= new_y < m:
                new_grid[new_x, new_y] = grid[i, j]
    return new_grid


def collision(
    _=None, object1=None, object2=None, x1=0, y1=0, x2=0, y2=0, background=Color.BLACK
):
    """
    Check if object1 and object2 collide when object1 is at (x1, y1) and object2 is at (x2, y2).

    Example usage:

    # Check if a sprite can be placed onto a grid at (X,Y)
    collision(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects collide
    collision(object1=object1, object2=object2, x1=X1, y1=Y1, x2=X2, y2=Y2)
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
                if (
                    0 <= new_x < n2
                    and 0 <= new_y < m2
                    and object2[new_x, new_y] != background
                ):
                    return True

    return False


def contact(
    _=None,
    object1=None,
    object2=None,
    x1=0,
    y1=0,
    x2=0,
    y2=0,
    background=Color.BLACK,
    connectivity=4,
):
    """
    Check if object1 and object2 touch each other (have contact) when object1 is at (x1, y1) and object2 is at (x2, y2).
    They are touching each other if they share a border, or if they overlap. Collision implies contact, but contact does not imply collision.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. (8-way counts diagonals as touching, 4-way only counts cardinal directions as touching)

    Example usage:

    # Check if a sprite touches anything if it were to be placed at (X,Y)
    contact(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects touch each other
    contact(object1=object1, object2=object2)
    """
    n1, m1 = object1.shape
    n2, m2 = object2.shape

    dx = x2 - x1
    dy = y2 - y1

    if connectivity == 4:
        moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        moves = [
            (0, 0),
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    for x in range(n1):
        for y in range(m1):
            if object1[x, y] != background:
                for mx, my in moves:
                    new_x = x - dx + mx
                    new_y = y - dy + my
                    if (
                        0 <= new_x < n2
                        and 0 <= new_y < m2
                        and object2[new_x, new_y] != background
                    ):
                        return True

    return False


def random_free_location_for_object(
    grid,
    sprite,
    background=Color.BLACK,
    border_size=0,
    padding=0,
    padding_connectivity=8,
):
    """
    Find a random free location for the sprite in the grid
    Returns a tuple (x, y) of the top-left corner of the sprite in the grid, which can be passed to `blit`

    border_size: minimum distance from the edge of the grid
    background: color treated as transparent
    padding: if non-zero, the sprite will be padded with a non-background color before checking for collision
    padding_connectivity: 4 or 8, for 4-way or 8-way connectivity when padding the sprite

    Example usage:
    x, y = random_free_location_for_object(grid, sprite) # find the location
    assert not collision(object1=grid, object2=sprite, x2=x, y2=y)
    blit(grid, sprite, x, y)
    """
    n, m = grid.shape

    sprite_mask = 1 * (sprite != background)

    # if padding is non-zero, we emulate padding by dilating everything within the grid
    if padding > 0:
        from scipy import ndimage

        if padding_connectivity == 4:
            structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        elif padding_connectivity == 8:
            structuring_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:
            raise ValueError("padding_connectivity must be 4 or 8.")

        # use binary dilation to pad the sprite with a non-background color
        grid_mask = ndimage.binary_dilation(
            grid != background, iterations=padding, structure=structuring_element
        ).astype(int)
    else:
        grid_mask = 1 * (grid != background)

    possible_locations = [
        (x, y)
        for x in range(border_size, n + 1 - border_size - sprite.shape[0])
        for y in range(border_size, m + 1 - border_size - sprite.shape[1])
    ]

    non_background_grid = np.sum(grid_mask)
    non_background_sprite = np.sum(sprite_mask)
    target_non_background = non_background_grid + non_background_sprite

    # Scale background pixels to 0 so np.maximum can be used later
    scaled_grid = grid.copy()
    scaled_grid[scaled_grid == background] = Color.BLACK

    # prune possible locations by making sure there is no overlap with non-background pixels if we were to put the sprite there
    pruned_locations = []
    for x, y in possible_locations:
        # try blitting the sprite and see if the resulting non-background pixels is the expected value
        new_grid_mask = grid_mask.copy()
        blit(new_grid_mask, sprite_mask, x, y, background=0)
        if np.sum(new_grid_mask) == target_non_background:
            pruned_locations.append((x, y))

    if len(pruned_locations) == 0:
        raise ValueError("No free location for sprite found.")

    return random.choice(pruned_locations)

def object_interior(grid, background_color=Color.BLACK):
    """
    Computes the interior of the object (including edges)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's interior.

    Example usage:
    interior = object_interior(obj, background_color=Color.BLACK)
    for x, y in np.argwhere(interior):
        # x,y is either inside the object or at least on its edge
    """

    mask = 1*(grid != background_color)

    # March around the border and flood fill (with 42) wherever we find zeros
    n, m = grid.shape
    for i in range(n):
        if grid[i, 0] == background_color:
            flood_fill(mask, i, 0, 42)
        if grid[i, m-1] == background_color: flood_fill(mask, i, m-1, 42)
    for j in range(m):
        if grid[0, j] == background_color: flood_fill(mask, 0, j, 42)
        if grid[n-1, j] == background_color: flood_fill(mask, n-1, j, 42)
    
    return mask != 42

def object_boundary(grid, background_color=Color.BLACK):
    """
    Computes the boundary of the object (excluding interior)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's boundary.

    Example usage:
    boundary = object_boundary(obj, background_color=Color.BLACK)
    assert np.all(obj[boundary] != Color.BLACK)
    """

    # similar idea: first get the exterior, but then we search for all the pixels that are part of the object and either adjacent to 42, or are part of the boundary

    exterior = ~object_interior(grid, background_color)

    # Now we find all the pixels that are part of the object and adjacent to the exterior, or which are part of the object and on the boundary of the canvas
    canvas_boundary = np.zeros_like(grid, dtype=bool)
    canvas_boundary[0, :] = True
    canvas_boundary[-1, :] = True
    canvas_boundary[:, 0] = True
    canvas_boundary[:, -1] = True

    from scipy import ndimage
    adjacent_to_exterior = ndimage.binary_dilation(exterior, iterations=1)

    boundary = (grid != background_color) & (adjacent_to_exterior | canvas_boundary)

    return boundary
    



def detect_horizontal_periodicity(grid, ignore_color=None):
    """
    Finds the period of a grid that was produced by repeated horizontal translation (tiling) of a smaller grid.

    Horizontal period satisfies: grid[x, y] == grid[x + i*period, y] for all x, y, i, as long as neither pixel is `ignore_color`.

    ignore_color: optional color that should be ignored when checking for periodicity

    Example usage:
    period = detect_horizontal_periodicity(grid, ignore_color=None)
    assert grid[:period, :] == grid[period:2*period, :]
    """

    if ignore_color is None:
        ignore_color = 99  # ignore nothing

    for h_period in range(1, grid.shape[0]):
        pattern = grid[:h_period, :]
        h_repetitions = int(grid.shape[0] / h_period + 0.5)

        success = True
        for i in range(1, h_repetitions):
            sliced_input = grid[i * h_period : (i + 1) * h_period, :]
            sliced_pattern = pattern[: sliced_input.shape[0], : sliced_input.shape[1]]
            # Check that they are equal except where one of them is black
            if np.all(
                (sliced_input == sliced_pattern)
                | (sliced_input == ignore_color)
                | (sliced_pattern == ignore_color)
            ):
                # Update the pattern to include the any new nonblack pixels
                sliced_pattern[sliced_input != ignore_color] = sliced_input[
                    sliced_input != ignore_color
                ]
            else:
                success = False
                break
        if success:
            return h_period

def detect_vertical_periodicity(grid, ignore_color=None):
    """
    Finds the period of a grid that was produced by repeated vertical translation (tiling) of a smaller grid.

    Vertical period satisfies: grid[x, y] == grid[x, y + i*period] for all x, y, i, as long as neither pixel is `ignore_color`.

    ignore_color: optional color that should be ignored when checking for periodicity

    Example usage:
    period = detect_vertical_periodicity(grid, ignore_color=None)
    assert grid[:, :period] == grid[:, period:2*period]
    """

    if ignore_color is None:
        ignore_color = 99  # ignore nothing

    for v_period in range(1, grid.shape[1]):
        pattern = grid[:, :v_period]
        v_repetitions = int(grid.shape[1] / v_period + 0.5)

        success = True
        for i in range(1, v_repetitions):
            sliced_input = grid[:, i * v_period : (i + 1) * v_period]
            sliced_pattern = pattern[: sliced_input.shape[0], : sliced_input.shape[1]]
            # Check that they are equal except where one of them is black
            if np.all(
                (sliced_input == sliced_pattern)
                | (sliced_input == ignore_color)
                | (sliced_pattern == ignore_color)
            ):
                # Update the pattern to include the any new nonblack pixels
                sliced_pattern[sliced_input != ignore_color] = sliced_input[
                    sliced_input != ignore_color
                ]
            else:
                success = False
                break
        if success:
            break

    return v_period


def detect_rotational_symmetry(grid, ignore_color=0):
    """
    Finds the center of rotational symmetry of a grid.
    Satisfies: grid[x, y] == grid[y - rotate_center_y + rotate_center_x, -x + rotate_center_y + rotate_center_x] # clockwise
               grid[x, y] == grid[-y + rotate_center_y + rotate_center_x, x - rotate_center_y + rotate_center_x] # counterclockwise
               for all x, y, as long as neither pixel is `ignore_color`.

    Example:
    rotate_center_x, rotate_center_y = detect_rotational_symmetry(grid, ignore_color=Color.BLACK) # ignore_color: In case parts of the object have been removed and occluded by black
    for x, y in np.argwhere(grid != Color.BLACK):
        # IMPORTANT! cast to int
        rotated_x, rotated_y = y + int(rotate_center_x - rotate_center_y), -x + int(rotate_center_y + rotate_center_x)
        assert grid[rotated_x, rotated_y] == grid[x, y] or grid[rotated_x, rotated_y] == Color.BLACK
    """

    # Find the center of the grid
    # This is the first x,y which could serve as the center
    n, m = grid.shape

    # compute the center of mass, we search outward from that point
    total_mass = np.sum(grid != ignore_color)
    com_x = np.sum(np.arange(n)[:, None] * (grid != ignore_color)) / total_mass
    com_y = np.sum(np.arange(m)[None, :] * (grid != ignore_color)) / total_mass

    possibilities = [
        (x_center + z, y_center + z)
        for x_center in range(n)
        for y_center in range(m)
        for z in [0, 0.5]
    ]

    occupied_pixels = np.argwhere(grid != ignore_color)

    best_rotation, best_overlap = None, 0
    for x_center, y_center in possibilities:
        # Check if this is a valid center, meaning that every turned on pixel maps to another turned on pixel of the same color

        # rotate all the occupied pixels at once to get their new locations
        translated_pixels = occupied_pixels - np.array([[x_center, y_center]])
        translated_rotated_pixels = np.stack(
            [translated_pixels[:, 1], -translated_pixels[:, 0]], axis=1
        )
        rotated_pixels = translated_rotated_pixels + np.array([[x_center, y_center]])

        if (
            np.any(rotated_pixels < 0)
            or np.any(rotated_pixels[:, 0] >= n)
            or np.any(rotated_pixels[:, 1] >= m)
        ):
            continue

        # convert the rotated pixel coordinates to integers, so that we can use them as indices to see what's in the grid at those spots
        rotated_pixels = rotated_pixels.astype(int)
        rotated_values = grid[rotated_pixels[:, 0], rotated_pixels[:, 1]]
        original_values = grid[occupied_pixels[:, 0], occupied_pixels[:, 1]]

        if np.all(
            (rotated_values == original_values) | (rotated_values == ignore_color)
        ):
            overlap = np.sum(rotated_values == original_values)
            if overlap > best_overlap:
                best_rotation = (x_center, y_center)
                best_overlap = overlap

    return best_rotation


def show_colored_grid(grid, text=True):
    """
    internal function not used by LLM
    Not used by the language model, used by the rest of the code for debugging
    """

    if not text:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # RGB
        colors_rgb = {
            0: (0x00, 0x00, 0x00),
            1: (0x00, 0x74, 0xD9),
            2: (0xFF, 0x41, 0x36),
            3: (0x2E, 0xCC, 0x40),
            4: (0xFF, 0xDC, 0x00),
            5: (0xA0, 0xA0, 0xA0),
            6: (0xF0, 0x12, 0xBE),
            7: (0xFF, 0x85, 0x1B),
            8: (0x7F, 0xDB, 0xFF),
            9: (0x87, 0x0C, 0x25),
        }

        _float_colors = [tuple(c / 255 for c in col) for col in colors_rgb.values()]
        arc_cmap = ListedColormap(_float_colors)
        grid = grid.T
        plt.figure()
        plot_handle = plt.gca()
        plot_handle.pcolormesh(
            grid,
            cmap=arc_cmap,
            rasterized=True,
            vmin=0,
            vmax=9,
        )
        plot_handle.set_xticks(np.arange(0, grid.shape[1], 1))
        plot_handle.set_yticks(np.arange(0, grid.shape[0], 1))
        plot_handle.grid()
        plot_handle.set_aspect(1)
        plot_handle.invert_yaxis()
        plt.show()
        return

    color_names = [
        "black",
        "blue",
        "red",
        "green",
        "yellow",
        "grey",
        "pink",
        "orange",
        "teal",
        "maroon",
    ]
    color_8bit = {
        "black": 0,
        "blue": 4,
        "red": 1,
        "green": 2,
        "yellow": 3,
        "grey": 7,
        "pink": 13,
        "orange": 202,
        "teal": 6,
        "maroon": 196,
    }

    for y in range(grid.shape[1]):
        for x in range(grid.shape[0]):
            cell = grid[x, y]
            color_code = color_8bit[color_names[cell]]
            print(f"\033[38;5;{color_code}m{cell}\033[0m", end="")
        print()


def visualize(input_generator, transform, n_examples=5):
    """
    internal function not used by LLM
    """

    for index in range(n_examples):
        input_grid = input_generator()
        print("Input:")
        show_colored_grid(input_grid)

        output_grid = transform(input_grid)
        print("Output:")
        show_colored_grid(output_grid)

        if index < n_examples - 1:
            print("\n\n---------------------\n\n")


def apply_symmetry(sprite, symmetry_type):
    """
    internal function not used by LLM
    Apply the specified symmetry within the bounds of the sprite.
    """
    n, m = sprite.shape
    if symmetry_type == "horizontal":
        for y in range(m):
            for x in range(n // 2):
                sprite[x, y] = sprite[n - 1 - x, y] = (
                    sprite[x, y] or sprite[n - 1 - x, y]
                )
    elif symmetry_type == "vertical":
        for x in range(n):
            for y in range(m // 2):
                sprite[x, y] = sprite[x, m - 1 - y] = (
                    sprite[x, y] or sprite[x, m - 1 - y]
                )
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")
    return sprite


def apply_diagonal_symmetry(sprite):
    """
    internal function not used by LLM
    Apply diagonal symmetry within the bounds of the sprite. Assumes square sprite.
    """
    n, m = sprite.shape
    if n != m:
        raise ValueError("Diagonal symmetry requires a square sprite.")
    for x in range(n):
        for y in range(x + 1, m):
            sprite[x, y] = sprite[y, x] = sprite[x, y] or sprite[y, x]
    return sprite


def is_contiguous(bitmask, background=Color.BLACK, connectivity=4):
    """
    Check if an array is contiguous.

    background: Color that counts as transparent (default: Color.BLACK)
    connectivity: 4 or 8, for 4-way (only cardinal directions) or 8-way connectivity (also diagonals) (default: 4)

    Returns True/False
    """
    from scipy.ndimage import label

    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    labeled, n_objects = label(bitmask != background, structure)
    return n_objects == 1


def generate_sprite(
    n,
    m,
    symmetry_type,
    fill_percentage=0.5,
    max_colors=9,
    color_palate=None,
    connectivity=4,
):
    """ "
    internal function not used by LLM
    """
    # pick random colors, number of colors follows a geometric distribution truncated at 9
    if color_palate is None:
        n_colors = 1
        while n_colors < max_colors and random.random() < 0.3:
            n_colors += 1
        color_palate = random.sample(range(1, 10), n_colors)
    else:
        n_colors = len(color_palate)

    grid = np.zeros((n, m), dtype=int)
    if symmetry_type == "not_symmetric":
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
    elif symmetry_type == "horizontal":
        x, y = random.randint(0, n - 1), m // 2
    elif symmetry_type == "vertical":
        x, y = n // 2, random.randint(0, m - 1)
    elif symmetry_type == "diagonal":
        # coin flip for which diagonal orientation
        diagonal_orientation = random.choice([True, False])
        x = random.randint(0, n - 1)
        y = x if diagonal_orientation else n - 1 - x
    elif symmetry_type == "radial":
        # we are just going to make a single quadrant and then apply symmetry
        assert n == m, "Radial symmetry requires a square grid."
        original_length = n
        # shrink to quarter size, we are just making a single quadrant
        n, m = int(n / 2 + 0.5), int(m / 2 + 0.5)
        x, y = (
            n - 1,
            m - 1,
        )  # begin at the bottom corner which is going to become the middle, ensuring everything is connected
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")

    if connectivity == 4:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    color_index = 0
    while np.sum(grid > 0) < fill_percentage * n * m:
        grid[x, y] = color_palate[color_index]
        if random.random() < 0.33:
            color_index = random.choice(range(n_colors))
        dx, dy = random.choice(moves)
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < n and 0 <= new_y < m:
            x, y = new_x, new_y

    if symmetry_type in ["horizontal", "vertical"]:
        grid = apply_symmetry(grid, symmetry_type)
    elif symmetry_type == "radial":
        # this requires resizing
        output = np.zeros((original_length, original_length), dtype=int)
        blit(output, grid)
        for _ in range(3):
            blit(output, np.rot90(output), background=Color.BLACK)
        grid = output

    elif symmetry_type == "diagonal":
        # diagonal symmetry goes both ways, flip a coin to decide which way
        if diagonal_orientation:
            grid = np.flipud(grid)
            grid = apply_diagonal_symmetry(grid)
            grid = np.flipud(grid)
        else:
            grid = apply_diagonal_symmetry(grid)

    return grid


def random_sprite(n, m, density=0.5, symmetry=None, color_palette=None, connectivity=4):
    """
    Generate a sprite (an object), represented as a numpy array.

    n, m: dimensions of the sprite. If these are lists, then a random value will be chosen from the list.
    symmetry: optional type of symmetry to apply to the sprite. Can be 'horizontal', 'vertical', 'diagonal', 'radial', 'not_symmetric'. If None, a random symmetry type will be chosen.
    color_palette: optional list of colors to use in the sprite. If None, a random color palette will be chosen.

    Returns an (n,m) NumPy array representing the sprite.
    """

    # canonical form: force dimensions to be lists
    if not isinstance(n, list):
        n = [n]
    if not isinstance(m, list):
        m = [m]

    # radial and diagonal require target shape to be square
    can_be_square = any(n_ == m_ for n_ in n for m_ in m)

    # Decide on symmetry type before generating the sprites
    symmetry_types = ["horizontal", "vertical", "not_symmetric"]
    if can_be_square:
        symmetry_types = symmetry_types + ["diagonal", "radial"]

    symmetry = symmetry or random.choice(symmetry_types)

    # Decide on dimensions
    has_to_be_square = symmetry in ["diagonal", "radial"]
    if has_to_be_square:
        n, m = random.choice([(n_, m_) for n_ in n for m_ in m if n_ == m_])
    else:
        n = random.choice(n)
        m = random.choice(m)

    # if one of the dimensions is 1, then we need to make sure the density is high enough to fill the entire sprite
    if n == 1 or m == 1:
        density = 1
    # small sprites require higher density in order to have a high probability of reaching all of the sides
    elif n == 2 or m == 2:
        density = max(density, 0.7)
    elif density == 1:
        pass
    # randomly perturb the density so that we get a wider variety of densities
    else:
        density = max(0.4, min(0.95, random.gauss(density, 0.1)))

    while True:
        sprite = generate_sprite(
            n,
            m,
            symmetry_type=symmetry,
            color_palate=color_palette,
            fill_percentage=density,
            connectivity=connectivity,
        )
        assert is_contiguous(
            sprite, connectivity=connectivity
        ), "Generated sprite is not contiguous."
        # check that the sprite has pixels that are flushed with the border
        if (
            np.sum(sprite[0, :]) > 0
            and np.sum(sprite[-1, :]) > 0
            and np.sum(sprite[:, 0]) > 0
            and np.sum(sprite[:, -1]) > 0
        ):
            return sprite

def detect_objects(grid, _=None, predicate=None, background=Color.BLACK, monochromatic=False, connectivity=None, allowed_dimensions=None, colors=None, can_overlap=False):
    """
    Detects and extracts objects from the grid that satisfy custom specification.

    predicate: a function that takes a candidate object as input and returns True if it counts as an object
    background: color treated as transparent
    monochromatic: if True, each object is assumed to have only one color. If False, each object can include multiple colors.
    connectivity: 4 or 8, for 4-way or 8-way connectivity. If None, the connectivity is determined automatically.
    allowed_dimensions: a list of tuples (n, m) specifying the allowed dimensions of the objects. If None, objects of any size are allowed.
    colors: a list of colors that the objects are allowed to have. If None, objects of any color are allowed.
    can_overlap: if True, objects can overlap. If False, objects cannot overlap.

    Returns a list of objects, where each object is a numpy array.
    """

    objects = []

    if connectivity:
        objects.extend(find_connected_components(grid, background=background, connectivity=connectivity, monochromatic=monochromatic))
        if colors:
            objects = [obj for obj in objects if all((color in colors) or color == background for color in obj.flatten())]
        if predicate:
            objects = [obj for obj in objects if predicate(crop(obj, background=background))]
    
    if allowed_dimensions:
        objects = [obj for obj in objects if obj.shape in allowed_dimensions]

        # Also scan through the grid
        scan_objects = []
        for n, m in allowed_dimensions:
            for i in range(grid.shape[0] - n + 1):
                for j in range(grid.shape[1] - m + 1):
                    candidate_sprite = grid[i:i+n, j:j+m]

                    if np.any(candidate_sprite != background) and \
                        (colors is None or all((color in colors) or color == background for color in candidate_sprite.flatten())) and \
                        (predicate is None or predicate(candidate_sprite)):
                        candidate_object = np.full(grid.shape, background)
                        candidate_object[i:i+n, j:j+m] = candidate_sprite
                        if not any( np.all(candidate_object == obj) for obj in objects):
                            scan_objects.append(candidate_object)
        objects.extend(scan_objects)
    
    if not can_overlap:
        # sort objects by size, breaking ties by mass
        objects.sort(key=lambda obj: (crop(obj, background).shape[0] * crop(obj, background).shape[1], np.sum(obj!=background)), reverse=True)
        overlap_matrix = np.full((len(objects), len(objects)), False)
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    overlap_matrix[i, j] = np.any((obj1 != background) & (obj2 != background))
                
        # Pick a subset of objects that don't overlap and which cover as many pixels as possible
        # First, we definitely pick everything that doesn't have any overlaps
        keep_objects = [obj for i, obj in enumerate(objects) if not np.any(overlap_matrix[i])]

        # Second, we might pick the remaining objects
        remaining_indices = [i for i, obj in enumerate(objects) if np.any(overlap_matrix[i])]

        # Figure out the best possible score we could get if we cover everything
        best_possible_mask = np.zeros_like(grid, dtype=bool)
        for i in remaining_indices:
            best_possible_mask |= objects[i] != background
        best_possible_score = np.sum(best_possible_mask)

        # Now we just do a brute force search recursively
        def pick_objects(remaining_indices, current_indices, current_mask):
            nonlocal overlap_matrix 
            
            if not remaining_indices:
                solution = [objects[i] for i in current_indices]
                solution_goodness = np.sum(current_mask)
                return solution, solution_goodness
            
            first_index, *rest = remaining_indices
            # Does that object have any overlap with the current objects? If so don't pick it
            if any( overlap_matrix[i, first_index] for i in current_indices):
                return pick_objects(rest, current_indices, current_mask)
            
            # Try picking it
            with_index, with_goodness = pick_objects(rest, current_indices + [first_index], current_mask | (objects[first_index] != background))

            # Did we win?
            if with_goodness == best_possible_score:
                return with_index, with_goodness

            # Try not picking it
            without_index, without_goodness = pick_objects(rest, current_indices, current_mask)

            if with_goodness > without_goodness:
                return with_index, with_goodness
            else:
                return without_index, without_goodness
        
        solution, _ = pick_objects(remaining_indices, [], np.zeros_like(grid, dtype=bool))

        objects = keep_objects + solution

    return objects

