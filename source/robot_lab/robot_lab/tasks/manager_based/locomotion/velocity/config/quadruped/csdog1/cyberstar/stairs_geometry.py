"""NumPy geometry and stratified patch allocation for Cyberstar stairs."""
import numpy as np


def stair_patch_specs(count, flat_ratio, up_ratio, height_bands, rng):
    """Allocate flat/direction/height bands with the source largest-remainder rule."""
    if count <= 0 or not 0 <= flat_ratio <= 1 or not 0 <= up_ratio <= 1:
        raise ValueError("Invalid stair patch count or terrain proportions")
    bands = np.asarray(height_bands, dtype=float)
    if bands.ndim != 2 or bands.shape[1] != 3 or np.any(bands[:, 0] <= 0) or np.any(bands[:, 1] < bands[:, 0]):
        raise ValueError("Invalid stair height bands")
    weights = bands[:, 2]
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Stair height weights must be non-negative and sum above zero")
    flat_count = int(round(count * flat_ratio))
    category_weights = (weights[:, None] / weights.sum() * np.array([up_ratio, 1 - up_ratio])).flatten()
    raw = (count - flat_count) * category_weights
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts), kind="stable")[:count - flat_count - counts.sum()]] += 1
    specs = [(0., 0.)] * flat_count
    for index, amount in enumerate(counts):
        low, high = bands[index // 2, :2]
        # Negative pyramid heights ascend when moving outward from the spawn.
        sign = -1 if index % 2 == 0 else 1
        specs.extend((sign * rng.uniform(low, high), rng.uniform(.31, .39)) for _ in range(amount))
    rng.shuffle(specs)
    return specs


def pyramid_stair_heights(size, horizontal_scale, vertical_scale, step_height, step_width, platform_size):
    """Match Isaac Gym's quantized pyramid stairs, including the center platform."""
    shape = tuple(int(length / horizontal_scale) for length in size)
    heights = np.zeros(shape, dtype=np.int16)
    if step_height == 0:
        return heights
    width = int(step_width / horizontal_scale)
    platform = int(platform_size / horizontal_scale)
    if width < 1 or min(shape) <= platform:
        raise ValueError("Step width and platform must fit the stair patch")
    step = int(step_height / vertical_scale)
    start_x = start_y = 0
    stop_x, stop_y = shape
    height = 0
    while stop_x - start_x > platform and stop_y - start_y > platform:
        start_x += width
        stop_x -= width
        start_y += width
        stop_y -= width
        height += step
        heights[start_x:stop_x, start_y:stop_y] = height
    return heights
