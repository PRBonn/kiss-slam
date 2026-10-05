# SPDX-FileCopyrightText: 2026 Jinhang Dong
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import yaml
from PIL import Image

from kiss_slam.tools.map_io import write_2d_occupancy_grid


def export_grid(tmp_path, grid, lower_bound=(-3, 7, 0), resolution=0.5):
    write_2d_occupancy_grid(tmp_path, grid, np.array(lower_bound), resolution, 0.65, 0.2)
    with Image.open(tmp_path / "map.png") as image:
        pixels = np.array(image)
    with open(tmp_path / "map.yaml") as stream:
        metadata = yaml.safe_load(stream)
    return pixels, metadata


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (1, 3), (3, 1)])
def test_image_width_follows_x_and_height_follows_y(tmp_path, shape):
    export_grid(tmp_path, np.ones(shape))
    with Image.open(tmp_path / "map.png") as image:
        assert image.size == shape


@pytest.mark.parametrize("shape", [(3, 2), (2, 3), (3, 3), (1, 1)])
@pytest.mark.parametrize("lower_bound,resolution", [((-3, 7, 0), 0.5), ((4, -2, 0), 0.25)])
def test_ros_map_loading_preserves_world_cell_locations(tmp_path, shape, lower_bound, resolution):
    grid = np.resize([0.0, 0.5, 1.0], shape)
    pixels, metadata = export_grid(tmp_path, grid, lower_bound, resolution)

    # ROS map loading places the bottom image row at origin, with x along
    # image columns and y increasing upwards. Decode occupied/free/unknown.
    loaded_cells = {}
    height, width = pixels.shape
    for row in range(height):
        for column in range(width):
            occupancy = 1.0 - pixels[row, column] / 255.0
            state = (
                100
                if occupancy > metadata["occupied_thresh"]
                else (0 if occupancy < metadata["free_thresh"] else -1)
            )
            x = metadata["origin"][0] + (column + 0.5) * metadata["resolution"]
            y = metadata["origin"][1] + (height - row - 0.5) * metadata["resolution"]
            loaded_cells[x, y] = state

    expected_cells = {}
    for x_index, y_index in np.ndindex(shape):
        x = (lower_bound[0] + x_index + 0.5) * resolution
        y = (lower_bound[1] + y_index + 0.5) * resolution
        expected_cells[x, y] = {0.0: 100, 0.5: -1, 1.0: 0}[grid[x_index, y_index]]
    assert loaded_cells == expected_cells


def test_yaml_metadata_is_preserved(tmp_path):
    _, metadata = export_grid(tmp_path, np.ones((1, 1)))
    assert metadata == {
        "image": "map.png",
        "resolution": 0.5,
        "origin": [-1.5, 3.5, 0.0],
        "occupied_thresh": 0.65,
        "free_thresh": 0.2,
        "negate": 0,
    }
