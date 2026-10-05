# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os

import numpy as np
import yaml
from PIL import Image


def write_2d_occupancy_grid(
    output_dir, occupancy_grid, lower_bound, resolution, occupied_threshold, free_threshold
):
    """Write an [x, y] grid as a ROS map image and YAML metadata."""
    image_filename = "map.png"
    # Image columns increase with x; rows increase downwards from maximum y.
    occupancy_image = np.rot90(occupancy_grid) * 255.0
    Image.fromarray(np.asarray(occupancy_image, np.uint8)).save(
        os.path.join(output_dir, image_filename)
    )
    grid_info = {
        "image": image_filename,
        "resolution": resolution,
        "origin": [
            float(lower_bound[0]) * resolution,
            float(lower_bound[1]) * resolution,
            0.0,
        ],
        "occupied_thresh": occupied_threshold,
        "free_thresh": free_threshold,
        "negate": 0,
    }
    with open(os.path.join(output_dir, "map.yaml"), "w") as yaml_file:
        yaml.dump(grid_info, yaml_file, default_flow_style=False, sort_keys=False)
