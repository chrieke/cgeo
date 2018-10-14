# img.py

from typing import Tuple, Generator
from collections import namedtuple

import numpy as np
from rasterio.windows import Window
from rasterio.coords import BoundingBox


def windows_from_chipsize(raster_height: int,
                          raster_width: int,
                          chip_height: int=128,
                          chip_width: int=128,
                          skip_partial_chips: bool=False) -> Generator[Tuple[Window, BoundingBox], any, None]:
    """Generator yields rasterio windows and bounds to iterate over an image in chips.

    From top left chip to the right, then row by row.
    Another good solution for projected raster/chips:
    https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-
        rasterio/290059#290059

    Args:
        raster_height (int):
        raster_width (int):
        chip_height (int):
        chip_width (int):
        skip_partial_chips (bool): Skip image chips at the edge of the raster that do not result in a full size chip.

    Returns (Generator[Tuple[Window, BoundingBox], any, None]): tuple of rasterio window and bounding box.

    """
    # TODO: https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio

    for row_start in np.arange(0, raster_height, chip_height):
        for col_start in np.arange(0, raster_width, chip_width):
            row_stop = row_start + chip_height
            col_stop = col_start + chip_width

            if skip_partial_chips:
                if row_stop > raster_height or col_stop > raster_width:
                    break

            # print(row_start, row_stop, col_start, col_stop)
            window = Window(col_off=col_start, row_off=row_start, width=chip_width, height=chip_height)
            # window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
            Bounds = namedtuple('BoundingBox', ('left', 'bottom', 'right', 'top'))
            bounds = Bounds(col_start, row_start, col_stop, row_stop)
            # print(window, bounds)
            yield (window, bounds)
