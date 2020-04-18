# img.py

from typing import Tuple, Generator, Union

import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import Polygon
import itertools
import shapely
import affine


def get_chip_windows(
    meta_raster,
    chip_width: int = 256,
    chip_height: int = 256,
    skip_partial_chips: bool = False,
) -> Generator[Tuple[Window, Polygon, affine.Affine], any, None]:
    """Generator for rasterio windows of specified pixel size to iterate over an image in chips.

    Chips are created row wise, from top to bottom of the raster.

    Args:
        meta_raster: rasterio src.meta or src.profile
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        skip_partial_chips: Skip image chips at the edge of the raster that do not result in a full size chip.

    Returns : Yields tuple of rasterio window, Polygon and transform.

    """

    raster_width, raster_height = meta_raster["width"], meta_raster["height"]
    big_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    col_row_offsets = itertools.product(
        range(0, raster_width, chip_width), range(0, raster_height, chip_height)
    )

    for col_off, row_off in col_row_offsets:

        chip_window = Window(
            col_off=col_off, row_off=row_off, width=chip_width, height=chip_height
        )

        if skip_partial_chips:
            if (
                row_off + chip_height > raster_height
                or col_off + chip_width > raster_width
            ):
                continue

        chip_window = chip_window.intersection(big_window)
        chip_transform = rasterio.windows.transform(
            chip_window, meta_raster["transform"]
        )
        chip_bounds = rasterio.windows.bounds(
            chip_window, meta_raster["transform"]
        )  # Use the transform of the full
        #  raster here!
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_poly, chip_transform)
