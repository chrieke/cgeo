import functools
from typing import Union

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame as GDF
import shapely
from shapely.geometry import Polygon
import pyproj
import rasterio.crs


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    Args:
        poly: Input shapely Polygon

    Example:
        in_geo.geometry.apply(lambda p: _close_holes(p))
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def explode_mp(df: GDF) -> GDF:
    """
    Explode all multi-polygon geometries in a geodataframe into individual polygon
    geometries.

    Adds exploded polygons as rows at the end of the geodataframe and resets its index.

    Args:
        df: Input GeoDataFrame
    """
    outdf = df[df.geom_type != "MultiPolygon"]

    df_mp = df[df.geom_type == "MultiPolygon"]
    for idx, row in df_mp.iterrows():
        df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
        df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
        for i in range(len(row.geometry)):
            df_temp.loc[i, "geometry"] = row.geometry[i]
        outdf = outdf.append(df_temp, ignore_index=True)

    outdf = outdf.reset_index(drop=True)
    return outdf


def dissolve_mp_biggest(df: GDF) -> GDF:
    """
    Keep the biggest area-polygon of geodataframe rows with multipolygon geometries.

    Args:
        df: Input GeoDataFrame
    """
    row_idxs_mp = df.index[df.geometry.geom_type == "MultiPolygon"].tolist()
    for idx in row_idxs_mp:
        mp = df.loc[idx].geometry
        poly_areas = [p.area for p in mp]
        max_area_poly = mp[poly_areas.index(max(poly_areas))]
        df.loc[idx, "geometry"] = max_area_poly
    return df


def reduce_precision(poly: Polygon, precision: int = 3) -> Polygon:
    """
    Reduces the number of after comma decimals of a shapely Polygon or geodataframe
    geometries.

    GeoJSON specification recommends 6 decimal places for lat & lon which equates to
    roughly 10cm of precision (https://github.com/perrygeo/geojson-precision).

    Args:
        poly: Input shapely Polygon.
        precision: number of after comma values that should remain.
    """
    geojson = shapely.geometry.mapping(poly)
    geojson["coordinates"] = np.round(np.array(geojson["coordinates"]), precision)
    poly = shapely.geometry.shape(geojson)
    if (
        not poly.is_valid
    ):  # Too low precision can lead to invalid polys due to line overlap effects.
        poly = poly.buffer(0)
    return poly


def to_pixelcoords(
    poly: Polygon,
    reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
    scale: bool = False,
    nrows: int = None,
    ncols: int = None,
) -> Polygon:
    """
    Converts projected polygon coordinates to pixel coordinates of an image array.

    Subtracts point of origin, scales to pixelcoordinates.

    Args:
        poly: Input shapely Polygon.
        reference_bounds:  Bounding box object or tuple of reference (e.g. image chip)
            in format (left, bottom, right, top). Can be delineated from transform,
            nrows, ncols via rasterio.transform.reference_bounds.
        scale: Scale the polygons to the image size/resolution. Requires image array
            nrows and ncols parameters.
        nrows: image array nrows, required for scale.
        ncols: image array ncols, required for scale.
    """
    try:
        minx, miny, maxx, maxy = reference_bounds
        w_poly, h_poly = (maxx - minx, maxy - miny)
    except (TypeError, ValueError):
        raise Exception(
            f"reference_bounds needs to be a tuple or rasterio bounding box instance."
        )

    # Subtract point of origin of image bbox.
    x_coords, y_coords = poly.exterior.coords.xy
    p_origin = shapely.geometry.Polygon(
        [[x - minx, y - miny] for x, y in zip(x_coords, y_coords)]
    )

    if scale is False:
        return p_origin
    elif scale is True:
        if ncols is None or nrows is None:
            raise ValueError("ncols and nrows required for scale")
        x_scaler = ncols / w_poly
        y_scaler = nrows / h_poly
        return shapely.affinity.scale(
            p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)
        )


def invert_y_axis(
    poly: Polygon, reference_height: int
) -> Polygon:
    """
    Invert y-axis of polygon in reference to a bounding box e.g. of an image chip.

    Usage e.g. for COCOJson format.

    Args:
        ingeo: Input Polygon or geodataframe.
        reference_height: Height (in coordinates or rows) of reference object
            (polygon or image, e.g. image chip.
    """
    x_coords, y_coords = poly.exterior.coords.xy
    p_inverted_y_axis = shapely.geometry.Polygon(
        [[x, reference_height - y] for x, y in zip(x_coords, y_coords)]
    )
    return p_inverted_y_axis


def reproject_shapely(
    geometry: shapely.geometry, epsg_in: Union[str, int], epsg_out: Union[str, int]
) -> shapely.geometry:
    """
    Reproject shapely geometry to different epsg crs.

    Example: reproject_shapely(mp, 'epsg:28992', 'epsg:32631')

    Args:
        geometry: input shapely geometry
        epsg_in: input epsg code
        epsg_out: epsg code for reprojection
    """
    project = functools.partial(
        pyproj.transform,
        pyproj.Proj(init=f"epsg{str(epsg_in)}"),
        pyproj.Proj(init=f"epsg{str(epsg_out)}"),
    )
    geometry = shapely.ops.transform(project, geometry)
    return geometry
