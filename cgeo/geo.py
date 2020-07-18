from typing import Union
import math

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame as GDF
import shapely
from shapely.geometry import Polygon
from shapely.ops import transform
import rasterio.crs
import pyproj


# pylint: disable=chained-comparison
def get_utm_zone_epsg(lon: float, lat: float) -> int:
    """
    Calculates the suitable UTM crs epsg code for an input geometry point.

    Args:
        lon: Longitude of point
        lat: Latitude of point

    Returns:
        EPSG code i.e. 32658
    """
    zone_number = int((math.floor((lon + 180) / 6) % 60) + 1)

    # Special zones for Norway
    if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
        zone_number = 32
    # Special zones for Svalbard
    elif lat >= 72.0 and lat < 84.0:
        if lon >= 0.0 and lon < 9.0:
            zone_number = 31
        elif lon >= 9.0 and lon < 21.0:
            zone_number = 33
        elif lon >= 21.0 and lon < 33.0:
            zone_number = 35
        elif lon >= 33.0 and lon < 42.0:
            zone_number = 37

    if lat > 0:
        epsg_utm = zone_number + 32600
    else:
        epsg_utm = zone_number + 32700
    return epsg_utm


def buffer_meter(
    poly: Polygon,
    distance: float,
    epsg_in: int,
    use_centroid=True,
    lon: float = None,
    lat: float = None,
    **kwargs,
) -> Polygon:
    """
    Buffers a polygon in meter by temporarily reprojecting to appropiate equal-area UTM
    projection.

    Args:
        poly: Input shapely polygon.
        distance: Buffer size.
        epsg_in: Polygon input crs epsg.
        use_centroid: Uses the polygon centroid to calculate the suitable UTM epsg
            code (default). If false, uses the provided lon lat coordinates instead.
        lon: Optional lon coordinate for UTM epsg calculation.
        lat: Optional lat coordinate for UTM epsg calculation.
        kwargs: Rest of the shapely .buffer() args.

    Returns:
        Buffered polygon (by distance) in original epsg crs.
    """
    if use_centroid:
        lon = poly.centroid.x
        lat = poly.centroid.y

    epsg_utm = get_utm_zone_epsg(lon=lon, lat=lat)  # type: ignore
    poly_utm = reproject_shapely(geometry=poly, epsg_in=epsg_in, epsg_out=epsg_utm)
    poly_buff = poly_utm.buffer(distance, **kwargs)
    poly_buff_original_epsg = reproject_shapely(
        geometry=poly_buff, epsg_in=epsg_utm, epsg_out=epsg_in
    )

    return poly_buff_original_epsg


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    Args:
        poly: Input shapely Polygon

    Example:
        df.geometry.apply(lambda p: close_holes(p))
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
    for _, row in df_mp.iterrows():
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


def invert_y_axis(poly: Polygon, reference_height: int) -> Polygon:
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
            "reference_bounds needs to be a tuple or rasterio bounding box instance."
        )

    # Subtract point of origin of image bbox.
    x_coords, y_coords = poly.exterior.coords.xy
    p_origin = shapely.geometry.Polygon(
        [[x - minx, y - miny] for x, y in zip(x_coords, y_coords)]
    )

    if scale is True:
        if ncols is None or nrows is None:
            raise ValueError("ncols and nrows required for scale")
        x_scaler = ncols / w_poly
        y_scaler = nrows / h_poly
        p_origin = shapely.affinity.scale(
            p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0)
        )

    return p_origin


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


def reproject_shapely(
    geometry: shapely.geometry, epsg_in: Union[str, int], epsg_out: Union[str, int]
) -> shapely.geometry:
    """
    Reprojects shapely geometry to different epsg crs.

    Example:
        reproject_shapely(poly, 4326, 32631)

    Args:
        geometry: input shapely geometry
        epsg_in: input epsg code
        epsg_out: epsg code for reprojection
    """
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(f"epsg:{str(epsg_in)}"), pyproj.Proj(f"epsg:{str(epsg_out)}")
    )
    geometry_reprojected = transform(project.transform, geometry)
    return geometry_reprojected
