# geo.py

import functools
import warnings
from typing import Union, Dict

import numpy as np
import geopandas as gpd
from geopandas import GeoDataFrame as GDF
from pandas import DataFrame as DF
import shapely
from shapely.geometry import Polygon
import pyproj
import rasterio.crs


def close_holes(poly: Polygon) -> Polygon:
    """
    Close polygon holes by limitation to the exterior ring.

    Example: in_geo.geometry.apply(lambda _p: _close_holes(_p))
    """
    if poly.interiors:
        return Polygon(list(poly.exterior.coords))
    else:
        return poly


def explode_mp(df: GDF) -> GDF:
    """
    Explode all multi-polygon geometries in a geodataframe into individual polygon geometries.

    Adds exploded polygons as rows at the end of the geodataframe and resets its index.
    """
    outdf = df[df.geom_type == 'Polygon']

    df_mp = df[df.geom_type == 'MultiPolygon']
    for idx, row in df_mp.iterrows():
        df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
        df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
        for i in range(len(row.geometry)):
            df_temp.loc[i, 'geometry'] = row.geometry[i]
        outdf = outdf.append(df_temp, ignore_index=True)

    outdf.reset_index(drop=True, inplace=True)
    return outdf


def keep_biggest_poly(df: GDF) -> GDF:
    """
    Keep the biggest area-polygon of geodataframe rows with multipolygon geometries.
    """
    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()
    for idx in row_idxs_mp:
        mp = df.loc[idx].geometry
        poly_areas = [p.area for p in mp]
        max_area_poly = mp[poly_areas.index(max(poly_areas))]
        df.loc[idx, 'geometry'] = max_area_poly
    return df


def reduce_precision(ingeo: Union[Polygon, GDF], precision: int=3) -> Union[Polygon, GDF]:
    """Reduces the number of after comma decimals of a shapely Polygon or geodataframe geometries.

    GeoJSON specification recommends 6 decimal places for latitude and longitude which equates to roughly 10cm of
    precision (https://github.com/perrygeo/geojson-precision).

    Args:
        ingeo: input geodataframe or shapely Polygon.
        precision: number of after comma values that should remain.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _reduce_precision(poly: Polygon, precision: int) -> Polygon:
        geojson = shapely.geometry.mapping(poly)
        geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision)
        poly = shapely.geometry.shape(geojson)
        if not poly.is_valid:  # Too low precision can potentially lead to invalid polygons due to line overlap effects.
            poly = poly.buffer(0)
        return poly

    if isinstance(ingeo, Polygon):
        return _reduce_precision(poly=ingeo, precision=precision)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _reduce_precision(poly=_p, precision=precision))
        return ingeo


def to_pixelcoords(ingeo: Union[Polygon, GDF],
                   reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
                   scale: bool=False,
                   nrows: int=None,
                   ncols: int=None
                   ) -> Union[Polygon, GDF]:
    """Converts projected polygon coordinates to pixel coordinates of an image array.

    Subtracts point of origin, scales to pixelcoordinates.

    Input:
        ingeo: input geodataframe or shapely Polygon.
        reference_bounds:  Bounding box object or tuple of reference (e.g. image chip) in format (left, bottom,
            right, top)
        scale: Scale the polygons to the image size/resolution. Requires image array nrows and ncols parameters.
        nrows: image array nrows, required for scale.
        ncols: image array ncols, required for scale.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _to_pixelcoords(poly: Polygon, reference_bounds, scale, nrows, ncols):
        try:
            minx, miny, maxx, maxy = reference_bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'reference_bounds argument is of type {type(reference_bounds)}, needs to be a tuple or rasterio bounding box '
                f'instance. Can be delineated from transform, nrows, ncols via rasterio.transform.reference_bounds')

        # Subtract point of origin of image bbox.
        x_coords, y_coords = poly.exterior.coords.xy
        p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            if ncols is None or nrows is None:
                raise ValueError('ncols and nrows required for scale')
            x_scaler = ncols / w_poly
            y_scaler = nrows / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    if isinstance(ingeo, Polygon):
        return _to_pixelcoords(poly=ingeo, reference_bounds=reference_bounds, scale=scale, nrows=nrows, ncols=ncols)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, reference_bounds=reference_bounds,
                                                                         scale=scale, nrows=nrows, ncols=ncols))
        return ingeo


def invert_y_axis(ingeo: Union[Polygon, GDF],
                  reference_height: int
                  ) -> Union[Polygon, GDF]:
    """Invert y-axis of polygon or geodataframe geometries in reference to a bounding box e.g. of an image chip.

    Usage e.g. for COCOJson format.

    Args:
        ingeo: Input Polygon or geodataframe.
        reference_height: Height (in coordinates or rows) of reference object (polygon or image, e.g. image chip.

    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _invert_y_axis(poly: Polygon=ingeo, reference_height=reference_height):
        x_coords, y_coords = poly.exterior.coords.xy
        p_inverted_y_axis = shapely.geometry.Polygon([[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        return p_inverted_y_axis

    if isinstance(ingeo, Polygon):
        return _invert_y_axis(poly=ingeo, reference_height=reference_height)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _invert_y_axis(poly=_p, reference_height=reference_height))
        return ingeo


def reclassify_col(df: Union[GDF, DF],
                   rcl_scheme: Dict,
                   col_classlabels: str= 'lcsub',
                   col_classids: str= 'lcsub_id',
                   drop_other_classes: bool=True
                   ) -> Union[GDF, DF]:
    """Reclassify class label and class ids in a dataframe column.

    # TODO: Make more efficient!
    Args:
        df: input geodataframe.
        rcl_scheme: Reclassification scheme, e.g. {'springcereal': [1,2,3], 'wintercereal': [10,11]}
        col_classlabels: column with class labels.
        col_classids: column with class ids.
        drop_other_classes: Drop classes that are not contained in the reclassification scheme.

    Returns:
        Result dataframe.
    """
    if drop_other_classes is True:
        classes_to_drop = [v for values in rcl_scheme.values() for v in values]
        df = df[df[col_classids].isin(classes_to_drop)].copy()

    rcl_dict = {}
    rcl_dict_id = {}
    for i, (key, value) in enumerate(rcl_scheme.items(), 1):
        for v in value:
            rcl_dict[v] = key
            rcl_dict_id[v] = i

    df[f'rcl_{col_classlabels}'] = df[col_classids].copy().map(rcl_dict)  # map name first, id second!
    df[f'rcl_{col_classids}'] = df[col_classids].map(rcl_dict_id)
    return df


def reproject_shapely(geometry: shapely.geometry,
                      epsg_in: Union[str, int],
                      epsg_out: Union[str, int]
                      ) -> shapely.geometry:
    """Reproject shapely geometry to different epsg crs.

    Example: my_reproject(mp, 'epsg:28992', 'epsg:32631')

    Args:
        geometry: input shapely geometry
        epsg_in: input epsg code
        epsg_out: epsg code for reprojection

    Returns:
         Results shapely geometry.
    """
    project = functools.partial(
        pyproj.transform,
        pyproj.Proj(init=f'epsg{str(epsg_in)}'),
        pyproj.Proj(init=f'epsg{str(epsg_out)}'))
    geometry = shapely.ops.transform(project, geometry)  # apply projection
    return geometry


def set_crs(df: GDF, epsg_code: Union[int, str]) -> GDF:
    """Sets dataframe crs in geopandas pipeline.

    TODO: Deprecate with next rasterio version that will integrate set_crs method.
    """
    df.crs = {'init': f'epsg:{str(epsg_code)}'}
    return df
