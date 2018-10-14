# geo.py

import functools
import warnings
from typing import Union, Dict

import numpy as np
import rasterio
import geopandas as gpd
from geopandas import GeoDataFrame as GDF
from pandas import DataFrame as DF
import shapely
from shapely.geometry import Polygon, MultiPolygon
import pyproj


def buffer_zero(df: GDF) -> GDF:
    """
    Make self-intersecting, invalid geometries in geodataframe valid by buffering with 0.
    """
    if False in df.geometry.is_valid.unique():
        df.geometry = df.geometry.apply(lambda p: p.buffer(0))
    return df


def explode_mp(df: GDF) -> GDF:
    """
    Explode all MultiPolygon geometries in a geodataframe into individual Polygon geometries.

    Adds exploded Polygons as rows at the end of the geodataframe and resets index!
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


def intersect_with_geo(df: GDF,
                       clip_geo: Polygon,
                       explode_mp_: bool=False,
                       keep_biggest: bool=False) -> GDF:
    """
    Filter and clip geodataframe to clipping geometry.

    Args:
        df (GDF): input geodataframe
        clip_geo (Polygon): clipping geometry
        explode_mp_ (bool): Apply explode_mp function. Append dataframe rows for each polygon in potential
                            multipolygons that were created by the intersection. Resets the dataframe index!
        keep_biggest (bool): Drops Multipolygons by only keeping the Polygon with the greatest area.

    Returns (GDF)
    """
    df = df[df.geometry.intersects(clip_geo)].copy()
    df.geometry = df.geometry.apply(lambda p: p.intersection(clip_geo))
    df = buffer_zero(df)

    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()

    if not row_idxs_mp:
        return df
    elif not explode_mp_ and (not keep_biggest):
        if row_idxs_mp:
            warnings.warn(f"Warning, intersection resulted in {len(row_idxs_mp)} split polygons. Use explode_mp_=True "
                          f"or keep_biggest=True!")
        return df
    elif explode_mp_ and keep_biggest:
        raise ValueError('You can only use only "explode_mp" or "keep_biggest"!')
    elif explode_mp_:
        return explode_mp(df)
    elif keep_biggest:
        for idx in row_idxs_mp:
            mp = df.loc[idx].geometry
            poly_areas = [p.area for p in mp]
            max_area_poly = mp[poly_areas.index(max(poly_areas))]
            df.loc[idx, 'geometry'] = max_area_poly
        return df


def reduce_precision(ingeo: Union[Polygon, GDF], precision: int=3) -> Union[Polygon, GDF]:
    """
    Reduces the after comma precision of a shapely Polygon or geodataframe geometries.

    GeoJSON specification recommends 6 decimal places for latitude and longitude which equates to roughly 10cm of
    precision (https://github.com/perrygeo/geojson-precision).

    Args:
        ingeo (Union[Polygon, GDF]): input geodataframe or shapely Polygon.
        precision (int): number of after comma values that should be kept.

    Returns (Union[GDF, Polygon]): Same type as input.
    """
    def _set_precision(poly: Polygon, precision: int):
        geojson = shapely.geometry.mapping(poly)
        geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision)
        poly = shapely.geometry.shape(geojson)
        if not poly.is_valid:
            # Too low precision would potentially lead to invalid polygons due to line overlap effects.
            poly = poly.buffer(0)
        return poly

    if isinstance(ingeo, Polygon):
        return _set_precision(poly=ingeo, precision=precision)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _set_precision(poly=_p, precision=precision))
        return ingeo


def to_pixelcoords(ingeo: Union[Polygon, GDF],
                   bounds: Union[rasterio.coords.BoundingBox, tuple],
                   invert_y: bool=False,
                   scale: bool=False,
                   height: int=None,
                   width: int=None) -> Union[Polygon, GDF]:
    """
    Converts projected polygon coordinates to pixel coordinates of an image array.

    Subtracts point of origin, scales to pixelcoordinates. Optional y-axis inversion (e.g. for COCOjson format).

    Example:
        df.geometry = df.   geometry.apply(lambda p: to_pixelcoords(p, img_meta, scale=True, invert_y=True))

    Input:
        ingeo (Union[Polygon, GDF]): input geodataframe or shapely Polygon.
        bounds (Union[rasterio.coords.BoundingBox, tuple]):  Bounding box.
        invert_y (bool): e.g. for COCOjson format.
        scale (bool): Scale the polygons to the image size/resolution. Requires image array height and width parameters.
        height (int): image array height, optional, required for scaling.
        width (int): image array width, optional, required for scaling.

    Returns (Union[GDF, Polygon]): Same type as input.
    """
    def _to_pixelcoords(poly: Polygon, bounds: Union[rasterio.coords.BoundingBox, tuple],
                        invert_y: bool, scale: bool, height: int, width: int):
        try:
            minx, miny, maxx, maxy = bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'bounds argument is of type {type(bounds)}, needs to be a tuple or rasterio bounding box instance. '
                f' Can be delineated from transform, height, width via rasterio.transform.array_bounds')

        if invert_y is False:
            # Subtract point of origin of image bbox. Shpbbox could be different if polygons not directly at the edges.
            x_coords, y_coords = poly.exterior.coords.xy
            p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])
        elif invert_y is True:
            # Subtract point of origin of poly_bbox, invert y-axis (required by COCO format!)
            x_coords, y_coords = poly.exterior.coords.xy
            p_origin = shapely.geometry.Polygon([[x - minx, h_poly - (y - miny)] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            x_scaler = width / w_poly
            y_scaler = height / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    if isinstance(ingeo, Polygon):
        return _to_pixelcoords(poly=ingeo, bounds=bounds, invert_y=invert_y, scale=scale, height=height, width=width)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, bounds=bounds, invert_y=invert_y,
                                                                         scale=scale, height=height, width=width))
        return ingeo


def reclassify(df: Union[GDF, DF],
               rcl_scheme: Dict,
               col_classname: str='lcsub',
               col_classid: str='lcsub_id',
               drop_other_classes: bool=True) -> Union[GDF, DF]:
    """
    Reclassifies the input dataframe

    # TODO: Make more efficient!
    Args:
        df (GDF): input geodataframe.
        rcl_scheme (Dictionary): Reclassification scheme, e.g. {'springcereal': [1,2,3], 'wintercereal': [10,11]}
        col_classname (str):
        col_classid (str):
        drop_other_classes (bool): Drop classes that are not contained in the reclassification scheme.

    Returns:

    """
    if drop_other_classes is True:
        classes_to_drop = [v for values in rcl_scheme.values() for v in values]
        df = df[df[col_classid].isin(classes_to_drop)].copy()

    rcl_dict = {}
    rcl_dict_id = {}
    for i, (key, value) in enumerate(rcl_scheme.items(), 1):
        for v in value:
            rcl_dict[v] = key
            rcl_dict_id[v] = i

    df[f'rcl_{col_classname}'] = df[col_classid].copy().map(rcl_dict)  # map name first, id second!
    df[f'rcl_{col_classid}'] = df[col_classid].map(rcl_dict_id)
    return df


def reproject_shapely(geometry: shapely.geometry,
                      epsg_in: Union[str, int],
                      epsg_out: Union[str, int]) -> shapely.geometry:
    """
    Reproject shapely feature to different epsg crs.

    Example: my_reproject(mp, 'epsg:28992', 'epsg:32631')

    Args:
        geometry (shapely.geometry): input shapely geometry
        epsg_in (Union[str, int]): input epsg code
        epsg_out (Union[str, int]): epsg code for reprojection

    Returns (shapely.geometry):
    """
    project = functools.partial(
        pyproj.transform,
        pyproj.Proj(init=f'epsg{str(epsg_in)}'),
        pyproj.Proj(init=f'epsg{str(epsg_out)}'))
    geometry = shapely.ops.transform(project, geometry)  # apply projection
    return geometry


def set_crs(df: GDF, epsg_code: Union[int, str]) -> GDF:
    """
    Sets dataframe crs in geopandas pipeline.
    TODO: Deprecate with next rasterio version, will integrate set_crs method.
    """
    df.crs = {'init': f'epsg:{str(epsg_code)}'}
    return df
