from typing import Union, Dict

from geopandas import GeoDataFrame as GDF
from pandas import DataFrame as DF


def reclassify_col(
    df: Union[GDF, DF],
    rcl_scheme: Dict,
    col_classlabels: str = "lcsub",
    col_classids: str = "lcsub_id",
    drop_other_classes: bool = True,
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

    df[f"rcl_{col_classlabels}"] = (
        df[col_classids].copy().map(rcl_dict)
    )  # map name first, id second!
    df[f"rcl_{col_classids}"] = df[col_classids].map(rcl_dict_id)
    return df
