# rs.py

import numpy as np
from typing import List, Dict


class Indices(object):
    """Provides calculation of various remote sensing indices from numpy arrays.

    Currently implemented: ndvi, evi, ndwi, brightness
    # TODO: Add more indices

    Args:
        bands_names: List of band names. E.g. ['coastal', 'blue', 'green', 'red', 'rededge', 'nir', 'swir']
        array: 3d input numpy array image. Channel order must correspond to the order in input_bands_names.

    Returns:
        Results index numpy array

    Example:
        Indices(band_names=['red', 'nir'], array=np.array([[0.5, 0,1], [0.7, 02]]).ndvi()
    """

    def __init__(self, bands_names: List[str], array: np.ndarray):
        self.input_band_names = bands_names
        for i, band_name in enumerate(bands_names):
            setattr(self, band_name, array[i])

    def ndvi(self):
        nir = self.nir.astype(np.float32)
        red = self.red.astype(np.float32)
        ndvi = ((nir) - (red)) / ((nir) + (red))
        return ndvi

    def evi(self):
        nir = self.nir.astype(np.float32)
        red = self.red.astype(np.float32)
        blue = self.blue.astype(np.float32)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        return evi

    def ndwi(self):
        swir = self.swir.astype(np.float32)
        nir = self.nir.astype(np.float32)
        index = (nir - swir) / (nir + swir)
        return index

    def brightness(self):
        swir = self.swir.astype(np.float32)
        nir = self.nir.astype(np.float32)
        red = self.red.astype(np.float32)
        green = self.green.astype(np.float32)
        brightness = np.sqrt(
            np.power(green, 2) + np.power(red, 2) + np.power(nir, 2) + np.power(swir, 2)
        )
        return brightness


class Sensors(object):
    """Provides sensor-specific band information, band combinations and metadata

    # TODO: Add more info
    Args:
        sensor: Name abbreviation of the sensor, e.g. 'S2', 'L8', 'S1'.
    """

    def __init__(self, sensor: str = None):
        if sensor not in ["S2", "L8", "S1"]:
            raise ValueError("Select the correct sensor!")
        self.sensor = sensor

        sensor_bands = {
            "S2": {
                1: "coastal",
                2: "blue",
                3: "green",
                4: "red",
                5: "rededge1",
                6: "rededge2",
                7: "rededge3",
                8: "nir",
                9: "watervapour",
                10: "cirrus",
                11: "swir1",
                12: "swir2",
            },
            "L8": {},
            "S1": {"VH", "VV"},
        }
        self.sensor_bands = sensor_bands

        band_combinations = {"S2": {"RGB": [4, 3, 2]}, "L8": {}, "S1": {}}
        self.band_combinations = band_combinations

    def bands_info(self) -> Dict:
        return self.sensor_bands[self.sensor]
