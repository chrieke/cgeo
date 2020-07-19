import numpy as np


class Indices:
    """
    Provides various remote sensing indices from numpy arrays.

    Example:
        Indices(band_names=['red', 'nir'], array=np.array([[0.5, 0,1], [0.7, 02]]).ndvi()
    """

    @staticmethod
    def ndvi(nir, red):
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        ndvi = ((nir) - (red)) / ((nir) + (red))
        return ndvi

    @staticmethod
    def evi(nir, red, blue):
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        blue = blue.astype(np.float32)
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        return evi

    @staticmethod
    def ndwi(swir, nir):
        swir = swir.astype(np.float32)
        nir = nir.astype(np.float32)
        index = (nir - swir) / (nir + swir)
        return index

    @staticmethod
    def brightness(swir, nir, red, green):
        swir = swir.astype(np.float32)
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        brightness = np.sqrt(
            np.power(green, 2) + np.power(red, 2) + np.power(nir, 2) + np.power(swir, 2)
        )
        return brightness


class Sensors:
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

    def bands_info(self):
        return self.sensor_bands[self.sensor]
