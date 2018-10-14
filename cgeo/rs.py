# rs.py

import numpy as np


class Indices(object):
    """
    Remote sensing indices from numpy arrays.
    """
    @staticmethod
    def ndvi(nir, red):
        # nir = nir.astype(np.float32)
        # red = red.astype(np.float32)
        ndvi = ((nir) - (red)) / ((nir) + (red))
        return ndvi

    @staticmethod
    def evi(nir, red, blue):
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        blue = blue.astype(np.float32)
        evi = 2.5*((nir-red)/(nir+6*red - 7.5 * blue + 1))
        return evi

    @staticmethod
    def ndwi(nir, swir):
        swir = swir.astype(np.float32)
        nir = nir.astype(np.float32)
        index = ((nir - swir) / (nir + swir))
        return index

    @staticmethod
    def brightness(nir, green, red, swir):
        swir = swir.astype(np.float32)
        nir = nir.astype(np.float32)
        red = red.astype(np.float32)
        green = green.astype(np.float32)
        brightness = np.sqrt(np.power(green, 2) + np.power(red, 2) + np.power(nir, 2) + np.power(swir, 2))
        return brightness
