import numpy as np

from .context import rs


def test_ndvi():
    nir = np.array(([0.008], [0.807]), dtype=np.float32)
    red = np.array(([1.0], [0.33]), dtype=np.float32)
    ndvi = rs.Indices.ndvi(nir=nir, red=red)

    comp_ndvi = np.array(([[-0.9841269], [0.41952506]]))
    assert np.allclose(ndvi, comp_ndvi)
