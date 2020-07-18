import pytest

from .context import geo


@pytest.mark.parametrize(
    "lon, lat, epsg_expected",
    [
        (-79.52826976776123, 8.847423357771518, 32617),  # Panama
        (9.95121, 49.79391, 32632),  # Wuerzburg
        (9.767417, 62.765571, 32632),  # Norway special zone
        (12.809028, 79.026583, 32633),  # Svalbard special zone
    ],
)
def test_get_utm_zone_epsg(lat, lon, epsg_expected):
    epsg = geo.get_utm_zone_epsg(lat=lat, lon=lon)

    assert epsg == epsg_expected
