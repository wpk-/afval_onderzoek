from collections.abc import Sequence

from numpy import asarray, dstack, ndarray
from pyproj import CRS, Transformer

gps = 4326
rd = rijksdriehoek = 28992


def projecteer_epsg(coordinates: ndarray | Sequence[Sequence[float]],
                    naar: int = gps, van: int = gps) -> ndarray:
    """Projecteert coördinaten van projectie `van` naar projectie `naar`.

    :param coordinates: Numpy n x 2 array of een vergelijkbare lijst voor
        `numpy.asarray()`. De coordinaten zijn floats in de projectie
        `van`.
    :param naar: EPSG-nummeraanduiding van de projectie waarnaar de
        coordinaten omgezet worden. Standaard is 4326 (GPS).
    :param van: EPSG-nummeraanduiding van de projectie waarin de
        coordinaten gegeven zijn. Standaard is 4326 (GPS).
    :return: Numpy n x 2 array met de geprojecteerde coordinaten.
    """
    crs_from = CRS.from_epsg(van)
    crs_to = CRS.from_epsg(naar)
    transformer = Transformer.from_crs(crs_from, crs_to)

    coordinates = asarray(coordinates)
    fx, fy = transformer.transform(coordinates[:, 0], coordinates[:, 1])
    return dstack([fx, fy])[0]  # Re-create (n,2) coordinates


if __name__ == '__main__':
    import numpy as np
    coords = np.asarray([
        [52.3470533, 4.996248],
        [52.3533863, 5.008157],
    ])
    print(projecteer_epsg(coords, naar=rijksdriehoek))
