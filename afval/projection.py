from collections.abc import Sequence

from numpy import asarray, dstack, ndarray
from pyproj import CRS, Transformer


def project_array(coordinates: ndarray | Sequence[Sequence[float]],
                  to_epsg: int = 4326, from_epsg: int = 4326) -> ndarray:
    """Project coordinates in projection from_epsg to projection to_epsg.
    Returns a numpy (n,2) array.
    """
    crs_from = CRS.from_epsg(from_epsg)
    crs_to = CRS.from_epsg(to_epsg)
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
    print(project_array(coords, to_epsg=28992))
