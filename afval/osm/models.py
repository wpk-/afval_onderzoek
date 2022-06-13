from typing import Any, NamedTuple, TypeVar

import numpy as np

JSON = dict[str, Any]


class Graaf(NamedTuple):
    vertices: np.ndarray
    edges: np.ndarray


Model = TypeVar('Model')    # Graaf, ...
