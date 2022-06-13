from collections import defaultdict
from itertools import count, pairwise

import numpy as np

from .models import Graaf, JSON


def parse_graaf(json: JSON) -> Graaf:
    # json['elements'] = [
    #   {'type': 'node', 'id': int, 'lat': float, 'lon': float,
    #    'tags': {'railway': 'crossing'}},
    #   {'type': 'node', 'id': int, 'lat': float, 'lon': float,
    #    'tags': {'barrier': 'bollard', 'foot': 'yes'}},
    #   ...,
    #   {'type': 'way', 'id': int,
    #    'nodes': [int, int, int, ...],
    #    'tags': {'highway': 'unclassified', 'oneway': 'yes'}},
    #   ...
    # ]
    vertex_index = defaultdict(count().__next__)

    edges = np.array(
        [
            (vertex_index[a], vertex_index[b])
            for node in json['elements']
            if node['type'] == 'way'
            for a, b in pairwise(node['nodes'])
        ],
        dtype=int
    )

    vertex_index = dict(vertex_index)

    vertices = np.array(
        [
            (vertex_index[node['id']], node['lat'], node['lon'])
            for node in json['elements']
            if node['type'] == 'node'
        ],
        dtype=float
    )
    vertices = vertices[np.argsort(vertices[:, 0]), 1:]

    assert vertices.shape[0] == max(vertex_index.values()) + 1

    return Graaf(vertices, edges)
