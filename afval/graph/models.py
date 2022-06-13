from typing import NamedTuple

import numpy as np
from scipy.spatial import KDTree

__all__ = ['Graph', 'KDGraph', 'NearestTarget']


class Graph:
    """A graph with vertices and edges.
    """
    def __init__(self, vertices: np.ndarray, edges: np.ndarray) -> None:
        self.vertices = vertices
        self.edges = edges

    @property
    def a(self) -> np.ndarray:
        return self.vertices[self.edges[:, 0]]

    @property
    def b(self) -> np.ndarray:
        return self.vertices[self.edges[:, 1]]

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def edges_length(self) -> np.ndarray:
        ab = self.b - self.a
        return np.sqrt(np.einsum('ij,ij->i', ab, ab))


class KDGraph(Graph):
    """A graph that stores its vertices in a KDTree.
    """
    def __init__(self, vertices: np.ndarray, edges: np.ndarray) -> None:
        self.tree: KDTree = None
        super().__init__(vertices, edges)

    @property
    def num_vertices(self) -> int:
        return self.tree.n

    @property
    def vertices(self) -> np.ndarray:
        return self.tree.data

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        self.tree = KDTree(vertices)

    @classmethod
    def from_graph(cls, g: Graph) -> 'KDGraph':
        return cls(g.vertices, g.edges)


class NearestTarget(NamedTuple):
    distance: np.ndarray
    index: np.ndarray
