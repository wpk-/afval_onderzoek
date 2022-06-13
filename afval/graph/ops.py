from heapq import heapify, heappop, heappush

import numpy as np

from .models import Graph, KDGraph, NearestTarget

__all__ = ['nearest_target', 'shortest_distance', 'subdivide']


def nearest_target(targets: np.ndarray, g: KDGraph,
                   edges_length: np.ndarray | None = None) -> NearestTarget:
    """For every vertex in the graph, finds its nearest target (along the
    edges).

    :param targets: (n, 2) numpy array of target coordinates.
    :param g: The graph with vertices and edges.
    :param edges_length: Optional length-n numpy array with precomputed
    edge lengths. By default this will be computed.
    :return: An instance of NearestTarget. It stores for every vertex in
    the graph, its distance (along the graph) to the nearest target and
    the index of the vertex that is closest to that target.
    """
    targets = np.atleast_2d(targets)
    d, ix = g.tree.query(targets)

    el = g.edges_length() if edges_length is None else edges_length

    n = g.num_vertices
    z = np.ones(n, dtype=float) * np.inf    # Shortest distance to target.
    r = -np.ones(n, dtype=int)              # Index of the nearest target.

    a = {v: [] for v in range(n)}           # Actions: vertex -> next.
    for (ve1, ve2), l in zip(g.edges, el):
        a[ve1].append((l, ve2))
        a[ve2].append((l, ve1))

    # z: vertex -> distance
    # r: vertex -> target
    # a: vertex -> [(length, next vertex), ...]

    front = list(zip(d, ix, range(d.size)))
    heapify(front)

    while front:
        zi, vi, ri = heappop(front)
        if z[vi] <= zi:
            continue
        else:
            if not np.isinf(z[vi]):
                raise ValueError('Algorithmic error!')
            z[vi] = zi
            r[vi] = ri
        # logging.debug(f'{vi}: {zi:.2f}')
        for l, vj in a[vi]:
            heappush(front, (zi + l, vj, ri))

    return NearestTarget(z, r)


def shortest_distance(sources: np.ndarray, targets: NearestTarget | np.ndarray,
                      g: KDGraph) -> NearestTarget:
    """Computes for each source its nearest target over the graph.

    :param sources: (n, 2) numpy array of spatial coordinates.
    :param targets: (m, 2) numpy array of spatial coordinates or a
    NearestTarget object that holds precomputed shortest distances for
    all vertices in the graph.
    :param g: A KDGraph with its vertices stored in a KDTree.
    :return: A NearestTarget object with n entries. It assigns every
    source a nearest target (0..m-1) and the distance to that target
    along the edges of the graph.
    The path length consists of three parts that are summed together:
    1. the distance from source to the nearest vertex on the graph,
    2. the distance along the edges of the graph, and
    3. the distance from the graph to the target.
    """
    if not isinstance(targets, NearestTarget):
        targets = nearest_target(targets, g)

    sources = np.atleast_2d(sources)
    d, ix = g.tree.query(sources)
    return NearestTarget(d + targets.distance[ix], targets.index[ix])


def subdivide(max_edges_length: float, g: Graph) -> tuple[Graph, np.ndarray]:
    """Knipt alle edges in stukken niet langer dan max_edges_length.

    :param max_edges_length: Maximale lengte van een lijnstuk tussen twee
        vertices.
    :param g: Oorspronkelijke graaf.
    :return: Een tuple van twee waardes:
        1. Een nieuwe graaf met extra vertices en edges zodanig dat geen
        van de edges langer is dan max_edges_length.
        2. Een numpy array met alle lengtes van de edges.
    """
    edges = g.edges
    a = g.a
    b = g.b
    length = np.sqrt(np.einsum('ij,ij->i', b - a, b - a))

    add_verts = []
    add_edges = []
    add_length = []
    keep_edges = length <= max_edges_length

    # l = 7: 7/2 = 3+ = 4 parts
    # l = 8: 8/2 = 4  = 4 parts
    ix = np.flatnonzero(~keep_edges)
    nv = g.num_vertices

    for ei in ix:
        ne_i = np.ceil(length[ei] / max_edges_length).astype(int)
        nv_i = ne_i - 1

        interp = np.linspace(a[ei], b[ei], ne_i, endpoint=False)
        add_verts.append(interp[1:])

        add_edges.append(
            [
                (edges[ei][0], nv),
                (nv + nv_i - 1, edges[ei][1])
            ] + [
                (nv + i, nv + i + 1)
                for i in range(ne_i - 2)
            ]
        )

        add_length.append([length[ei] / ne_i] * ne_i)

        nv += nv_i

    g = Graph(np.vstack([g.vertices] + add_verts),
              np.vstack([edges[keep_edges]] + add_edges))
    l = np.hstack([length[keep_edges]] + add_length)

    return g, l
