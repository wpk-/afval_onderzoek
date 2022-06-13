import logging
from collections import defaultdict

import numpy as np
from aapi.csv import ContainersCsv, VerblijfsobjectenCsv

from afval import graph
from afval.graph.vis import plot_distances, plot_graph
from afval.osm import WandelwegenJson
from afval.projection import project_array

containers = ContainersCsv(
    'cache/Containers papier 2022-05-23.csv',
    {
        'fractieOmschrijving': 'Papier',
        'status': 1,
        'verwijderdDp': False,
        'geometrie[isnull]': False,
    }
)

huishoudens = VerblijfsobjectenCsv(
    'cache/Verblijfsobjecten met woonfunctie status-347 2022-05-16.csv',
    {
        'statusCode[in]': '3,4,7',
        'gebruiksdoel': 'woonfunctie',
    }
)

wegen = WandelwegenJson(
    'cache/Wandelwegen Amsterdam 2022-05-16.json',
    (52.2880962, 4.7543934, 52.4252595, 5.0212665),
)


def argsort_rows(a: np.ndarray) -> np.ndarray:
    """Return the sorting argument for rows of a.

    Implementation based on post by Jaime:
    https://stackoverflow.com/a/16973510
    and then https://stackoverflow.com/a/46953582

    :param a: Numpy (n, 2) array. All values must be numeric and positive!
    :return: An index vector `i` so that `a[i]` contains the rows of `a`
        in sorted order. Ties in the j-th column are broken by values in
        the j+1-th column. So, `[3, 3]` sorts after `[2, 4]` and `[3, 2]`
        and before `[4, 2]` and `[3, 4]`.
    """
    b = np.ascontiguousarray(a).view(
        np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    return np.argsort(b.ravel())


def main():
    from time import time

    # 52.3470533,4.996248   <- linksonder: lat (onder), long (links)
    # 52.3533863,5.008157   <- rechtsboven: lat (boven), long (rechts)
    # bbox = (lat_min, lng_max, lat_max, lng_min)
    # bbox = (52.3470533, 4.9962480, 52.3533863, 5.0081570)   # IJburg
    # bbox = (52.3432026, 4.9763181, 52.3613222, 5.0201921)   # IJburg redone
    t0 = time()
    kaart = wegen.read()
    kaart = graph.Graph(project_array(kaart.vertices, 28992), kaart.edges)
    logging.debug(f'Kaart gelezen in {time() - t0:.2f} sec.'
                  f' ({kaart.num_vertices} verts, {kaart.num_edges} edges)')

    # fig = plot_graph(kaart)

    t0 = time()
    kaart, el = graph.subdivide(max_edges_length=3.0, g=kaart)
    logging.debug(f'Opgeknipt in wegdelen van max 3m'
                  f' in {time() - t0:.2f} sec.')

    t0 = time()
    kaart = graph.KDGraph.from_graph(kaart)
    logging.debug(f'KDTree gemaakt in {time() - t0:.2f} sec.')

    t0 = time()
    clusters = defaultdict(list)
    for c in containers:
        clusters[c.clusterId].append(c.geometrie)
    targets_xy = np.vstack([np.mean(cc, axis=0) for cc in clusters.values()])
    # targets_xy = np.array([c.geometrie for c in containers])
    logging.debug(f'{targets_xy.shape[0]} containers gelezen'
                  f' in {time() - t0:.2f} sec.')

    t0 = time()
    sources_xy = np.array([h.geometrie for h in huishoudens])
    logging.debug(f'{sources_xy.shape[0]} huishoudens gelezen'
                  f' in {time() - t0:.2f} sec.')
    logging.debug(f'Huishouden nul ligt op coordinaat: {sources_xy[0]}')

    # Optioneel. Kan ook i.p.v. targets_index direct targets_xy gebruiken.
    t0 = time()
    targets_index = graph.nearest_target(targets_xy, kaart, el)
    logging.debug(f'{targets_index.distance.size} afstanden tot de container'
                  f' in {time() - t0:.2f} sec.')

    t0 = time()
    source_target = graph.shortest_distance(sources_xy, targets_index, kaart)
    logging.debug(f'{source_target.distance.size} huishouden-loopafstanden'
                  f' in {time() - t0:.2f} sec.')

    print(f'Kortste afstand tot een container: {source_target.distance.min()}')
    print(f'Langste afstand tot een container: {source_target.distance.max()}')
    # ix_source = np.argmax(source_target.distance)
    # ix_target = source_target.index[ix_source]
    # print(f'Source {ix_source}: {sources_xy[ix_source]}')
    # print(f'Target {ix_target}: {targets_xy[ix_target]}')
    # # print(f'Minste huishoudens per container: {}')
    # # print(f'Meeste huishoudens per container: {}')
    #
    # fig = plot_distances(kaart, sources_xy, targets_xy, source_target)
    # fig.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    main()
