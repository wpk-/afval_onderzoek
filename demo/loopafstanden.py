import logging
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from aapi.csv import ContainersCsv, VerblijfsobjectenCsv
from diskcache import Cache
from scipy.spatial import KDTree

from afval.graaf import (
    Graaf, KDGraaf, knip, dichtstbijzijnde_doel, kortste_afstand,
    herverdeel_hemelsbreed, kortste_afstand)
from afval.osm import wandelwegen, amsterdam
from afval.projectie import projecteer_epsg
from afval.types import AfstandIndex

containers = ContainersCsv(
    'cache/Containers rest 2022-01-10.csv',
    {
        'fractieOmschrijving': 'Rest',
        'status': 1,
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

bbox_amsterdam = (52.2880962, 4.7543934, 52.4252595, 5.0212665)

cache = Cache('cache/')


@cache.memoize(name='cached_amsterdam', expire=60*60*24*1000)
def cached_amsterdam(*args, **kwargs) -> tuple[Graaf, np.ndarray]:
    return amsterdam(*args, **kwargs)


def plot_graph(g: Graaf, fig: go.Figure | None = None) -> go.Figure:
    nan = np.nan * np.ones((g.aantal_lijnen, 1))
    x = np.hstack([g.punten[:, 0][g.lijnen], nan])
    y = np.hstack([g.punten[:, 1][g.lijnen], nan])

    fig = go.Figure() if fig is None else fig

    fig.add_trace(go.Scattergl(
        x=np.ravel(x),
        y=np.ravel(y),
        mode='markers+lines',
        line={'width': 1},
        marker={'color': 'black', 'size': 3},
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def plot_distances(g: Graaf, sources: np.ndarray, targets: np.ndarray,
                   source_target: AfstandIndex) -> go.Figure:
    cmap = np.arange(targets.shape[0])
    np.random.shuffle(cmap)

    _, index = source_target

    fig = plot_graph(g)

    fig.add_trace(go.Scattergl(
        x=sources[:, 0],
        y=sources[:, 1],
        mode='markers',
        marker={'color': cmap[index]},
    ))
    fig.add_trace(go.Scattergl(
        x=targets[:, 0],
        y=targets[:, 1],
        mode='markers',
        marker={'color': cmap, 'size': 10,
                'line': {'width': 2, 'color': 'black'}},
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


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


def kortst():
    knip = 3.
    dubbel_cluster = 25.

    print('Lees clusters...')
    clusters = defaultdict(list)
    for c in containers:
        clusters[c.clusterId].append(c.geometrie)
    targets_xy = np.vstack([np.mean(cc, axis=0) for cc in clusters.values()])

    print('Lees huishoudens...')
    sources_xy = np.array([h.geometrie for h in huishoudens])

    afstand = np.inf * np.ones(len(clusters), dtype=float)
    index = np.zeros(len(clusters), dtype=int)

    for i, (fietspaden, voetpaden) in enumerate((
            (False, False), (False, True), (True, False), (True, True))):
        print('Lees de kaart van Amsterdam...')
        kaart, lijn_lengtes = cached_amsterdam(
            knip_lengte=knip, fietspaden=fietspaden, voetpaden=voetpaden)

        print('Bereken loopafstanden...')
        d, ix = kortste_afstand(kaart, sources_xy, targets_xy,
                                lijn_lengtes=lijn_lengtes,
                                correctie_op_afstand=dubbel_cluster)

        if i == 0:
            afstand = d
            index = ix
        else:
            ii = d < afstand
            afstand[ii] = d[ii]
            index[ii] = ix[ii]
            print(ii.sum(), 'afstanden verkort.')

    print('Plot...')
    kaart0, _ = cached_amsterdam(fietspaden=True, voetpaden=True)
    fig = plot_distances(kaart0, sources_xy, targets_xy, (afstand, index))
    fig.show()


def main():
    from time import time

    # 52.3470533,4.996248   <- linksonder: lat (onder), long (links)
    # 52.3533863,5.008157   <- rechtsboven: lat (boven), long (rechts)
    # bbox = (lat_min, lng_max, lat_max, lng_min)
    # bbox = (52.3470533, 4.9962480, 52.3533863, 5.0081570)   # IJburg
    # bbox = (52.3432026, 4.9763181, 52.3613222, 5.0201921)   # IJburg redone
    # bbox = (52.3625584, 4.9303917, 52.3642942, 4.9339828)   # Javastraat 9
    # bbox = (52.3589340, 4.8612879, 52.3617979, 4.8704289)   # Overtoom
    t0 = time()
    # kaart = wandelwegen(bbox, timeout=10)
    kaart = wandelwegen(bbox_amsterdam, timeout=10)
    kaart0 = kaart = Graaf(projecteer_epsg(kaart.punten, 28992), kaart.lijnen)
    print(f'Kaart gelezen, {time() - t0:.2f} sec.'
          f' ({kaart.aantal_punten} punten, {kaart.aantal_lijnen} lijnen)')

    # fig = plot_graph(kaart)
    # fig.show()
    # return

    t0 = time()
    kaart, el = knip(kaart, max_lengte=3.0)
    print(f'Opgeknipt in wegdelen van max 3m, {time() - t0:.2f} sec.')

    # fig = plot_graph(kaart)
    # fig.show()
    # return

    t0 = time()
    kaart = KDGraaf.van_graaf(kaart)
    print(f'KDTree gemaakt, {time() - t0:.2f} sec.')

    t0 = time()
    clusters = defaultdict(list)
    for c in containers:
        clusters[c.clusterId].append(c.geometrie)
    targets_xy = np.vstack([np.mean(cc, axis=0) for cc in clusters.values()])
    print(f'{targets_xy.shape[0]} containers gelezen, {time() - t0:.2f} sec.')

    # ixt = ((targets_xy[:, 0] > 126250) &
    #       (targets_xy[:, 1] > 483450) &
    #       (targets_xy[:, 1] < 486500))
    # t = targets_xy[ixt]
    # fig = plot_graph(kaart)
    # cmap = np.arange(t.shape[0])
    # np.random.shuffle(cmap)
    # fig.add_trace(go.Scattergl(
    #     x=t[:, 0],
    #     y=t[:, 1],
    #     mode='markers',
    #     marker={'color': cmap, 'size': 10,
    #             'line': {'width': 2, 'color': 'black'}},
    # ))
    # fig.show()
    # return

    t0 = time()
    sources_xy = np.array([h.geometrie for h in huishoudens])
    print(f'{sources_xy.shape[0]} huishoudens gelezen, {time() - t0:.2f} sec.')
    print(f'Huishouden nul ligt op coordinaat: {sources_xy[0]}')

    # Optioneel. Kan ook i.p.v. targets_index direct targets_xy gebruiken.
    t0 = time()
    targets_index = dichtstbijzijnde_doel(kaart, targets_xy, el)
    print(f'{targets_index[0].size} afstanden tot de container,'
          f' {time() - t0:.2f} sec.')

    t0 = time()
    source_target = kortste_afstand(kaart, sources_xy, targets_index)
    print(f'{source_target[0].size} huishouden-loopafstanden'
          f' in {time() - t0:.2f} sec.')

    # Voor dicht bij elkaar gelegen clusters (max afstand 25 meter),
    # herverdeel de huishoudens op hemelsbrede afstand.
    source_target = herverdeel_hemelsbreed(source_target, sources_xy,
                                           targets_xy, max_onderling=25)

    print(f'Kortste afstand tot een container: {source_target[0].min()}')
    print(f'Langste afstand tot een container: {source_target[0].max()}')
    # ix_source = np.argmax(source_target.distance)
    # ix_target = source_target.index[ix_source]
    # print(f'Source {ix_source}: {sources_xy[ix_source]}')
    # print(f'Target {ix_target}: {targets_xy[ix_target]}')
    # # print(f'Minste huishoudens per container: {}')
    # # print(f'Meeste huishoudens per container: {}')
    #
    fig = plot_distances(kaart0, sources_xy, targets_xy, source_target)
    fig.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger('requests').setLevel(logging.WARNING)
    # logging.getLogger('urllib3').setLevel(logging.WARNING)
    # main()
    kortst()
