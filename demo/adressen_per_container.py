"""
Telt per container het aantal huishoudens en bedrijfspanden.
Dit op basis van loopafstanden (OSM-wegen), BAG-nummeraanduidingen en de
containers API. Hierbij wordt ook het containervolume meegenomen met ook
onder-/bovengronds onderscheid en indicatie van pers.

Het resultaat wordt opgeslagen in een CSV-bestand met de Nederlandstalige
indeling (puntkomma-gescheiden en komma als decimaal teken).

out/adressen_per_container_glas.csv
out/adressen_per_container_papier.csv
out/adressen_per_container_rest.csv

Maximale afstand: 210 meter
Adressen verder weg dan dat worden genegeerd.
"""
import csv
from collections.abc import Iterator

import numpy as np
from aapi.csv import (
    ContainersCsv, VerblijfsobjectenCsv, ContainertypesCsv, StandplaatsenCsv,
    LigplaatsenCsv
)

from afval.graaf import (
    KDGraaf, dichtstbijzijnde_doel, herverdeel_hemelsbreed, kortste_afstand
)
from afval.osm import amsterdam

FRACTIE = 'Rest'
# Maximum loopafstand van een huishouden tot de container is gecapt op
# 210 meter, in overeenstemming met api.data.amsterdam.nl.
MAX_AFSTAND_TOT_CONTAINER = 210
# Tussen clusters met een kleinere onderlinge afstand worden huishoudens
# herveredeld op basis van hemelsbrede (Euclidische) afstand.
MIN_AFSTAND_TUSSEN_CLUSTERS = 25
FILE_OUT = f'out/adressen_per_container_{FRACTIE.lower()}.csv'


class Stacked:
    def __init__(self, *iters) -> None:
        self.iters = iters

    def __iter__(self) -> Iterator:
        def avg_geometrie(x: list[list]):
            try:
                return np.mean(x[0], axis=0).tolist()
            except np.AxisError:
                print(x)
        for x in self.iters[0]:
            yield x
        for iterable in self.iters[1:]:
            for x in iter(iterable):
                if x.geometrie:
                    yield x._replace(geometrie=avg_geometrie(x.geometrie))


containers = ContainersCsv(
    f'cache/Containers {FRACTIE.lower()} 2022-06-23.csv',
    {
        'fractieOmschrijving': FRACTIE,
        'status': 1,
        'geometrie[isnull]': False,
    },
)

containertypes = ContainertypesCsv(
    'cache/Containertypes 2022-06-23.csv',
    {},
)

huishoudens = Stacked(
    VerblijfsobjectenCsv(
        'cache/Verblijfsobjecten met woonfunctie status-347 2022-05-16.csv',
        {
            'statusCode[in]': '3,4,7',
            'gebruiksdoel.omschrijving': 'woonfunctie',
        },
    ),
    LigplaatsenCsv(
        'cache/Ligplaatsen met woonfunctie status-1 2022-06-23.csv',
        {
            'statusCode': '1',
            'gebruiksdoel.omschrijving[in]': 'woonfunctie,Woonfunctie',
        },
    ),
    StandplaatsenCsv(
        'cache/Standplaatsen met woonfunctie status-1 2022-06-23.csv',
        {
            'statusCode': '1',
            'gebruiksdoel.omschrijving[in]': 'woonfunctie,Woonfunctie',
        },
    ),
)

bedrijven = Stacked(
    VerblijfsobjectenCsv(
        'cache/Verblijfsobjecten zonder woonfunctie status-347 2022-06-23.csv',
        {
            'statusCode[in]': '3,4,7',
            'gebruiksdoel.omschrijving[not]': 'woonfunctie',
        },
    ),
    # @FIXME: de query krijgt nu wel gebruiksdoel=woonfunctie (case sensitive).
    #         maar het zijn er gelukkig maar 6 of 8 of zo.
    LigplaatsenCsv(
        'cache/Ligplaatsen zonder woonfunctie status-1 2022-06-23.csv',
        {
            'statusCode': '1',
            'gebruiksdoel.omschrijving[not]': 'Woonfunctie',
        },
    ),
    StandplaatsenCsv(
        'cache/Standplaatsen zonder woonfunctie status-1 2022-06-23.csv',
        {
            'statusCode': '1',
            'gebruiksdoel.omschrijving[not]': 'Woonfunctie',
        },
    ),
)


def format_float(x: float, n_decimals: int) -> str:
    return str(round(float(x), n_decimals)).replace('.', ',')


def main(file_out: str = FILE_OUT,
         max_afstand: int = MAX_AFSTAND_TOT_CONTAINER,
         afstand_cluster_clusters: float = MIN_AFSTAND_TUSSEN_CLUSTERS,
         ) -> None:
    """
    Maakt een CSV met de volgende kolommen
    Container ID nummer, Type, Volume, Cluster ID, Aantal huishoudens, Aantal bedrijven
    :return:
    """
    hh = np.array([h.geometrie for h in huishoudens])
    bb = np.array([b.geometrie for b in bedrijven])
    cc = np.array([c.geometrie for c in containers])

    kaart, el = amsterdam(knip_lengte=3.0)
    kaart = KDGraaf.van_graaf(kaart)

    tot_container = dichtstbijzijnde_doel(kaart, cc)
    hh_cc = kortste_afstand(kaart, hh, tot_container)
    hh_cc = herverdeel_hemelsbreed(hh_cc, hh, cc, afstand_cluster_clusters)
    bb_cc = kortste_afstand(kaart, bb, tot_container)
    bb_cc = herverdeel_hemelsbreed(bb_cc, bb, cc, afstand_cluster_clusters)
    bins = np.arange(cc.shape[0] + 1)
    aantal_hh, _ = np.histogram(hh_cc[1][hh_cc[0] <= max_afstand], bins=bins)
    aantal_bb, _ = np.histogram(bb_cc[1][bb_cc[0] <= max_afstand], bins=bins)

    volume = {t.id: format_float(t.volumeM3, 2) for t in containertypes}
    bovengronds = {t.id: t.containertypeContainerType for t in containertypes}
    pers = {t.id: 'WAAR' if t.containertypeCompressieContainerInd
            or 'pers' in t.naam.lower() else 'ONWAAR'
            for t in containertypes}

    with open(file_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"')
        writer.writerow(('Container ID Nummer', 'Containertype', 'Volume',
                         'Pers', 'Bovengronds', 'Cluster ID',
                         'Aantal Huishoudens', 'Aantal Bedrijven'))
        writer.writerows(
            (c.idNummer, c.typeId, volume[c.typeId], pers[c.typeId],
             bovengronds[c.typeId], c.clusterId, nh, nb)
            for c, nh, nb in zip(containers, aantal_hh, aantal_bb)
        )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    main()
