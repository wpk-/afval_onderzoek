"""
Completeert het reinigingsrecht bestand met cluster ID's voor de fracties
glas, papier en restafval, inclusief loopafstand.

Stappen te doorlopen:
1. Lees postcode + huisnummer uit het reinigingsrecht.
2. Koppel aan een BAG-nummeraanduiding.
3. Koppel aan verblijfsobject/ligplaats/standplaats.
4. Vind dichtstbijzijnde container.

Extra kolommen:
X, Y,
Cluster Glas, Afstand Glas,
Cluster Papier, Afstand Papier,
Cluster Rest, Afstand Rest
"""
import csv
from bisect import bisect_left
from operator import itemgetter
from time import time

import numpy as np
import pylightxl as xl
from aapi.csv import (
    ContainersCsv, LigplaatsenCsv, NummeraanduidingenCsv, StandplaatsenCsv,
    VerblijfsobjectenCsv
)

from afval.graaf import (
    Graaf, KDGraaf, knip, kortste_afstand, herverdeel_hemelsbreed
)
from afval.osm import amsterdam
from demo.adressen_per_container import Stacked, format_float

FILE_IN = 'in/2022-03-28 Reinigingsrecht week 13.xlsx'
FILE_OUT = 'out/reinigingsrecht_met_clusters.csv'

# Tussen clusters met een kleinere onderlinge afstand worden huishoudens
# herveredeld op basis van hemelsbrede (Euclidische) afstand.
MIN_AFSTAND_TUSSEN_CLUSTERS = 25


class Tikker:
    def __init__(self, cumulatief: bool = False):
        self.t0 = time()
        self.cumulatief = cumulatief

    def __call__(self) -> float:
        t1 = time()
        dt = t1 - self.t0
        if not self.cumulatief:
            self.t0 = t1
        return dt


objecten = Stacked(
    VerblijfsobjectenCsv(
        'cache/Verblijfsobjecten met status-347 2022-06-23.csv',
        {
            'statusCode[in]': '3,4,7',
        },
    ),
    LigplaatsenCsv(
        'cache/Ligplaatsen met status-1 2022-06-23.csv',
        {
            'statusCode': '1',
        },
    ),
    StandplaatsenCsv(
        'cache/Standplaatsen met status-1 2022-06-23.csv',
        {
            'statusCode': '1',
        },
    ),
)

huisnummers = NummeraanduidingenCsv(
    'cache/Nummeraanduidingen 2022-06-23.csv',
    {
        # 'statusCode': '1',
    },
)


def huisnummer_sort(hn: tuple[str, int]) -> tuple[str, int, int]:
    postcode, nummer = hn
    return postcode, nummer % 2, nummer


def goede_buur(key: tuple[str, int], keys: list[tuple[str, int]]
               ) -> tuple[str, int] | None:
    ix = bisect_left(keys, huisnummer_sort(key), key=huisnummer_sort)
    if keys[ix][0] == key[0]:
        return keys[ix]
    elif keys[ix + 1][0] == key[0]:
        return keys[ix + 1]


def main(file_in: str = FILE_IN, file_out: str = FILE_OUT) -> None:
    tik = Tikker()
    nan = (np.nan, np.nan)
    fracties = ('Glas', 'Papier', 'Rest')

    print('Lees nummeraanduidingen...')
    geometrie = {o.id.split('.')[0]: o.geometrie
                 for o in objecten
                 if o.geometrie}
    nummer_geo = {(n.postcode, n.huisnummer or 0): geometrie.get(
                  n.adresseertVerblijfsobjectId or
                  n.adresseertLigplaatsId or
                  n.adresseertStandplaatsId, nan)
                  for n in huisnummers
                  if n.adresseertVerblijfsobjectId
                  or n.adresseertLigplaatsId
                  or n.adresseertStandplaatsId}
    nummers = sorted(nummer_geo.keys(), key=huisnummer_sort)

    print(f'{tik():.2f} sec.\nLees reinigingsrecht...')
    db = xl.readxl(fn=file_in, ws=('Inzicht',))
    sheet = db.ws(ws='Inzicht')
    assert sheet.address('N1') == 'HUISNR'
    assert sheet.address('R1') == 'POSTK_N'
    assert sheet.address('S1') == 'POSTK_A'

    rows = sheet.rows
    header = next(rows) + ['X', 'Y'] + [f'{f} {t}' for f in fracties
                                        for t in ('Cluster ID', 'Afstand')]
    rows = [r + ((2 + 2 * len(fracties)) * [None]) for r in rows]
    del sheet
    del db

    print(f'{tik():.2f} sec.\n'
          f'Match reinigingsrecht op nummeraanduidingen -> geo...')
    for r in rows:
        postcode = f'{r[17]}{r[18]}'
        huisnummer = int(r[13] or 0)
        index = goede_buur((postcode, huisnummer), nummers)
        x, y = nummer_geo.get(index, (np.nan, np.nan))
        r[27] = x
        r[28] = y
    geo = np.array(list(map(itemgetter(27, 28), rows)))
    heeft_geo = np.flatnonzero(np.logical_not(np.isnan(geo[:, 0])))
    # rij = np.flatnonzero(np.logical_not(np.isnan(geo[:, 0])))
    # geo = geo[rij]

    kaart, el = amsterdam(knip_lengte=3.0)
    kaart = KDGraaf.van_graaf(kaart)

    for k, fractie in enumerate(('Glas', 'Papier', 'Rest')):
        containers = ContainersCsv(
            f'cache/Containers {fractie.lower()} 2022-06-23.csv',
            {
                'fractieOmschrijving': fractie,
                'status': 1,
                'geometrie[isnull]': False,
            },
        )
        cc = np.array([c.geometrie for c in containers])
        cluster = [c.clusterId for c in containers]

        rr_cc = kortste_afstand(kaart, geo[heeft_geo], cc)
        rr_cc = herverdeel_hemelsbreed(rr_cc, geo[heeft_geo], cc,
                                       MIN_AFSTAND_TUSSEN_CLUSTERS)

        for r, d, i in zip(heeft_geo.tolist(), rr_cc[0].tolist(), rr_cc[1].tolist()):
            rows[r][29 + 2 * k] = cluster[i]
            rows[r][30 + 2 * k] = format_float(d, 1)

    # xl.writexl(db=db, fn=file_out)
    with open(file_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';', quotechar='"')
        w.writerow(header)
        w.writerows(rows)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    main()
