"""
Gebruik als volgt:

    from afval.osm import wandelwegen

    kaart = wandelwegen(bbox=(lat0, lon0, lat1, lon1), timeout=10)
    # --> kaart is een `afval.graph.Graaf`.
"""
import logging
from collections import defaultdict
from itertools import count, pairwise

import numpy as np
from orjson import orjson
from requests import HTTPError, Session

from afval.graaf import Graaf
from afval.session import make_session
from afval.types import JSON

logger = logging.getLogger(__name__)


def json(query: str, timeout: int = 10, session: Session = None) -> JSON:
    """Downloadt de JSON-response van de Overpass OSM API.

    :param query: De Overpass query.
    :param timeout: Aantal seconden voordat de request wordt afgekapt.
        Deze waarde wordt ook door de Overpass server gebruikt om queries
        te prioriteren. Een lagere timeout krijgt een hogere prioriteit.
        Maar als het antwoord niet binnen de tijd berekend kan worden
        faalt de hele query. Standaard is 10.
    :param session: Het sessie-object voor het HTTP-verzoek. Dit is een
        python requests module Session.
        Standaard is `afval.session.make_session()`
    :return: De JSON-respons van de Overpass server.
    """
    session = session or make_session()
    timeout = timeout or 10

    url = 'https://www.overpass-api.de/api/interpreter'
    data = {'data': query}
    logging.debug(query)
    res = session.post(url, data=data, timeout=timeout)

    try:
        res.raise_for_status()
    except HTTPError as err:
        if res.status_code in (429, 504):
            # 429 Too Many Requests
            # 504 Gateway Timeout
            logger.warning(
                f'Server responded with status {res.status_code}.'
                f' You can retry in a bit.')
            logger.debug(res.content)
        else:
            logger.error(
                f'Server responsed with status {res.status_code}.'
                f' Please interpret and take action accordingly.')
        raise err

    return orjson.loads(res.content)


def json_graaf(data: JSON) -> Graaf:
    """Zet Overpass JSON om in een graaf met punten en lijnen.

    :param data: De JSON.
    :return: De graaf.
    """
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
    punt_index = defaultdict(count().__next__)

    lijnen = np.array(
        [
            (punt_index[a], punt_index[b])
            for node in data['elements']
            if node['type'] == 'way'
            for a, b in pairwise(node['nodes'])
        ],
        dtype=int
    )

    punt_index = dict(punt_index)

    punten = np.array(
        [
            (punt_index[node['id']], node['lat'], node['lon'])
            for node in data['elements']
            if node['type'] == 'node'
        ],
        dtype=float
    )
    punten = punten[np.argsort(punten[:, 0]), 1:]

    assert punten.shape[0] == max(punt_index.values()) + 1

    return Graaf(punten, lijnen)


def graaf(query: str, *args, **kwargs) -> Graaf:
    """Stuurt de query naar Overpass en parset de response tot graaf.

    Dit is een wrapper om json() die de JSON als Graaf teruggeeft.

    :param query: De Overpass query om naar de server te sturen als
        rekenopdracht.
    :param args: Een timeout en sessie object zijn optioneel.
    :param kwargs: Een timeout en sessie object (session) zijn optioneel.
    :return:
    """
    data = json(query, *args, **kwargs)
    return json_graaf(data)


def wandelwegen(bbox: tuple[float, float, float, float], timeout: int,
                *args, **kwargs) -> Graaf:
    """Haalt alle wandelwegen op als een graaf.

    Dit is een wrapper om graaf() met een voorgedefinieerde query voor
    alle wandelwegen.

    :param bbox: Rechthoek die het gebied markeert, als volgt:
        (lat_min, long_min, lat_max, long_max).
    :param timeout: Maximale (reken-/request-)tijd voor het verzoek.
    :param args: Optionele extra argumenten voor graaf().
    :param kwargs: Optionele extra keyword argumenten voor graaf().
    :return: Een graaf van alle wandelwegen in het gemarkeerde gebied.
    """
    query = wandelwegen_query(bbox, timeout)
    return graaf(query, timeout, *args, **kwargs)


def wandelwegen_query(bbox: tuple[float, float, float, float], timeout: int
                      ) -> str:
    """Bouwt de Overpass query voor wandelwegen in het gebied.

    :param bbox: Rechthoek die het gebied markeert, als volgt:
        (lat_min, long_min, lat_max, long_max).
    :param timeout: Claim op rekentijd bij de Overpass server. Hoe groter
        de waarde, hoe lager de prioriteit. Maar bij een te lage waarde,
        als de benodigde rekentijd de timeout overschrijdt, zal de query
        falen.
    :return:
    """
    base_query = (
        'way["highway"]'
        '["highway"!~"motor|proposed|abandoned|platform|raceway"]'
        '["foot"!~"no"]'
        '["pedestrians"!~"no"]'
    )
    return (f'[out:json][timeout:{timeout}];'
            f'{base_query}{bbox};(._;>;);'
            f'out skel;')
