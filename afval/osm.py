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

from afval.graaf import Graaf, knip
from afval.projectie import projecteer_epsg, rijksdriehoek
from afval.session import make_session
from afval.types import JSON

logger = logging.getLogger(__name__)

bbox_amsterdam_gps = (52.2777, 4.7280, 52.4314, 5.1075)


def json(query: str, timeout: int = None, session: Session = None) -> JSON:
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
    :param kwargs: Optionele extra keyword argumenten voor
        wandelwegen_query() en graaf(). Zie beide functies voor
        documentatie.
    :return: Een graaf van alle wandelwegen in het gemarkeerde gebied.
    """
    query_kwargs = {
        k: kwargs.pop(k)
        for k in ('fietspaden', 'voetpaden', 'voetgangers', 'trappen')
        if k in kwargs
    }
    query = wandelwegen_query(bbox, timeout, **query_kwargs)
    return graaf(query, *args, **kwargs)


def wandelwegen_query(bbox: tuple[float, float, float, float], timeout: int,
                      fietspaden: bool = False, voetpaden: bool = False,
                      voetgangers: bool = False, trappen: bool = False) -> str:
    """Bouwt de Overpass query voor wandelwegen in het gebied.

    :param bbox: Rechthoek die het gebied markeert, als volgt:
        (lat_min, long_min, lat_max, long_max).
    :param timeout: Claim op rekentijd bij de Overpass server. Hoe groter
        de waarde, hoe lager de prioriteit. Maar bij een te lage waarde,
        als de benodigde rekentijd de timeout overschrijdt, zal de query
        falen.
    :param fietspaden: De graaf bevat ook fietspaden. Standaard is False.
    :param voetpaden: De graaf bevat ook voetpaden. Standaard is False.
    :param voetgangers: De graaf bevat ook voetgangers wegdelen.
        Standaard is False.
    :param trappen: De graaf bevat ook trappen. Standaard is False.
    :return: Een graaf van alle wegdelen in het geselecteerde gebied.
    """
    base_query = (
        'way["highway"]'
        '["highway"!~"motor|railway|busway|platform|service'
        f'{"" if fietspaden else "|cycleway"}{"" if voetpaden else "|footway"}'
        f'{"" if voetgangers else "|pedestrian"}{"" if trappen else "|steps"}"'
        ']'
    )
    return (f'[out:json][timeout:{timeout}];'
            f'{base_query}{bbox};(._;>;);'
            f'out skel;')


def amsterdam(knip_lengte: float = None, **wegen) -> tuple[Graaf, np.ndarray]:
    """Downloadt de basiskaart van Amsterdam.

    :param knip_lengte: De maximale lengte van een lijn in de graaf. Deze
        parameter is optioneel. Alleen als een maximale lengte wordt
        opgegeven worden langere lijnen opgedeeld.
    :param wegen: Optionele extra argumenten voor welke wegen opgevraagd
        worden. Zie de documentatie van wandelwegen_query() voor uitleg.
    :return: Twee waardes: Een graaf met de wegen van Amsterdam en een
        numpy array met de lengtes van alle lijnen in de graaf.
    """
    timeout = 15

    logging.debug('Download wegen van OSM...')
    kaart = wandelwegen(bbox_amsterdam_gps, timeout=timeout, **wegen)
    kaart = Graaf(projecteer_epsg(kaart.punten, rijksdriehoek), kaart.lijnen)
    logger.debug(f'{kaart.aantal_punten} punten, '
                 f'{kaart.aantal_lijnen} lijnen.')

    if knip_lengte:
        logger.debug('Prepareer voor berekeningen...')
        return knip(kaart, max_lengte=knip_lengte)
    else:
        return kaart, kaart.lijn_lengtes()
