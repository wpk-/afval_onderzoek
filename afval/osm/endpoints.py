from typing import Callable

from .base import JsonEndpoint
from .models import Graaf, JSON
from .parse import parse_graaf

__all__ = ['WegenkaartJson', 'WandelwegenJson']


class WegenkaartJson(JsonEndpoint[Graaf]):
    """Leest alle wegen als een Graaf met vertices en edges.
    """
    model = Graaf
    base_query = 'way["highway"]'

    def json_parser(self) -> Callable[[JSON], Graaf]:
        """Geeft een functie die Overpass JSON omzet in een Graaf.
        """
        return parse_graaf


class WandelwegenJson(WegenkaartJson):
    """Handige WegenkaartJson subklasse om de wandelwegen query niet te
    hoeven onthouden.

    Wegen waar volgens OSM aan gewerkt wordt, worden wel meegenomen.
    Wegen waarvoor de plannen er liggen maar die nog niet aangelegd
    worden (status proposed), worden niet meegenomen.
    """
    base_query = ('way["highway"]'
                  '["highway"!~"motor|proposed|abandoned|platform|raceway"]'
                  '["foot"!~"no"]["pedestrians"!~"no"]')
