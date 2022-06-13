"""
Gebruik als volgt:

    from afval.osm import WandelwegenJson

    wegen = WandelwegenJson(
        'cache/wandelwegen.json',
        bounding_box=(lat0, lon0, lat1, lon1),
        timeout=10,
    )

    kaart = wegen.read()    # kaart is een Graaf.
"""
from .endpoints import *
