afval
-----

Een Python module om alle afval-data van Amsterdam gemakkelijk te kunnen
raadplegen en verwerken.


Overzicht
---------

[graaf.py](afval/graaf.py) Een graaf bestaat uit punten en lijnen. Lijnen
kunnen geknipt worden op een maximale lengte. We kunnen ook afstanden berekenen
over de graaf. Standaard wordt de Euclidische afstand gebruikt maar het is ook
mogelijk om zelf de afstanden te berekenen en mee te geven.

[osm.py](afval/osm.py) Interface naar Open Street Maps. Raadpleeg OSM data via
de Overpass API. Voorgedefinieerde queries bestaan voor wandelpaden maar elke
query is mogelijk.

[projectie.py](afval/projectie.py) Functies om te projecteren tussen
verschillende coordinaatsystemen. Bijvoorbeeld van GPS (EPSG 4326) naar
Rijksdriehoek (EPSG 28992) of andersom. Dit is handig omdat de Amsterdamse data
vaak in RD gegeven wordt terwijl de online kaartsystemen GPS gebruiken. Ook de
andere kant op projecteren heeft voordelen, bijvoorbeeld omdat RD in meters
rekent en de Euclidische afstand tussen coördinaten dus overeen komt met de
fysieke afstand in meters.

[session.py](afval/session.py) Gemaksfuncties voor HTTP sessiebeheer in
scripts. Zo kan je eenvoudig een default sessie maken.

[types.py](afval/types.py) Algemene types zijn altijd goed om expliciet te
maken zodat duidelijk is wat voor data functies verwachten en teruggeven.
Bijvoorbeeld `JSON`.


Licentie
--------

[MIT](LICENSE.txt).
