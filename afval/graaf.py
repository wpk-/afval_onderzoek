from heapq import heapify, heappop, heappush
from typing import TypeVar

import numpy as np
from scipy.spatial import KDTree

from afval.types import AfstandIndex


class Graaf:
    """Een graaf met punten en lijnen.
    """
    def __init__(self, punten: np.ndarray, lijnen: np.ndarray) -> None:
        self.punten = punten
        self.lijnen = lijnen

    @property
    def a(self) -> np.ndarray:
        """De vertrekpunten van alle lijnen."""
        return self.punten[self.lijnen[:, 0]]

    @property
    def b(self) -> np.ndarray:
        """De doelpunten van alle lijnen."""
        return self.punten[self.lijnen[:, 1]]

    @property
    def aantal_lijnen(self) -> int:
        """Het aantal lijnen in de graaf."""
        return self.lijnen.shape[0]

    @property
    def aantal_punten(self) -> int:
        """Het aantal punten in de graaf."""
        return self.punten.shape[0]

    def lijn_lengtes(self) -> np.ndarray:
        """Berekent de lengte van elke lijn.

        De huidige berekening is op basis van de Euclidische afstand
        tussen de twee eindpunten. Subklassen kunnen andere maten
        gebruiken, zoals de haversine formule.
        """
        ab = self.b - self.a
        return np.sqrt(np.einsum('ij,ij->i', ab, ab))


class KDGraaf(Graaf):
    """A graph that stores its vertices in a KDTree.
    """
    def __init__(self, punten: np.ndarray, lijnen: np.ndarray) -> None:
        self.kd: KDTree = KDTree(np.empty((0, 2)))
        super().__init__(punten, lijnen)

    @property
    def aantal_punten(self) -> int:
        return self.kd.n

    @property
    def punten(self) -> np.ndarray:
        return self.kd.data

    @punten.setter
    def punten(self, punten: np.ndarray) -> None:
        self.kd = KDTree(punten)

    @classmethod
    def van_graaf(cls, g: Graaf) -> 'KDGraaf':
        return cls(g.punten, g.lijnen)


G = TypeVar('G', Graaf, KDGraaf)


def dichtstbijzijnde_doel(graaf: G, doelen: np.ndarray,
                          lijn_lengtes: np.ndarray | None = None,
                          doel_tot_graaf: AfstandIndex | None = None,
                          ) -> AfstandIndex:
    """Berekent voor elk punt in de graaf het dichtstbijzijnde doel.

    De doelen kunnen buiten de graaf liggen.

    Stel graaf G heeft punten A: (0, 0), B: (0, 5) en C: (2, 0) en lijnen
    AB en AC, maar niet BC. En stel dat we een doel hebben op D: (2, 2).
    Dan:
    - is de kortste afstand van doel tot de graaf |CD| = 2.
    - is de afstand van A tot het doel: |AD| = |AC| + |CD| = 4.
    - is de afstand van B tot het doel: |BD| = |AB| + |AC| + |CD| = 9.

    Maar voor een enkel doel op E: (2, 3) geldt:
    - de kortste afstand van doel tot de graaf is: |BE| = sqrt(8).
    - de afstand van A tot het doel: |AE| = |AB| + |BE| = 5 + sqrt(8).
    - de afstand van C tot het doel: |CE| = |AC| + |AB| + |BE|.

    Als we de twee doelen samen nemen geldt voor elk punt op de graaf het
    minimum van de afstanden tot alle doelen. Dus met doelen D en E:
    - de kortste afstand voor B is |BE| = sqrt(8).
    - de kortste afstand voor C is |CD| = 2.
    - de kortste afstand voor A is |AD| = 4.

    :param graaf: Graaf met n punten en m lijnen.
    :param doelen: Numpy k x 2 array van doel coördinaten. Deze mogen
        buiten de graaf liggen. De afstand van elk doel tot het
        dichtstbijzijnde punt op de graaf wordt Euclidisch berekend of
        kan opgegeven worden in `doel_tot_graaf`.
    :param lijn_lengtes: Numpy array met m vooraf uitgerekende
        lijnlengtes. Dit is een optioneel argument. Standaard worden de
        lengtes van de lijnen in de graaf uitgerekend.
    :param doel_tot_graaf: Dit is een AfstandIndex tuple met
        array-lengtes k. De eerste array geeft per doel de kortste
        afstand tot de graaf. De tweede array geeft per doel de index van
        het dichtstbijzijnde punt op de graaf. Optioneel. Standaard wordt
        dit (op basis van Euclidische afstand) uitgerekend.
    :return: Voor elk punt in de graaf de kortste afstand tot een doel.
        Dit is een AfstandIndex tuple met array-lengtes n. De eerste
        array geeft de afstand tot het dichtstbijzijnde doel. De tweede
        array geeft de index van dat doel (integer waardes van 0 tot k).
    """
    doelen = np.atleast_2d(doelen)

    try:
        d, ix = doel_tot_graaf
    except TypeError:
        try:
            d, ix = graaf.kd.query(doelen)
        except AttributeError:
            graaf = KDGraaf.van_graaf(graaf)
            d, ix = graaf.kd.query(doelen)

    if lijn_lengtes is None:
        lijn_lengtes = graaf.lijn_lengtes()

    n = graaf.aantal_punten
    z = np.ones(n, dtype=float) * np.inf    # Kortste afstand tot doel.
    r = -np.ones(n, dtype=int)              # Index van dat doel.

    a = {v: [] for v in range(n)}           # Acties: punt -> volgende.

    for (ve1, ve2), l in zip(graaf.lijnen, lijn_lengtes):
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
            assert np.isinf(z[vi]), 'Algorithmic error!'
            z[vi] = zi
            r[vi] = ri
        # logging.debug(f'{vi}: {zi:.2f}')
        for l, vj in a[vi]:
            heappush(front, (zi + l, vj, ri))

    return z, r


def knip(graaf: G, max_lengte: float) -> tuple[G, np.ndarray]:
    """Knipt de lijnen van de graaf in delen van maximaal `max_lengte`.

    :param graaf: De graaf. Dit kan een `Graaf` of een `KDGraaf` zijn.
    :param max_lengte: Maximale lengte van een lijnstuk. Voor langere
        lijnen worden een of meerdere punten aan de graaf toegevoegd en
        het lijnstuk vervangen door kortere lijnstukken. De nieuwe punten
        worden lineair tussen de oorspronkelijke punten geplaatst.
    :return: Een tuple van twee waardes: 1. De nieuwe graaf. Deze is van
        dezelfde klasse als de input graaf. 2. De lengtes van alle lijnen
        in de nieuwe graaf. Dit is handig om te hebben aangezien de graaf
        dat niet opslaat.
    """
    lijnen = graaf.lijnen
    a = graaf.a
    b = graaf.b
    lengtes = np.sqrt(np.einsum('ij,ij->i', b - a, b - a))

    extra_punten = []
    extra_lijnen = []
    extra_lengtes = []
    kort = lengtes <= max_lengte

    # l = 7: 7/2 = 3+ = 4 parts
    # l = 8: 8/2 = 4  = 4 parts
    ix = np.flatnonzero(~kort)
    nv = graaf.aantal_punten

    for ei in ix:
        ne_i = np.ceil(lengtes[ei] / max_lengte).astype(int)
        nv_i = ne_i - 1

        interp = np.linspace(a[ei], b[ei], ne_i, endpoint=False)
        extra_punten.append(interp[1:])

        extra_lijnen.append(
            [
                (lijnen[ei][0], nv),
                (nv + nv_i - 1, lijnen[ei][1])
            ] + [
                (nv + i, nv + i + 1)
                for i in range(ne_i - 2)
            ]
        )

        extra_lengtes.append([lengtes[ei] / ne_i] * ne_i)

        nv += nv_i

    graaf = graaf.__class__(np.vstack([graaf.punten] + extra_punten),
                            np.vstack([lijnen[kort]] + extra_lijnen))
    lengtes = np.hstack([lengtes[kort]] + extra_lengtes)

    return graaf, lengtes


def kortste_afstand(graaf: G, bronnen: np.ndarray,
                    doelen: AfstandIndex | np.ndarray,
                    lijn_lengtes: np.ndarray | None = None,
                    bron_tot_graaf: AfstandIndex | None = None,
                    doel_tot_graaf: AfstandIndex | None = None,
                    ) -> AfstandIndex:
    """Vindt voor elke bron het dichtstbijzijnde doel over de graaf.

    Zie ook uitleg in de documentatie van `dichtstbijzijnde_doel()`.
    Nu worden de bronnen ook expliciet genoemd. Net als de doelen mogen
    de bronnen ook buiten de graaf liggen en worden zij eerst op de graaf
    geprojecteerd.

    :param graaf: Graaf met n punten en m lijnen.
    :param bronnen: Numpy p x 2 array van bron coördinaten. Deze mogen
        buiten de graaf liggen. De afstand van elke bron tot de graaf is
        de afstand tot het dichtstbijzijnde punt. Dit wordt Euclidisch
        berekend of kan opgegeven worden in `bron_tot_graaf`.
    :param doelen: Numpy k x 2 array van doel coördinaten. Deze mogen
        buiten de graaf liggen. De afstand van elk doel tot het
        dichtstbijzijnde punt op de graaf wordt Euclidisch berekend of
        kan opgegeven worden in `doel_tot_graaf`.
    :param lijn_lengtes: Numpy array met m vooraf uitgerekende
        lijnlengtes. Dit is een optioneel argument. Standaard worden de
        lengtes van de lijnen in de graaf uitgerekend.
    :param bron_tot_graaf: Dit is een AfstandIndex tuple met
        array-lengtes p. De eerste array geeft per bron de kortste
        afstand tot de graaf. De tweede array geeft per bron de index van
        het dichtstbijzijnde punt op de graaf. Optioneel. Standaard wordt
        dit (op basis van Euclidische afstand) uitgerekend.
    :param doel_tot_graaf: Dit is een AfstandIndex tuple met
        array-lengtes k. De eerste array geeft per doel de kortste
        afstand tot de graaf. De tweede array geeft per doel de index van
        het dichtstbijzijnde punt op de graaf. Optioneel. Standaard wordt
        dit (op basis van Euclidische afstand) uitgerekend.
    :return: Voor elke bron de kortste afstand tot een doel. Dit is een
        AfstandIndex tuple met array-lengtes p. De arrays geven voor elke
        bron de afstand tot het dichtstbijzijnde doel en de index van dat
        doel (een integer van 0 tot k).
    """
    try:
        afstand, index = doelen
    except TypeError:
        afstand, index = dichtstbijzijnde_doel(graaf, doelen, lijn_lengtes,
                                               doel_tot_graaf)

    bronnen = np.atleast_2d(bronnen)

    try:
        d, ix = bron_tot_graaf
    except TypeError:
        try:
            d, ix = graaf.kd.query(bronnen)
        except AttributeError:
            graaf = KDGraaf.van_graaf(graaf)
            d, ix = graaf.kd.query(bronnen)

    return d + afstand[ix], index[ix]
