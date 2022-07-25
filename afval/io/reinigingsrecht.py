from typing import NamedTuple

import pylightxl as xl


class Reinigingsrecht(NamedTuple):
    jjheffing: str
    heffing_sys: str
    ddingang_obj: str
    ddeinde_obj: str
    ddingang_rel: str
    ddeinde_rel: str
    reintarcode: str
    contractletter: str
    uitsluitcode: str       # @TODO
    vrijstelcode: str
    belobjnr: str
    wozbelobjnr: str
    straat: str
    huisnr: int
    huislt: str
    toev: str
    aand: str
    postk_n: str
    postk_a: str
    woonplaats: str
    stadsdeel: str
    hoofdcode_gebr: str
    hoofd_oms: str
    subjectnr: str
    voorl: str
    voorv: str
    naam: str

    @property
    def postcode(self) -> str:
        return f'{self.postk_n}{self.postk_a}'


def lees(filename: str) -> list[Reinigingsrecht]:
    db = xl.readxl(fn=filename, ws=('Inzicht',))
    sheet = db.ws(ws='Inzicht')

    rows = sheet.rows
    header = next(rows)

    assert all(a.lower() == b
               for a, b in zip(header, Reinigingsrecht.__annotations__))

    return [Reinigingsrecht(*row) for row in rows]
