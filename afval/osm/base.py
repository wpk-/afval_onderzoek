import logging
from os.path import isfile
from typing import Callable, Generic, Optional, Type

from orjson import orjson
from requests import Session, HTTPError

from .models import JSON, Model

logger = logging.getLogger(__name__)


class JsonEndpoint(Generic[Model]):
    model: Type[Model]
    base_query: str
    url: str = 'https://www.overpass-api.de/api/interpreter'

    def __init__(self, cache_filename: str,
                 bounding_box: tuple[float, float, float, float],
                 query_filter: str = '', timeout: int = 20,
                 session: Optional[Session] = None) -> None:
        self.cache_filename = cache_filename
        self.bounding_box = bounding_box
        self.query_filter = query_filter
        self.timeout = timeout
        self.session = session or Session()

    @property
    def query(self) -> str:
        """De Overpass query."""
        query = f'{self.base_query}{self.query_filter}{self.bounding_box}'
        return f'[out:json][timeout:{self.timeout}];({query});out;'

    def download(self) -> None:
        """Downloadt de JSON naar cache.
        """
        data = {'data': self.query}

        logger.info(f'Fetch {self.url!r} met data {data!r}.')
        res = self.session.post(self.url, data=data, timeout=self.timeout)

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

        with open(self.cache_filename, 'wb') as fd:
            for chunk in res.iter_content():
                fd.write(chunk)

    def json_parser(self) -> Callable[[JSON], Model]:
        """Geeft een functie die JSON parset naar het model.
        """
        raise NotImplementedError('Subclasses must implement the JSON parser.')

    def read(self) -> Model:
        """Leest het model van cache. (Downloadt als cache ontbreekt.)
        """
        parse = self.json_parser()

        if not isfile(self.cache_filename):
            self.download()

        with open(self.cache_filename, 'rb') as f:
            json = orjson.loads(f.read())

        if 'remark' in json:
            logger.warning(f'Server remark: {json["remark"]!r}')

        return parse(json)
