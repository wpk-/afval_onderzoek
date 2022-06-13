from typing import Optional

from requests import Session

DEFAULT_HEADERS = {
    # 'User-Agent': 'Mozilla/5.0 AppleWebKit/537.36 Chrome/92.0 Safari/537.36',
}


def make_session(headers: Optional[dict[str, str]] = None) -> Session:
    session = Session()
    session.headers.update(headers or DEFAULT_HEADERS)
    return session
