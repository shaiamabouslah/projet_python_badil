import time

import requests

USER_AGENT = "BadilBot/0.1 (student project, non-commercial)"


def get(url: str, delay: float = 1.0, retries: int = 2) -> requests.Response:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
            response.raise_for_status()
            time.sleep(delay)
            return response
        except requests.exceptions.RequestException as error:
            last_error = error
            time.sleep(delay * (attempt + 1))
    raise last_error
