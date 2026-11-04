"""HTTP client for the Sejm API."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sejm.gov.pl/sejm"
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


class SejmClient:
    """GET helper that retries rate limits, server errors, and dropped connections."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            headers={"User-Agent": "data-mining-project/0.1"},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SejmClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, path: str, *, accept: str = "application/json") -> httpx.Response | None:
        """Return the response, or None when the resource does not exist."""
        delay = 1.0
        response: httpx.Response | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.get(path, headers={"Accept": accept})
            except httpx.TransportError as exc:
                if attempt == self._max_retries:
                    raise
                logger.warning("Request failed (%s): %s", path, exc)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue

            if response.status_code == 404:
                logger.warning("Not found: %s", path)
                return None

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self._max_retries:
                    response.raise_for_status()
                wait = _retry_delay(response, delay)
                logger.warning(
                    "HTTP %s for %s, retrying in %.1fs",
                    response.status_code,
                    path,
                    wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            response.raise_for_status()
            return response

        if response is not None:
            response.raise_for_status()
        raise RuntimeError(f"Request failed: {path}")

    def get_json(self, path: str) -> Any | None:
        response = self.get(path, accept="application/json")
        if response is None:
            return None
        return response.json()

    def get_text(self, path: str) -> str | None:
        response = self.get(path, accept="text/html")
        if response is None:
            return None
        response.encoding = "utf-8"
        return response.text


def _retry_delay(response: httpx.Response, fallback: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return fallback
    try:
        return max(fallback, float(retry_after))
    except ValueError:
        return fallback
