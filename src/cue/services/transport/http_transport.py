import logging
from typing import Any, Optional, Protocol
from typing_extensions import runtime_checkable

import aiohttp
from httpx import HTTPError

logger = logging.getLogger(__name__)


@runtime_checkable
class HTTPTransport(Protocol):
    """Protocol for HTTP transport operations"""

    async def request(
        self, method: str, endpoint: str, data: Optional[dict[str, Any]] = None, params: Optional[dict[str, Any]] = None
    ) -> Any: ...


class AioHTTPTransport(HTTPTransport):
    """AIOHTTP implementation of HTTP transport"""

    def __init__(self, base_url: str, api_key: str, session: Optional[aiohttp.ClientSession] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.is_server_available = False
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": f"{self.api_key}",
        }
        self.session = session or aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def request(
        self, method: str, endpoint: str, data: Optional[dict[str, Any]] = None, params: Optional[dict[str, Any]] = None
    ) -> Any:
        if not self.is_server_available:
            logger.warning("Server is not available.")
            return
        url = f"{self.base_url}{endpoint}"
        try:
            async with getattr(self.session, method.lower())(
                url, json=data, params=params, headers=self.headers
            ) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    logger.error(f"HTTP {response.status}: {error_data}")
                    raise HTTPError(f"HTTP {response.status}: {error_data.get('detail', 'Unknown error')}")
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {str(e)}, url: {url}")
            raise ConnectionError(f"Request failed: {str(e)}")
