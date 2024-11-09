# src/cue/memory/client.py
import logging
from typing import List

from ..schemas import (
    Assistant,
    AssistantCreate,
)
from .transport import ResourceClient

logger = logging.getLogger(__name__)


class AssistantClient(ResourceClient):
    """Client for assistant-related operations"""

    async def create(self, assistant: AssistantCreate) -> Assistant:
        response = await self._http.request("POST", "/assistants/", data=assistant.model_dump())
        if not response:
            return
        return Assistant(**response)

    async def get(self, assistant_id: str) -> Assistant:
        response = await self._http.request("GET", f"/assistants/{assistant_id}")
        return Assistant(**response)

    async def list(self, skip: int = 0, limit: int = 100) -> List[Assistant]:
        response = await self._http.request("GET", f"/assistants/?skip={skip}&limit={limit}")
        return [Assistant(**asst) for asst in response]

    async def delete(self, assistant_id: str) -> None:
        await self._http.request("DELETE", f"/assistants/{assistant_id}")

    async def create_default_assistant(self):
        """Create an default assistant to persist memories across multiple conversation"""
        assistant = await self.create(AssistantCreate(name="default"))
        if not assistant:
            return
        self._default_assistant_id = assistant.id