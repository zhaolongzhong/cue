from typing import List, Union, Optional

from pydantic import BaseModel

from .author import Author


class AgentTransfer(BaseModel):
    to_agent_id: str
    context: Optional[Union[str, List]] = None
    message: Optional[str] = None


class ToolResponseWrapper(BaseModel):
    author: Optional[Author] = None
    tool_messages: Optional[List[dict]] = None
    tool_result_message: Optional[dict] = None
    agent_transfer: Optional[AgentTransfer] = None
    base64_images: list = None
