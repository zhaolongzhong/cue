from typing import Literal, ClassVar
from pathlib import Path

from .base import BaseTool, ToolError
from ..schemas.tool_response_wrapper import AgentTransfer

Command = Literal["transfer"]


class CoordinateTool(BaseTool):
    """
    An agent coordination tool that handle agent coordination like transfer task
    """

    name: ClassVar[Literal["coordinate"]] = "coordinate"

    _file_history: dict[Path, list[str]]

    def __init__(self):
        self._function = self.coordinate
        super().__init__()

    async def __call__(
        self,
        *,
        command: Command,
        to_agent_id: str,
        message: str,
        **kwargs,
    ):
        return await self.coordinate(
            command=command,
            to_agent_id=to_agent_id,
            message=message,
            **kwargs,
        )

    async def coordinate(
        self,
        *,
        command: Command,
        to_agent_id: str,
        message: str,
        **kwargs,
    ):
        if command == "transfer":
            return self.transfer(
                to_agent_id=to_agent_id,
                message=message,
                **kwargs,
            )
        else:
            return ToolError(f"Unsupport command '{command}', supported command are {Command}")

    def transfer(
        self,
        *,
        to_agent_id: str,
        message: str,
        **kwargs,
    ) -> AgentTransfer:
        return AgentTransfer(
            to_agent_id=to_agent_id,
            message=message,
        )
