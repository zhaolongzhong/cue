import logging
from enum import Enum
from typing import Dict, List, Union, Optional

from .base import BaseTool
from .edit import EditTool
from .drive import GoogleDriveTool
from .email import EmailTool
from .browse import BrowseTool
from .memory import MemoryTool
from .bash_tool import BashTool
from .coordinate import CoordinateTool
from .read_image import ReadImageTool
from .run_script import PythonRunner
from .utils.function_utils import get_definition_by_model
from .utils.function_to_json import function_to_json
from ..memory.memory_service_client import MemoryServiceClient

logger = logging.getLogger(__name__)


class Tool(Enum):
    Bash = BashTool.name
    Edit = EditTool.name
    Python = PythonRunner.name
    Browse = BrowseTool.name
    Email = EmailTool.name
    Drive = GoogleDriveTool.name
    Image = ReadImageTool.name
    Memory = MemoryTool.name
    Coordinate = CoordinateTool.name


class ToolManager:
    def __init__(self, memory_service: Optional[MemoryServiceClient] = None):
        self.tools: Dict[str, BaseTool] = {
            Tool.Bash.value: BashTool(),
            Tool.Edit.value: EditTool(),
            Tool.Python.value: PythonRunner(),
            Tool.Browse.value: BrowseTool(),
            Tool.Email.value: EmailTool(),
            Tool.Drive.value: GoogleDriveTool(),
            Tool.Drive.value: GoogleDriveTool(),
            Tool.Image.value: ReadImageTool(),
            Tool.Memory.value: MemoryTool(memory_service),
            Tool.Coordinate.value: CoordinateTool(),
        }
        self._definition_cache: Dict[str, dict] = {}

    def _get_cache_key(self, tool_id: str, model: str) -> str:
        """Generate a unique cache key for a tool definition."""
        return f"{tool_id}_{model}"

    def _get_cached_definition(self, tool_id: str, model: str) -> Optional[dict]:
        """Retrieve cached tool definition if it exists."""
        return self._definition_cache.get(self._get_cache_key(tool_id, model))

    def _cache_definition(self, tool_id: str, model: str, definition: dict):
        """Cache a tool definition."""
        self._definition_cache[self._get_cache_key(tool_id, model)] = definition

    def get_tool_definitions(self, model: str, tools: Optional[Union[List[Tool], List[str]]] = None) -> List[dict]:
        """Iterate through the tools and gather their JSON configurations.

        Args:
            model: Original model name from original provider
            tools: Optional list of Tool enum values or tool string identifiers

        Returns:
            List of tool parameter dictionaries
        """
        if tools is None:
            tools_to_iterate = list(self.tools.keys())
        else:
            tools_to_iterate = []
            for tool in tools:
                if isinstance(tool, Tool):
                    tools_to_iterate.append(tool.value)
                elif isinstance(tool, str):
                    tools_to_iterate.append(tool)
                elif callable(tool):
                    # Convert callable to JSON definition and add to tools
                    tool_id = tool.__name__
                    cached_def = self._get_cached_definition(tool_id, model)
                    if cached_def is None:
                        definition = get_definition_by_model(function_to_json(tool), model)
                        self.tools[tool_id] = definition
                        self._cache_definition(tool_id, model, definition)
                    else:
                        self.tools[tool_id] = cached_def
                    tools_to_iterate.append(tool_id)
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")

        func_definitions = []
        for tool_id in tools_to_iterate:
            if tool_id not in self.tools:
                raise ValueError(f"Unsupported tool type: {type(tool)}, make sure to register first in ToolManager")

            cached_def = self._get_cached_definition(tool_id, model)
            if cached_def is not None:
                func_definitions.append(cached_def)
            else:
                tool = self.tools[tool_id]
                if isinstance(tool, dict):
                    definition = tool
                else:
                    definition = tool.to_json(model, use_file_definition=True)
                self._cache_definition(tool_id, model, definition)
                func_definitions.append(definition)

        return func_definitions
