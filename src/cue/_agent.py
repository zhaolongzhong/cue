import asyncio
import json
import logging
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from .llm.llm_client import LLMClient
from .memory.memory import InMemoryStorage
from .schemas import AgentConfig, CompletionRequest, CompletionResponse, RunMetadata, SystemMessage, ToolResponseWrapper
from .schemas.anthropic import ToolResultContent, ToolResultMessage, ToolUseContent
from .schemas.chat_completion import ToolCall, ToolMessage
from .tool_manager import ToolManager

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig, agent_manager: "AgentManager"):  # type: ignore # noqa: F821
        self.id = config.id
        self.config = config
        self.tool_manager = ToolManager()
        self.memory = InMemoryStorage()
        self.client: LLMClient = LLMClient(self.config)
        self.metadata: Optional[RunMetadata] = None
        self.agent_manager = agent_manager
        self.description = self._generate_description()
        self.other_agents_info = ""

    def get_system_message(self) -> SystemMessage:
        instruction = f"{self.config.instruction} Your idenity is id: {self.config.id}, name: {self.config.name}"
        instruction += f"\n\nYou are aware of the following other agents:\n{self.other_agents_info}"
        return SystemMessage(role="system", name=self.config.name, content=instruction)

    def get_messages(self) -> List:
        """
        Retrieve the original list of messages from the Pydantic model.

        Returns:
            List: A list of message objects stored in the memory.
        """
        return self.memory.messages

    def get_message_params(self) -> List[Dict]:
        """
        Retrieve a list of message parameter dictionaries for the completion API call.

        Returns:
            List[Dict]: A list of dictionaries containing message parameters
                        required for the completion API call, based on the model ID
                        from the current configuration.
        """
        return self.memory.get_message_params(self.config.model.id)

    def _generate_description(self) -> str:
        description = ""
        if self.config.description:
            description = self.config.description
        tool_names = [tool.value for tool in self.config.tools]
        if not tool_names:
            return description
        description += f"Agent {self.config.id} is able to use these tools: {', '.join(tool_names)}"
        return description

    def _get_tool_json(self) -> Optional[List[Dict]]:
        if self.config.tools:
            return self.tool_manager.get_tool_definitions(self.config.model, self.config.tools)
        else:
            return None

    async def send_messages(
        self, messages: List[Union[BaseModel, Dict]], metadata: Optional[RunMetadata] = None
    ) -> Union[CompletionResponse, ToolCall, ToolUseContent]:
        if not self.metadata:
            self.metadata = metadata

        messages_dict = [
            msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
            for msg in messages
        ]
        # Add system message if it doesn't exist
        if messages_dict[0]["role"] != "system":
            messages_dict.insert(0, self.get_system_message().model_dump(exclude_none=True))

        completion_request = CompletionRequest(
            model=self.config.model.id,
            messages=messages_dict,
            metadata=metadata,
            tool_json=self._get_tool_json(),
        )
        response = await self.client.send_completion_request(completion_request)
        usage = response.get_usage()
        if usage:
            logger.debug(f"completion response usage: {usage.model_dump(exclude_none=True)}")
        return response

    async def process_tools_with_timeout(
        self, tool_calls: Union[List[ToolCall], List[ToolUseContent]], timeout: int = 30
    ) -> ToolResponseWrapper:
        tool_responses = []
        tasks = []

        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                tool_name = tool_call.function.name
                tool_id = tool_call.id
                args = tuple(json.loads(tool_call.function.arguments).values())
            elif isinstance(tool_call, ToolUseContent):
                tool_name = tool_call.name
                tool_id = tool_call.id
                args = tuple(tool_call.input.values())
            else:
                raise ValueError(f"Unsupported tool call type: {type(tool_call)}")

            if tool_name not in self.tool_manager.tools and tool_name not in ["call_agent", "transfer_to_agent"]:
                error_message = f"Tool '{tool_name}' not found. {self.tool_manager.tools.keys()}"
                logger.error(error_message)
                tool_responses.append(self.create_error_response(tool_id, tool_name, error_message))
                continue

            if tool_name == "transfer_to_agent":
                tool_func = self.agent_manager.transfer_to_agent
            else:
                tool_func = self.tool_manager.tools[tool_name]
            task = asyncio.create_task(self.run_tool(tool_func, *args))
            tasks.append((task, tool_id, tool_name))

        for task, tool_id, tool_name in tasks:
            try:
                tool_response = await asyncio.wait_for(task, timeout=timeout)
                tool_responses.append(self.create_success_response(tool_id, tool_name, str(tool_response)))
            except asyncio.TimeoutError:
                error_message = f"Timeout while calling tool <{tool_name}> after {timeout}s."
                tool_responses.append(self.create_error_response(tool_id, tool_name, error_message))
            except Exception as e:
                error_message = f"Error while calling tool <{tool_name}>: {e}"
                tool_responses.append(self.create_error_response(tool_id, tool_name, error_message))

        response = None
        if "claude" in self.config.model.id:
            tool_result_message = ToolResultMessage(role="user", content=tool_response)
            response = ToolResponseWrapper(tool_result_message == tool_result_message)
        else:
            response = ToolResponseWrapper(tool_messages=tool_responses)

        return response

    async def run_tool(self, tool_func, *args):
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(*args)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, tool_func, *args)

    def create_success_response(self, tool_id: str, tool_name: str, content: str):
        if "claude" in self.config.model.id:
            return ToolResultContent(tool_use_id=tool_id, content=content)
        else:
            return ToolMessage(tool_call_id=tool_id, name=tool_name, role="tool", content=content)

    def create_error_response(self, tool_id: str, tool_name: str, error_message: str):
        if "claude" in self.config.model.id:
            return ToolResultContent(tool_use_id=tool_id, content=error_message, is_error=True)
        else:
            return ToolMessage(tool_call_id=tool_id, name=tool_name, role="tool", content=error_message)