import os
import copy
import json
import asyncio
import logging
from typing import Dict, List, Union, Callable, Optional
from pathlib import Path

from pydantic import BaseModel
from anthropic.types import ToolUseBlock
from openai.types.chat import (
    ChatCompletionMessageToolCall as ToolCall,
    ChatCompletionToolMessageParam as ToolMessageParam,
)
from anthropic.types.beta import BetaTextBlockParam, BetaImageBlockParam, BetaToolResultBlockParam

from .llm import LLMClient
from .tools import Tool, MemoryTool, ToolResult, ToolManager
from .utils import DebugUtils, record_usage
from .schemas import (
    Author,
    AgentConfig,
    RunMetadata,
    MessageParam,
    AgentTransfer,
    CompletionRequest,
    CompletionResponse,
    ConversationContext,
    ToolResponseWrapper,
    ToolCallToolUseBlock,
)
from .memory.memory_manager import DynamicMemoryManager
from .system_message_builder import SystemMessageBuilder
from .context.context_manager import DynamicContextManager

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: AgentConfig, agent_manager: "AgentManager"):  # type: ignore # noqa: F821
        self.id = config.id.strip().replace(" ", "_")
        self.config = config
        self.tool_manager: Optional[ToolManager] = None
        self.memory_manager = DynamicMemoryManager(max_tokens=1000)
        self.context = DynamicContextManager(max_tokens=12000 if self.config.is_primary else 4096)
        self.client: LLMClient = LLMClient(self.config)
        self.metadata: Optional[RunMetadata] = None
        self.agent_manager = agent_manager
        self.description = self._generate_description()
        self.other_agents_info = ""
        self.tool_json = None
        self.conversation_context: Optional[ConversationContext] = None  # current conversation context
        self.system_message_builder = SystemMessageBuilder(self.id, self.config)
        self.setup_feedback()

    def get_system_message(self) -> MessageParam:
        self.system_message_builder.set_conversation_context(self.conversation_context)
        self.system_message_builder.set_other_agents_info(self.other_agents_info)
        return self.system_message_builder.build()

    def _get_message_params(self) -> List[Dict]:
        """Retrieve a list of message parameter dictionaries for the completion API call."""
        return self.context.messages

    async def _get_recent_memories(self) -> Optional[dict]:
        """Should be called whenever there is a memory update"""
        if not self.config.enable_external_memory:
            return None
        try:
            memory_tool: MemoryTool = self.tool_manager.tools[Tool.Memory.value]
            memories = await memory_tool.get_recent_memories(limit=5)
            if memories:
                memories.reverse()  # # oldest -> newest order, oldest first, FIFO when limit reached
                for memory in memories:
                    if memory.strip():
                        self.memory_manager.add_memory(memory)

                # Get formatted memories respecting token limits
                return self.memory_manager.get_formatted_memories()

        except Exception as e:
            logger.error(f"Ran into {e} while trying to get recent memories.")
            return None

    def _generate_description(self) -> str:
        """Return the description about the agent."""
        description = ""
        if self.config.description:
            description = self.config.description
        if not self.config.tools:
            return
        tool_names = [tool.value for tool in self.config.tools]
        if not tool_names:
            return description
        description += f" Agent {self.id} is able to use these tools: {', '.join(tool_names)}"
        return description

    def _init_tools(self):
        if not self.tool_manager:
            self.tool_manager = self.agent_manager.tool_manager
        if not self.tool_json and self.config.tools:
            self.tool_json = self.tool_manager.get_tool_definitions(self.config.model, self.config.tools)

    def add_message(self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]) -> None:
        self.context.add_message(message)

    def snapshot(self) -> str:
        """Take a snapshot of current message list and save to a file"""
        return DebugUtils.take_snapshot(self.context.messages)

    def setup_feedback(self) -> str:
        """Setup system feedback"""
        if not self.config.feedback_path:
            self.config.feedback_path = Path(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/feedbacks"))
            )
            self.config.feedback_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"System feedback: {self.config.feedback_path}")

    async def run(self, author: Optional[Author] = None):
        self._init_tools()
        messages = self._get_message_params()  # convert message params
        logger.debug(f"{self.id}, size: {len(messages)}")
        history = [
            msg.model_dump() if hasattr(msg, "model_dump") else msg.dict() if hasattr(msg, "dict") else msg
            for msg in messages
        ]
        memories_param = await self._get_recent_memories()
        if memories_param:
            history.insert(0, memories_param)
            logger.debug(f"Recent memories: {json.dumps(memories_param, indent=4)}")
        return await self.send_messages(messages=history, author=author)

    async def send_messages(
        self,
        messages: List[Union[BaseModel, Dict]],
        metadata: Optional[RunMetadata] = None,
        author: Optional[Author] = None,
    ) -> Union[CompletionResponse, ToolCallToolUseBlock]:
        if not self.metadata:
            self.metadata = metadata

        messages_dict = [
            msg.model_dump(exclude_none=True, exclude_unset=True) if isinstance(msg, BaseModel) else msg
            for msg in messages
        ]

        completion_request = CompletionRequest(
            author=Author(name=self.id, role="assistant") if not author else author,
            model=self.config.model,
            messages=messages_dict,
            metadata=metadata,
            tool_json=self.tool_json,
            system_prompt_suffix=self.get_system_message().content,
        )
        response = await self.client.send_completion_request(completion_request)
        record_usage(response)
        logger.debug(f"{self.id} response: {response}")
        return response

    async def process_tools_with_timeout(
        self,
        tool_calls: List[ToolCallToolUseBlock],
        timeout: int = 30,
        author: Optional[Author] = None,
    ) -> ToolResponseWrapper:
        tool_results = []
        tasks = []

        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                tool_name = tool_call.function.name
                tool_id = tool_call.id
                kwargs = json.loads(tool_call.function.arguments)
            elif isinstance(tool_call, ToolUseBlock):
                tool_name = tool_call.name
                tool_id = tool_call.id
                kwargs = tool_call.input
            else:
                raise ValueError(f"Unsupported tool call type: {type(tool_call)}")

            create_agent_handoff = "create_agent_handoff"
            if tool_name not in self.tool_manager.tools and tool_name not in create_agent_handoff:
                error_message = f"Tool '{tool_name}' not found. {self.tool_manager.tools.keys()}"
                logger.error(f"{error_message}, tool_call: {tool_call}")
                tool_results.append(
                    self.create_error_response(
                        tool_id,
                        error_message,
                        tool_name,
                    )
                )
                continue

            tool_func = self.tool_manager.tools[tool_name]
            task = asyncio.create_task(self.run_tool(tool_func, **kwargs))
            tasks.append((task, tool_id, tool_name))

        base64_images = []
        for task, tool_id, tool_name in tasks:
            try:
                tool_result = await asyncio.wait_for(task, timeout=timeout)
                if isinstance(tool_result, AgentTransfer):
                    return ToolResponseWrapper(agent_transfer=tool_result)
                base64_image = tool_result.base64_image
                if base64_image:
                    base64_images.append(base64_image)
                tool_results.append(self.create_success_response(tool_id, tool_result, tool_name))
            except asyncio.TimeoutError:
                error_message = f"Timeout while calling tool <{tool_name}> after {timeout}s."
                tool_results.append(self.create_error_response(tool_id, error_message, tool_name))
            except Exception as e:
                error_message = f"Error while calling tool <{tool_name}>: {e}"
                tool_results.append(self.create_error_response(tool_id, error_message, tool_name))

        response = None
        if "claude" in self.config.model:
            tool_result_message = {"role": "user", "content": tool_results}
            response = ToolResponseWrapper(tool_result_message=tool_result_message, author=author)
        else:
            response = ToolResponseWrapper(tool_messages=tool_results, author=author, base64_images=base64_images)

        return response

    async def run_tool(self, tool_func, **kwargs):
        return await tool_func(**kwargs)

    def create_success_response(self, tool_id: str, result: ToolResult, tool_name: Optional[str] = None):
        if "claude" in self.config.model:
            tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
            is_error = False
            if result.error:
                is_error = True
                tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
            else:
                if result.output:
                    tool_result_content.append(
                        {
                            "type": "text",
                            "text": _maybe_prepend_system_tool_result(result, result.output),
                        }
                    )
                if result.base64_image:
                    tool_result_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": result.base64_image,
                            },
                        }
                    )

            # BetaToolResultBlockParam
            tool_result_block_param = {
                "tool_use_id": tool_id,
                "content": tool_result_content,
                "type": "tool_result",
                "is_error": is_error,
            }
            return tool_result_block_param
        else:
            tool_result_content = ""
            if result.error:
                is_error = True
                tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
            else:
                if result.output:
                    tool_result_content = _maybe_prepend_system_tool_result(result, result.output)
                # if result.base64_image:
                #     # handle in loop
                #     pass

            # ChatCompletionToolMessageParam
            tool_message_param = {
                "tool_call_id": tool_id,
                "name": tool_name,
                "role": "tool",
                "content": tool_result_content,
            }
            return tool_message_param

    def create_error_response(self, tool_id: str, error_message: str, tool_name: str):
        if "claude" in self.config.model:
            result_param = BetaToolResultBlockParam(
                tool_use_id=tool_id, content=error_message, type="tool_result", is_error=True
            )
            return result_param
        else:
            return ToolMessageParam(tool_call_id=tool_id, name=tool_name, role="tool", content=error_message)

    def build_context_for_next_agent(self) -> str:
        history = copy.deepcopy(self._get_message_params())
        # remove the transfer tool call and append message from calling agent
        # in this way, we keep the conversation cohensive with less noise
        max_messages = 6
        start_idx = max(0, len(history) - (max_messages + 1))
        messages = history[start_idx:-1]
        messages_content = ",".join([str(msg) for msg in messages])

        messages_content = ",".join([str(msg) for msg in messages])
        return messages_content


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
