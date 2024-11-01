import time
import asyncio
import logging
from typing import Dict, List, Union, Callable, Optional

from ._agent import Agent
from .schemas import (
    Author,
    AgentConfig,
    RunMetadata,
    MessageParam,
    AgentHandoffResult,
    CompletionResponse,
    ConversationContext,
    ToolResponseWrapper,
)
from .tools._tool import Tool, ToolManager
from .memory.memory_service_client import MemoryServiceClient

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self):
        logger.info("AgentManager initialized")
        self._agents: Dict[str, Agent] = {}
        self.active_agent: Optional[Agent] = None
        self.primary_agent: Optional[Agent] = None
        self.memory_service: Optional[MemoryServiceClient] = None
        self.tool_manager: Optional[ToolManager] = None

    async def run(
        self,
        agent_identifier: str,
        message: str,
        run_metadata: RunMetadata = RunMetadata(),
    ) -> CompletionResponse:
        self.primary_agent = self.get_agent(agent_identifier)
        # Update other agents info once we set primary agent
        self.update_other_agents_info()
        if run_metadata.enable_external_memory and not self.memory_service:
            self.memory_service = MemoryServiceClient()
            await self.memory_service.create_default_assistant()
        if not self.tool_manager:
            self.tool_manager = ToolManager(self.memory_service)
        try:
            logger.info(f"Starting run with agent {agent_identifier}")
            start_time = time.time()
            response = await self._execute_run(self.primary_agent, message, run_metadata)
            duration = time.time() - start_time
            logger.info(f"Run completed in {duration:.2f}s, metadata:\n{run_metadata.model_dump_json(indent=4)}")

            return response
        except Exception as e:
            logger.error(f"Error in run: {str(e)}", exc_info=True)
            raise

    async def _execute_run(self, agent: Agent, message: str, run_metadata: RunMetadata):
        self.active_agent = agent
        user_message = MessageParam(role="user", content=message)
        self.active_agent.add_message(user_message)

        response = None
        turns_count = 0

        author = Author(role="user", name="")
        while True:
            turns_count += 1
            if not self.should_continue_run(turns_count, run_metadata):
                break

            response: CompletionResponse = await self.active_agent.run(author)
            author = None

            if not isinstance(response, CompletionResponse):
                if response.error:
                    logger.error(response.error.model_dump())
                    # Even if there's an error, store the interaction in memory
                    # This preserves the tool call/response pair for:
                    # - Maintaining conversation context
                    # - Avoiding repeated failed attempts
                    self.active_agent.add_message(response)
                    continue
                else:
                    raise Exception(f"Unexpected response: {response}")

            self.active_agent.add_message(response)
            tool_calls = response.get_tool_calls()
            if not tool_calls:
                if self.active_agent.config.is_primary:
                    return response
                else:
                    # auto switch to primary agent
                    hand_off_result = await self.active_agent.chat_with_agent(
                        self.primary_agent.id, response.get_text()
                    )
                    await self._handle_hand_off_result(hand_off_result)
                    continue

            tool_result_wrapper = await self.active_agent.process_tools_with_timeout(
                tool_calls=tool_calls, timeout=30, author=response.author
            )
            if isinstance(tool_result_wrapper, ToolResponseWrapper):
                if tool_result_wrapper.agent_handoff_result:
                    hand_off_result = tool_result_wrapper.agent_handoff_result
                    await self._handle_hand_off_result(hand_off_result)
                    # pick up new active agent in next loop
                    continue
                else:
                    self.active_agent.add_message(tool_result_wrapper)
                    if tool_result_wrapper.base64_images:
                        tool_result_content = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{tool_result_wrapper.base64_images[0]}"},
                        }

                        contents = [
                            {"type": "text", "text": "Please check previous query info related to this image"},
                            tool_result_content,
                        ]
                        # ChatCompletionUserMessageParam
                        message_param = {"role": "user", "content": contents}
                        self.active_agent.add_message(message_param)
            else:
                raise Exception(f"Unexpected response: {tool_result_wrapper}")

        return response

    async def _handle_hand_off_result(self, hand_off_result: AgentHandoffResult) -> None:
        """Process agent handoff by updating active agent and transferring context.

        Args:
            hand_off_result: Contains target agent ID and context to transfer

        The method clears previous memory and transfers either single or list context
        to the new active agent.
        """
        self.active_agent = self._agents[hand_off_result.to_agent_id]
        self.active_agent.conversation_context = ConversationContext(
            participants=[hand_off_result.from_agent_id, hand_off_result.to_agent_id]
        )
        if isinstance(hand_off_result.context, List):
            for msg in hand_off_result.context:
                self.active_agent.add_message(msg)  # TODO: should we combine them as single message in the first place
        else:
            self.active_agent.add_message(hand_off_result.context)

    def should_continue_run(self, turns_count: int, run_metadata: RunMetadata):
        """
        Determines if the game loop should continue based on turn count and user input.

        Args:
            turns_count: Current number of turns
            run_metadata: Metadata containing max turns and debug settings

        Returns:
            bool: True if the loop should continue, False otherwise
        """
        if run_metadata.enable_turn_debug:
            response = input(
                f"Maximum turn {run_metadata.max_turns}, current: {turns_count}. Debug. Continue? (y/n, press Enter to continue): "
            )
            if response.lower() not in ["y", "yes", ""]:
                logger.warning("Stopped by user.")
                return False
        if turns_count >= run_metadata.max_turns:
            if self.active_agent.config.is_test:
                return False
            logger.warning(f"Run reaches max turn: {run_metadata.max_turns}")
            response = input(
                f"Maximum turn {run_metadata.max_turns} reached. Continue? (y/n, press Enter to continue): "
            )
            if response.lower() not in ["y", "yes", ""]:
                logger.warning("Stopped by user.")
                return False

        return True

    async def initialize(self, active_agent_id: str):
        self.active_agent = self.get_agent(active_agent_id)

    async def clean_up(self):
        if self.memory_service:
            await self.memory_service.disconnect()

        cleanup_tasks = []
        for agent_id in list(self._agents.keys()):
            try:
                if hasattr(self._agents[agent_id], "cleanup"):
                    cleanup_tasks.append(asyncio.create_task(self._agents[agent_id].cleanup()))
            except Exception as e:
                logger.error(f"Error cleaning up agent {agent_id}: {e}")

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self._agents.clear()
        logger.info("All agents cleaned up and removed.")

    def register_agent(self, config: AgentConfig) -> Agent:
        if config.id in self._agents:
            logger.warning(f"Agent with id {config.id} already exists, returning existing agent.")
            return self._agents[config.id]

        agent = Agent(config=config, agent_manager=self)
        self._agents[agent.config.id] = agent
        logger.info(
            f"register_agent {agent.config.id} (name: {config.id}), available agents: {list(self._agents.keys())}"
        )
        if config.is_primary:
            self.primary_agent = agent
        return agent

    def update_other_agents_info(self):
        if not self.primary_agent:
            for agent in self._agents.values():
                if agent.config.is_primary:
                    self.primary_agent = agent

        for agent_id, agent in self._agents.items():
            if agent.config.is_primary:
                agent.other_agents_info = self.list_agents(exclude=[agent_id])
            else:
                agent.other_agents_info = {
                    "id": self.primary_agent.id,
                    "description": self.primary_agent.description,
                }
            logger.debug(f"{agent_id} other_agents_info: {agent.other_agents_info}")

    def get_agent(self, identifier: str) -> Optional[Agent]:
        if identifier in self._agents:
            return self._agents[identifier]

        for agent in self._agents.values():
            if agent.config.id == identifier:
                return agent

        raise Exception(f"Agent '{identifier}' not found")

    def add_tool_to_agent(self, agent_id: str, tool: Union[Callable, Tool]) -> None:
        if agent_id in self._agents:
            self._agents[agent_id].config.tools.append(tool)
            logger.debug(f"Added tool {tool} to agent {agent_id}")

    def list_agents(self, exclude: List[str] = []) -> List[dict[str, str]]:
        return [
            {"id_or_name": agent.id, "description": agent.description}
            for agent in self._agents.values()
            if agent.id not in exclude
        ]
