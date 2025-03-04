from unittest.mock import Mock, AsyncMock

import pytest
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from cue.tools import Tool, ToolManager
from cue.types import Author, AgentConfig, FeatureFlag, RunMetadata, MessageParam, CompletionResponse
from cue._agent import Agent
from cue.services import ServiceManager
from cue.llm.llm_model import ChatModel
from cue.services.message_storage_service import MessageStorageService


@pytest.fixture
def agent_config() -> AgentConfig:
    """Create test agent configuration."""
    return AgentConfig(
        id="test_agent",
        name="test_agent",
        description="Test agent",
        instruction="You are a test agent",
        model=ChatModel.GPT_4O_MINI.id,
        tools=[Tool.Edit, Tool.Memory],
        feature_flag=FeatureFlag(enable_services=False, enable_storage=False),
    )


@pytest.fixture
def mock_tool_manager() -> Mock:
    """Create a mock tool manager."""
    manager = Mock(spec=ToolManager)
    manager.initialize = AsyncMock()
    manager.clean_up = AsyncMock()
    manager.get_tool_definitions = Mock(return_value=[{"name": "edit", "description": "Edit tool for testing"}])
    return manager


@pytest.fixture
def mock_service_manager():
    service_manager = AsyncMock(spec=ServiceManager)
    assistants = AsyncMock()
    assistants.get_system_context = AsyncMock(return_value="Test system context")
    service_manager.assistants = assistants
    service_manager.messages = AsyncMock()
    service_manager.message_storage_service = Mock(spec=MessageStorageService)
    return service_manager


@pytest.fixture
def agent(agent_config: AgentConfig) -> Agent:
    """Create a test agent."""
    return Agent(agent_config)


@pytest.fixture
def initialize_agent(agent: Agent, mock_tool_manager, mock_service_manager):
    """Return an async function that initializes the agent."""

    async def _initialize():
        await agent.initialize(tool_manager=mock_tool_manager, service_manager=mock_service_manager)
        return agent

    return _initialize


@pytest.mark.asyncio
async def test_agent_initialization(agent: Agent, agent_config: AgentConfig) -> None:
    """Test basic agent initialization."""
    assert agent.id == "test_agent"
    assert agent.config == agent_config
    assert not agent.state.has_initialized
    assert agent.state.get_token_stats() == {
        "system": 0,
        "tool": 0,
        "project": 0,
        "memories": 0,
        "summaries": 0,
        "messages": 0,
        "context_window": None,
        "actual_usage": {},
    }


@pytest.mark.asyncio
async def test_agent_initialization_with_tools(agent: Agent, mock_tool_manager: Mock) -> None:
    """Test agent initialization with tool manager."""
    await agent.initialize(mock_tool_manager)

    assert agent.state.has_initialized
    assert agent.tool_manager == mock_tool_manager
    mock_tool_manager.initialize.assert_called_once()
    mock_tool_manager.get_tool_definitions.assert_called_once()


@pytest.mark.asyncio
async def test_service_manager_integration(agent: Agent, mock_tool_manager: Mock) -> None:
    """Test service manager integration."""
    service_manager = Mock(spec=ServiceManager)
    service_manager.message_storage_service = Mock(spec=MessageStorageService)
    service_manager.messages = Mock()
    config = AgentConfig(model=ChatModel.GPT_4O_MINI.id)
    service_manager.get_agent_config = AsyncMock(return_value=config)

    await agent.initialize(tool_manager=mock_tool_manager, service_manager=service_manager)
    assert agent.state.has_initialized
    assert agent.service_manager == service_manager

    # Test model overwrite handling
    await agent.handle_overwrite_config()
    assert agent.config.model == ChatModel.GPT_4O_MINI.id


@pytest.mark.asyncio
async def test_add_message(initialize_agent) -> None:
    """Test adding a single message."""
    agent = await initialize_agent()
    message = MessageParam(role="user", content="Test message")
    updated_message = await agent.add_message(message)

    assert updated_message == message
    messages = agent.context_manager.context_window_manager.get_messages()
    assert len(messages) == 1
    assert messages[0]["content"] == "Test message"


@pytest.mark.asyncio
async def test_add_messages(initialize_agent) -> None:
    """Test adding multiple messages."""
    agent = await initialize_agent()
    messages = [
        MessageParam(role="user", content="Message 1"),
        MessageParam(role="assistant", content="Message 2"),
    ]
    updated_messages = await agent.add_messages(messages)

    assert len(updated_messages) == 2
    context_messages = agent.context_manager.context_window_manager.get_messages()
    assert len(context_messages) == 2
    assert context_messages[0]["content"] == "Message 1"
    assert context_messages[1]["content"] == "Message 2"


@pytest.mark.asyncio
async def test_build_message_params(initialize_agent) -> None:
    """Test building message parameters."""
    agent = await initialize_agent()
    # Add some test messages
    messages = [
        MessageParam(role="user", content="Test input"),
        MessageParam(role="assistant", content="Test response"),
    ]
    await agent.add_messages(messages)

    # Build message parameters
    message_params = await agent.build_message_params()

    assert len(message_params) == 2
    assert message_params[0]["role"] == "user"
    assert message_params[0]["content"] == "Test input"
    assert message_params[1]["role"] == "assistant"
    assert message_params[1]["content"] == "Test response"


@pytest.mark.asyncio
async def test_build_context_for_next_agent(initialize_agent) -> None:
    """Test building context for next agent."""
    agent = await initialize_agent()
    # Add some test messages
    messages = [
        MessageParam(role="user", content="Message 1"),
        MessageParam(role="assistant", content="Message 2"),
        MessageParam(role="user", content="Transfer message"),  # Will be excluded
        MessageParam(role="assistant", content="Transfer result"),  # Will be excluded
    ]
    await agent.add_messages(messages)

    # Test with different max_messages
    context_zero = agent.build_context_for_next_agent(max_messages=0)
    assert context_zero == ""

    context_one = agent.build_context_for_next_agent(max_messages=1)
    assert "Message 2" in context_one
    assert "Transfer" not in context_one

    context_all = agent.build_context_for_next_agent(max_messages=5)
    assert "Message 1" in context_all
    assert "Message 2" in context_all
    assert "Transfer" not in context_all


@pytest.mark.asyncio
async def test_run_message_flow(agent: Agent, mock_tool_manager: Mock) -> None:
    """Test the complete message flow through run()."""
    await agent.initialize(tool_manager=mock_tool_manager)
    # Mock the client's send_completion_request
    chat_message = ChatCompletionMessage(content="Test response", role="assistant")
    chat_completion = ChatCompletion(
        id="test_id",
        model=ChatModel.GPT_4O_MINI.id,
        object="chat.completion",
        choices=[Choice(finish_reason="stop", index=0, message=chat_message, logprobs=None)],
        created=123,
        usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    )
    mock_response = CompletionResponse(
        msg_id="test_id",
        model=ChatModel.GPT_4O_MINI.id,
        author=Author(name="test_agent", role="assistant"),
        response=chat_completion,
    )
    agent.client.send_completion_request = AsyncMock(return_value=mock_response)

    # Add a test message
    await agent.add_message(MessageParam(role="user", content="Test input"))

    # Run with metadata
    run_metadata = RunMetadata()
    response = await agent.run(run_metadata=run_metadata)

    assert response == mock_response
    assert agent.state.get_token_stats()["actual_usage"] != {}
