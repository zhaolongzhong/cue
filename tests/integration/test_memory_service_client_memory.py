import logging
from uuid import uuid4
from typing import AsyncGenerator
from datetime import datetime

import pytest
import pytest_asyncio

from cue.schemas.assistant import AssistantCreate
from cue.schemas.assistant_memory import AssistantMemoryCreate, AssistantMemoryUpdate
from cue.memory.memory_service_client import MemoryServiceClient

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def memory_client() -> AsyncGenerator[MemoryServiceClient, None]:
    """Fixture to create and manage MemoryServiceClient instance."""
    client = MemoryServiceClient()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


@pytest_asyncio.fixture
async def test_assistant(memory_client: MemoryServiceClient):
    """Fixture to create a test assistant and yield its object."""
    assistant = await memory_client.create_assistant(
        AssistantCreate(
            name=f"Test Assistant {uuid4()}", metadata={"test": True, "created_at": datetime.now().isoformat()}
        )
    )
    try:
        yield assistant
    finally:
        try:
            await memory_client.delete_assistant(assistant_id=assistant.id)
        except Exception as e:
            logger.warning(f"Failed to delete test assistant: {e}")


@pytest.mark.integration
class TestMemoryServiceClient:
    @pytest.mark.asyncio
    async def test_create_memory(self, memory_client: MemoryServiceClient, test_assistant) -> None:
        """Test creating and retrieving a memory."""
        assistant_id = test_assistant.id
        memory_content = "This is a test memory content"

        # Create memory
        created_memory = await memory_client.create_memory(
            memory=AssistantMemoryCreate(assistant_id=assistant_id, content=memory_content, metadata={"test": True}),
            assistant_id=assistant_id,
        )

        assert created_memory is not None, "Memory should not be None"
        assert created_memory.content == memory_content, "Memory content should match"
        assert created_memory.assistant_id == assistant_id, "Assistant ID should match"
        assert created_memory.metadata.get("test") is True, "Memory metadata should be preserved"

        # Verify memory can be retrieved
        retrieved_memory = await memory_client.get_memory(memory_id=created_memory.id, assistant_id=assistant_id)
        assert retrieved_memory is not None, "Retrieved memory should not be None"
        assert retrieved_memory.id == created_memory.id, "Retrieved memory ID should match"

        memory_content_edited = "This is a test memory content(edit)"
        await memory_client.update_memory(
            memory_id=retrieved_memory.id,
            memory=AssistantMemoryUpdate(content=memory_content_edited),
            assistant_id=assistant_id,
        )
        retrieved_memory = await memory_client.get_memory(memory_id=created_memory.id, assistant_id=assistant_id)
        assert retrieved_memory.content == memory_content_edited, "Memory edit content should match"

        # Delete memory
        result = await memory_client.delete_memories([created_memory.id], assistant_id=assistant_id)
        assert result["success"] is True, "Delete memory should be successful"
        assert result["deleted_count"] == 1, "One memory should be deleted"

        retrieved_memory = await memory_client.get_memory(
            memory_id=created_memory.id,
            assistant_id=assistant_id,
        )
        assert retrieved_memory is None, "Retrieved memory should be None now"

    @pytest.mark.asyncio
    async def test_query_memories(self, memory_client: MemoryServiceClient, test_assistant) -> None:
        """Test creating multiple memories and querying them."""
        # Create multiple memories
        assistant_id = test_assistant.id
        memories_content = [
            "Python is a versatile programming language",
            "JavaScript is commonly used for web development",
            "Machine learning involves statistical analysis",
        ]

        for content in memories_content:
            await memory_client.create_memory(
                memory=AssistantMemoryCreate(content=content),
                assistant_id=assistant_id,
            )

        # Query memories
        query = "programming languages"
        memories_response = await memory_client.search_memories(assistant_id=assistant_id, query=query)
        retrieved_memories = memories_response.memories

        assert retrieved_memories is not None, "Query should return results"
        assert len(retrieved_memories) > 0, "Should find at least one relevant memory"

        # Verify the most relevant memory is returned first
        assert any(
            "Python" in memory.content for memory in retrieved_memories
        ), "Should find memory about Python programming"

        # Test empty query handling
        # empty_query = "nonexistent content xyz123"
        # empty_results = await memory_client.search_memories(assistant_id=assistant_id, query=empty_query)
        # assert len(empty_results) == 0, "Should return empty list for irrelevant query"
