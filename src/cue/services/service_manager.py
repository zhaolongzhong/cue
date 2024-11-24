import json
import uuid
import asyncio
import logging
import platform
from typing import Union, Callable, Optional, Awaitable

import aiohttp
from aiohttp import ClientResponseError, ClientConnectionError
from pydantic import BaseModel

from ..config import get_settings
from ..schemas import AgentConfig, FeatureFlag, RunMetadata, MessageParam, CompletionResponse, ToolResponseWrapper
from .transport import (
    AioHTTPTransport,
    AioHTTPWebSocketTransport,
)
from .memory_client import MemoryClient
from .message_client import MessageClient
from .assistant_client import AssistantClient
from .websocket_manager import WebSocketManager
from .conversation_client import ConversationClient
from ..schemas.event_message import EventMessage, MessagePayload, EventMessageType, ClientEventPayload

logger = logging.getLogger(__name__)


class ServiceManager:
    """Manager for coordinating service clients and handling WebSocket interactions."""

    def __init__(
        self,
        run_metadata: RunMetadata,
        feature_flag: FeatureFlag,
        base_url: str,
        session: aiohttp.ClientSession,
        on_message: Callable[[dict[str, any]], Awaitable[None]] = None,
        agent: Optional[AgentConfig] = None,
    ):
        self.run_metadata = run_metadata
        self.feature_flag = feature_flag
        self.base_url = base_url
        self.access_token = get_settings().ACCESS_TOKEN
        self.agent = agent
        self.assistant_id: Optional[str] = agent.id if agent else None
        self.client_id: Optional[str] = agent.client_id if agent else run_metadata.id
        self.user_id: Optional[str] = None
        self.is_server_available = False
        if platform.system() != "Darwin" and "http://localhost" in self.base_url:
            self.base_url = self.base_url.replace("http://localhost", "http://host.docker.internal")
        self._session = session
        self.on_message = on_message
        self.recipient: Optional[str] = None  # either user id or assistant runner id
        self.runner_id: Optional[str] = None
        if self.run_metadata.mode != "client":
            # it will be used as participant so we can find the websocket session of this client by using this id
            self.runner_id = self.client_id

        self._http = AioHTTPTransport(base_url=self.base_url, access_token=self.access_token, session=self._session)
        self._ws = AioHTTPWebSocketTransport(
            ws_url=self.base_url.replace("http", "ws") + "/ws",
            client_id=self.client_id,
            access_token=self.access_token,
            runner_id=self.runner_id if self.run_metadata.mode != "client" else None,
            session=self._session,
        )

        self._ws_manager = WebSocketManager(
            ws_transport=self._ws,
            message_handlers={
                "client_disconnect": self._handle_client_connect,
                "client_connect": self._handle_client_connect,
                "client_status": self._handle_client_connect,
                "ping": self._handle_ping,
                "pong": self._handle_pong,
                "generic": self._handle_generic,
                "message": self._handle_message,
                "user": self._handle_message,
                "assistant": self._handle_message,
            },
        )

        # Initialize resource clients
        self.assistants = AssistantClient(self._http)
        self.memories = MemoryClient(self._http)
        self.conversations = ConversationClient(self._http)
        self.messages = MessageClient(self._http)

    @classmethod
    async def create(
        cls,
        run_metadata: RunMetadata,
        feature_flag: Optional[FeatureFlag] = FeatureFlag(),
        base_url: Optional[str] = None,
        on_message: Callable[[dict[str, any]], Awaitable[None]] = None,
        agent: Optional[AgentConfig] = None,
    ):
        settings = get_settings()
        base_url = base_url or settings.API_URL
        session = aiohttp.ClientSession()
        service_manager = cls(
            run_metadata=run_metadata,
            feature_flag=feature_flag,
            base_url=base_url,
            session=session,
            on_message=on_message,
            agent=agent,
        )
        await service_manager.initialize()
        return service_manager

    async def initialize(self) -> None:
        try:
            self.is_server_available = await self._check_server_availability()
            self._http.is_server_available = self.is_server_available
            if not self.is_server_available:
                return
        except Exception as e:
            logger.error(f"Server availability check failed: {e}")
            return
        if self.feature_flag.enable_storage:
            await self._create_default_assistant(self.agent.name)

    async def close(self) -> None:
        """Close all connections"""
        await self._ws.disconnect()
        await self._ws_manager.disconnect()
        await self._session.close()

    async def connect(self) -> None:
        """Establish connection to the service"""

        if not self.is_server_available:
            logger.error("Server is not available.")
            return
        await self._ws_manager.connect()

    async def disconnect(self) -> None:
        """Close all connections"""
        await self._ws_manager.disconnect()
        await self._session.close()

    async def broadcast(self, message: str) -> None:
        if not self.is_server_available:
            return
        await self._ws_manager.send_message(message)

    async def send_message_to_assistant(self, message: str) -> None:
        websocket_request_id = str(uuid.uuid4())
        msg = EventMessage(
            type=EventMessageType.USER,
            payload=MessagePayload(
                message=message,
                recipient="all",
                websocket_request_id=websocket_request_id,
                metadata={"author": {"role": "user"}, "recipient": self.assistant_id},
            ),
            websocket_request_id=websocket_request_id,
        )
        await self.broadcast(msg.model_dump_json())

    async def send_message_to_user(self, message: Union[CompletionResponse, ToolResponseWrapper, MessageParam]) -> None:
        payload = None
        role = "assistant"
        name = self.assistant_id

        if isinstance(message, ToolResponseWrapper):
            payload = message.model_dump(exclude_none=True)
        elif isinstance(message, CompletionResponse):
            if isinstance(message.response, BaseModel):
                payload = message.response.model_dump(exclude_none=True)
        elif isinstance(message, MessageParam):
            role = message.role
            name = message.name

        logger.debug(f"inx send_message_to_user message.msg_id: {message.msg_id}")
        msg = EventMessage(
            type=EventMessageType.ASSISTANT,
            payload=MessagePayload(
                message=message.get_text(),
                sender=self.run_metadata.id,
                recipient="",  # empty or user id
                payload=payload,
                websocket_request_id=str(uuid.uuid4()),
                metadata={"author": {"role": role, "name": name}},
                msg_id=message.msg_id,
            ),
            websocket_request_id=str(uuid.uuid4()),
        )
        await self.broadcast(msg.model_dump_json())

    async def broadcast_client_status(self) -> None:
        logger.debug("broadcast client status")
        msg = EventMessage(
            type=EventMessageType.CLIENT_STATUS,
            client_id=self.client_id,
            payload=ClientEventPayload(
                client_id=self.client_id,
                sender="",
                recipient="",
                payload={"runner_id": self.runner_id, "assistant_id": self.assistant_id},
            ),
        )
        await self.broadcast(msg.model_dump_json())

    async def _handle_client_connect(self, message: EventMessage) -> None:
        if message.type.value == EventMessageType.CLIENT_CONNECT:
            self.user_id = message.payload.user_id
            await self.broadcast_client_status()

        logger.info(f"Client connect event {message.payload.client_id} event: {message.model_dump_json(indent=4)}")

    async def _handle_ping(self, message: EventMessage) -> None:
        logger.debug(f"Received ping message: {message}")

    async def _handle_pong(self, message: EventMessage) -> None:
        logger.debug(f"Received pong message: {message}.")
        self._ws.handle_pong(message=message)

    async def _handle_generic(self, message: EventMessage) -> None:
        logger.debug(f"Handling generic message: {message}")

    async def _handle_message(self, message: EventMessage) -> None:
        if self.on_message:
            try:
                await self.on_message(message)
            except Exception as e:
                # Catch any other errors in on_message handling
                logger.error(f"Error in on_message handling: {e}. Message content: {message.model_dump_json(indent=4)}")

    async def _check_server_availability(self) -> bool:
        """Check if the server is running by performing an HTTP GET request to the health endpoint."""
        health_url = f"{self.base_url}/health"
        logger.debug(f"Performing health check at {health_url}")
        try:
            async with self._session.get(health_url, timeout=10) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Health check failed with status {response.status}: {text}")
                    raise ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Health check failed: {text}",
                        headers=response.headers,
                    )
                data = await response.json()
                if data.get("status") != "ok":
                    logger.error(f"Health check returned unexpected status: {data}")
                    raise ValueError(f"Unexpected health check status: {data.get('status')}")
                logger.debug("Server is available based on health check.")
                return True
        except asyncio.TimeoutError:
            logger.error("Health check request timed out.")
            raise
        except ClientConnectionError as e:
            logger.warning(f"Failed to connect to server for health check: {e}")
            return False
        except ClientResponseError as e:
            logger.error(f"Server responded with an error during health check: {e}")
        except json.JSONDecodeError:
            logger.error("Health check response is not valid JSON.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            raise

    async def _create_default_assistant(self, name: Optional[str] = None):
        self.assistant_id = await self.assistants.create_default_assistant(name)
        if not self.assistant_id:
            raise
        self.memories.set_default_assistant_id(self.assistant_id)
        conversation_id = await self.conversations.create_default_conversation(self.assistant_id)
        self.messages.set_default_conversation_id(conversation_id)
