# https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/bash.py
import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import ClassVar, Literal, Optional

from .base import BaseTool, CLIResult, ToolError, ToolResult


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    @staticmethod
    @asynccontextmanager
    async def async_timeout(timeout: float):
        """Backwards compatible implementation of asyncio.timeout(). Works with Python versions before 3.11."""
        try:
            if hasattr(asyncio, "timeout"):
                # Python 3.11+ - use built-in timeout
                async with asyncio.timeout(timeout):
                    yield
            else:
                # Earlier versions - use wait_for
                try:
                    async with AsyncExitStack() as _stack:
                        cancel_scope = asyncio.create_task(asyncio.sleep(timeout))
                        try:
                            yield
                        finally:
                            cancel_scope.cancel()
                            try:
                                await cancel_scope
                            except asyncio.CancelledError:
                                pass
                except asyncio.TimeoutError:
                    raise
        except asyncio.TimeoutError:
            raise

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(command.encode() + f"; echo '{self._sentinel}'\n".encode())
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with self.async_timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = self._process.stdout._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = self._process.stderr._buffer.decode()  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class BashTool(BaseTool):
    """A tool that allows the agent to run bash commands. The tool parameters are defined by Anthropic and are not editable."""

    _session: Optional[_BashSession]
    name: ClassVar[Literal["bash"]] = "bash"

    def __init__(self):
        self._function = self.bash
        self._session = None
        super().__init__()

    async def __call__(self, command: Optional[str] = None, restart: bool = False, **kwargs):
        return await self.bash(command=command, restart=restart, **kwargs)

    async def bash(self, command: Optional[str] = None, restart: bool = False, **kwargs):
        """Run bash commands.

        This tool allows the agent to execute bash commands.

        Args:
            command (Optional[str]): The bash command to execute. If None, defaults to an empty command.
            restart (bool): Whether to restart the process after executing the command. Defaults to False.
            **kwargs: Additional keyword arguments for further customization.

        Returns:
            None

        """
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")
