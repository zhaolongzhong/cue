import json
from pathlib import Path

from tools.execute_shell_command import execute_shell_command
from tools.read_file import read_file
from tools.run_python_script import run_python_script
from tools.scan_folder import scan_folder
from tools.write_to_file import write_to_file
from utils.logs import logger


class ToolManager:
    def __init__(self):
        self.tools = {
            "read_file": read_file,
            "write_to_file": write_to_file,
            "scan_folder": scan_folder,
            "run_python_script": run_python_script,
            "execute_shell_command": execute_shell_command,
        }

    def get_tool_config(self, tool_name: str) -> dict | None:
        """Load the JSON configuration for a specific tool from its respective file."""
        # TODO: use dynamic path
        config_path = Path(f"src/tools/{tool_name}.json")
        if not config_path.exists():
            logger.error(f"Configuration file for {tool_name} does not exist")
            return None

        with config_path.open("r", encoding="utf-8") as file:
            # TODO: validate the JSON schema
            return json.load(file)

    def get_tools_json(self) -> list[dict]:
        """Iterate through the tools and gather their JSON configurations."""
        tools_configs = []
        for tool_name in self.tools:
            config = self.get_tool_config(tool_name)
            if config:
                tools_configs.append(config)
        return tools_configs

    def get_tool(self, tool_name: str):
        return self.tools.get(tool_name)