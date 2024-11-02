from typing import Dict

from ..schemas import AgentConfig
from ..tools._tool import Tool
from ..llm.llm_model import ChatModel

default_model = ChatModel.GPT_4O_MINI.id

main_agent = AgentConfig(
    id="main",
    is_primary=True,
    description="Is main task executor and collaborate with other agents.",
    instruction="Analyze requests, collaborate with specialists when appropriate, maintain context, and synthesize outputs.",
    model=ChatModel.GPT_4O_MINI.id,
    temperature=0.8,
    max_tokens=2000,
    tools=[Tool.Edit, Tool.Memory, Tool.Coordinate],
    enable_external_memory=True,
)

agent_o = AgentConfig(
    id="agent_o",
    description="Is very good at readoning, analyzing problems, be able to deep dive on a topic.",
    instruction="You are an expert AI assistant with advanced reasoning capabilities.",
    model=ChatModel.O1_MINI.id,
    tools=[Tool.Edit],
)

agent_claude = AgentConfig(
    id="agent_claude",
    description="Is very good at coding and also provide detail reasoning on a topic.",
    instruction="You are an expert AI assistant with advanced reasoning capabilities.",
    model=ChatModel.GPT_4O.id,
    tools=[Tool.Edit],
)

system_operator = AgentConfig(
    id="system_operator",
    description="Is system operations specialist, be able to run python code or script, and execute bash command.",
    instruction="You are able to read file, write content to file, run python code or script, and execute bash command.",
    model=ChatModel.GPT_4O.id,
    tools=[Tool.Edit, Tool.Python, Tool.Bash],
)

browse_agent = AgentConfig(
    id="browse_agent",
    description="Is able to search internet and browse web page or search news.",
    instruction="Search internet, extract relevant information, verify reliability, report limitations.",
    model=default_model,
    tools=[Tool.Browse],
)

email_agent = AgentConfig(
    id="email_manager",
    description="Is able to read and sand emails and other email operations.",
    instruction="Manage email operations using available Gmail commands.",
    model=default_model,
    tools=[Tool.Email],
)

drive_agent = AgentConfig(
    id="google_drive_manager",
    description="Is able to access google drive, read file from drive or upload file to drive, or other operations.",
    instruction="Manage Google Drive operations using available commands.",
    model=default_model,
    tools=[Tool.Drive],
)


def get_agent_configs() -> tuple[Dict[str, AgentConfig], str]:
    """Get predefined agent configurations and main agent ID.

    Returns:
        Agent configurations dictionary and main agent ID
    """
    configs = {
        "main": main_agent,
        # "agent_o": agent_o,
        # "agent_claude": agent_claude,
        # "system_operator": system_operator,
        # "browse_agent": browse_agent,
        # "email_manager": email_agent,
        # "google_drive_manager": drive_agent,
    }
    return (configs, "main")
