import json
import shutil
import asyncio
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path

from tool_use_task import ToolTaskFamily

from cue import Tool, Agent, ChatModel, AgentConfig, FeatureFlag, AgentManager, CompletionResponse

logger = logging.getLogger(__name__)


def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
    if result.returncode != 0:
        raise Exception(f"Command exited with non-zero exit code: {result.returncode}")

    if result.stdout:
        print(f"[stdout]\n{result.stdout}")
        return result.stdout

    return ""


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run agent")
    parser.add_argument("--task_id", type=str, required=True, help="Task ID")
    parser.add_argument("--instruction", type=str, required=False, help="Instruction")
    parser.add_argument("--output-folder", type=str, required=False, help="Output folder to write the file")
    args = parser.parse_args()

    if not args.task_id.strip():
        parser.error("The --task_id argument cannot be empty.")

    return args


async def run_task(args: argparse.Namespace, task_family: ToolTaskFamily, temp_dir: str):
    model = ChatModel.GPT_4O_MINI
    task_id = args.task_id
    output_path = getattr(args, "output_path", None)

    if not output_path:
        output_path = str(Path(__file__).parent / "output")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    task = task_family.get_tasks()[task_id]

    agent_config = AgentConfig(
        id="cue_cli",
        model=ChatModel.GPT_4O_MINI,
        temperature=0.8,
        max_tokens=2000,
        conversation_id="",
        feature_flag=FeatureFlag(is_cli=True, is_eval=False),
        enable_external_memory=False,
        tools=[Tool.Edit, Tool.Bash],
        is_test=True,
    )
    manager = AgentManager()
    agent: Agent = await manager.register_agent(agent_config)

    # set up task
    init_cmd = task["start"]["code"]
    if init_cmd:
        run_command(f"cd {temp_dir} && {init_cmd}")

    instruction = task_family.get_instructions(task, temp_dir)
    response = await agent.send_messages(instruction)
    metadata = agent.get_metadata()
    print(f"task:{task['id']}, metadata: {metadata}, agent response: {response}")
    if isinstance(response, CompletionResponse):
        submission = response.get_text()
    else:
        submission = str(response)

    # check if there is a check command
    check_cmd = task["evaluation"]["check"]
    if check_cmd:
        output = run_command(f"cd {temp_dir} && {check_cmd}")
        submission = output
    score = task_family.score(task, str(submission))
    result = {
        "task_id": task["id"],
        "model": model,
        "submission": submission,
        "score": score,
        "metadata": metadata.model_dump() if metadata else None,
    }

    result_path = output_path / f"result_{task_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as result_file:
        json.dump(result, result_file, indent=4)


async def main() -> None:
    temp_dir = tempfile.mkdtemp()
    try:
        args: argparse.Namespace = parse_args()
        await run_task(args, ToolTaskFamily(), temp_dir)
    finally:
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except PermissionError as e:
        logger.error(f"Failed due to permission issues. {str(e)}")
        exit(2)
    except OSError:
        logger.error("Failed due to an I/O error.")
        exit(3)
    except Exception as e:
        logger.error(f"An unexpected error occurred in run_task: {e}")
        exit(1)

"""
python3 run_task.py --task_id "read"
python3 assets/run_task.py --task_id "read"
"""
