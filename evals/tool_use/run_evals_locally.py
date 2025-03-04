import os
import json
import time
import asyncio
import logging
import argparse
import subprocess
from typing import Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import docker
from tqdm.asyncio import tqdm

from environment.logs import get_logger, setup_logger
from environment.task_run import TaskRun
from environment.id_generator import generate_run_id
from evals.tool_use.assets.tool_use_task import ToolTaskFamily

logger = get_logger(__name__)
logger.setLevel(level=logging.DEBUG)

RUN_DIR = Path(__file__).parent / "logs/run_dir"


def setup_logging(log_dir: Path, run_id: str, task_id: str) -> logging.Logger:
    log_dir = log_dir / run_id / task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_task.log"
    logger = setup_logger(task_id, log_file)
    logger.setLevel(logging.DEBUG)
    return logger


def get_run_assets_path() -> Path:
    return Path(__file__).parent / "assets"


def run_locally(task_run: TaskRun, results: dict):
    _logger = setup_logging(
        log_dir=RUN_DIR,
        run_id=task_run.run_id,
        task_id=task_run.task_id,
    )
    _logger.debug("start ...")
    # Offload run_in_container to the ThreadPoolExecutor
    start_time = time.time()
    logger.debug("Running task locally")
    output_path = task_run.run_dir / task_run.run_id / task_run.task_id
    command = [
        "python3",
        "assets/run_task.py",
        "--output-folder",
        str(output_path),
        "--task_id",
        task_run.task_id,
        "--instruction",
        task_run.instruction,
    ]
    logger.debug(f"Executing command: {' '.join(command)}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        env=os.environ,
    )
    logger.debug(f"Task output: {result.stdout}")
    end_time = time.time()
    # Store the result; assuming run_in_container returns a dictionary or similar
    # Save the full stdout to a file
    stdout_file = output_path / f"{task_run.task_id}_output.log"
    with open(stdout_file, "w") as file:
        file.write(result.stdout)

    results[task_run.task_id] = {
        "success": True,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "total_duration": round(end_time - start_time, 2),
    }


async def task_wrapper(
    client: docker.DockerClient, executor: ThreadPoolExecutor, loop, pbar, task_run: TaskRun, results: dict
):
    try:
        run_locally(task_run, results)
    except Exception as e:
        logger.error(f"Error in task {task_run.task_id}: {e}")
        results[task_run.task_id] = {"success": False, "error": str(e)}
    finally:
        pbar.update(1)


async def main(args):
    MAX_CONCURRENT_TASKS = 10
    run_id = generate_run_id()
    run_assets_path = get_run_assets_path()
    task_family = ToolTaskFamily()
    # for task in task_family.get_tasks().values():

    tasks = [
        TaskRun(
            task_id=task["id"],
            run_id=run_id,
            run_dir=RUN_DIR,
            run_asset_path=run_assets_path,
            image="tool_usage",
            instruction="",
        )
        for task in task_family.get_tasks().values()
    ]
    logger.debug(f"Starting tasks: {len(tasks)}")

    # Container run results
    results: dict[str, Any] = {}

    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)
    loop = asyncio.get_running_loop()

    # Create a list of coroutine tasks
    client = docker.from_env()
    # Initialize the asyncio-compatible progress bar
    pbar = tqdm(total=len(tasks), desc="Running Tasks")
    coroutine_tasks = [task_wrapper(client, executor, loop, pbar, task, results) for task in tasks]
    # Run tasks concurrently with limited concurrency
    await asyncio.gather(*coroutine_tasks)
    # Clean up
    executor.shutdown(wait=True)
    logger.info("All tasks have been processed.")

    # Save results
    success_count = sum(1 for res in results.values() if res.get("success"))
    failure_count = len(results) - success_count
    logger.info(f"Tasks completed: {success_count} succeeded, {failure_count} failed.")

    for task_id, res in list(results.items())[:5]:
        logger.info(f"Task {task_id}: {res}")
    logger.debug(f"results:\n{json.dumps(results, indent=4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple Docker container tasks concurrently.")
    args = parser.parse_args()
    asyncio.run(main(args))
