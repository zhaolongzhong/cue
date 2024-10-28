import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict

from colorama import Fore, Style, init

from cue.utils.id_generator import generate_id

default_model_4o = "gpt-4o"
o1_preview = "gpt-4o-mini"
# o1_preview = "o1-preview-2024-09-12"

# Load agent configurations from JSON file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_CONFIG_FILE = os.path.join(BASE_DIR, "agents.json")


def debug_pause():
    response = input("Continue? (y/n, press Enter to continue): ")
    if response.lower() not in ["y", "yes", ""]:
        # logger.warning("Stopped by user.")
        return


def load_agents_config():
    try:
        with open(AGENTS_CONFIG_FILE, encoding="utf-8") as f:
            agents_data = json.load(f)
        print(Fore.YELLOW + f"Successfully loaded agents configuration from '{AGENTS_CONFIG_FILE}'." + Style.RESET_ALL)
        return agents_data.get("agents", [])
    except FileNotFoundError:
        print(Fore.YELLOW + f"Agents configuration file '{AGENTS_CONFIG_FILE}' not found." + Style.RESET_ALL)
        logging.error(f"Agents configuration file '{AGENTS_CONFIG_FILE}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(Fore.YELLOW + f"Error parsing '{AGENTS_CONFIG_FILE}': {e}" + Style.RESET_ALL)
        logging.error(f"Error parsing '{AGENTS_CONFIG_FILE}': {e}")
        return []


# Custom Formatter for Logging
class ColoredFormatter(logging.Formatter):
    # Colors for log levels
    LEVEL_COLORS = {
        logging.DEBUG: Fore.LIGHTYELLOW_EX,
        logging.INFO: Fore.WHITE,  # Default to white for INFO
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    # Specific coloring for messages containing certain keywords
    KEYWORD_COLORS = {
        "HTTP Request": Fore.LIGHTYELLOW_EX,  # Use Fore.LIGHTYELLOW_EX to avoid conflict
    }

    def format(self, record):
        message = super().format(record)

        # Check for specific keywords to apply color
        for keyword, color in self.KEYWORD_COLORS.items():
            if keyword in message:
                return color + message + Style.RESET_ALL

        # Otherwise, color based on the log level
        color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
        return color + message + Style.RESET_ALL


# Remove existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a console handler with the custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# console_formatter = ColoredFormatter("%(asctime)s %(levelname)s:%(message)s")
# console_handler.setFormatter(console_formatter)

# Create a file handler without colors
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../logs/reasoning.log"))
file_handler = logging.FileHandler(base_dir)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
file_handler.setFormatter(file_formatter)


def setup_logging() -> None:
    init(autoreset=True)
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )


def print_divider(char="═", length=100, color=Fore.YELLOW):
    """
    Prints a divider line.

    Args:
        char (str): The character to use for the divider.
        length (int): The length of the divider.
        color: The color to use for the divider.
    """
    print(color + char * length + Style.RESET_ALL)


def print_header(title, color=Fore.YELLOW):
    """
    Prints a formatted header.

    Args:
        title (str): The title to display.
        color: The color to use for the header.
    """
    border = "═" * 58
    print(color + f"╔{border}╗")
    print(color + f"║{title.center(58)}║")
    print(color + f"╚{border}╝" + Style.RESET_ALL)


def log_reasoning(results: Dict):
    id = generate_id(prefix="res_")
    results["id"] = id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}_{id}.json"

    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    log_path = base_dir / "logs/reasoning" / file_name

    # Create directory if it doesn't exist
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare usage entry

    # Append to JSONL file
    with log_path.open("w") as f:
        json.dump(results, f, indent=4)
