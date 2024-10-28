# Constants
MAX_TOTAL_TOKENS = 4096  # Adjust based on OpenAI's token limit per request
MAX_REFINEMENT_ATTEMPTS = 3
MAX_CHAT_HISTORY_TOKENS = 4096  # Max tokens for the chat mode
RETRY_LIMIT = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor

# Load agent configurations from JSON file
AGENTS_CONFIG_FILE = "agents.json"
