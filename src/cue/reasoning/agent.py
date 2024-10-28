import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import tiktoken
from openai import OpenAI

from .system_prompt import get_shared_system_message

# o1_preview = "o1-preview-2024-09-12"

# Initialize the OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logging.error(
        "OpenAI API key not found in environment variable 'OPENAI_API_KEY'. Please set it and rerun the script."
    )
    exit(1)

client = OpenAI(api_key=api_key)

MAX_TOTAL_TOKENS = 4096
RETRY_LIMIT = 3
RETRY_BACKOFF_FACTOR = 2


class Agent:
    """
    Represents an agent that can perform various reasoning actions.
    """

    ACTION_DESCRIPTIONS = {
        "discuss": "formulating a response",
        "verify": "verifying data",
        "refine": "refining the response",
        "critique": "critiquing another agent's response",
    }

    def __init__(self, **kwargs):
        """
        Initialize an agent with custom instructions.
        """
        self.name = kwargs.get("name", "AI Assistant")
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.messages = []
        self.system_purpose = kwargs.get("system_purpose", "")
        additional_attributes = {k: v for k, v in kwargs.items() if k not in ["name", "system_purpose"]}
        # Build the full instructions
        self.instructions = self.system_purpose
        for attr_name, attr_value in additional_attributes.items():
            if isinstance(attr_value, dict):
                details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in attr_value.items())
                self.instructions += f"\n\n{attr_name.replace('_', ' ').title()}:\n{details}"
            else:
                self.instructions += f"\n\n{attr_name.replace('_', ' ').title()}: {attr_value}"
        self.other_agents_info = ""
        self.context = None

    def _add_message(self, role, content, mode="reasoning"):
        """
        Adds a message to the agent's message history and ensures token limit is not exceeded.
        """
        self.messages.append({"role": role, "content": content})
        # Enforce maximum token limit for message history
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logging.error(f"Error getting encoding: {e}")
            raise e
        total_tokens = sum(len(encoding.encode(msg["content"])) for msg in self.messages)
        if total_tokens > MAX_TOTAL_TOKENS:
            # Trim messages from the beginning
            while total_tokens > MAX_TOTAL_TOKENS and len(self.messages) > 1:
                self.messages.pop(0)
                total_tokens = sum(len(encoding.encode(msg["content"])) for msg in self.messages)

    def _handle_chat_response(self, prompt) -> Tuple[str, float, Dict]:
        """
        Handles the chat response for reasoning logic using o1-preview model.
        """
        # Use the shared system message
        shared_system_message = get_shared_system_message()

        # Combine shared system message and agent-specific instructions
        system_message = f"{shared_system_message}\n\n{self.instructions}"
        if self.context:
            system_message = f"{system_message}\n\n{self.context}"

        # Prepare messages with static content at the beginning
        messages = [{"role": "user", "content": system_message}]

        # Add message history
        messages.extend(self.messages)

        # Add the dynamic prompt at the end
        messages.append({"role": "user", "content": prompt})

        # Start timing
        start_time = time.time()

        # Initialize retry parameters
        retries = 0
        backoff = 1  # Initial backoff time in seconds

        while retries < RETRY_LIMIT:
            try:
                # Agent generates a response
                response = client.chat.completions.create(model=self.model, messages=messages)

                # End timing
                end_time = time.time()
                duration = end_time - start_time

                # Extract and return reply
                assistant_reply = response.choices[0].message.content.strip()
                self._add_message("assistant", assistant_reply)
                usage = response.usage
                return assistant_reply, duration, usage.model_dump()

            except Exception as e:
                error_type = type(e).__name__
                logging.error(f"Error in agent '{self.name}': {error_type}: {e}")
                retries += 1
                if retries >= RETRY_LIMIT:
                    logging.error(f"Agent '{self.name}' reached maximum retry limit.")
                    break
                backoff_time = backoff * (RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)

        return "An error occurred while generating a response.", time.time() - start_time

    def discuss(self, prompt, context: Optional[str] = None) -> Tuple[str, float, Dict]:
        """
        Agent formulates a response to the user's prompt.
        """
        self.context = context
        return self._handle_chat_response(prompt)

    def verify(self, data) -> Tuple[str, float, Dict]:
        """
        Agent verifies the accuracy of the provided data.
        """
        verification_prompt = f"Verify the accuracy of the following information:\n\n{data}"
        return self._handle_chat_response(verification_prompt)

    def refine(self, data, more_time=False, iterations=2) -> Tuple[str, float, List[Dict]]:
        """
        Agent refines the response to improve its accuracy and completeness.
        """
        refinement_prompt = f"Please refine the following response to improve its accuracy and completeness:\n\n{data}"
        if more_time:
            refinement_prompt += "\nTake additional time to improve the response thoroughly."

        total_duration = 0
        refined_response = data
        usages = []
        for _ in range(iterations):
            refined_response, duration, usage = self._handle_chat_response(refinement_prompt)
            total_duration += duration
            usages.append(usage)
            # Update the prompt for the next iteration
            refinement_prompt = f"Please further refine the following response:\n\n{refined_response}"

        return (refined_response, total_duration, usages)

    def critique(self, other_agent_response) -> Tuple[str, float, Dict]:
        """
        Agent critiques another agent's response for accuracy and completeness.
        """
        critique_prompt = f"Critique the following response for accuracy and completeness:\n\n{other_agent_response}"
        return self._handle_chat_response(critique_prompt)
