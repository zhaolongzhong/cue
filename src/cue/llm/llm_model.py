import os
from enum import Enum


class ChatModel(Enum):
    # Claude https://docs.anthropic.com/en/docs/about-claude/models#model-names
    CLAUDE_3_OPUS_20240229 = ("claude-3-opus-20240229", True, "anthropic")
    CLAUDE_3_5_SONNET_20241022 = ("claude-3-5-sonnet-20241022", True, "anthropic")
    CLAUDE_3_5_HAIKU_20241022 = ("claude-3-5-haiku-20241022", True, "anthropic")
    CLAUDE_3_7_SONNET_20241022 = ("claude-3-7-sonnet-20250219", True, "anthropic")

    # Gemini https://ai.google.dev/gemini-api/docs/models/gemini#model-variations
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library#supported_models
    GEMINI_1_5_FLASH = ("gemini-1.5-flash", True, "google")
    GEMINI_EXP_1206 = ("gemini-exp-1206", True, "google")
    GEMINI_1_5_PRO = ("gemini-1.5-pro", True, "google")
    GEMINI_2_0_FLASH = ("gemini-2.0-flash", True, "google")

    # OpenAI https://platform.openai.com/docs/models/overview
    GPT_4O = ("gpt-4o", True, "openai")
    GPT_4O_MINI = ("gpt-4o-mini", True, "openai")
    O1 = ("o1", True, "openai")
    O1_MINI = ("o1-mini", False, "openai")
    O1_PREVIEW = ("o1-preview", False, "openai")
    O3_MINI = ("o3-mini", True, "openai")

    DEEP_SEEK_R1_7B = ("deepseek-r1:7b", False, "ollama")

    def __init__(self, id, tool_use_support, provider):
        self.id = id
        self.tool_use_support = tool_use_support
        self.provider = provider

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def api_key_env(self) -> str:
        return f"{self.provider.upper()}_API_KEY"

    @classmethod
    def from_model_id(cls, model_id: str) -> "ChatModel":
        for model in cls:
            if model.model_id == model_id:
                return model
        raise ValueError(f"Model with id '{model_id}' not found.")

    @classmethod
    def get_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"API key for {self.model.api_key_env} not found in environment variables")
        return api_key
