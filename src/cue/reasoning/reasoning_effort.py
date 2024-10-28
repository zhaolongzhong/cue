import logging
from typing import Optional, Tuple

from pydantic import BaseModel

from ..llm import ChatModel, LLMClient
from ..schemas import AgentConfig, CompletionRequest, MessageParam

logger = logging.getLogger(__name__)


class ReasoningEffortResult(BaseModel):
    needs_reasoning: bool = False
    confidence: float = 1.0


class ReasoningEffortEvaluator:
    def __init__(
        self,
        model=ChatModel.GPT_4O_MINI,
    ):
        self.llm_client = LLMClient(
            config=AgentConfig(
                id="reasoning_effort_evaluator",
                name="reasoning_effort_evaluator",
                model=ChatModel.GPT_4O_MINI,
            ),
        )

    async def evaluate(self, message: str, context: Optional[str] = None) -> ReasoningEffortResult:
        """Evaluate if a reasoning step is needed for the given message and context."""
        try:
            logger.debug(f"evaluate: {message}, context: {context}")
            evaluation_result = await self._get_evaluation_result(message, context)
            needs_reasoning, confidence = self._parse_evaluation_result(evaluation_result)
            logger.info(f"Reasoning needed: {needs_reasoning}, Confidence: {confidence}")
            return ReasoningEffortResult(needs_reasoning=needs_reasoning, confidence=confidence)
        except Exception as e:
            logger.error(f"Error in reasoning evaluation: {str(e)}")
            return ReasoningEffortResult(needs_reasoning=True, confidence=0.5)  # Default to reasoning when in doubt

    async def _get_evaluation_result(self, message: str, context: dict = None) -> str:
        """Send evaluation request to LLM."""
        if message is None or message.strip() == "":
            return "NO, 1.0"

        context_str = str(context) if context else "No additional context"

        system_message = MessageParam(
            role="system",
            content="""
            Evaluate if a reasoning step is needed before processing the user's message. A reasoning step helps analyze complex problems, break down tasks, or plan approaches.

            Respond with YES/NO followed by a confidence score between 0 and 1:
            - YES: Complex query requiring analysis, planning, or multi-step reasoning
            - NO: Simple query that can be handled directly

            Example responses:
            "YES, 0.9"
            "NO, 0.95"

            Consider these guidelines:
            1. Messages requiring analysis of multiple aspects or planning need reasoning (YES)
            2. Direct tool usage requests (read file, execute command) don't need reasoning (NO),
            3. Once we receiveing tool result, it's YES, but based on the context, we can evaluate the difficulty if we need thorough reasoning, high difficulty means high score.
            4. Simple queries, greetings, or straightforward questions don't need reasoning (NO)
            5. Queries involving problem-solving, explanation, or multiple steps need reasoning (YES)

            Examples:
            - "Hello" -> "NO, 1.0"
            - "Read file /path/to/file.txt" -> "NO, 0.9"
            - "Analyze this data and explain the trends" -> "YES, 0.9"
            """,
            name="reasoning_evaluator",
        )

        user_message = {
            "role": "user",
            "content": f"Evaluate roughly about the reasoning effort: <message>{message}</message> <context>{context_str}</context>",
        }

        request = CompletionRequest(
            messages=[system_message.model_dump(), user_message],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=10,
        )

        try:
            completion_response = await self.llm_client.send_completion_request(request=request)
            return completion_response.get_text()
        except Exception as e:
            logger.error(f"Error in evaluate_reasoning_need: {str(e)}")
            return "YES, 0.5"

    def _parse_evaluation_result(self, result: str) -> Tuple[bool, float]:
        """Parse the evaluation result and extract decision and confidence."""
        parts = result.strip().split(",")
        if len(parts) != 2:
            return True, 0.5  # Default values

        decision = parts[0].strip().upper()
        try:
            confidence = float(parts[1].strip())
            needs_reasoning = decision == "YES"
            return needs_reasoning, confidence
        except ValueError:
            return True, 0.5
