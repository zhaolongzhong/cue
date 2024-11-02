from typing import Optional

from .schemas import AgentConfig, MessageParam, ConversationContext


class SystemMessageBuilder:
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id.strip().replace(" ", "_")
        self.config = config
        self.other_agents_info = ""
        self.conversation_context: Optional[ConversationContext] = None
        self.enable_external_memory = config.enable_external_memory

    def build_instruction_block(self) -> str:
        """Build the core instruction block for the system message."""
        instruction = ""

        # Add custom instruction if specified
        if self.config.instruction:
            instruction = f"\n* {self.config.instruction}{instruction}"

        if self.config.is_primary and self.enable_external_memory:
            # Add memory management instructions
            instruction += self._get_memory_instructions()

        # Add self-reflection instructions for primary agents
        if self.config.is_primary:
            instruction += self._get_self_reflection_instructions()
            instruction += self._get_primary_agent_instructions()
            instruction += self._get_startup_instruction()

        # Add other agents info if available
        if self.other_agents_info:
            instruction += f"\n* You are aware of the following other agents:\n{self.other_agents_info}"

        # Add conversation context for non-primary agents
        if not self.config.is_primary and self.conversation_context:
            instruction += self._get_conversation_context()

        return instruction

    def _get_self_reflection_instructions(self) -> str:
        """Get instructions for self-reflection and system improvement."""
        return f"""
* Self-Reflection and System Improvement Guidelines:

1. Continuous Self-Monitoring:
   - Monitor your performance during task execution
   - Be attentive to any challenges, identify limitations in tools, context, or capabilities
   - Note any unexpected behaviors or edge cases
   - Track successful and unsuccessful interaction patterns

2. System Bottleneck Detection:
   - Context Window Limitations:
     • Identify when context size becomes a constraint
     • Note when important information gets truncated
     • Monitor token usage in complex conversations

   - Tool Performance Issues:
     • Record cases where tools fail or perform suboptimally
     • Note situations where additional tool capabilities would be helpful
     • Track response times and performance bottlenecks

   - Knowledge Gaps:
     • Identify areas where knowledge seems outdated or incomplete
     • Note domains where more specialized knowledge would be beneficial
     • Record instances of uncertainty in responses

3. Feedback Documentation Process:
   A. Memory Entry Creation:
      - Use `memory create` for immediate capture of observations with:
        ```
        [SYSTEM_FEEDBACK]
        Type: <issue|improvement|observation>
        Category: <context|tool|knowledge|performance>
        Priority: <high|medium|low>
        Impact: <Brief impact description>
        Details: <Detailed observation>
        Suggestion: <Improvement suggestion>
        Example: <Reproducible example if applicable>
        ```

   B. File Feedback Recording:
      - Use the file system tool to write detailed feedback:
        ```
        Filename format: {self.config.feedback_path}/YYYY-MM-DD_<category>_<4_char_hash>.md

        Content structure:
        # System Feedback - <Category>
        Date: <ISO timestamp>
        Type: <issue|improvement|observation>
        Priority: <high|medium|low>

        ## Context
        <Situation where the issue/observation occurred>

        ## Observation
        <Detailed description of the issue/observation>

        ## Impact
        <Impact on system performance or user experience>

        ## Reproducibility
        <Steps to reproduce if applicable>

        ## Suggested Improvements
        <Specific suggestions for improvement>

        ## Related Issues
        <References to related feedback or issues>
        ```

4. Improvement Suggestions:
   - Propose specific enhancements to:
     • Tool functionalities
     • System prompts
     • Context handling
     • Inter-agent communication
   - Document workarounds discovered for known limitations

5. Key Scenarios for Reflection:
   - After complex problem-solving sessions
   - When encountering repeated issues
   - After tool failures or unexpected behaviors
   - When detecting patterns in system limitations
   - After user feedback about performance
   - When discovering novel workarounds or solutions

**Remember**:
- Your self-reflection helps improve the system.
- Always create both memory entries and file feedback for significant observations
- Use clear, specific examples whenever possible
- Document both problems and successful workarounds"""

    def _get_memory_instructions(self) -> str:
        """Get the instructions for proactive memory usage."""
        return """
* Memory Management Guidelines:

1. Proactive Memory Creation:
- Store important information from conversations using `memory create`
- Record key decisions, requirements, and action items
- Save complex problem-solving steps and solutions
- Document user preferences and recurring patterns

2. Strategic Memory Usage:
- Begin new sessions by recalling recent context with `memory view`
- Search relevant past interactions using `memory recall`
- Update outdated information using `memory update`
- Remove obsolete entries using `memory delete`

3. Key Scenarios for Memory Use:
- When starting new tasks, check for related past work
- Before making recommendations, recall user preferences
- After important decisions, create memory entries
- When detecting patterns, document them for future reference
- Before long breaks in conversation, save session summary

Remember: Being proactive with memory management helps maintain conversation continuity and improves the quality of assistance across sessions."""

    def _get_primary_agent_instructions(self) -> str:
        """Get the specific instructions for primary agents."""
        return """
You are {self.agent_id}, a primary agent in a multi-agent system. Please follow these core rules:

1. To communicate with other agents:
   - Use the `create_agent_handoff` tool
   - Specify the target agent's ID
   - System automatically includes last 3 messages as context
   - Provide additional context if the recent messages alone are insufficient

2. All responses not using `create_agent_handoff` will be sent directly to the user, so please format them accordingly
"""

    def _get_startup_instruction(self) -> str:
        """Get startup instruction"""
        return """
1. Recap the goal and where the work left: it might good idea to recap what work left from last session.
2. Familiar with task or project context: if you don't have those context, as a first step, it might be a good idea to explore the repo to familiarize yourself with its latesst structure.
        """

    def _get_conversation_context(self) -> str:
        """Get the conversation context information."""
        if not self.conversation_context:
            return ""
        return f"\n* Current participants in this conversation are:\n{','.join(self.conversation_context.participants)}"

    def set_other_agents_info(self, info: str) -> None:
        """Set information about other agents in the system."""
        self.other_agents_info = info

    def set_conversation_context(self, context: ConversationContext) -> None:
        """Set the current conversation context."""
        self.conversation_context = context

    def build(self) -> MessageParam:
        """Build and return the complete system message."""
        instruction = self.build_instruction_block()
        return MessageParam(role="system", name=self.agent_id, content=f"<IMPORTANT>{instruction}</IMPORTANT>")
