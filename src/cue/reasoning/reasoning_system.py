import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion

from cue.reasoning.reasoning_effort import ReasoningEffortEvaluator

from .agent import Agent
from .utils import load_agents_config, log_reasoning, print_divider, print_header, setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


o1_preview = "gpt-4o-mini"
# o1_preview = "o1-preview-2024-09-12"


# Initialize the OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logging.error(
        "OpenAI API key not found in environment variable 'OPENAI_API_KEY'. Please set it and rerun the script."
    )
    exit(1)

client = OpenAI(api_key=api_key)


def get_completion(messages: List[Dict]) -> ChatCompletion:
    response = client.chat.completions.create(model=o1_preview, messages=messages)
    return response


MAX_REFINEMENT_ATTEMPTS = 3


class ReasoningCore:
    def __init__(self, **kwargs):
        self.agents = None
        self.context = None

    def initialize_agents(self):
        print("initialize_agents")
        """
        Initializes agents based on the configurations loaded from the JSON file.
        """
        agents_data = load_agents_config()
        agents = []
        agent_data_dict = {}

        for agent_data in agents_data:
            name = agent_data.get("name", "Unnamed Agent")
            agent = Agent(**agent_data)
            agents.append(agent)
            agent_data_dict[name] = agent_data

        # Set other agents' info for each agent
        for agent in agents:
            other_agents_info = ""
            for other_agent in agents:
                if other_agent.name != agent.name:
                    info = f"Name: {other_agent.name}"
                    # Get agent_data for other_agent
                    other_agent_data = agent_data_dict[other_agent.name]
                    # Include system purpose and other traits
                    system_purpose = other_agent_data.get("system_purpose", "")
                    info += f"\nSystem Purpose: {system_purpose}"
                    other_attributes = {
                        k: v for k, v in other_agent_data.items() if k not in ["name", "system_purpose"]
                    }
                    for attr_name, attr_value in other_attributes.items():
                        if isinstance(attr_value, dict):
                            details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in attr_value.items())
                            info += f"\n{attr_name.replace('_', ' ').title()}:\n{details}"
                        else:
                            info += f"\n{attr_name.replace('_', ' ').title()}: {attr_value}"
                    other_agents_info += f"\n\n{info}"
            # Append information about other agents to the agent's instructions
            agent.instructions += f"\n\nYou are aware of the following other agents:\n{other_agents_info.strip()}"

        return agents

    def blend_responses(self, agent_responses, user_prompt):
        """
        Combines multiple agent responses into a single, optimal response.

        Args:
            agent_responses (list): List of tuples containing agent names and their responses.
            user_prompt (str): The original user prompt.

        Returns:
            str: The blended response.
        """
        combined_prompt = (
            "Please combine the following responses into a single, optimal answer to the question.\n"
            f"Question: '{user_prompt}'\n"
            "Responses:\n"
            + "\n\n".join(f"Response from {agent_name}:\n{response}" for agent_name, response in agent_responses)
            + "\n\nProvide a concise and accurate combined response."
        )

        try:
            start_time = time.time()
            response = get_completion([{"role": "user", "content": combined_prompt}])
            blended_reply = response.choices[0].message.content.strip()
            end_time = time.time()
            duration = end_time - start_time
            usage = response.usage

            return blended_reply, duration, usage.model_dump()
        except Exception as e:
            logging.error(f"Error in blending responses: {e}")
            return "An error occurred while attempting to blend responses."

    def process_agent_action(self, agent: Agent, action: str, *args, **kwargs):
        """
        Processes a specific action for an agent.

        Args:
            agent (Agent): The agent performing the action.
            action (str): The action to perform.
            *args: Arguments required for the action.
            **kwargs: Keyword arguments for the action.

        Returns:
            tuple: The result of the action and the duration.
        """
        action_method = getattr(agent, action)
        action_description = agent.ACTION_DESCRIPTIONS.get(action, "performing an action")

        print_divider()
        logger.debug(f"System Message: {agent.name} is {action_description}...")

        try:
            result, duration, usage = action_method(*args, **kwargs)
            logger.debug(f"{agent.name}'s action completed in {duration:.2f} seconds.")
            return result, duration, usage
        except Exception as e:
            logging.error(f"Error during {action} action for {agent.name}: {e}")
            return "An error occurred.", 0

    async def reasoning(
        self,
        user_prompt: str,
        context: Optional[str] = None,
        agents: Optional[List[Agent]] = None,
    ) -> Optional[str]:
        """
        Handles the reasoning logic where agents collaborate to provide an optimal response.

        Args:
            agents (list): List of Agent instances.
        """

        effort_evaluator = ReasoningEffortEvaluator()
        effort_result = await effort_evaluator.evaluate(user_prompt, context)
        if not effort_result.needs_reasoning or effort_result.confidence < 0.7:
            logger.debug(f"Skip reasoning, {effort_result.model_dump()}")
            return None

        self.context = context

        # ------------------ Reasoning Step 1: Agents Discuss the Prompt ------------------
        print_header("Reasoning Step 1: Discussing the Prompt")
        results = {}
        opinions = {}
        durations = {}
        usages = {}
        results["user_prompt"] = user_prompt
        results["context"] = context

        if agents:
            self.agents = agents
        elif not self.agents:
            self.agents = self.initialize_agents()
        agents = self.agents

        for agent in agents:
            opinion, duration, usage = self.process_agent_action(agent, "discuss", user_prompt, context=self.context)
            opinions[agent.name] = opinion
            durations[agent.name] = duration
            usages[agent.name] = usage

        total_discussion_time = sum(durations.values())
        results["opinions"] = {"opinions": opinions, "duration": total_discussion_time, "usages": usages}
        print_divider()
        logger.debug(f"opinions result: {json.dumps(results['opinions'], indent=4)}")

        # ------------------ Reasoning Step 2: Agents Verify Their Responses ------------------
        print_header("Reasoning Step 2: Verifying Responses")
        verified_opinions = {}
        verify_durations = {}
        verify_usage = {}

        with ThreadPoolExecutor() as executor:
            futures = {}
            for agent in agents:
                futures[executor.submit(self.process_agent_action, agent, "verify", opinions[agent.name])] = agent

            for future in futures:
                agent = futures[future]
                verified_opinion, duration, usage = future.result()
                verified_opinions[agent.name] = verified_opinion
                verify_durations[agent.name] = duration
                verify_usage[agent.name] = usage

        total_verification_time = sum(verify_durations.values())
        results["verified_opinions"] = {
            "verified_opinions": verified_opinions,
            "duration": total_verification_time,
            "usages": verify_usage,
        }
        print_divider()
        logger.debug(f"verified_opinions result: {json.dumps(results['verified_opinions'], indent=4)}")

        # ------------------ Reasoning Step 3: Agents Critique Each Other's Responses ------------------
        print_header("Reasoning Step 3: Critiquing Responses")
        critiques = {}
        critique_durations = {}
        critique_usages = {}

        # Agents critique each other's verified responses
        num_agents = len(agents)
        for i, agent in enumerate(agents):
            other_agent = agents[(i + 1) % num_agents]  # Get the next agent in the list
            critique, duration, usage = self.process_agent_action(
                agent, "critique", verified_opinions[other_agent.name]
            )
            critiques[agent.name] = critique
            critique_durations[agent.name] = duration
            critique_usages[agent.name] = usage

        total_critique_time = sum(critique_durations.values())
        results["critiques"] = {
            "critiques": verified_opinions,
            "duration": total_critique_time,
            "usages": critique_usages,
        }
        print_divider()
        logger.debug(f"Critiques result: {json.dumps(results['critiques'], indent=4)}")

        # ------------------ Reasoning Step 4: Agents Refine Their Responses ------------------
        print_header("Reasoning Step 4: Refining Responses")
        refined_opinions = {}
        refine_durations = {}
        refined_usages = {}

        for agent in agents:
            refined_opinion, duration, usages = self.process_agent_action(agent, "refine", opinions[agent.name])
            refined_opinions[agent.name] = refined_opinion
            refine_durations[agent.name] = duration
            total_usages = []
            for usage in usages:
                total_usages.append(usage)
            refined_usages[agent.name] = total_usages

        total_refinement_time = sum(refine_durations.values())
        results["refined_opinions"] = {
            "refined_opinions": refined_opinions,
            "duration": total_refinement_time,
            "usages": refined_usages,
        }
        print_divider()
        logger.debug(f"Total refinement time: {total_refinement_time:.2f} seconds.")
        logger.debug(f"refined_opinions result: {json.dumps(results['refined_opinions'], indent=4)}")

        # ------------------ Reasoning Step 5: Blending Refined Responses ------------------
        print_header("Reasoning Step 5: Blending Responses")
        agent_responses = [(agent.name, refined_opinions[agent.name]) for agent in agents]
        start_blend_time = time.time()
        optimal_response, duration, usage = self.blend_responses(agent_responses, user_prompt)
        end_blend_time = time.time()
        blend_duration = end_blend_time - start_blend_time

        # Output the optimal response with enhanced formatting
        print_divider()
        print_header("Optimal Response")
        logger.debug(optimal_response)
        print_divider()

        # self.run_feedback_loop(self.agents, user_prompt, refined_opinions, refine_durations)

        results["optimal_response"] = {
            "optimal_response": optimal_response,
            "duration": blend_duration,
            "usages": usage,
        }
        logger.debug(f"optimal_response {json.dumps(results['optimal_response'], indent=4)}.")
        log_reasoning(results)
        return optimal_response

    def run_feedback_loop(self, agents: List[Agent], user_prompt: str, refined_opinions: Dict, refine_durations: Dict):
        # ------------------ Feedback Loop for Refinement ------------------
        refine_count = 0
        more_time = False
        while refine_count < MAX_REFINEMENT_ATTEMPTS:
            logger.debug("\nWas this response helpful and accurate? (yes/no): ")
            user_feedback = input().strip().lower()

            if user_feedback == "yes":
                logger.debug("Thank you for your feedback!")
                break  # Exit the feedback loop
            elif user_feedback != "no":
                logger.debug("Please answer 'yes' or 'no'.")
                continue

            # After the second "no," ask if the user wants the agents to take more time
            refine_count += 1
            if refine_count >= 2:
                logger.debug("Would you like the agents to take more time refining the response? (yes/no): ")
                more_time_input = input().strip().lower()
                more_time = more_time_input == "yes"

            # Agents can try to improve the response
            logger.debug("We're sorry to hear that. Let's try to improve the response.")

            # Agents refine their responses again
            for agent in agents:
                refined_opinion, duration = self.process_agent_action(
                    agent, "refine", refined_opinions[agent.name], more_time=more_time
                )
                refined_opinions[agent.name] = refined_opinion
                refine_durations[agent.name] += duration  # Accumulate duration

            total_refinement_time = sum(refine_durations.values())
            print_divider()
            logger.debug(f"Total refinement time: {total_refinement_time:.2f} seconds.")

            # Blend refined responses again
            print_divider()
            print_header("Blending Refined Responses")
            agent_responses = [(agent.name, refined_opinions[agent.name]) for agent in agents]
            start_blend_time = time.time()
            optimal_response = self.blend_responses(agent_responses, user_prompt)
            end_blend_time = time.time()
            blend_duration = end_blend_time - start_blend_time

            # Output the new optimal response with enhanced formatting
            print_divider()
            print_header("New Optimal Response")
            logger.debug(optimal_response)
            print_divider()

            logger.debug(f"Response generated in {blend_duration:.2f} seconds.")
        else:
            logger.debug("Maximum refinement attempts reached.")

        # ------------------ Asking to Retain Context ------------------
        logger.debug("Would you like to retain this conversation context for the next prompt? (yes/no): ")
        retain_context_input = input().strip().lower()
        if retain_context_input != "yes":
            # Clear agents' message histories
            for agent in agents:
                agent.messages = []
            logger.debug("Conversation context has been reset.")
        else:
            logger.debug("Conversation context has been retained for the next prompt.")
