# python module1.py

# uv venv
# source .venv/bin/activate
# uv pip install openai-agents
# uv pip install python-dotenv
# python module1.py  # Input your goal or idea and get success criteria
# python module2.py  # Creates and selects a plan
# python module3.py  # Expands and evaluates the plan
# python module3-claude.py  # Better expands and evaluates the plan
# python module3-gem.py  # Expands and evaluates the plan
# python module4.py  # Identifies needed revisions
# python module5.py  # Implements revisions into a final plan
# python module6.py  # Generate easy to read markdown of final plan

# I need a plan for a workshop on legal aspects of AI agents that conduct transactions to purchase items for users.  It needs to look at 1) the contractual and general liability aspects for users, agent providers, and third party merchants that actually conduct the transactions with the user's AI agents, and 2) fiduciary duties for providers of the agents to the users who are deemed principals. The plan needs a simple scenario for the user/principal, AI agent provider (who is also the legal agent of the user/principal), and the third party merchants so participants can brainstorm the types of contractual provisions the users, agent providers, and third parties would all seek to have in place for these transactions.  The fiduciary relationship between the agent provider and the user will require corresponding contractual provisions.

import asyncio
import json
import os
import logging
from datetime import datetime, timezone
import re
from typing import Any, List, Dict, Optional

from agents import trace as agent_trace_context
from streamlit_app.utils.tracing_utils import init_tracing

# Import OpenAI libraries
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from agents import Agent, GuardrailFunctionOutput, OutputGuardrail, Runner, WebSearchTool
from agents.run_context import RunContextWrapper
from agents.lifecycle import AgentHooks

load_dotenv()  # Load environment variables

# --- Setup Logging (Modified for Verbosity) ---
def setup_logging(module_name):
    """Set up logging to console, a standard file, and a verbose file."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(logs_dir, f"{module_name}_{timestamp}.log")
    verbose_log_file = os.path.join(logs_dir, f"{module_name}_verbose_{timestamp}.log")

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Verbose logger (no truncation)
    verbose_logger = logging.getLogger(f"{module_name}_verbose")
    verbose_logger.setLevel(logging.INFO)
    if verbose_logger.handlers:
        verbose_logger.handlers = []
    verbose_file_handler = logging.FileHandler(verbose_log_file)
    verbose_file_handler.setLevel(logging.INFO)
    verbose_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    verbose_file_handler.setFormatter(verbose_format)
    verbose_logger.addHandler(verbose_file_handler)

    return logger, verbose_logger

# Initialize loggers
logger, verbose_logger = setup_logging("module1")

# Helper function to log to both loggers
def log_info(message, truncate=False, max_length=5000):
    verbose_logger.info(message)  # Always log full message to verbose
    if truncate:
        if len(message) > max_length:
            message = message[:max_length] + "... [truncated, see verbose log]"
        logger.info(message)
    else:
        logger.info(message)

# --- Text Validation Functions ---
def sanitize_text(text: str) -> str:
    """Clean and validate text to prevent corruption."""
    if not isinstance(text, str):
        return str(text)

    # Remove any non-printable or control characters
    text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t', ' '])

    # Check for obvious corruption patterns (random Unicode characters, etc.)
    # This regex looks for clusters of non-English characters that might indicate corruption
    corruption_pattern = r'[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3040-\u309F\u30A0-\u30FF\u3130-\u318F\uAC00-\uD7AF]{3,}'

    # Replace corrupted sections with a note
    text = re.sub(corruption_pattern, '[corrupted text removed]', text)

    # Ensure the text doesn't exceed a reasonable size (50KB) - adjust as necessary
    max_length = 50000
    if len(text) > max_length:
        text = text[:max_length] + "...[text truncated due to length]"

    return text

# --- Pydantic Models --- (No changes)
class SuccessCriteria(BaseModel):
    criteria: str
    reasoning: str
    rating: int = Field(..., description="Rating of the criterion (1-10)")
    
    @field_validator('rating')
    def check_rating(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Rating must be between 1 and 10')
        return v

class Module1Output(BaseModel):
    goal: str
    success_criteria: list[SuccessCriteria]
    selected_criteria: list[SuccessCriteria]  # Changed to a list for multiple criteria

    @field_validator('selected_criteria')
    def validate_selected_criteria(cls, v):
        if not v:
            raise ValueError("At least one criterion must be selected")
        return v

# --- Custom Agent Hooks for Detailed Logging --- (Modified for verbosity)
class DetailedLoggingHooks(AgentHooks):
    def __init__(self, logger, verbose_logger):
        self.logger = logger
        self.verbose_logger = verbose_logger

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ):
        """Called before the agent is invoked."""
        inputs_json = json.dumps(agent.model_dump() if hasattr(agent, 'model_dump') else {"name": agent.name}, indent=2)
        log_info(f"===== API CALL: {agent.name} =====", truncate=True)
        log_info(f"Agent start: {agent.name}", truncate=True)
        self.verbose_logger.info(f"===== API CALL: {agent.name} =====") # Redundant, but consistent
        return

    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ):
        """Called when the agent produces a final output."""
        log_info(f"===== API RESPONSE: {agent.name} =====", truncate=True)
        self.verbose_logger.info(f"===== API RESPONSE: {agent.name} =====")

        try:
            if hasattr(output, 'final_output'):
                # Sanitize if the final output has text
                if isinstance(output.final_output, str):
                    output.final_output = sanitize_text(output.final_output)
                elif isinstance(output.final_output, list):
                    for item in output.final_output:
                         if hasattr(item, "criteria"):
                            item.criteria = sanitize_text(item.criteria)
                         if hasattr(item, "reasoning"):
                            item.reasoning = sanitize_text(item.reasoning)

                response_content = json.dumps(output.final_output, indent=2) if hasattr(output, 'final_output') else str(output)
                log_info(f"Response from {agent.name}: {response_content}", truncate=True)
                self.verbose_logger.info(f"Response from {agent.name}: {response_content}")
            else:
                log_info(f"Response from {agent.name}: {str(output)}", truncate=True)
                self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
        except Exception as e:
            log_info(f"Response from {agent.name}: {str(output)}", truncate=True)
            log_info(f"Could not format response as JSON: {e}", truncate=True)
            self.verbose_logger.info(f"Response from {agent.name}: {str(output)}")
            self.verbose_logger.info(f"Could not format response as JSON: {e}")
        return output

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any
    ):
        """Called before a tool is invoked."""
        log_info(f"===== TOOL CALL: {agent.name} =====", truncate=True)
        self.verbose_logger.info(f"===== TOOL CALL: {agent.name} =====")
        return

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Any, result: str
    ):
        """Called after a tool is invoked."""
        try:
            response_content = json.dumps(result, indent=2)
            log_info(f"Tool Result from {agent.name}: {response_content}", truncate=True)
            self.verbose_logger.info(f"Tool Result from {agent.name}: {response_content}")
        except Exception as e:  # JSON decoding might fail
            log_info(f"Tool Result from {agent.name}: {str(result)}", truncate=True)
            self.verbose_logger.info(f"Tool Result from {agent.name}: {str(result)}")
            log_info(f"Could not format response as JSON: {e}", truncate=True)
            self.verbose_logger.info(f"Could not format response as JSON: {e}")

        return result

# Create logging hooks
logging_hooks = DetailedLoggingHooks(logger, verbose_logger)


# --- Search Agent ---
web_search_tool = WebSearchTool()  # Instantiate the tool

search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You are a web search assistant. Given a user's goal, "
        "perform a web search to find information relevant to defining success criteria. "
        "Return a concise summary of your findings, including citations to sources."
    ),
    model="gpt-4o",
    tools=[web_search_tool],  # Pass the *instance* of the tool
    hooks=logging_hooks,
)

# --- Other Agents ---
generate_criteria_agent = Agent(
    name="CriteriaGenerator",
    instructions=(
        "You are a helpful assistant. Given a user's goal or idea, and the results of a web search,"
        "generate five distinct and measurable success criteria. "
        "Provide a brief reasoning for each criterion. "
        "Rate each criterion on a scale of 1-10 based on how strongly it indicates goal achievement."
    ),
    model="gpt-4o",
    output_type=list[SuccessCriteria],
    hooks=logging_hooks,
)

evaluate_criteria_agent = Agent(
    name="CriteriaEvaluator",
    instructions=(
        "You are an expert evaluator. Given a goal/idea, search results, and a list of "
        "potential success criteria, select the THREE criteria that, if met together, "
        "would most strongly indicate that the goal has been achieved. "
        "Choose criteria that complement each other and cover different aspects of the goal. "
        "Consider information found by search to assist with your selection. "
        "Provide detailed reasoning for each of your selections."
    ),
    model="gpt-4o",
    output_type=list[SuccessCriteria],  # Changed to expect a list
    hooks=logging_hooks,
)

async def validate_module1_output(
    context: RunContextWrapper[None], agent: Agent, agent_output: Any
) -> GuardrailFunctionOutput:
    """Validates the output of Module 1."""
    try:
        log_info("Validating Module 1 output...", truncate=True)
        verbose_logger.info("Validating Module 1 output...")

        # Log only key parts for the standard log
        truncated_output = {
            "goal": agent_output.goal,
            "selected_criteria_count": len(agent_output.selected_criteria),
        }

        log_info(f"Output to validate (truncated): {json.dumps(truncated_output, indent=2)}", truncate=True)
        verbose_logger.info(f"Output to validate: {json.dumps(agent_output.model_dump() if hasattr(agent_output, 'model_dump') else agent_output, indent=2)}")

        Module1Output.model_validate(agent_output)
        log_info("Module 1 output validation passed", truncate=True)
        verbose_logger.info("Module 1 output validation passed")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)
    except ValidationError as e:
        logger.error(f"Module 1 output validation failed: {e}")
        verbose_logger.error(f"Module 1 output validation failed: {e}")
        return GuardrailFunctionOutput(
            output_info={"error": str(e)}, tripwire_triggered=True
        )

async def run_module_1(user_goal: str, output_file: str) -> Optional[Dict[str, Any]]:
    """Runs Module 1 and returns generated trace file paths."""
    context = RunContextWrapper(context=None)

    module_name = "module1"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    trace_processor = init_tracing(module_name=module_name, run_id=run_id)

    final_module_output_data_dict = None
    generated_trace_files = None

    try:
        async with agent_trace_context(f"{module_name}_MainWorkflow_{run_id}"):
            log_info(f"Starting Module 1 with goal: {user_goal}", truncate=True)
            verbose_logger.info(f"Starting Module 1 with goal: {user_goal}")

            log_info("Running Search Agent...", truncate=True)
            verbose_logger.info("Running Search Agent...")
            try:
                search_result = await Runner.run(
                    search_agent,
                    input=f"Find information about success criteria for: {user_goal}",
                    context=context,
                )
                search_summary = search_result.final_output
                log_info(f"Search Agent returned (truncated): {search_summary[:200]}...", truncate=True)
                verbose_logger.info(f"Search Agent returned (full): {search_summary}")
            except Exception as e:
                logger.warning(f"Search Agent failed: {e}. Proceeding without search results.")
                verbose_logger.warning(f"Search Agent failed: {e}. Proceeding without search results.")
                search_summary = "No search results available."

            log_info("GENERATING CANDIDATE SUCCESS CRITERIA...", truncate=True)
            verbose_logger.info("GENERATING CANDIDATE SUCCESS CRITERIA...")
            criteria_result = await Runner.run(
                generate_criteria_agent,
                input=f"The user's goal is: {user_goal}\n\nSearch Results:\n{search_summary}",
                context=context,
            )
            generated_criteria = criteria_result.final_output
            log_info(f"Generated {len(generated_criteria)} success criteria", truncate=True)
            verbose_logger.info(f"Generated {len(generated_criteria)} success criteria")

            for i, criterion in enumerate(generated_criteria, 1):
                log_info(f"Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})", truncate=True)
                verbose_logger.info(f"Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})")

            log_info("EVALUATING AND SELECTING TOP CRITERIA...", truncate=True)
            verbose_logger.info("EVALUATING AND SELECTING TOP CRITERIA...")
            criteria_json = json.dumps([c.model_dump() for c in generated_criteria], indent=2)
            evaluation_input = (
                f"Goal: {user_goal}\n\nSearch Results:\n{search_summary}\n\nCriteria:\n{criteria_json}"
            )
            log_info(f"Evaluation input (truncated): {evaluation_input[:500]}...", truncate=True)
            verbose_logger.info(f"Evaluation input (full): {evaluation_input}")
            evaluation_result = await Runner.run(
                evaluate_criteria_agent,
                input=evaluation_input,
                context=context,
            )
            selected_criteria = evaluation_result.final_output
            log_info(f"Selected {len(selected_criteria)} top criteria", truncate=True)
            verbose_logger.info(f"Selected {len(selected_criteria)} top criteria")

            for i, criterion in enumerate(selected_criteria, 1):
                log_info(f"Selected Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})", truncate=True)
                verbose_logger.info(f"Selected Criterion {i}: {criterion.criteria} (Rating: {criterion.rating})")

            log_info("CREATING MODULE 1 OUTPUT OBJECT...", truncate=True)
            verbose_logger.info("CREATING MODULE 1 OUTPUT OBJECT...")
            module_1_output_pydantic = Module1Output(
                goal=user_goal,
                success_criteria=generated_criteria,
                selected_criteria=selected_criteria,
            )
            verbose_logger.info(
                f"Complete Module 1 output: {json.dumps(module_1_output_pydantic.model_dump(), indent=2)}"
            )

            log_info("Applying output guardrail...", truncate=True)
            verbose_logger.info("Applying output guardrail...")
            guardrail = OutputGuardrail(guardrail_function=validate_module1_output)
            guardrail_result = await guardrail.run(
                agent=evaluate_criteria_agent,
                agent_output=module_1_output_pydantic,
                context=context,
            )
            if guardrail_result.output.tripwire_triggered:
                logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
                verbose_logger.error(f"Guardrail failed: {guardrail_result.output.output_info}")
                return None

            final_module_output_data_dict = module_1_output_pydantic.model_dump()

    except Exception as e_core_logic:
        logger.error(f"Error during Module 1 core logic (within agent_trace_context): {e_core_logic}")
        verbose_logger.error(f"Error during Module 1 core logic: {e_core_logic}", exc_info=True)
        if final_module_output_data_dict is None:
            final_module_output_data_dict = {"error": f"Core logic failed: {e_core_logic}"}
    finally:
        if trace_processor:
            try:
                trace_processor.finalize_and_write_files()
                generated_trace_files = trace_processor.get_generated_file_paths()
                log_info(
                    f"Trace files finalized for Module 1, Run ID {run_id}: {generated_trace_files}",
                    truncate=True,
                )
            except Exception as e_finalize:
                logger.error(f"Error finalizing/writing trace files for Module 1, Run ID {run_id}: {e_finalize}")
                verbose_logger.error(f"Error finalizing/writing trace files: {e_finalize}", exc_info=True)

        if final_module_output_data_dict:
            try:
                module_output_dir = os.path.dirname(output_file)
                os.makedirs(module_output_dir, exist_ok=True)

                module_output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                base_output_filename, output_ext = os.path.splitext(os.path.basename(output_file))

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(final_module_output_data_dict, f, indent=4)
                log_info(f"Module 1 primary output saved to {output_file}", truncate=True)

                timestamped_module_output_filename = os.path.join(
                    module_output_dir, f"{base_output_filename}_{module_output_timestamp}{output_ext}"
                )
                with open(timestamped_module_output_filename, "w", encoding="utf-8") as f_ts:
                    json.dump(final_module_output_data_dict, f_ts, indent=4)
                log_info(
                    f"Timestamped Module 1 primary output saved to {timestamped_module_output_filename}",
                    truncate=True,
                )
            except Exception as e_save_output:
                logger.error(f"Error saving Module 1 primary output file: {e_save_output}")
                verbose_logger.error(f"Error saving Module 1 primary output: {e_save_output}", exc_info=True)
        else:
            log_info("No final module output data to save for Module 1.", truncate=False)

    return generated_trace_files

async def main():
    log_info("Starting main function", truncate=True)
    verbose_logger.info("Starting main function")

    user_goal = input("Please enter your goal or idea: ")
    log_info(f"User input goal: {user_goal}", truncate=True)
    verbose_logger.info(f"User input goal: {user_goal}")


    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "module1_output.json")

    await run_module_1(user_goal, output_file)
    log_info("Main function completed", truncate=True)
    verbose_logger.info("Main function completed")

if __name__ == "__main__":
    log_info("Module 1 script starting", truncate=True)
    verbose_logger.info("Module 1 script starting")

    asyncio.run(main())
    log_info("Module 1 script completed", truncate=True)
    verbose_logger.info("Module 1 script completed")