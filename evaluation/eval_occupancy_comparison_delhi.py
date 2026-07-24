"""
Evaluation: Occupancy Comparison for New Delhi (March +50% Occupancy)

Use Case: User asks to compare basecase electricity usage in New Delhi during March against a scenario with 50% increased occupancy.
Expected Tool Sequence: 
  1. list_available_files
  2. run_energyplus_simulation (Basecase)
  3. modify_people (Increase occupancy by 50%)
  4. run_energyplus_simulation (Modified case)
  5. calculate_rmse_tool (Compare electricity usage)

This script runs the evaluation in two modes:
  1) WITHOUT memory — each run is independent (fresh thread_id)
  2) WITH memory — all runs share the same thread (agent remembers prior runs)
"""

import asyncio
import os
import sys

# Add parent directory so we can import eval_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import run_evaluation, generate_report

# ===========================================================================
# Configuration for this specific use case
# ===========================================================================
AGENT_URL = "http://127.0.0.1:5000"
USE_CASE_NAME = "occupancy_comparison_delhi"
QUERY = (
    "I am in New Delhi, next month March 50 percent more people are going to come to the building. "
    "I need to compare the new electricity usage with the basecase. Give me the comparison for that."
)

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "load_idf_model",
    "inspect_people",
    "check_simulation_settings",
    "modify_run_period",
    "add_output_meters",
    "add_output_meters",
    "modify_people",
    "run_energyplus_simulation",
    "run_energyplus_simulation",
    "create_interactive_plot",
    "create_interactive_plot",
    "create_interactive_plot",
    "get_server_logs",
    "copy_file",
    "copy_file",
    "run_energyplus_simulation",
    "copy_file",
    "copy_file"
]

NUM_RUNS = 5

# Argument validation rules for this use case
EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "run_energyplus_simulation": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
    },
    "modify_people": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
    },
    "calculate_rmse_tool": {
        "target_csv_path": lambda v: isinstance(v, str) and ".csv" in v,
        "simulation_csv_path": lambda v: isinstance(v, str) and ".csv" in v,
    },
}


# ===========================================================================
# Main
# ===========================================================================
async def main():
    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=USE_CASE_NAME,
        query=QUERY,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=NUM_RUNS,
    )

    # Generate report
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")

    generate_report(
        no_memory_results=no_memory_results,
        with_memory_results=with_memory_results,
        use_case_name=USE_CASE_NAME,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
