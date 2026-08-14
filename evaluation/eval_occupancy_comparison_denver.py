"""
Evaluation: Occupancy Comparison for Denver (March +50% Occupancy)

Use Case: User asks to compare basecase electricity usage in Denver during March against a scenario with 50% increased occupancy.
Expected Tool Sequence: 
  1. list_available_files
  2. load_idf_model
  3. inspect_people
  4. check_simulation_settings
  5. modify_run_period
  6. modify_people
  7. add_output_meters (x4)
  8. run_energyplus_simulation (x2)
  9. extract_annual_energy_tool (x2)
  10. create_interactive_plot (x2)
  11. calculate_rmse_tool

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
AGENT_URL = "http://127.0.0.1:5001"
USE_CASE_NAME = "occupancy_comparison_delhi"
QUERY = (
    "I am in New_delhi large building , next month March 50 percent more people are going to come to the building. "
    "I need to compare the new electricity usage with the basecase."
)

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "inspect_people",
    "check_simulation_settings",
    "modify_run_period",
    "add_output_meters",
    "add_output_meters",
    "run_energyplus_simulation", # Run basecase
    "modify_people",             # Increase occupancy
    "run_energyplus_simulation", # Run modified case
    "extract_annual_energy_tool",
    "extract_annual_energy_tool",
    "calculate_rmse_tool"        # Compare
]


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
import argparse

async def main():
    parser = argparse.ArgumentParser(description="Evaluate natural language occupancy comparison")
    parser.add_argument("--model", type=str, default="qwen3.6:27b", help="LLM model to evaluate")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per mode")
    args = parser.parse_args()

    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=USE_CASE_NAME,
        query=QUERY,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=args.runs,
        model=args.model,
        run_with_memory=False,
        strict_tool_penalty=False
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
