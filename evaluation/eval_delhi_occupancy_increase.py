"""
Evaluation: New Delhi Occupancy Increase (+50%) Simulation and Comparison

Use Case: User asks to take a New Delhi IDF and weather file, run a baseline simulation,
increase the building occupancy by 50%, run the simulation again, and compare the energy results.

Expected Tool Sequence:
  1. list_available_files
  2. run_energyplus_simulation (Baseline)
  3. modify_people (Increase occupancy by 50% / multiplier=1.5)
  4. run_energyplus_simulation (Modified)
  5. calculate_rmse_tool (Compare electricity usage CSVs)

Runs WITHOUT memory only — each run is independent (fresh thread_id).
"""

import argparse
import asyncio
import os
import sys

# Add parent directory so we can import eval_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import run_evaluation, generate_report

MODEL = "qwen3.6:27b"  # Default model to evaluate

# ===========================================================================
# Configuration for this specific use case
# ===========================================================================
AGENT_URL = "http://127.0.0.1:5000"
USE_CASE_NAME = "delhi_occupancy_increase"
QUERY = (
    "Follow these exact steps in order:\n"
    "1. Call list_available_files to see what files are available.\n"
    "2. Call run_energyplus_simulation with "
    "idf_path=/workspace/all_files/ASHRAE901_OfficeLarge_STD2019_NewDelhi.idf, "
    "epw_path=/workspace/all_files/IND_DL_New.Delhi-Gandhi.Intl.AP.421810_TMYx.2009-2023.epw, "
    "output_dir=/workspace/outputs/delhi_baseline. This is the baseline simulation.\n"
    "3. Call modify_people on /workspace/all_files/ASHRAE901_OfficeLarge_STD2019_NewDelhi.idf "
    "with multiplier=1.5 to increase occupancy by 50%, saving the result to "
    "/workspace/outputs/delhi_modified/modified.idf.\n"
    "4. Call run_energyplus_simulation again with the modified IDF "
    "/workspace/outputs/delhi_modified/modified.idf and the same weather file, "
    "output_dir=/workspace/outputs/delhi_modified_sim.\n"
    "5. Call calculate_rmse_tool comparing the baseline electricity CSV "
    "/workspace/outputs/delhi_baseline/eplusout.csv against the modified simulation CSV "
    "/workspace/outputs/delhi_modified_sim/eplusout.csv. "
    "Then stop and report the RMSE comparison result."
)

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "run_energyplus_simulation",
    "modify_people",
    "run_energyplus_simulation",
    "calculate_rmse_tool"
]

NUM_RUNS = 3

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
    parser = argparse.ArgumentParser(description="Run Delhi Occupancy Increase Eval against a specified LLM model")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model to evaluate (e.g. qwen3.6:27b, llama3.1:8b)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs (default: 3)")
    args = parser.parse_args()

    model_tag = args.model.replace(":", "_").replace(".", "_")
    use_case_name = f"{model_tag}_{USE_CASE_NAME}"

    print(f"\nEvaluating Model: {args.model} on {USE_CASE_NAME}\n")

    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=use_case_name,
        query=QUERY,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=args.runs,
        run_with_memory=False,
        model=args.model,
    )

    # Generate report
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")

    generate_report(
        no_memory_results=no_memory_results,
        with_memory_results=with_memory_results,
        use_case_name=use_case_name,
        reports_dir=reports_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
