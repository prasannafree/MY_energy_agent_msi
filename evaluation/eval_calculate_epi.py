"""
Evaluation: End-to-End EPI Calculation (calculate_epi_tool)

Use Case: User asks to run an EnergyPlus simulation on a building model and calculate its
Energy Performance Index (EPI) and code compliance.

Expected Tool Sequence:
  1. list_available_files
  2. add_output_meters (Electricity:Facility)
  3. add_output_meters (NaturalGas:Facility)
  4. run_energyplus_simulation
  5. calculate_epi_tool

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
USE_CASE_NAME = "calculate_epi"
QUERY = (
    "Follow these exact steps in order:\n"
    "1. Call list_available_files to see what files are available.\n"
    "2. Call add_output_meters on /workspace/all_files/1ZoneUncontrolled.idf to add Electricity:Facility meter "
    "and save the updated IDF to /workspace/outputs/epi_run/1ZoneUncontrolled_with_meters.idf.\n"
    "3. Call add_output_meters again on the updated IDF to add NaturalGas:Facility meter.\n"
    "4. Call run_energyplus_simulation with the updated IDF and weather file "
    "/workspace/all_files/USA_CO_Denver.Intl.AP.725650_TMY3.epw, output to /workspace/outputs/epi_run.\n"
    "5. Call calculate_epi_tool with idf_path=/workspace/outputs/epi_run/1ZoneUncontrolled_with_meters.idf, "
    "simulation_output_dir=/workspace/outputs/epi_run, building_type=office, climate_zone=composite. "
    "Then stop and report the EPI result."
)

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "add_output_meters",
    "add_output_meters",
    "run_energyplus_simulation",
    "calculate_epi_tool",
]
NUM_RUNS = 3

# Argument validation rules for this use case
EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "add_output_meters": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
    },
    "run_energyplus_simulation": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
    },
    "calculate_epi_tool": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
        "simulation_output_dir": lambda v: isinstance(v, str) and len(v) > 0,
        "building_type": lambda v: isinstance(v, str) and len(v) > 0,
        "climate_zone": lambda v: isinstance(v, str) and len(v) > 0,
    },
}


# ===========================================================================
# Main
# ===========================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run EPI Calculation Eval against a specified LLM model")
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
