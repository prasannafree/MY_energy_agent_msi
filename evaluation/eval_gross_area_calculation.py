"""
Evaluation: Gross Floor Area Calculation

Use Case: User asks to calculate the gross floor area of a specific building model (e.g. New Delhi building model) and list the tools used.
Expected Tool Sequence:
  1. list_available_files
  2. calculate_gross_floor_area_tool

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
USE_CASE_NAME = "Conditioned area calculation"
QUERY = "calculate total conditioned building area of the new_delhi small office building 2022 ."
EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "calculate_gross_floor_area_tool",
]
NUM_RUNS = 5

# Argument validation rules for this use case
EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "calculate_gross_floor_area_tool": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
    },
}


# ===========================================================================
# Main
# ===========================================================================
import argparse

async def main():
    parser = argparse.ArgumentParser(description="Evaluate Gross Area Calculation")
    parser.add_argument("--model", type=str, default="qwen3.6:27b", help="LLM model to evaluate")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs")
    args = parser.parse_args()

    model_tag = args.model.replace(":", "_").replace(".", "_")
    use_case_name = f"{model_tag}_{USE_CASE_NAME.replace(' ', '_').lower()}"

    print(f"\nEvaluating Model: {args.model} on {USE_CASE_NAME}\n")

    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=use_case_name,
        query=QUERY,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=args.runs,
        model=args.model,
        run_with_memory=False,
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
