"""
Evaluation: Automated Occupancy Calibration

Use Case: User asks to calibrate building occupancy for 5ZoneAirCooled.idf using Denver.epw against measured_target.csv.
Expected Tool Sequence: 
  1. list_available_files
  2. calibrate_occupancy_tool

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
USE_CASE_NAME = "occupancy_calibration"
QUERY = "Calibrate the building occupancy for 5ZoneAirCooled.idf in Denver against measured_target_5x.csv and give me the calibrated multiplier."

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "calibrate_occupancy_tool"
]

NUM_RUNS = 5

# Argument validation rules for this use case
EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "calibrate_occupancy_tool": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
        "target_csv_path": lambda v: isinstance(v, str) and ".csv" in v,
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
