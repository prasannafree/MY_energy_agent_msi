"""
Evaluation: End-to-End EPI Calculation (calculate_epi_tool)

Use Case: User asks to run an EnergyPlus simulation on a building model and calculate its Energy Performance Index (EPI) and code compliance.
Expected Tool Sequence:
  1. list_available_files
  2. add_output_meters (Electricity:Facility)
  3. add_output_meters (NaturalGas:Facility)
  4. run_energyplus_simulation
  5. calculate_epi_tool

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
USE_CASE_NAME = "calculate_epi"
QUERY = (
    "Run an EnergyPlus simulation on the 1ZoneUncontrolled IDF model "
    "and calculate its EPI (Energy Performance Index) and code compliance for an office building in a composite climate."
)
EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "add_output_meters",
    "add_output_meters",
    "run_energyplus_simulation",
    "calculate_epi_tool",
]
NUM_RUNS = 5

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
