"""
Evaluation: EPI Benchmark Table Retrieval (get_epi_benchmark_tool)

Use Case: User asks to generate a complete comparison table of EPI code benchmarks for hotels across all 5 climate zones (cold, composite, hot_dry, warm_humid, temperate).
Expected Tool Sequence:
  1-5. get_epi_benchmark_tool (5 calls, one per climate zone)

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
USE_CASE_NAME = "get_epi_benchmark"
QUERY = "Give me a complete comparison table of EPI benchmarks for hotels across all climate zones (cold, composite, hot_dry, warm_humid, temperate)."
EXPECTED_TOOL_SEQUENCE = ["get_epi_benchmark_tool"] * 5
NUM_RUNS = 5

# Argument validation rules for this use case
EXPECTED_ARG_RULES = {
    "get_epi_benchmark_tool": {
        "building_type": lambda v: isinstance(v, str) and "hotel" in v.lower(),
        "climate_zone": lambda v: isinstance(v, str) and v.lower() in ["cold", "composite", "hot_dry", "warm_humid", "temperate", "summer", "hot"],
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
