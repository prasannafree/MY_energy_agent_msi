"""
Evaluation: Calibration Methods

Tests the occupancy calibration tool using four different algorithms:
Nelder-Mead, Differential Evolution, PSO, and Bayesian Optimization.

Extracts standard agent metrics (tool accuracy) and calibration metrics
(Final RMSE, Optimal Multiplier).
"""

import asyncio
import os
import sys
import re

# Add parent directory so we can import eval_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import run_evaluation, generate_report

AGENT_URL = "http://127.0.0.1:5000"
NUM_RUNS = 2  # As requested

EXPECTED_TOOL_SEQUENCE = [
    "list_available_files",
    "calibrate_occupancy_tool"
]

EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "calibrate_occupancy_tool": {
        "idf_path": lambda v: isinstance(v, str) and ".idf" in v,
        "target_csv_path": lambda v: isinstance(v, str) and ".csv" in v,
        "optimization_method": lambda v: isinstance(v, str) and len(v) > 2,
    },
}

def extract_calibration_metrics(data: dict) -> dict:
    """Extract Optimal Multiplier and Final RMSE from the agent's markdown response."""
    response_text = data.get("response", "")
    metrics = {
        "Final RMSE (W)": None,
        "Optimal Multiplier": None,
        "Optimization Method": None
    }
    
    # Extract Optimal Multiplier
    mult_match = re.search(r'\|\s*Optimal Multiplier\s*\|\s*([0-9.]+)\s*\|', response_text, re.IGNORECASE)
    if mult_match:
        try:
            metrics["Optimal Multiplier"] = float(mult_match.group(1))
        except ValueError:
            pass

    # Extract Final RMSE
    rmse_match = re.search(r'\|\s*Final RMSE\s*\|\s*([0-9.]+)\s*W?\s*\|', response_text, re.IGNORECASE)
    if rmse_match:
        try:
            metrics["Final RMSE (W)"] = float(rmse_match.group(1))
        except ValueError:
            pass
            
    # Extract Optimization Method
    method_match = re.search(r'\|\s*Optimization Method\s*\|\s*([^|]+)\s*\|', response_text, re.IGNORECASE)
    if method_match:
        metrics["Optimization Method"] = method_match.group(1).strip()
        
    return metrics


async def evaluate_method(method_name: str, display_name: str):
    use_case_name = f"occupancy_calibration_{display_name.lower()}"
    query = (
        f"Calibrate the building occupancy for 5ZoneAirCooled.idf in Denver against "
        f"measured_target.csv using the {method_name} algorithm. "
        f"Save the outputs to /workspace/outputs/test_calibration_{display_name} "
        f"and tell me the final calibrated multiplier."
    )

    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=use_case_name,
        query=query,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=NUM_RUNS,
        custom_metric_extractor=extract_calibration_metrics
    )

    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")

    generate_report(
        no_memory_results=no_memory_results,
        with_memory_results=with_memory_results,
        use_case_name=use_case_name,
        reports_dir=reports_dir,
    )

async def main():
    methods = [
        ("Nelder-Mead", "Nelder_Mead"),
        ("Differential Evolution", "DE"),
        ("Particle Swarm Optimization", "PSO"),
        ("Bayesian Optimization", "Bayesian")
    ]
    
    for method, display_name in methods:
        await evaluate_method(method, display_name)

if __name__ == "__main__":
    asyncio.run(main())
