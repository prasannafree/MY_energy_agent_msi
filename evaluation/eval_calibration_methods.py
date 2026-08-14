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

import argparse

AGENT_URL = "http://127.0.0.1:5000"
MODEL = "qwen3.6:27b"  # Default model to evaluate
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


async def evaluate_method(method_name: str, display_name: str, model: str = MODEL, runs: int = NUM_RUNS):
    model_tag = model.replace(":", "_").replace(".", "_")
    use_case_name = f"{model_tag}_occupancy_calibration_{display_name.lower()}"
    query = (
        f"List available files first, then calibrate the building occupancy for "
        f"/workspace/all_files/ASHRAE901_OfficeSmall_STD2022_NewDelhi.idf using weather file /workspace/all_files/IND_DL_New.Delhi-Gandhi.Intl.AP.421810_TMYx.2009-2023.epw "
        f"against /workspace/all_files/my_target_delhi_small.csv using the {method_name} algorithm. "
        f"Save the outputs to /workspace/outputs/test_calibration_{display_name} "
        f"and tell me the final calibrated multiplier."
    )

    no_memory_results, with_memory_results = await run_evaluation(
        agent_url=AGENT_URL,
        use_case_name=use_case_name,
        query=query,
        expected_tool_sequence=EXPECTED_TOOL_SEQUENCE,
        expected_arg_rules=EXPECTED_ARG_RULES,
        num_runs=runs,
        custom_metric_extractor=extract_calibration_metrics,
        run_with_memory=False,
        model=model,
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
    parser = argparse.ArgumentParser(description="Evaluate Calibration Methods against a specified LLM model")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model to evaluate (e.g. qwen3.6:27b, gemini-2.5-flash, llama3.1:8b)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs per method (default: 2)")
    parser.add_argument("--method", type=str, choices=["nelder_mead", "de", "pso", "bayesian", "all"], default="all", help="Specific method to evaluate")
    args = parser.parse_args()

    all_methods = [
        ("Nelder-Mead", "Nelder_Mead"),
        ("Differential Evolution", "DE"),
        ("Particle Swarm Optimization", "PSO"),
        ("Bayesian Optimization", "Bayesian")
    ]

    if args.method != "all":
        methods_to_run = [m for m in all_methods if m[1].lower() == args.method.lower()]
    else:
        methods_to_run = all_methods
    
    print(f"\nEvaluating Model: {args.model}")
    print(f"Methods to run:  {[m[0] for m in methods_to_run]}\n")

    for method, display_name in methods_to_run:
        await evaluate_method(method, display_name, model=args.model, runs=args.runs)

if __name__ == "__main__":
    asyncio.run(main())
