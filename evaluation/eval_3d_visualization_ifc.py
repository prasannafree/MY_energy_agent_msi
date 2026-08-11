"""
Evaluation: 3D Visualization of IFC File

Use Case: User asks to generate a 3D visualization of an IFC building model.
Expected Tool Sequence: list_available_files → inspect_and_visualize_ifc_tool

Runs WITHOUT memory only — each run is independent (fresh thread_id).
"""

import argparse
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
MODEL = "qwen3.6:27b"  # Default model to evaluate
USE_CASE_NAME = "3d_visualization_ifc"
QUERY = (
    "List available files, then generate an interactive 3D HTML visualization "
    "for /workspace/all_files/20160414office_model_CV2_fordesign.ifc. "
    "Save the output to /workspace/outputs."
)
EXPECTED_TOOL_SEQUENCE = ["list_available_files", "inspect_and_visualize_ifc_tool"]
NUM_RUNS = 5

# Argument validation rules for this use case
# Each tool maps to {arg_name: validator_function}
EXPECTED_ARG_RULES = {
    "list_available_files": {},
    "inspect_and_visualize_ifc_tool": {
        "ifc_path": lambda v: isinstance(v, str) and v.endswith(".ifc"),
        "output_dir": lambda v: v is None or isinstance(v, str),
    },
}


# ===========================================================================
# Main
# ===========================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run 3D Visualization IFC Eval against a specified LLM model")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model to evaluate (e.g. qwen3.6:27b, llama3.1:8b)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs (default: 5)")
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
