import asyncio
import os
import sys
import shutil
import importlib
import pandas as pd
from datetime import datetime

# Add the evaluation directory to sys.path so we can import the eval scripts dynamically
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import run_evaluation

MODELS = [
    "models/qwen3:8b",
    "models/llama3.1:8b",
    "models/gemma4:latest",
    "models/mistral-nemo:12b"
]

# List of all evaluation modules in the folder
EVAL_MODULES = [
    "eval_3d_visualization_ifc",
    "eval_calculate_epi",
    "eval_delhi_occupancy_increase",
    "eval_get_epi_benchmark",
    "eval_gross_area_calculation",
    "eval_occupancy_calibration",
    "eval_occupancy_comparison_delhi",
]

async def main():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(workspace_dir, "outputs")
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")

    print(f"{'='*80}")
    print("MASTER EVALUATION SCRIPT")
    print(f"Models: {MODELS}")
    print(f"Use Cases: {EVAL_MODULES}")
    print("Mode: WITHOUT MEMORY ONLY")
    print(f"{'='*80}\n")

    # Clear files inside outputs directory
    if os.path.exists(outputs_dir):
        print(f"Cleaning outputs directory: {outputs_dir}")
        for item in os.listdir(outputs_dir):
            item_path = os.path.join(outputs_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.unlink(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
        print("Outputs directory cleaned.\n")

    # Clear files inside evaluation_reports directory
    if os.path.exists(reports_dir):
        print(f"Cleaning evaluation_reports directory: {reports_dir}")
        for item in os.listdir(reports_dir):
            item_path = os.path.join(reports_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.unlink(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")
        print("Evaluation reports directory cleaned.\n")

    # To store summary results for the final table
    # Columns will be: Model, Use Case, Success Rate (%), Avg Time (s), Tool Selection Accuracy
    summary_data = []

    for model in MODELS:
        model_name = model.split('/')[-1]
        print(f"\n{'#'*80}")
        print(f"STARTING EVALUATION FOR MODEL: {model_name}")
        print(f"{'#'*80}\n")

        for module_name in EVAL_MODULES:
            try:
                mod = importlib.import_module(module_name)
            except Exception as e:
                print(f"Failed to import {module_name}: {e}")
                continue
            
            use_case_name = getattr(mod, "USE_CASE_NAME", module_name)
            query = getattr(mod, "QUERY", "")
            agent_url = getattr(mod, "AGENT_URL", "http://127.0.0.1:5000")
            expected_tool_sequence = getattr(mod, "EXPECTED_TOOL_SEQUENCE", [])
            expected_arg_rules = getattr(mod, "EXPECTED_ARG_RULES", {})
            num_runs = getattr(mod, "NUM_RUNS", 5)

            print(f"\n--- Running Use Case: {use_case_name} with {model_name} ---")

            try:
                # We only want to run "without memory" case, so we pass run_with_memory=False
                no_memory_results, _ = await run_evaluation(
                    agent_url=agent_url,
                    use_case_name=use_case_name,
                    query=query,
                    expected_tool_sequence=expected_tool_sequence,
                    expected_arg_rules=expected_arg_rules,
                    num_runs=num_runs,
                    model=model,
                    run_with_memory=False
                )
                
                # Calculate metrics for the summary table
                if no_memory_results:
                    df = pd.DataFrame(no_memory_results)
                    total = len(df)
                    success_rate = (df["Success"].sum() / total) * 100 if total > 0 else 0
                    avg_time = df["Time Taken (s)"].mean() if total > 0 else 0
                    tool_acc = df["Tool Selection Accuracy"].mean() if "Tool Selection Accuracy" in df else 0
                    task_success_rate = (df["Task Success"].sum() / total) * 100 if total > 0 else 0
                else:
                    success_rate = 0
                    avg_time = 0
                    tool_acc = 0
                    task_success_rate = 0
                
                summary_data.append({
                    "Model": model_name,
                    "Use Case": use_case_name,
                    "Success Rate (%)": round(success_rate, 2),
                    "Task Success Rate (%)": round(task_success_rate, 2),
                    "Tool Selection Acc.": round(tool_acc, 4),
                    "Avg Time (s)": round(avg_time, 2)
                })

            except Exception as e:
                print(f"Error running evaluation for {module_name} with {model_name}: {e}")

    # Generate final comparison table
    print(f"\n{'='*80}")
    print("MASTER EVALUATION SUMMARY - MODELS vs USE CASES")
    print(f"{'='*80}\n")

    if not summary_data:
        print("No evaluation data generated.")
        return

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_markdown(index=False))

    # Pivot table to make it easier to compare models side-by-side
    print(f"\n{'='*80}")
    print("SUCCESS RATE (%) BY MODEL AND USE CASE")
    print(f"{'='*80}\n")
    
    pivot_df = summary_df.pivot(index="Use Case", columns="Model", values="Success Rate (%)")
    print(pivot_df.to_markdown())

    # Save to CSV
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"master_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(report_path, index=False)
    print(f"\nMaster summary report saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
