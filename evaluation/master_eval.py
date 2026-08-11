"""
Master Evaluation Script — Run Any Model × Any Eval Combination
(Without-memory mode only: each run gets a fresh thread)

Usage:
  # Run a specific model against specific evals
  python evaluation/master_eval.py \
    --models llama3.1:8b \
    --evals eval_calibration_methods eval_3d_visualization_ifc eval_delhi_occupancy_increase \
    --runs 3

  # Run multiple models against all evals
  python evaluation/master_eval.py \
    --models "gemini-2.5-flash" "qwen3:8b" "llama3.1:8b" \
    --evals all \
    --runs 5

  # Dry run — just show what would be executed
  python evaluation/master_eval.py \
    --models gemini-2.5-flash \
    --evals all \
    --dry-run
"""

import argparse
import asyncio
import importlib
import inspect
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ensure the evaluation directory is importable
EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR))

from eval_utils import run_evaluation, generate_report


# ---------------------------------------------------------------------------
# Eval Registry — Auto-discovery of eval_*.py modules
# ---------------------------------------------------------------------------

# Each entry: {
#     "module_name": str,
#     "use_case_name": str,
#     "query": str,
#     "expected_tool_sequence": list,
#     "expected_arg_rules": dict,
#     "custom_metric_extractor": callable or None,
#     "num_runs": int (default from module),
# }

def _discover_eval_modules() -> dict:
    """
    Scan the evaluation/ directory for eval_*.py files (excluding eval_utils.py
    and master_eval.py) and extract their configurations.

    Returns a dict of {module_name: config_dict}.
    """
    registry = {}

    for py_file in sorted(EVAL_DIR.glob("eval_*.py")):
        mod_name = py_file.stem
        if mod_name in ("eval_utils", "master_eval"):
            continue

        try:
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"  ⚠  Skipping {mod_name}: import failed → {e}")
            continue

        # --- Special case: eval_calibration_methods has sub-evaluations ---
        if mod_name == "eval_calibration_methods":
            methods = getattr(mod, "methods", None)
            # fallback: look inside main() for the methods list
            if methods is None:
                # Attempt to pull methods from the source as a constant
                try:
                    source = inspect.getsource(mod.main)
                    # The module defines methods inline in main(); we duplicate
                    # the list here for clarity.
                    methods = [
                        ("Nelder-Mead", "Nelder_Mead"),
                        ("Differential Evolution", "DE"),
                        ("Particle Swarm Optimization", "PSO"),
                        ("Bayesian Optimization", "Bayesian"),
                    ]
                except Exception:
                    methods = []

            expected_tool_sequence = getattr(mod, "EXPECTED_TOOL_SEQUENCE", [])
            expected_arg_rules = getattr(mod, "EXPECTED_ARG_RULES", {})
            custom_extractor = getattr(mod, "extract_calibration_metrics", None)

            for method_name, display_name in methods:
                use_case = f"occupancy_calibration_{display_name.lower()}"
                query = (
                    f"List available files first, then calibrate the building occupancy for "
                    f"/workspace/all_files/5ZoneAirCooled.idf using weather file /workspace/all_files/USA_CO_Denver.Intl.AP.725650_TMY3.epw "
                    f"against /workspace/all_files/measured_target.csv using the {method_name} algorithm. "
                    f"Save the outputs to /workspace/outputs/test_calibration_{display_name} "
                    f"and tell me the final calibrated multiplier."
                )
                registry[f"{mod_name}_{display_name}"] = {
                    "module_name": mod_name,
                    "use_case_name": use_case,
                    "query": query,
                    "expected_tool_sequence": expected_tool_sequence,
                    "expected_arg_rules": expected_arg_rules,
                    "custom_metric_extractor": custom_extractor,
                    "num_runs": getattr(mod, "NUM_RUNS", 2),
                }
            continue

        # --- Standard eval modules ---
        use_case = getattr(mod, "USE_CASE_NAME", mod_name.replace("eval_", ""))
        query = getattr(mod, "QUERY", None)
        expected_tool_sequence = getattr(mod, "EXPECTED_TOOL_SEQUENCE", [])
        expected_arg_rules = getattr(mod, "EXPECTED_ARG_RULES", {})
        num_runs = getattr(mod, "NUM_RUNS", 5)

        # Look for a custom metric extractor function
        custom_extractor = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if callable(obj) and "metric" in attr_name.lower() and "extract" in attr_name.lower():
                custom_extractor = obj
                break

        if query is None:
            print(f"  ⚠  Skipping {mod_name}: no QUERY constant found")
            continue

        registry[mod_name] = {
            "module_name": mod_name,
            "use_case_name": use_case,
            "query": query,
            "expected_tool_sequence": expected_tool_sequence,
            "expected_arg_rules": expected_arg_rules,
            "custom_metric_extractor": custom_extractor,
            "num_runs": num_runs,
        }

    return registry


# ---------------------------------------------------------------------------
# Master Evaluation Runner
# ---------------------------------------------------------------------------

async def run_master_evaluation(
    models: list,
    eval_keys: list,
    registry: dict,
    num_runs: int,
    agent_url: str,
    timeout: int,
    output_dir: str,
):
    """
    Run every (model × eval) combination and collect results.
    """
    workspace_dir = EVAL_DIR.parent
    reports_dir = Path(output_dir) if os.path.isabs(output_dir) else workspace_dir / output_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    master_rows = []
    all_detailed_rows = []
    total_combos = len(models) * len(eval_keys)
    combo_idx = 0

    for model in models:
        print(f"\n{'█' * 70}")
        print(f"  MODEL: {model}")
        print(f"{'█' * 70}")

        for eval_key in eval_keys:
            combo_idx += 1
            cfg = registry[eval_key]
            use_case = cfg["use_case_name"]

            print(f"\n  [{combo_idx}/{total_combos}] {model} × {use_case}")
            print(f"  {'─' * 60}")

            eval_start = time.time()

            try:
                no_mem_results, with_mem_results = await run_evaluation(
                    agent_url=agent_url,
                    use_case_name=f"{model.replace(':', '_').replace('.', '_')}_{use_case}",
                    query=cfg["query"],
                    expected_tool_sequence=cfg["expected_tool_sequence"],
                    expected_arg_rules=cfg["expected_arg_rules"],
                    num_runs=num_runs,
                    sleep_between_runs=10,
                    custom_metric_extractor=cfg["custom_metric_extractor"],
                    run_with_memory=False,
                    model=model,
                    timeout=timeout,
                )
            except Exception as e:
                print(f"  ❌ FATAL ERROR: {e}")
                master_rows.append({
                    "Model": model,
                    "Use Case": use_case,
                    "Eval Key": eval_key,
                    "Success Rate (%)": 0.0,
                    "Task Success Rate (%)": 0.0,
                    "Tool Selection Acc.": 0.0,
                    "Avg Time (s)": 0.0,
                    "Avg LLM Steps": 0,
                    "Avg Tool Calls": 0,
                    "Avg Tool Error Rate": 1.0,
                    "Avg Step Efficiency": 0.0,
                    "Avg Redundant Ratio": 0.0,
                    "Avg Looping Rate": 0.0,
                    "Avg Prompt Tokens": 0,
                    "Avg Completion Tokens": 0,
                    "Error": str(e),
                })
                continue

            eval_elapsed = time.time() - eval_start

            # All results (without-memory only)
            all_results = no_mem_results

            # Tag every row with the model name
            for row in all_results:
                row["Model"] = model
                row["Use Case"] = use_case
                row["Eval Key"] = eval_key
            all_detailed_rows.extend(all_results)

            # Generate per-eval report
            generate_report(
                no_memory_results=no_mem_results,
                with_memory_results=[],
                use_case_name=f"{model.replace(':', '_').replace('.', '_')}_{use_case}",
                reports_dir=str(reports_dir),
            )

            # Build master summary row (aggregate across no-memory runs)
            summary_source = no_mem_results
            if summary_source:
                df = pd.DataFrame(summary_source)
                total = len(df)
                master_rows.append({
                    "Model": model,
                    "Use Case": use_case,
                    "Eval Key": eval_key,
                    "Success Rate (%)": round((df["Success"].sum() / total) * 100, 1),
                    "Task Success Rate (%)": round((df["Task Success"].sum() / total) * 100, 1),
                    "Tool Selection Acc.": round(df["Tool Selection Accuracy"].mean(), 4),
                    "Avg Time (s)": round(df["Time Taken (s)"].mean(), 2),
                    "Avg LLM Steps": round(df["LLM Steps"].mean(), 1),
                    "Avg Tool Calls": round(df["Total Tool Calls"].mean(), 1),
                    "Avg Tool Error Rate": round(df["Tool Error Rate"].mean(), 4),
                    "Avg Step Efficiency": round(df["Step Efficiency"].mean(), 4),
                    "Avg Redundant Ratio": round(df["Redundant Call Ratio"].mean(), 4),
                    "Avg Looping Rate": round(df["Looping Rate"].mean(), 4),
                    "Avg Prompt Tokens": round(df["Prompt Tokens"].mean(), 1) if "Prompt Tokens" in df.columns else 0,
                    "Avg Completion Tokens": round(df["Completion Tokens"].mean(), 1) if "Completion Tokens" in df.columns else 0,
                    "Error": "",
                })
            else:
                master_rows.append({
                    "Model": model,
                    "Use Case": use_case,
                    "Eval Key": eval_key,
                    "Success Rate (%)": 0.0,
                    "Task Success Rate (%)": 0.0,
                    "Tool Selection Acc.": 0.0,
                    "Avg Time (s)": 0.0,
                    "Avg LLM Steps": 0,
                    "Avg Tool Calls": 0,
                    "Avg Tool Error Rate": 0.0,
                    "Avg Step Efficiency": 0.0,
                    "Avg Redundant Ratio": 0.0,
                    "Avg Looping Rate": 0.0,
                    "Avg Prompt Tokens": 0,
                    "Avg Completion Tokens": 0,
                    "Error": "No results collected",
                })

    # ---------------------------------------------------------------------------
    # Generate Master Summary Report
    # ---------------------------------------------------------------------------
    if master_rows:
        master_df = pd.DataFrame(master_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_csv = reports_dir / f"master_evaluation_summary_{timestamp}.csv"
        master_df.to_csv(master_csv, index=False)

        # Console summary
        print(f"\n{'═' * 80}")
        print(f"  MASTER EVALUATION SUMMARY")
        print(f"{'═' * 80}")
        print(f"  Models tested:  {', '.join(models)}")
        print(f"  Evals run:      {len(eval_keys)}")
        print(f"  Runs per eval:  {num_runs}")
        print(f"{'═' * 80}\n")

        display_cols = [
            "Model", "Use Case", "Success Rate (%)", "Task Success Rate (%)",
            "Tool Selection Acc.", "Avg Time (s)", "Avg LLM Steps",
            "Avg Tool Calls", "Avg Tool Error Rate", "Avg Step Efficiency",
        ]
        available = [c for c in display_cols if c in master_df.columns]
        print(master_df[available].to_markdown(index=False))

        print(f"\n  📄 Master summary saved: {master_csv}")

        # Also save detailed results
        if all_detailed_rows:
            detailed_df = pd.DataFrame(all_detailed_rows)
            detailed_csv = reports_dir / f"master_detailed_results_{timestamp}.csv"
            detailed_df.to_csv(detailed_csv, index=False)
            print(f"  📄 Detailed results saved: {detailed_csv}")

        # Per-model summary
        print(f"\n{'─' * 80}")
        print(f"  PER-MODEL AVERAGES")
        print(f"{'─' * 80}")
        for model in models:
            model_df = master_df[master_df["Model"] == model]
            if model_df.empty:
                continue
            print(f"\n  🤖 {model}:")
            print(f"     Avg Success Rate:       {model_df['Success Rate (%)'].mean():.1f}%")
            print(f"     Avg Task Success Rate:  {model_df['Task Success Rate (%)'].mean():.1f}%")
            print(f"     Avg Tool Selection Acc: {model_df['Tool Selection Acc.'].mean():.4f}")
            print(f"     Avg Response Time:      {model_df['Avg Time (s)'].mean():.2f}s")
            print(f"     Avg Step Efficiency:    {model_df['Avg Step Efficiency'].mean():.4f}")

    print(f"\n{'═' * 80}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'═' * 80}\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Master Evaluation — Run any model against any eval script(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model, specific evals
  python evaluation/master_eval.py \\
    --models llama3.1:8b \\
    --evals eval_calibration_methods eval_3d_visualization_ifc eval_delhi_occupancy_increase \\
    --runs 3

  # Multiple models, all evals
  python evaluation/master_eval.py \\
    --models "gemini-2.5-flash" "qwen3:8b" \\
    --evals all --runs 5

  # Dry run
  python evaluation/master_eval.py --models test --evals all --dry-run
        """
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model name(s) to evaluate (e.g. gemini-2.5-flash qwen3:8b llama3.1:8b)"
    )
    parser.add_argument(
        "--evals", nargs="+", required=True,
        help="Eval script names (without .py) or 'all' (e.g. eval_calibration_methods eval_3d_visualization_ifc)"
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per eval per model (default: 3)"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Timeout in seconds per agent call (default: 1800s / 30 mins)"
    )
    parser.add_argument(
        "--agent-url", type=str, default="http://127.0.0.1:5000",
        help="Agent base URL (default: http://127.0.0.1:5000)"
    )

    parser.add_argument(
        "--output-dir", type=str, default="evaluation_reports",
        help="Output directory for reports (default: evaluation_reports/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be run without executing"
    )

    args = parser.parse_args()

    # Discover available evals
    print(f"\n{'═' * 70}")
    print(f"  MASTER EVALUATION SCRIPT")
    print(f"{'═' * 70}")
    print(f"  Discovering eval modules...")

    registry = _discover_eval_modules()

    if not registry:
        print("  ❌ No eval modules found in evaluation/ directory!")
        sys.exit(1)

    print(f"  Found {len(registry)} eval configurations:")
    for key, cfg in registry.items():
        print(f"    • {key} → use_case={cfg['use_case_name']}")

    # Resolve --evals
    if args.evals == ["all"] or args.evals == ["ALL"]:
        eval_keys = list(registry.keys())
    else:
        eval_keys = []
        for name in args.evals:
            # Match by exact key, module_name, or use_case_name
            matched = False
            for key, cfg in registry.items():
                if name in (key, cfg["module_name"], cfg["use_case_name"]):
                    eval_keys.append(key)
                    matched = True
            if not matched:
                # Try partial match
                for key in registry:
                    if name in key or name in registry[key]["use_case_name"]:
                        eval_keys.append(key)
                        matched = True
            if not matched:
                print(f"  ⚠  Unknown eval: '{name}' — skipping")

    eval_keys = list(dict.fromkeys(eval_keys))  # deduplicate preserving order

    if not eval_keys:
        print("  ❌ No matching eval scripts found!")
        print(f"  Available: {', '.join(registry.keys())}")
        sys.exit(1)

    # Summary
    print(f"\n  Configuration:")
    print(f"    Models:     {args.models}")
    print(f"    Evals:      {eval_keys}")
    print(f"    Runs/eval:  {args.runs}")
    print(f"    Timeout:    {args.timeout}s")
    print(f"    Agent URL:  {args.agent_url}")
    print(f"    Mode:       without-memory (fresh thread per run)")
    print(f"    Output dir: {args.output_dir}")
    print(f"    Total combos: {len(args.models)} models × {len(eval_keys)} evals = {len(args.models) * len(eval_keys)}")

    if args.dry_run:
        print(f"\n  🏁 DRY RUN — would execute the following:")
        for model in args.models:
            for eval_key in eval_keys:
                cfg = registry[eval_key]
                print(f"    → {model} × {cfg['use_case_name']}: \"{cfg['query'][:80]}...\"")
        print(f"\n  (No API calls made)")
        return

    # Pre-flight: verify agent is reachable and models are available
    import httpx as _httpx
    print(f"\n  Pre-flight checks...")
    try:
        resp = _httpx.get(f"{args.agent_url}/api/health", timeout=5.0)
        if resp.status_code != 200:
            print(f"  ❌ Agent not healthy at {args.agent_url} (HTTP {resp.status_code})")
            sys.exit(1)
        health = resp.json()
        if not health.get("agent_ready"):
            print(f"  ❌ Agent not ready: {health.get('init_error', 'unknown')}")
            sys.exit(1)
        print(f"  ✔  Agent healthy (default model: {health.get('model', '?')})")
    except Exception as e:
        print(f"  ❌ Cannot connect to agent at {args.agent_url}: {e}")
        print(f"     Start it with: python agent.py")
        sys.exit(1)

    # Check Ollama model availability for local models
    ollama_models = set()
    try:
        resp = _httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            for m in resp.json().get("models", []):
                ollama_models.add(m["name"])
    except Exception:
        pass

    for model in args.models:
        if model.startswith("gemini") or model.startswith("deepseek"):
            continue  # API models — checked at runtime by agent
        if ollama_models and model not in ollama_models:
            print(f"  ⚠  Model '{model}' not found in Ollama. Available: {', '.join(sorted(ollama_models))}")
            print(f"     Pull it with: ollama pull {model}")
            sys.exit(1)
        elif ollama_models:
            print(f"  ✔  Ollama model '{model}' available")

    print(f"  ✔  All pre-flight checks passed\n")

    # Execute
    asyncio.run(run_master_evaluation(
        models=args.models,
        eval_keys=eval_keys,
        registry=registry,
        num_runs=args.runs,
        agent_url=args.agent_url,
        timeout=args.timeout,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
