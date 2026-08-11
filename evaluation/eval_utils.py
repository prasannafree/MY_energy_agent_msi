"""
Shared evaluation utilities for the EnergyPlus MCP Agent evaluation framework.

Provides metric computation functions and a reusable evaluation runner
that supports dual-mode testing (with-memory vs without-memory).
"""

import asyncio
import time
from collections import Counter
import httpx
import pandas as pd
from datetime import datetime
import os


# ---------------------------------------------------------------------------
# Metric Computation Functions
# ---------------------------------------------------------------------------

def compute_tool_selection_accuracy(expected_tools: list, actual_tools: list) -> float:
    """
    Measures how accurately the agent selected the right tools.
    Uses multiset (Counter) comparison to properly handle expected duplicates
    (e.g., run_energyplus_simulation called twice, get_epi_benchmark_tool ×5).

    Returns a ratio between 0.0 and 1.0.

    Formula: multiset_intersection_size / max(|expected|, |actual|)
    - 1.0 = perfect match (exact same multiset of tools)
    - <1.0 = either missing expected tools or called extra wrong tools
    """
    if not expected_tools and not actual_tools:
        return 1.0
    expected_counter = Counter(expected_tools)
    actual_counter = Counter(actual_tools)
    # Multiset intersection: min count for each tool
    intersection_size = sum((expected_counter & actual_counter).values())
    denominator = max(len(expected_tools), len(actual_tools))
    return intersection_size / denominator if denominator > 0 else 0.0


def compute_argument_precision(tool_call_details: list, arg_rules: dict) -> float:
    """
    Checks if tool arguments match expected schema/validation rules.

    arg_rules: dict mapping tool_name -> {arg_name: validator_fn}
      e.g. {"inspect_and_visualize_ifc_tool": {"ifc_path": lambda v: v.endswith(".ifc")}}

    Returns ratio of valid arguments to total checked arguments.
    Returns 1.0 ONLY when all checked args pass. Returns 0.0 when the agent
    called tools that have no matching rules at all (prevents inflated scores
    for completely wrong tool selections).
    """
    total_checks = 0
    passed_checks = 0
    has_any_rules = False

    for tc in tool_call_details:
        tool_name = tc.get("tool_name", "")
        args = tc.get("args", {})
        rules = arg_rules.get(tool_name, {})

        if rules:
            has_any_rules = True

        for arg_name, validator in rules.items():
            total_checks += 1
            arg_value = args.get(arg_name)
            try:
                if validator(arg_value):
                    passed_checks += 1
            except Exception:
                pass  # validator crashed = failed check

    if total_checks > 0:
        return passed_checks / total_checks
    # No rules matched any tool called — if the agent called tools but none
    # had validation rules, return 1.0 only if all called tools are known
    if not tool_call_details:
        return 1.0  # no tools called, nothing to check
    if not has_any_rules:
        return 0.0  # agent called tools with zero matching rules → bad signal
    return 1.0


def compute_error_rate(tool_call_details: list) -> float:
    """
    Ratio of tool calls that resulted in errors to total tool calls.
    0.0 = no errors, 1.0 = all errors.
    """
    if not tool_call_details:
        return 0.0
    error_count = sum(1 for tc in tool_call_details if tc.get("status") == "error")
    return error_count / len(tool_call_details)


def compute_redundant_ratio(tool_call_details: list, expected_tools: list = None) -> float:
    """
    Ratio of TRULY redundant (unexpected duplicate) tool calls to total tool calls.
    Accounts for expected duplicates — e.g., run_energyplus_simulation called
    twice is expected in Delhi occupancy evals and should NOT be penalized.

    0.0 = no unnecessary redundancy, >0 = some tools were called more than expected.
    """
    if not tool_call_details:
        return 0.0
    actual_counter = Counter(tc.get("tool_name") for tc in tool_call_details)
    expected_counter = Counter(expected_tools) if expected_tools else Counter()

    # Redundant = actual calls beyond what's expected for each tool
    redundant_count = 0
    for tool, actual_count in actual_counter.items():
        expected_count = expected_counter.get(tool, 1)  # default: expect once
        excess = max(0, actual_count - expected_count)
        redundant_count += excess

    return redundant_count / len(tool_call_details)


def compute_step_efficiency(expected_tools: list, tool_call_details: list) -> float:
    """
    Ratio of optimal steps to actual steps taken.
    1.0 = perfect efficiency (exact number of steps).
    <1.0 = agent took extra unnecessary steps.
    Capped at 1.0 — fewer steps than expected means the agent SKIPPED
    required steps, which is captured by Task Success / Tool Accuracy instead.
    """
    if not tool_call_details:
        return 0.0
    optimal = len(expected_tools)
    actual = len(tool_call_details)
    raw = optimal / actual if actual > 0 else 0.0
    return min(raw, 1.0)  # cap at 1.0 — skipping steps isn't "efficient"


def compute_looping_rate(tool_call_details: list) -> float:
    """
    Detects pathological looping patterns in tool call sequences:
      - Direct repeats: A → A
      - Oscillation: A → B → A → B
    Returns ratio of detected loop steps to total tool calls.
    """
    if len(tool_call_details) < 2:
        return 0.0

    tool_names = [tc.get("tool_name") for tc in tool_call_details]
    loop_count = 0

    # Detect direct consecutive repeats: A → A
    for i in range(1, len(tool_names)):
        if tool_names[i] == tool_names[i - 1]:
            loop_count += 1

    # Detect oscillation: A → B → A → B (pairs repeating)
    for i in range(2, len(tool_names)):
        if (tool_names[i] == tool_names[i - 2]
                and tool_names[i] != tool_names[i - 1]):
            loop_count += 1

    return loop_count / len(tool_names)


# ---------------------------------------------------------------------------
# Single Run Executor
# ---------------------------------------------------------------------------

async def run_single_eval(
    client: httpx.AsyncClient,
    agent_url: str,
    query: str,
    thread_id: str,
    model: str = None,
) -> dict:
    """
    Execute a single evaluation run against the agent.
    Returns the raw JSON response from the agent.
    """
    payload = {
        "message": query,
        "thread_id": thread_id,
        "session_id": thread_id,  # agent.py expects session_id
    }
    if model:
        payload["model"] = model
    response = await client.post(f"{agent_url}/api/chat", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"HTTP {response.status_code}", "response": "", "tools_used": [], "trace": {}}


# ---------------------------------------------------------------------------
# Evaluation Runner (Dual Mode: with-memory & without-memory)
# ---------------------------------------------------------------------------

async def run_evaluation(
    agent_url: str,
    use_case_name: str,
    query: str,
    expected_tool_sequence: list,
    expected_arg_rules: dict,
    num_runs: int = 5,
    sleep_between_runs: int = 15,
    custom_metric_extractor: callable = None,
    run_with_memory: bool = True,
    model: str = None,
    timeout: int = 1800,
):
    """
    Run the full evaluation in two modes:
      1) WITHOUT memory: each run gets a fresh thread_id (independent runs)
      2) WITH memory: all runs share the same thread_id (agent remembers previous runs)

    Returns (no_memory_results, with_memory_results) as lists of dicts.
    """
    print(f"{'=' * 60}")
    print(f"  EVALUATION: {use_case_name}")
    print(f"{'=' * 60}")
    print(f"Query:              '{query}'")
    print(f"Expected Tools:     {expected_tool_sequence}")
    print(f"Runs per mode:      {num_runs}")
    print()

    no_memory_results = []
    with_memory_results = []

    async with httpx.AsyncClient(timeout=float(timeout)) as client:
        # Health check
        try:
            resp = await client.get(f"{agent_url}/api/health")
            if resp.status_code != 200:
                print(f"Agent is not healthy (HTTP {resp.status_code}). Start it with `python agent.py` first.")
                return [], []
        except Exception:
            print(f"Cannot connect to agent at {agent_url}. Start it with `python agent.py` first.")
            return [], []

        # ---------------------------------------------------------------
        # MODE 1: WITHOUT MEMORY (fresh thread_id per run)
        # ---------------------------------------------------------------
        print(f"\n{'─' * 60}")
        print(f"  MODE 1: WITHOUT MEMORY (fresh thread per run)")
        print(f"{'─' * 60}\n")

        for i in range(num_runs):
            run_id = i + 1
            print(f"Run {run_id}/{num_runs}...")

            # Fresh thread_id ensures no memory carryover
            thread_id = f"eval_nomem_{use_case_name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{run_id}"

            start_time = time.time()
            try:
                data = await run_single_eval(client, agent_url, query, thread_id, model=model)
                elapsed_time = time.time() - start_time

                if "error" not in data or data.get("tools_used"):
                    row = _build_result_row(
                        run_id, elapsed_time, data,
                        expected_tool_sequence, expected_arg_rules, "no_memory", custom_metric_extractor
                    )
                else:
                    row = _build_error_row(run_id, elapsed_time, data, expected_tool_sequence, "no_memory", custom_metric_extractor)

                no_memory_results.append(row)
                print(f"  -> {elapsed_time:.2f}s | Success: {row['Success']} | Tools: {row['Actual Tools']}")

            except Exception as e:
                elapsed_time = time.time() - start_time
                no_memory_results.append(_build_exception_row(
                    run_id, elapsed_time, e, expected_tool_sequence, "no_memory"
                ))
                print(f"  -> Error: {e}")

            if i < num_runs - 1:
                print(f"  -> Sleeping {sleep_between_runs}s...")
                await asyncio.sleep(sleep_between_runs)

        # ---------------------------------------------------------------
        # MODE 2: WITH MEMORY (shared thread_id across all runs)
        # ---------------------------------------------------------------
        if run_with_memory:
            print(f"\n{'─' * 60}")
            print(f"  MODE 2: WITH MEMORY (shared thread across runs)")
            print(f"{'─' * 60}\n")
    
            shared_thread_id = f"eval_mem_{use_case_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
            for i in range(num_runs):
                run_id = i + 1
                print(f"Run {run_id}/{num_runs}...")
    
                start_time = time.time()
                try:
                    data = await run_single_eval(client, agent_url, query, shared_thread_id, model=model)
                    elapsed_time = time.time() - start_time
    
                    if "error" not in data or data.get("tools_used"):
                        row = _build_result_row(
                            run_id, elapsed_time, data,
                            expected_tool_sequence, expected_arg_rules, "with_memory", custom_metric_extractor
                        )
                    else:
                        row = _build_error_row(run_id, elapsed_time, data, expected_tool_sequence, "with_memory", custom_metric_extractor)
    
                    with_memory_results.append(row)
                    print(f"  -> {elapsed_time:.2f}s | Success: {row['Success']} | Tools: {row['Actual Tools']}")
    
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    with_memory_results.append(_build_exception_row(
                        run_id, elapsed_time, e, expected_tool_sequence, "with_memory"
                    ))
                    print(f"  -> Error: {e}")
    
                if i < num_runs - 1:
                    print(f"  -> Sleeping {sleep_between_runs}s...")
                    await asyncio.sleep(sleep_between_runs)

    return no_memory_results, with_memory_results


# ---------------------------------------------------------------------------
# Result Row Builders
# ---------------------------------------------------------------------------

def _build_result_row(run_id, elapsed_time, data, expected_tools, arg_rules, mode, custom_metric_extractor=None):
    """Build a result dict from a successful agent response."""
    tools_used_data = data.get("tools_used", [])
    trace = data.get("trace", {})
    tool_call_details = trace.get("tool_call_details", [])
    agent_reply = data.get("response", "")
    token_usage = trace.get("token_usage", {})

    # agent.py deduplicates tools_used_data. To get the TRUE sequence with duplicates,
    # we pull the names from tool_call_details in the trace.
    if tool_call_details:
        actual_tools = [tc.get("tool_name") for tc in tool_call_details]
    else:
        actual_tools = [t.get("name") for t in tools_used_data]

    # Core success: evaluate the complete start-to-end tool sequence exactly
    success = (actual_tools == expected_tools)

    # Task success: uses multiset comparison — all expected tool calls
    # (including expected duplicates) must be present in actual calls
    task_success = not (Counter(expected_tools) - Counter(actual_tools))

    # Compute all metrics
    tool_accuracy = compute_tool_selection_accuracy(expected_tools, actual_tools)
    arg_precision = compute_argument_precision(tool_call_details, arg_rules)
    error_rate = compute_error_rate(tool_call_details)
    redundant_ratio = compute_redundant_ratio(tool_call_details, expected_tools)
    step_efficiency = compute_step_efficiency(expected_tools, tool_call_details)
    looping_rate = compute_looping_rate(tool_call_details)
    llm_steps = trace.get("total_llm_calls", 0)

    result = {
        "Run ID": run_id,
        "Mode": mode,
        "Time Taken (s)": round(elapsed_time, 2),
        "Success": success,
        "Task Success": task_success,
        "Expected Tools": ", ".join(expected_tools),
        "Actual Tools": ", ".join(actual_tools),
        "Tool Selection Accuracy": round(tool_accuracy, 4),
        "Argument Precision": round(arg_precision, 4),
        "Tool Error Rate": round(error_rate, 4),
        "Redundant Call Ratio": round(redundant_ratio, 4),
        "Step Efficiency": round(step_efficiency, 4),
        "Looping Rate": round(looping_rate, 4),
        "LLM Steps": llm_steps,
        "Total Tool Calls": trace.get("total_tool_calls", len(actual_tools)),
        "Prompt Tokens": token_usage.get("prompt_tokens") or 0,
        "Completion Tokens": token_usage.get("completion_tokens") or 0,
        "Agent Reply Snippet": agent_reply[:100].replace('\n', ' ') + "...",
    }

    if custom_metric_extractor:
        try:
            result.update(custom_metric_extractor(data))
        except Exception as e:
            print(f"Error in custom metric extractor: {e}")

    return result


def _build_error_row(run_id, elapsed_time, data, expected_tools, mode, custom_metric_extractor=None):
    """Build a result dict for an HTTP-level error (no tools were called)."""
    result = {
        "Run ID": run_id,
        "Mode": mode,
        "Time Taken (s)": round(elapsed_time, 2),
        "Success": False,
        "Task Success": False,
        "Expected Tools": ", ".join(expected_tools),
        "Actual Tools": data.get("error", "HTTP Error"),
        "Tool Selection Accuracy": 0.0,
        "Argument Precision": 0.0,
        "Tool Error Rate": 0.0,  # No tools were called — don't blame tools for infra errors
        "Redundant Call Ratio": 0.0,
        "Step Efficiency": 0.0,
        "Looping Rate": 0.0,
        "LLM Steps": 0,
        "Total Tool Calls": 0,
        "Prompt Tokens": 0,
        "Completion Tokens": 0,
        "Agent Reply Snippet": "",
    }

    if custom_metric_extractor:
        try:
            result.update(custom_metric_extractor(data))
        except Exception:
            pass

    return result


def _build_exception_row(run_id, elapsed_time, exception, expected_tools, mode):
    """Build a result dict for a Python-level exception (timeout, network, etc.)."""
    return {
        "Run ID": run_id,
        "Mode": mode,
        "Time Taken (s)": round(elapsed_time, 2),
        "Success": False,
        "Task Success": False,
        "Expected Tools": ", ".join(expected_tools),
        "Actual Tools": f"Exception: {type(exception).__name__}",
        "Tool Selection Accuracy": 0.0,
        "Argument Precision": 0.0,
        "Tool Error Rate": 0.0,  # No tools were called — infra error, not tool error
        "Redundant Call Ratio": 0.0,
        "Step Efficiency": 0.0,
        "Looping Rate": 0.0,
        "LLM Steps": 0,
        "Total Tool Calls": 0,
        "Prompt Tokens": 0,
        "Completion Tokens": 0,
        "Agent Reply Snippet": str(exception)[:100],
    }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(
    no_memory_results: list,
    with_memory_results: list,
    use_case_name: str,
    reports_dir: str,
):
    """Generate and print evaluation reports for both modes, and save combined CSV."""

    df_no_mem = pd.DataFrame(no_memory_results)
    df_with_mem = pd.DataFrame(with_memory_results)

    metric_cols = [
        "Tool Selection Accuracy", "Argument Precision", "Tool Error Rate",
        "Redundant Call Ratio", "Step Efficiency", "Looping Rate", "LLM Steps",
    ]

    display_cols = [
        "Run ID", "Success", "Task Success", "Time Taken (s)",
        "Tool Selection Accuracy", "Argument Precision", "Tool Error Rate",
        "Redundant Call Ratio", "Step Efficiency", "Looping Rate", "LLM Steps",
        "Expected Tools", "Actual Tools",
    ]

    # Dynamically detect any custom metrics added to the DataFrame
    all_cols = list(df_no_mem.columns) if not df_no_mem.empty else (list(df_with_mem.columns) if not df_with_mem.empty else [])
    custom_cols = [c for c in all_cols if c not in display_cols and c not in ("Mode", "Agent Reply Snippet", "Total Tool Calls")]
    display_cols.extend(custom_cols)

    for label, df in [("WITHOUT MEMORY", df_no_mem), ("WITH MEMORY", df_with_mem)]:
        total = len(df)
        if total == 0:
            continue

        success_rate = (df["Success"].sum() / total) * 100
        task_success_rate = (df["Task Success"].sum() / total) * 100
        avg_time = df["Time Taken (s)"].mean()

        print(f"\n{'=' * 60}")
        print(f"  {label} — EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Use Case:               {use_case_name}")
        print(f"  Total Runs:             {total}")
        print(f"  Strict Sequence Match:  {success_rate:.2f}%")
        print(f"  Task Success Rate:      {task_success_rate:.2f}%")
        print(f"  Avg Time Taken:         {avg_time:.2f}s")

        for col in metric_cols:
            if col in df.columns:
                avg_val = df[col].mean()
                if col in ("Tool Selection Accuracy", "Argument Precision", "Step Efficiency"):
                    print(f"  Avg {col}:  {avg_val:.4f}")
                elif col == "LLM Steps":
                    print(f"  Avg {col}:            {avg_val:.1f}")
                else:
                    print(f"  Avg {col}:   {avg_val:.4f}")

        print(f"{'=' * 60}")
        print(f"\n  Individual Runs ({label}):")
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols].to_markdown(index=False))

    # Save combined CSV
    os.makedirs(reports_dir, exist_ok=True)
    combined_df = pd.concat([df_no_mem, df_with_mem], ignore_index=True)
    report_filename = f"{use_case_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_path = os.path.join(reports_dir, report_filename)
    combined_df.to_csv(report_path, index=False)
    print(f"\nDetailed report saved to {report_path}")
