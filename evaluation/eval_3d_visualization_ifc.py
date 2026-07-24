import asyncio
import time
import httpx
import pandas as pd
from datetime import datetime
import os

# Configuration for this specific evaluation
AGENT_URL = "http://127.0.0.1:5000"
USE_CASE_NAME = "3d_visualization_ifc"
QUERY = "Generate an interactive 3D HTML visualization for the sample IFC file."
EXPECTED_TOOL_SEQUENCE = ["list_available_files", "inspect_and_visualize_ifc_tool"]
NUM_RUNS = 5

async def run_evaluation():
    print(f"Starting evaluation for use case: {USE_CASE_NAME}")
    print(f"Query: '{QUERY}'")
    print(f"Expected Tool Sequence: {EXPECTED_TOOL_SEQUENCE}")
    print(f"Number of runs: {NUM_RUNS}\n")

    results = []
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check if agent is up
        try:
            resp = await client.get(f"{AGENT_URL}/api/health")
            if resp.status_code != 200:
                print(f"Agent is not healthy (HTTP {resp.status_code}). Start it with `python agent.py` first.")
                return
        except Exception:
            print(f"Cannot connect to agent at {AGENT_URL}. Start it with `python agent.py` first.")
            return
            
        for i in range(NUM_RUNS):
            run_id = i + 1
            print(f"Run {run_id}/{NUM_RUNS}...")
            
            payload = {
                "message": QUERY,
                "thread_id": f"eval_{USE_CASE_NAME}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{run_id}"
            }
            
            start_time = time.time()
            try:
                response = await client.post(f"{AGENT_URL}/api/chat", json=payload)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    agent_reply = data.get("response", "")
                    tools_used_data = data.get("tools_used", [])
                    
                    # Extract the sequence of tool names called by the agent
                    actual_tools = [t.get("name") for t in tools_used_data]
                    
                    # Check if the expected tools were called (using set comparison to ignore order, as parallel tools can return in any order)
                    success = (set(actual_tools) == set(EXPECTED_TOOL_SEQUENCE))
                    
                    results.append({
                        "Run ID": run_id,
                        "Time Taken (s)": round(elapsed_time, 2),
                        "Success": success,
                        "Expected Tools": ", ".join(EXPECTED_TOOL_SEQUENCE),
                        "Actual Tools": ", ".join(actual_tools),
                        "Agent Reply Snippet": agent_reply[:100].replace('\n', ' ') + "..."
                    })
                    print(f"  -> Finished in {elapsed_time:.2f}s. Success: {success}.")
                    print(f"  -> Tools Used: {actual_tools}")
                    
                else:
                    elapsed_time = time.time() - start_time
                    results.append({
                        "Run ID": run_id,
                        "Time Taken (s)": round(elapsed_time, 2),
                        "Success": False,
                        "Expected Tools": ", ".join(EXPECTED_TOOL_SEQUENCE),
                        "Actual Tools": f"HTTP Error {response.status_code}",
                        "Agent Reply Snippet": ""
                    })
                    print(f"  -> Failed (HTTP {response.status_code}) in {elapsed_time:.2f}s")
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                results.append({
                    "Run ID": run_id,
                    "Time Taken (s)": round(elapsed_time, 2),
                    "Success": False,
                    "Expected Tools": ", ".join(EXPECTED_TOOL_SEQUENCE),
                    "Actual Tools": f"Exception: {type(e).__name__}",
                    "Agent Reply Snippet": str(e)[:100]
                })
                print(f"  -> Error: {e}")
                
            # Sleep to prevent rate limits on LLM APIs, unless it's the last run
            if i < NUM_RUNS - 1:
                print("  -> Sleeping for 15 seconds to respect API rate limits...")
                await asyncio.sleep(15)

    # Generate Report
    df = pd.DataFrame(results)
    
    # Calculate summary metrics
    total_runs = len(df)
    success_rate = (df["Success"].sum() / total_runs) * 100 if total_runs > 0 else 0
    avg_time = df["Time Taken (s)"].mean()
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Use Case:           {USE_CASE_NAME}")
    print(f"Total Runs:         {total_runs}")
    print(f"Success Rate:       {success_rate:.2f}%")
    print(f"Average Time Taken: {avg_time:.2f}s")
    print("="*50 + "\n")
    
    # Ensure reports directory exists relative to this script or workspace
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(workspace_dir, "evaluation_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    report_filename = f"{USE_CASE_NAME}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_path = os.path.join(reports_dir, report_filename)
    
    df.to_csv(report_path, index=False)
    
    print(f"Detailed report saved to {report_path}")
    print("\nIndividual Runs:")
    # Print a nice markdown table to console
    print(df[["Run ID", "Success", "Time Taken (s)", "Expected Tools", "Actual Tools"]].to_markdown(index=False))

if __name__ == "__main__":
    asyncio.run(run_evaluation())
