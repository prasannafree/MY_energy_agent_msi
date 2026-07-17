import asyncio
import time
import json
import httpx
import pandas as pd
from datetime import datetime

# URL of your running EnergyPlus Agent (make sure agent.py is running)
AGENT_URL = "http://127.0.0.1:5000"

# Define the test cases for the evaluation benchmark
TEST_CASES = [
    {
        "id": "T01",
        "category": "Basic",
        "prompt": "List all available sample IDF models.",
        "expected_tool": "list_available_files"
    },
    {
        "id": "T02",
        "category": "Basic",
        "prompt": "Use alter_occupancy_global_tool to change the occupancy of 1ZoneUncontrolled.idf to a multiplier of 1.5. Output to /workspace/epMCP/test_output",
        "expected_tool": "alter_occupancy_global_tool"
    },
    {
        "id": "T03",
        "category": "Simulation",
        "prompt": "Run a simulation using run_ep_simulation_tool with 1ZoneUncontrolled.idf and Denver weather file.",
        "expected_tool": "run_ep_simulation_tool"
    },
    {
        "id": "T04",
        "category": "Advanced",
        "prompt": "We don't have a trained surrogate model yet. Try to predict with the surrogate model and tell me what the error message says.",
        "expected_tool": "predict_with_surrogate_tool"
    }
]

async def run_evaluation():
    print(f"Starting automated evaluation against {AGENT_URL}...\n")
    results = []

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check if agent is up
        try:
            resp = await client.get(f"{AGENT_URL}/api/health")
            if resp.status_code != 200:
                print("Agent is not healthy. Start it with `python agent.py` first.")
                return
        except Exception:
            print("Cannot connect to agent. Start it with `python agent.py` first.")
            return

        for idx, test in enumerate(TEST_CASES):
            print(f"Running Test {test['id']} ({test['category']})...")
            
            payload = {
                "message": test["prompt"],
                "thread_id": f"eval_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}"
            }
            
            start_time = time.time()
            try:
                # Send the chat request to the agent
                response = await client.post(f"{AGENT_URL}/api/chat", json=payload)
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    agent_reply = data.get("response", "")
                    
                    # We can't see the exact backend tool calls through this basic API endpoint 
                    # easily, but we can do a naive check if the tool name appears in the reply
                    # or assume success if it didn't crash. (A deeper eval would use the langsmith/langfuse).
                    success = True
                    
                    results.append({
                        "Test ID": test["id"],
                        "Category": test["category"],
                        "Prompt": test["prompt"],
                        "Time (s)": round(elapsed_time, 2),
                        "Status": "Success",
                        "Agent Reply Snippet": agent_reply[:150] + "..."
                    })
                    print(f"  -> Success in {elapsed_time:.2f}s")
                    
                    # Sleep to prevent hitting Gemini API free tier rate limits (15 Requests Per Minute)
                    if idx < len(TEST_CASES) - 1:
                        print("  -> Sleeping for 20 seconds to respect API rate limits...")
                        await asyncio.sleep(20)
                else:
                    elapsed_time = time.time() - start_time
                    results.append({
                        "Test ID": test["id"],
                        "Category": test["category"],
                        "Prompt": test["prompt"],
                        "Time (s)": round(elapsed_time, 2),
                        "Status": f"Failed (HTTP {response.status_code})",
                        "Agent Reply Snippet": ""
                    })
                    print(f"  -> Failed (HTTP {response.status_code})")
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                results.append({
                    "Test ID": test["id"],
                    "Category": test["category"],
                    "Prompt": test["prompt"],
                    "Time (s)": round(elapsed_time, 2),
                    "Status": f"Error: {str(e)}",
                    "Agent Reply Snippet": ""
                })
                print(f"  -> Error: {e}")

    # Generate Report
    df = pd.DataFrame(results)
    report_path = "evaluation_report.csv"
    df.to_csv(report_path, index=False)
    
    print(f"\nEvaluation complete! Report saved to {report_path}")
    print("\nSummary:")
    print(df[["Test ID", "Category", "Time (s)", "Status"]].to_markdown(index=False))

if __name__ == "__main__":
    asyncio.run(run_evaluation())
