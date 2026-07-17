import sys
from pathlib import Path

# Add paths to allow import
sys.path.insert(0, "/workspace/epMCP")

from epmcp_mcp_server.tools import run_ep_simulation

try:
    result = run_ep_simulation(
        idf_path="/workspace/energyplus-mcp-server/sample_files/1ZoneUncontrolled.idf",
        epw_path="/workspace/epMCP/sample_models/USA_CO_Denver.Intl.AP.725650_TMY3.epw",
        output_dir="/workspace/epMCP/test_output"
    )
    print("Result:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
