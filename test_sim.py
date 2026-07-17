import sys
from pathlib import Path

# Add paths to allow import
sys.path.insert(0, str(Path("epMCP").resolve()))

from epmcp_mcp_server.tools import run_ep_simulation

result = run_ep_simulation(
    idf_path="EnergyPlus-MCP/energyplus-mcp-server/sample_files/1ZoneUncontrolled.idf",
    epw_path="epMCP/sample_models/USA_CO_Denver.Intl.AP.725650_TMY3.epw",
    output_dir="epMCP/test_output"
)
print("Result:", result)
