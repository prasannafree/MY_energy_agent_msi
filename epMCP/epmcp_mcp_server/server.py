"""
epMCP MCP Server — Exposes epMCP tools via the Model Context Protocol.

Runs inside the energyplus-mcp-dev Docker container using stdio transport.
Provides occupancy modification, simulation, calibration, and surrogate tools.

Usage (inside Docker):
    uv run --no-dev python -m epmcp_mcp_server.server
"""

import os
import json
import logging
from typing import Optional, List

from mcp.server.fastmcp import FastMCP

from epmcp_mcp_server.tools import (
    calculate_rmse,
    calibrate_occupancy,
    inspect_and_visualize_ifc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNS rebinding protection — same pattern as the EnergyPlus-MCP server
# ---------------------------------------------------------------------------
_allowed_hosts_env = os.getenv("MCP_ALLOWED_HOSTS", "").strip()
_mcp_kwargs: dict = {}

if _allowed_hosts_env:
    from mcp.server.transport_security import TransportSecuritySettings
    _transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[h.strip() for h in _allowed_hosts_env.split(",") if h.strip()],
    )
    _mcp_kwargs["transport_security"] = _transport_security
else:
    try:
        from mcp.server.transport_security import TransportSecuritySettings
        _transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        _mcp_kwargs["transport_security"] = _transport_security
    except ImportError:
        pass

mcp = FastMCP("epMCP-tools", **_mcp_kwargs)

logger.info("epMCP MCP Server initialized")


# ---------------------------------------------------------------------------
# Tool 1: calculate_rmse
# ---------------------------------------------------------------------------
@mcp.tool()
async def calculate_rmse_tool(
    target_csv_path: str,
    simulation_csv_path: str,
    column_name: str = "Whole Building:Facility Total Electricity Demand Rate [W](Hourly)",
) -> str:
    """
    Calculate RMSE (Root Mean Square Error) between a target CSV and a simulation CSV.

    Compares hourly electricity demand profiles to quantify simulation accuracy.
    Automatically detects the electricity demand column if exact name doesn't match.

    Args:
        target_csv_path: Path to the ground truth / measured data CSV
        simulation_csv_path: Path to the EnergyPlus simulation output CSV
        column_name: Column to compare (auto-detected if not found)

    Returns:
        JSON string with RMSE, MAE, CV-RMSE percentage, and data point count

    Examples:
        calculate_rmse_tool("/workspace/measured.csv", "/workspace/outputs/eplusout.csv")
    """
    try:
        logger.info(f"Calculating RMSE: {target_csv_path} vs {simulation_csv_path}")
        result = calculate_rmse(target_csv_path, simulation_csv_path, column_name)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error calculating RMSE: {e}", exc_info=True)
        return f"Error calculating RMSE: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: calibrate_occupancy
# ---------------------------------------------------------------------------
@mcp.tool()
async def calibrate_occupancy_tool(
    idf_path: str,
    epw_path: str,
    target_csv_path: str,
    output_dir: str,
    max_iterations: int = 25,
    multiplier_tolerance: float = 0.01,
    rmse_tolerance: float = 5.0,
    initial_guess: float = 1.0,
    search_min: float = 0.1,
    search_max: float = 5.0,
) -> str:
    """
    Automated calibration of occupancy multiplier using Nelder-Mead optimization.

    Iteratively adjusts the occupancy multiplier, runs EnergyPlus simulations,
    and compares results against measured/target data until the RMSE converges.

    WARNING: This tool runs multiple EnergyPlus simulations and can take significant time.

    Args:
        idf_path: Path to the baseline IDF model
        epw_path: Path to the EPW weather file
        target_csv_path: Path to the ground truth / measured data CSV
        output_dir: Working directory for intermediate simulation files
        max_iterations: Maximum number of optimization iterations (default: 25)
        multiplier_tolerance: Convergence tolerance for the multiplier value (default: 0.01)
        rmse_tolerance: Convergence tolerance for RMSE in watts (default: 5.0)
        initial_guess: Starting multiplier value (default: 1.0)
        search_min: Minimum allowed multiplier (default: 0.1)
        search_max: Maximum allowed multiplier (default: 5.0)

    Returns:
        JSON string with optimal multiplier, final RMSE, iteration log

    Examples:
        calibrate_occupancy_tool(
            "/workspace/model.idf",
            "/workspace/weather.epw",
            "/workspace/measured_data.csv",
            "/workspace/calibration_output"
        )
    """
    try:
        logger.info(f"Starting occupancy calibration: {idf_path}")
        result = calibrate_occupancy(
            idf_path, epw_path, target_csv_path, output_dir,
            max_iterations, multiplier_tolerance, rmse_tolerance,
            initial_guess, search_min, search_max,
        )
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error during calibration: {e}", exc_info=True)
        return f"Error during calibration: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: inspect_and_visualize_ifc
# ---------------------------------------------------------------------------
@mcp.tool()
async def inspect_and_visualize_ifc_tool(
    ifc_path: str,
    output_dir: str = "/workspace/outputs",
    output_html_name: str = "ifc_3d_visualization.html",
) -> str:
    """
    Inspect an IFC building model, extract storeys, spaces, and element metadata,
    and generate an interactive 3D HTML visualization file.

    Args:
        ifc_path: Path to the input .ifc file (e.g. /workspace/all_files/model.ifc)
        output_dir: Directory to save the 3D HTML visualization (default: /workspace/outputs)
        output_html_name: Name of output HTML visualization file (default: ifc_3d_visualization.html)

    Returns:
        JSON string with IFC metadata summary, storeys, spaces, element counts,
        and generated 3D HTML visualization path.

    Examples:
        inspect_and_visualize_ifc_tool("/workspace/all_files/20160414office_model_CV2_fordesign.ifc")
    """
    try:
        logger.info(f"Inspecting and visualizing IFC model: {ifc_path}")
        result = inspect_and_visualize_ifc(ifc_path, output_dir, output_html_name)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error inspecting IFC model: {e}", exc_info=True)
        return f"Error inspecting IFC model: {str(e)}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting epMCP MCP Server (stdio transport)")
    mcp.run(transport="stdio")

