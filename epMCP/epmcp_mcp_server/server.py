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
    alter_occupancy_global,
    run_ep_simulation,
    calculate_rmse,
    calibrate_occupancy,
    train_surrogate_model,
    predict_with_surrogate,
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
# Tool 1: alter_occupancy_global
# ---------------------------------------------------------------------------
@mcp.tool()
async def alter_occupancy_global_tool(
    idf_path: str,
    multiplier: float,
    output_dir: str,
    output_filename: str = "occupancy_modified.idf",
) -> str:
    """
    Scale occupancy (People objects) in ALL zones of an EnergyPlus IDF model by a multiplier.

    Handles all three EnergyPlus occupancy calculation methods:
    - People: Number_of_People * multiplier
    - People/Area: People_per_Floor_Area * multiplier
    - Area/Person: Floor_Area_per_Person / multiplier

    Args:
        idf_path: Path to the input IDF file
        multiplier: Factor to scale occupancy by (e.g., 1.5 for 50% increase, 0.5 for 50% decrease)
        output_dir: Directory to save the modified IDF file
        output_filename: Name for the output file (default: occupancy_modified.idf)

    Returns:
        JSON string with modification summary including old/new values per zone

    Examples:
        # Increase occupancy by 50%
        alter_occupancy_global_tool("/workspace/sample_files/model.idf", 1.5, "/workspace/outputs")

        # Decrease occupancy by 30%
        alter_occupancy_global_tool("/workspace/sample_files/model.idf", 0.7, "/workspace/outputs")
    """
    try:
        logger.info(f"Altering occupancy: {idf_path} x{multiplier}")
        result = alter_occupancy_global(idf_path, multiplier, output_dir, output_filename)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error altering occupancy: {e}", exc_info=True)
        return f"Error altering occupancy: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: run_ep_simulation
# ---------------------------------------------------------------------------
@mcp.tool()
async def run_ep_simulation_tool(
    idf_path: str,
    epw_path: str,
    output_dir: str,
    start_month: Optional[int] = None,
    start_day: Optional[int] = None,
    end_month: Optional[int] = None,
    end_day: Optional[int] = None,
) -> str:
    """
    Run an EnergyPlus simulation using geomeppy and produce output CSV.

    Executes the simulation with hourly Facility Total Electricity Demand Rate tracking.
    Optionally overrides the RunPeriod to a custom date range.

    Args:
        idf_path: Path to the IDF model file
        epw_path: Path to the EPW weather file
        output_dir: Directory for simulation output files
        start_month: Optional start month (1-12) for custom run period
        start_day: Optional start day (1-31) for custom run period
        end_month: Optional end month (1-12) for custom run period
        end_day: Optional end day (1-31) for custom run period

    Returns:
        JSON string with simulation results including output CSV path and row count

    Examples:
        # Full year simulation
        run_ep_simulation_tool("/workspace/model.idf", "/workspace/weather.epw", "/workspace/outputs")

        # January only
        run_ep_simulation_tool("/workspace/model.idf", "/workspace/weather.epw", "/workspace/outputs", 1, 1, 1, 31)
    """
    try:
        logger.info(f"Running simulation: {idf_path}")
        result = run_ep_simulation(
            idf_path, epw_path, output_dir,
            start_month, start_day, end_month, end_day,
        )
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error running simulation: {e}", exc_info=True)
        return f"Error running simulation: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: calculate_rmse
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
# Tool 4: calibrate_occupancy
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
# Tool 5a: train_surrogate_model
# ---------------------------------------------------------------------------
@mcp.tool()
async def train_surrogate_model_tool(
    cache_npz_path: str,
    epw_path: str,
    idf_path: Optional[str] = None,
) -> str:
    """
    Train a surrogate model from pre-computed EnergyPlus simulation cache data.

    Uses a Ridge regression model trained on simulation results to enable
    sub-second occupancy calibration without running additional EnergyPlus simulations.

    The cache NPZ file must contain:
    - 'multipliers': array of occupancy multipliers used in simulations
    - 'y_data': 2D array of hourly demand curves (shape: n_simulations x 8760)

    Args:
        cache_npz_path: Path to the simulation cache .npz file
        epw_path: Path to the EPW weather file (used for weather feature extraction)
        idf_path: Optional path to baseline IDF (not required if cache has baseline)

    Returns:
        JSON string with training summary including R² score

    Examples:
        train_surrogate_model_tool("/workspace/cache/simulation_cache.npz", "/workspace/weather.epw")
    """
    try:
        logger.info(f"Training surrogate model from: {cache_npz_path}")
        result = train_surrogate_model(cache_npz_path, epw_path, idf_path)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error training surrogate: {e}", exc_info=True)
        return f"Error training surrogate: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 5b: predict_with_surrogate
# ---------------------------------------------------------------------------
@mcp.tool()
async def predict_with_surrogate_tool(
    target_csv_path: Optional[str] = None,
    initial_guess: float = 0.0,
) -> str:
    """
    Use the trained surrogate model for instant occupancy calibration.

    Requires train_surrogate_model_tool to have been called first in this session.
    Runs Nelder-Mead optimization against the surrogate model predictions instead
    of full EnergyPlus simulations, achieving sub-second calibration.

    Args:
        target_csv_path: Optional path to measured/target data CSV. If not provided,
                         runs a demo with synthetic noise on the baseline.
        initial_guess: Starting multiplier for optimization (default: 0.0)

    Returns:
        JSON string with optimal multiplier, RMSE, and convergence time

    Examples:
        # Calibrate against measured data
        predict_with_surrogate_tool("/workspace/measured_data.csv")

        # Demo mode
        predict_with_surrogate_tool()
    """
    try:
        logger.info("Running surrogate prediction")
        result = predict_with_surrogate(target_csv_path, initial_guess)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in surrogate prediction: {e}", exc_info=True)
        return f"Error in surrogate prediction: {str(e)}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting epMCP MCP Server (stdio transport)")
    mcp.run(transport="stdio")
