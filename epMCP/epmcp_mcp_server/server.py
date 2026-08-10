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
    calculate_gross_floor_area,
    extract_annual_energy_kwh,
    get_epi_benchmark,
    calculate_epi,
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
    max_iterations: int = 20,
    multiplier_tolerance: float = 0.01,
    rmse_tolerance: float = 15.0,
    initial_guess: float = 1.0,
    search_min: float = 0.1,
    search_max: float = 5.0,
    optimization_method: str = "Nelder-Mead",
) -> str:
    """
    Automated calibration of occupancy multiplier using advanced optimization methods.

    Iteratively adjusts the occupancy multiplier, runs EnergyPlus simulations,
    and compares results against measured/target data until the RMSE converges.

    WARNING: This tool runs multiple EnergyPlus simulations and can take significant time.

    Args:
        idf_path: Path to the baseline IDF model
        epw_path: Path to the EPW weather file
        target_csv_path: Path to the ground truth / measured data CSV
        output_dir: Working directory for intermediate simulation files
        max_iterations: Maximum number of optimization iterations (default: 20, strictly capped)
        multiplier_tolerance: Convergence tolerance for the multiplier value (default: 0.01)
        rmse_tolerance: Convergence tolerance for RMSE in watts (default: 15.0)
        initial_guess: Starting multiplier value (default: 1.0)
        search_min: Minimum allowed multiplier (default: 0.1)
        search_max: Maximum allowed multiplier (default: 5.0)
        optimization_method: Choice of optimizer (e.g. 'Nelder-Mead', 'Bayesian Optimization (Gaussian Process)', 'Differential Evolution (Genetic Algorithm)', 'Particle Swarm Optimization (PSO)')

    Returns:
        JSON string with optimal multiplier, final RMSE, iteration log

    Examples:
        calibrate_occupancy_tool(
            "/workspace/model.idf",
            "/workspace/weather.epw",
            "/workspace/measured_data.csv",
            "/workspace/calibration_output",
            optimization_method="Particle Swarm Optimization (PSO)"
        )
    """
    try:
        logger.info(f"Starting occupancy calibration: {idf_path} with {optimization_method}")
        result = calibrate_occupancy(
            idf_path, epw_path, target_csv_path, output_dir,
            max_iterations, multiplier_tolerance, rmse_tolerance,
            initial_guess, search_min, search_max, optimization_method
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
        inspect_and_visualize_ifc_tool("/workspace/all_files/model.ifc")
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
# Tool 4: calculate_gross_floor_area
# ---------------------------------------------------------------------------
@mcp.tool()
async def calculate_gross_floor_area_tool(idf_path: str) -> str:
    """
    Calculate total gross conditioned floor area (m²) from IDF building geometry.

    Uses geomeppy surface geometry analysis with multiple fallback strategies.
    Returns per-zone floor area breakdown.

    Args:
        idf_path: Path to the EnergyPlus IDF file

    Returns:
        JSON string with total floor area, method used, and per-zone breakdown

    Examples:
        calculate_gross_floor_area_tool("/workspace/all_files/5ZoneAirCooled.idf")
    """
    try:
        logger.info(f"Calculating gross floor area: {idf_path}")
        result = calculate_gross_floor_area(idf_path)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error calculating floor area: {e}", exc_info=True)
        return f"Error calculating floor area: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 5: extract_annual_energy_kwh
# ---------------------------------------------------------------------------
@mcp.tool()
async def extract_annual_energy_tool(output_directory: str) -> str:
    """
    Extract annual energy consumption (kWh) from EnergyPlus simulation output CSV.

    Parses meter output files (eplusmeter.csv or eplusout.csv) and converts
    values from Joules to kWh. Detects Electricity:Facility and NaturalGas:Facility.

    Args:
        output_directory: Path to the directory containing EnergyPlus CSV output files

    Returns:
        JSON string with electricity_kwh, gas_kwh, total_kwh, and source info

    Examples:
        extract_annual_energy_tool("/workspace/outputs/simulation_run_01")
    """
    try:
        logger.info(f"Extracting annual energy from: {output_directory}")
        result = extract_annual_energy_kwh(output_directory)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error extracting energy data: {e}", exc_info=True)
        return f"Error extracting energy data: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 6: get_epi_benchmark
# ---------------------------------------------------------------------------
@mcp.tool()
async def get_epi_benchmark_tool(
    building_type: str,
    climate_zone: str,
) -> str:
    """
    Look up the code-compliant EPI benchmark (kWh/m²/year) for a building type
    and climate zone.

    Supported building types: apartment_highrise, office, hospital, school,
    retail, hotel.
    Supported climate zones: cold, composite, hot_dry, warm_humid, temperate.

    Args:
        building_type: Type of building (e.g. 'office', 'hospital', 'apartment_highrise')
        climate_zone: Climate zone (e.g. 'composite', 'cold', 'hot_dry')

    Returns:
        JSON string with benchmark EPI value and available options

    Examples:
        get_epi_benchmark_tool("office", "composite")
        get_epi_benchmark_tool("hospital", "hot_dry")
    """
    try:
        logger.info(f"Getting EPI benchmark: {building_type} / {climate_zone}")
        result = get_epi_benchmark(building_type, climate_zone)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting EPI benchmark: {e}", exc_info=True)
        return f"Error getting EPI benchmark: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 7: calculate_epi
# ---------------------------------------------------------------------------
@mcp.tool()
async def calculate_epi_tool(
    idf_path: str,
    simulation_output_dir: str,
    building_type: str = "office",
    climate_zone: str = "composite",
) -> str:
    """
    Compute the Energy Performance Index (EPI) for a building.

    This is a comprehensive tool that combines floor area calculation from IDF
    geometry, annual energy extraction from simulation output, benchmark lookup,
    and code compliance evaluation.

    Prerequisites: A simulation must have been run first (use run_energyplus_simulation)
    and output CSVs must exist in simulation_output_dir.

    Args:
        idf_path: Path to the IDF file (for floor area calculation)
        simulation_output_dir: Directory containing EnergyPlus simulation CSV output
        building_type: Building type for benchmark lookup
                       (apartment_highrise, office, hospital, school, retail, hotel)
        climate_zone: Climate zone for benchmark lookup
                      (cold, composite, hot_dry, warm_humid, temperate)

    Returns:
        JSON string with proposed EPI, benchmark EPI, EPI ratio, compliance status,
        energy breakdown, floor area, and performance summary

    Examples:
        calculate_epi_tool(
            "/workspace/all_files/5ZoneAirCooled.idf",
            "/workspace/outputs/sim_run",
            "office",
            "composite"
        )
    """
    try:
        logger.info(
            f"Calculating EPI: {idf_path} | output={simulation_output_dir} | "
            f"type={building_type} | climate={climate_zone}"
        )
        result = calculate_epi(idf_path, simulation_output_dir, building_type, climate_zone)
        return json.dumps(result, indent=2)
    except FileNotFoundError as e:
        return f"File not found: {str(e)}"
    except Exception as e:
        logger.error(f"Error calculating EPI: {e}", exc_info=True)
        return f"Error calculating EPI: {str(e)}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting epMCP MCP Server (stdio transport)")
    mcp.run(transport="stdio")

