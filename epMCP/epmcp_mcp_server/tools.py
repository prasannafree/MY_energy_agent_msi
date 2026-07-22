"""
epMCP Core Tools — Clean computational logic extracted from the epMCP Streamlit apps.

All functions are self-contained and accept explicit paths (no hardcoded paths,
no Streamlit dependencies). Designed to run inside the energyplus-mcp-dev Docker
container which has EnergyPlus 26.1.0 at /app/software/EnergyPlusV26-1-0/.
"""

import os
import sys
import copy
import json
import logging
import time
import types
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Tkinter shim — geomeppy imports tkinter for its optional 3D geometry viewer.
# The Docker container doesn't have Tk/X11 libraries. We mock the low-level
# _tkinter C extension so that `import tkinter` succeeds but the viewer is
# simply unavailable. This does NOT affect IDF read/write/simulation.
# ---------------------------------------------------------------------------
if "_tkinter" not in sys.modules:
    _fake_tk = types.ModuleType("_tkinter")
    _fake_tk.TclError = type("TclError", (Exception,), {})
    _fake_tk.TK_VERSION = "8.6"
    _fake_tk.TCL_VERSION = "8.6"
    _fake_tk.READABLE = 1
    _fake_tk.WRITABLE = 2
    _fake_tk.EXCEPTION = 4
    sys.modules["_tkinter"] = _fake_tk

from geomeppy import IDF
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IDD Setup — auto-detect from Docker container or environment
# ---------------------------------------------------------------------------
_IDD_INITIALIZED = False

def _ensure_idd():
    """Set the EnergyPlus IDD path once. Checks env var, then Docker default."""
    global _IDD_INITIALIZED
    if _IDD_INITIALIZED:
        return

    idd_candidates = [
        os.environ.get("EPLUS_IDD_PATH", ""),
        "/workspace/all_files/Energy+.idd",
        "/app/software/EnergyPlusV26-1-0/Energy+.idd",
        "/workspace/epMCP/sample_models/Energy+.idd",
    ]
    for path in idd_candidates:
        if path and Path(path).exists():
            IDF.setiddname(path)
            logger.info(f"IDD set to: {path}")
            _IDD_INITIALIZED = True
            return

    raise FileNotFoundError(
        "Cannot find Energy+.idd. Set EPLUS_IDD_PATH env var or ensure "
        "EnergyPlus is installed at /app/software/EnergyPlusV26-1-0/"
    )


# ---------------------------------------------------------------------------
# Tool 1: alter_occupancy_global
# ---------------------------------------------------------------------------

def alter_occupancy_global(
    idf_path: str,
    multiplier: float,
    output_dir: str,
    output_filename: str = "occupancy_modified.idf",
) -> dict:
    """
    Scale occupancy in ALL zones of an IDF model by a multiplier.

    Handles all three EnergyPlus occupancy calculation methods:
    - People: Number_of_People * multiplier
    - People/Area: People_per_Floor_Area * multiplier
    - Area/Person: Floor_Area_per_Person / multiplier

    Returns dict with modification summary.
    """
    _ensure_idd()

    idf_path = Path(idf_path)
    if not idf_path.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")

    idf_object = IDF(str(idf_path))
    modified_idf = copy.deepcopy(idf_object)
    modified_count = 0
    details = []

    people_objects = modified_idf.idfobjects.get("PEOPLE", [])
    if not people_objects:
        return {
            "status": "warning",
            "message": "No PEOPLE objects found in the IDF model.",
            "modified_count": 0,
        }

    for p in people_objects:
        method = str(getattr(p, "Number_of_People_Calculation_Method", "")).strip().lower()
        try:
            if method == "people":
                old_val = float(p.Number_of_People)
                p.Number_of_People = old_val * multiplier
                details.append({
                    "name": p.Name, "method": "People",
                    "old": old_val, "new": old_val * multiplier,
                })
                modified_count += 1
            elif method == "people/area":
                old_val = float(p.People_per_Floor_Area)
                p.People_per_Floor_Area = old_val * multiplier
                details.append({
                    "name": p.Name, "method": "People/Area",
                    "old": old_val, "new": old_val * multiplier,
                })
                modified_count += 1
            elif method == "area/person":
                old_val = float(p.Floor_Area_per_Person)
                p.Floor_Area_per_Person = old_val / multiplier
                details.append({
                    "name": p.Name, "method": "Area/Person",
                    "old": old_val, "new": old_val / multiplier,
                })
                modified_count += 1
            else:
                logger.warning(f"Unknown occupancy method '{method}' for {p.Name}")
        except Exception as e:
            logger.error(f"Failed to modify PEOPLE object '{p.Name}': {e}")

    # Save modified IDF
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    final_path = save_path / output_filename
    modified_idf.saveas(str(final_path))

    return {
        "status": "success",
        "modified_count": modified_count,
        "total_people_objects": len(people_objects),
        "multiplier": multiplier,
        "output_file": str(final_path),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Tool 2: run_ep_simulation
# ---------------------------------------------------------------------------

def run_ep_simulation(
    idf_path: str,
    epw_path: str,
    output_dir: str,
    start_month: int = None,
    start_day: int = None,
    end_month: int = None,
    end_day: int = None,
) -> dict:
    """
    Run an EnergyPlus simulation and produce eplusout.csv.

    Optionally overrides the RunPeriod to the specified date range.
    Returns dict with output paths and summary.
    """
    _ensure_idd()

    idf_path = Path(idf_path)
    epw_path = Path(epw_path)
    if not idf_path.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # EnergyPlus/eppy writes a temporary IDF next to the loaded model file.
    # Stage the model inside the writable output directory so simulations can
    # run even when the original sample model lives in a read-only location.
    work_dir = out_path / "_working"
    work_dir.mkdir(parents=True, exist_ok=True)
    staged_idf_path = work_dir / idf_path.name
    shutil.copy2(idf_path, staged_idf_path)

    idf_object = IDF(str(staged_idf_path))

    # Override RunPeriod if dates provided
    if all(v is not None for v in [start_month, start_day, end_month, end_day]):
        run_periods = idf_object.idfobjects.get("RUNPERIOD", [])
        if run_periods:
            rp = run_periods[0]
            rp.Begin_Month = start_month
            rp.Begin_Day_of_Month = start_day
            rp.End_Month = end_month
            rp.End_Day_of_Month = end_day
        else:
            idf_object.newidfobject(
                "RUNPERIOD",
                Name="Custom_Run_Period",
                Begin_Month=start_month,
                Begin_Day_of_Month=start_day,
                End_Month=end_month,
                End_Day_of_Month=end_day,
            )

    # Clear existing output configs and add our targets
    for obj_type in ["OUTPUT:TABLE:SUMMARYREPORTS", "OUTPUT:VARIABLE", "OUTPUT:METER"]:
        if obj_type in idf_object.idfobjects:
            while len(idf_object.idfobjects[obj_type]) > 0:
                idf_object.removeidfobject(idf_object.idfobjects[obj_type][0])

    idf_object.newidfobject(
        "OUTPUT:VARIABLE",
        Key_Value="*",
        Variable_Name="Facility Total Electricity Demand Rate",
        Reporting_Frequency="Hourly",
    )

    # Run simulation
    idf_object.epw = str(epw_path)
    idf_object.run(
        weather=str(epw_path),
        output_directory=str(out_path),
        readvars=True,
    )

    csv_path = out_path / "eplusout.csv"
    result = {
        "status": "success",
        "output_directory": str(out_path),
        "csv_exists": csv_path.exists(),
        "csv_path": str(csv_path) if csv_path.exists() else None,
    }

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        result["rows"] = len(df)
        result["columns"] = list(df.columns[:10])  # First 10 column names

    return result


# ---------------------------------------------------------------------------
# Tool 3: calculate_rmse
# ---------------------------------------------------------------------------

def calculate_rmse(
    target_csv_path: str,
    simulation_csv_path: str,
    column_name: str = "Whole Building:Facility Total Electricity Demand Rate [W](Hourly)",
) -> dict:
    """
    Calculate RMSE between a target (ground truth) CSV and a simulation CSV.

    Both CSVs must contain the specified column.
    Returns dict with RMSE value and statistics.
    """
    target_path = Path(target_csv_path)
    sim_path = Path(simulation_csv_path)

    if not target_path.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation CSV not found: {sim_path}")

    df_target = pd.read_csv(target_path)
    df_sim = pd.read_csv(sim_path)

    df_target.columns = df_target.columns.str.strip()
    df_sim.columns = df_sim.columns.str.strip()

    # Try exact match first, then fuzzy match
    if column_name not in df_target.columns:
        matches = [c for c in df_target.columns if "Electricity Demand Rate" in c]
        if matches:
            column_name = matches[0]
        else:
            return {
                "status": "error",
                "message": f"Column not found in target CSV. Available: {list(df_target.columns)}",
            }

    if column_name not in df_sim.columns:
        matches = [c for c in df_sim.columns if "Electricity Demand Rate" in c]
        if matches:
            column_name = matches[0]
        else:
            return {
                "status": "error",
                "message": f"Column not found in simulation CSV. Available: {list(df_sim.columns)}",
            }

    y_true = df_target[column_name].values
    y_pred = df_sim[column_name].values

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mean_true = float(np.mean(y_true))
    cv_rmse = (rmse / mean_true * 100) if mean_true != 0 else None

    return {
        "status": "success",
        "rmse_watts": round(rmse, 4),
        "mae_watts": round(mae, 4),
        "cv_rmse_percent": round(cv_rmse, 2) if cv_rmse else None,
        "data_points": min_len,
        "column_used": column_name,
        "target_mean_watts": round(mean_true, 2),
    }


# ---------------------------------------------------------------------------
# Tool 4: calibrate_occupancy
# ---------------------------------------------------------------------------

def calibrate_occupancy(
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
) -> dict:
    """
    Automated Nelder-Mead calibration of occupancy multiplier.

    Iteratively adjusts occupancy, runs EnergyPlus, and compares against
    a target CSV until RMSE converges or max_iterations is reached.

    Returns dict with optimal multiplier, final RMSE, iteration count.
    """
    _ensure_idd()

    idf_path = Path(idf_path)
    epw_path = Path(epw_path)
    target_csv = Path(target_csv_path)
    work_dir = Path(output_dir)

    if not idf_path.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    if not target_csv.exists():
        raise FileNotFoundError(f"Target CSV not found: {target_csv}")

    work_dir.mkdir(parents=True, exist_ok=True)

    # Load the target column
    df_target = pd.read_csv(target_csv)
    df_target.columns = df_target.columns.str.strip()
    target_col = [c for c in df_target.columns if "Electricity Demand Rate" in c]
    if not target_col:
        return {"status": "error", "message": "No electricity demand column in target CSV."}
    target_values = df_target[target_col[0]].values

    base_idf = IDF(str(idf_path))
    iteration_log = []

    def objective(guess):
        mult = guess[0]
        iter_num = len(iteration_log) + 1
        logger.info(f"Calibration iteration {iter_num}: multiplier={mult:.4f}")

        try:
            # Alter occupancy
            modified_idf = copy.deepcopy(base_idf)
            for p in modified_idf.idfobjects.get("PEOPLE", []):
                method = str(getattr(p, "Number_of_People_Calculation_Method", "")).strip().lower()
                if method == "people":
                    p.Number_of_People = float(p.Number_of_People) * mult
                elif method == "people/area":
                    p.People_per_Floor_Area = float(p.People_per_Floor_Area) * mult
                elif method == "area/person":
                    p.Floor_Area_per_Person = float(p.Floor_Area_per_Person) / mult

            # Save temp IDF
            iter_dir = work_dir / f"iter_{iter_num}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            temp_idf_path = iter_dir / "temp_model.idf"
            modified_idf.saveas(str(temp_idf_path))

            # Run simulation
            sim_idf = IDF(str(temp_idf_path))
            for obj_type in ["OUTPUT:TABLE:SUMMARYREPORTS", "OUTPUT:VARIABLE", "OUTPUT:METER"]:
                if obj_type in sim_idf.idfobjects:
                    while len(sim_idf.idfobjects[obj_type]) > 0:
                        sim_idf.removeidfobject(sim_idf.idfobjects[obj_type][0])
            sim_idf.newidfobject(
                "OUTPUT:VARIABLE", Key_Value="*",
                Variable_Name="Facility Total Electricity Demand Rate",
                Reporting_Frequency="Hourly",
            )
            sim_idf.epw = str(epw_path)
            sim_idf.run(weather=str(epw_path), output_directory=str(iter_dir), readvars=True)

            # Calculate RMSE
            csv_file = iter_dir / "eplusout.csv"
            if not csv_file.exists():
                logger.error(f"Simulation output not found for iter {iter_num}")
                return 1e10

            df_sim = pd.read_csv(csv_file)
            df_sim.columns = df_sim.columns.str.strip()
            sim_col = [c for c in df_sim.columns if "Electricity Demand Rate" in c]
            if not sim_col:
                return 1e10

            y_pred = df_sim[sim_col[0]].values
            min_len = min(len(target_values), len(y_pred))
            rmse = float(np.sqrt(np.mean((target_values[:min_len] - y_pred[:min_len]) ** 2)))

            iteration_log.append({"iteration": iter_num, "multiplier": mult, "rmse": rmse})
            logger.info(f"Iteration {iter_num}: multiplier={mult:.4f}, RMSE={rmse:.2f}")
            return rmse

        except Exception as e:
            logger.error(f"Calibration iteration {iter_num} failed: {e}")
            return 1e10

    result = minimize(
        objective,
        [initial_guess],
        method="Nelder-Mead",
        options={
            "xatol": multiplier_tolerance,
            "fatol": rmse_tolerance,
            "maxiter": max_iterations,
        },
    )

    return {
        "status": "success" if result.success else "converged_by_threshold",
        "optimal_multiplier": round(float(result.x[0]), 4),
        "final_rmse_watts": round(float(result.fun), 2),
        "total_iterations": result.nit,
        "scipy_message": result.message,
        "iteration_log": iteration_log,
    }


# ---------------------------------------------------------------------------
# Tool 5: train_surrogate_model & predict_with_surrogate
# ---------------------------------------------------------------------------

# In-memory storage for trained surrogate (per server session)
_surrogate_state = {
    "model": None,
    "base_demand": None,
    "weather_features": None,
    "training_info": None,
}


def _parse_epw_weather(epw_path: str):
    """Extract hourly weather features from an EPW file."""
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    dry_bulb = df[6].values
    solar_rad = df[14].values
    hours = np.tile(np.arange(1, 25), 365)
    return dry_bulb, solar_rad, hours


def train_surrogate_model(
    cache_npz_path: str,
    epw_path: str,
    idf_path: str = None,
) -> dict:
    """
    Train a surrogate model from a pre-computed simulation cache (.npz file).

    The cache file should contain:
    - 'multipliers': array of occupancy multipliers used
    - 'y_data': 2D array of hourly demand curves (n_sims x 8760)

    If idf_path is provided, runs a baseline simulation to compute delta features.
    Otherwise uses the first cache entry as baseline.

    Returns training summary with R² score.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    cache_path = Path(cache_npz_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    epw = Path(epw_path)
    if not epw.exists():
        raise FileNotFoundError(f"EPW file not found: {epw}")

    # Load cache
    cached = np.load(cache_path)
    multipliers = cached["multipliers"].tolist()
    y_data = cached["y_data"]

    # Parse weather features
    dry_bulb, solar_rad, hours = _parse_epw_weather(str(epw))

    # Use first entry as baseline (multiplier closest to 0)
    base_idx = min(range(len(multipliers)), key=lambda i: abs(multipliers[i]))
    base_demand = y_data[base_idx]

    # Build flattened feature matrix
    X_flat = []
    y_deltas = []
    for idx, mult in enumerate(multipliers):
        for h in range(min(8760, y_data.shape[1])):
            X_flat.append([mult, dry_bulb[h], solar_rad[h], hours[h]])
            y_deltas.append(y_data[idx][h] - base_demand[h])

    X_flat = np.array(X_flat)
    y_deltas = np.array(y_deltas)

    # Train Ridge regression
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_deltas, test_size=0.2, random_state=42
    )
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    r2_score = float(model.score(X_test, y_test))

    # Store in memory
    _surrogate_state["model"] = model
    _surrogate_state["base_demand"] = base_demand
    _surrogate_state["weather_features"] = (dry_bulb, solar_rad, hours)
    _surrogate_state["training_info"] = {
        "n_simulations": len(multipliers),
        "n_hours": y_data.shape[1],
        "multiplier_range": [float(min(multipliers)), float(max(multipliers))],
        "r2_score": round(r2_score, 4),
    }

    return {
        "status": "success",
        "r2_score": round(r2_score, 4),
        "n_simulations_used": len(multipliers),
        "multiplier_range": [float(min(multipliers)), float(max(multipliers))],
        "message": "Surrogate model trained and ready for predictions.",
    }


def predict_with_surrogate(
    target_csv_path: str = None,
    initial_guess: float = 0.0,
) -> dict:
    """
    Use trained surrogate model for instant occupancy calibration.

    If target_csv_path is provided, calibrates against that data.
    Otherwise uses synthetic noise on baseline as a demo.

    Returns optimal multiplier and RMSE in sub-second time.
    """
    if _surrogate_state["model"] is None:
        return {
            "status": "error",
            "message": "No surrogate model trained. Call train_surrogate_model first.",
        }

    model = _surrogate_state["model"]
    base_demand = _surrogate_state["base_demand"]
    dry_bulb, solar_rad, hours = _surrogate_state["weather_features"]

    # Determine target data
    if target_csv_path and Path(target_csv_path).exists():
        df = pd.read_csv(target_csv_path)
        df.columns = df.columns.str.strip()
        col = [c for c in df.columns if "Electricity Demand Rate" in c]
        if col:
            target_data = df[col[0]].values[:8760]
        else:
            return {"status": "error", "message": "No electricity demand column in target CSV."}
    else:
        # Demo mode: add noise to baseline
        target_data = base_demand + np.random.normal(0, 5, len(base_demand))

    n_hours = min(len(target_data), len(base_demand), 8760)

    def objective(x):
        mult = x[0]
        if mult < -1.0 or mult > 5.0:
            return 1e10
        query = np.zeros((n_hours, 4))
        query[:, 0] = mult
        query[:, 1] = dry_bulb[:n_hours]
        query[:, 2] = solar_rad[:n_hours]
        query[:, 3] = hours[:n_hours]
        predicted_deltas = model.predict(query)
        predicted_total = base_demand[:n_hours] + predicted_deltas
        return float(np.sqrt(np.mean((target_data[:n_hours] - predicted_total) ** 2)))

    start = time.time()
    result = minimize(objective, [initial_guess], method="Nelder-Mead")
    elapsed = time.time() - start

    return {
        "status": "success",
        "optimal_multiplier": round(float(result.x[0]), 4),
        "final_rmse_watts": round(float(result.fun), 4),
        "convergence_time_seconds": round(elapsed, 4),
        "scipy_converged": result.success,
        "message": f"Calibration complete in {elapsed:.3f}s",
    }


# ---------------------------------------------------------------------------
# Tool 6: inspect_and_visualize_ifc
# ---------------------------------------------------------------------------

def inspect_and_visualize_ifc(
    ifc_path: str,
    output_dir: str = "/workspace/outputs",
    output_html_name: str = "ifc_3d_visualization.html",
) -> dict:
    """
    Inspect an IFC building model, extract storeys, spaces, and element metadata,
    and generate an interactive 3D HTML visualization file.

    Args:
        ifc_path: Path to the input .ifc file (e.g. /workspace/all_files/model.ifc)
        output_dir: Directory to save the 3D HTML visualization (default: /workspace/outputs)
        output_html_name: Name of output HTML visualization file (default: ifc_3d_visualization.html)

    Returns:
        dict with IFC metadata summary, storey breakdown, space list, element counts,
        and generated HTML visualization path.
    """
    ifc_file = Path(ifc_path)
    if not ifc_file.exists():
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")

    try:
        from epmcp_mcp_server.ifc_visualizer import parse_ifc, build_3d_traces
    except ImportError:
        try:
            from visualization_and_converter import parse_ifc, build_3d_traces
        except ImportError as e:
            raise ImportError(f"Required IFC visualization modules not found: {e}")

    file_bytes = ifc_file.read_bytes()
    parsed = parse_ifc(file_bytes)

    # Generate 3D visualization if Plotly is installed
    html_saved_path = None
    try:
        import plotly.graph_objects as go
        traces, spaces, elements = build_3d_traces(parsed["ifc"])
        clean_traces = []
        for t in traces:
            t_copy = copy.deepcopy(t)
            t_copy.pop("global_id", None)
            clean_traces.append(t_copy)

        fig = go.Figure(data=clean_traces)
        fig.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis=dict(title="X (m)"),
                yaxis=dict(title="Y (m)"),
                zaxis=dict(title="Z (m)"),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title=f"3D IFC Model: {parsed.get('project', ifc_file.name)}",
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        html_target = out_path / output_html_name
        fig.write_html(str(html_target))
        html_saved_path = str(html_target)
    except Exception as e:
        logger.warning(f"Could not generate Plotly HTML visualization: {e}")

    element_counts = dict(parsed.get("element_counts", {}))

    return {
        "status": "success",
        "file_name": ifc_file.name,
        "project_name": parsed.get("project", "Unnamed"),
        "schema": parsed.get("schema", "IFC"),
        "total_storeys": len(parsed.get("storeys", [])),
        "storeys": [s.get("name") for s in parsed.get("storeys", [])],
        "total_spaces": len(parsed.get("spaces", [])),
        "space_names": [sp.get("name") for sp in parsed.get("spaces", [])],
        "element_counts": element_counts,
        "visualization_html": html_saved_path,
        "message": f"Successfully parsed IFC file '{ifc_file.name}' with {len(parsed.get('spaces', []))} spaces.",
    }
