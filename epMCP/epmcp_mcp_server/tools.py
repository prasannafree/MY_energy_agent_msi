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

try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

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
from scipy.optimize import minimize, differential_evolution

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
    output_filename: str = "occupancy_modified.idf",) -> dict:
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
    end_day: int = None,) -> dict:
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


def _find_facility_electricity_col(columns: list) -> str:
    """Find the total facility electricity demand or meter column, avoiding sub-component meters like chiller or pump electricity."""
    cols_clean = [c.strip() for c in columns]
    # Priority 1: Facility Total Electricity Demand Rate / Electricity:Facility
    for c in cols_clean:
        cl = c.lower()
        if "facility total electricity demand rate" in cl or "electricity:facility" in cl:
            return c
    # Priority 2: Generic electricity demand rate excluding sub-components
    for c in cols_clean:
        cl = c.lower()
        if "electricity demand rate" in cl and not any(sub in cl for sub in ["chiller", "boiler", "fan", "pump", "coil"]):
            return c
    # Priority 3: Any electricity meter excluding sub-components
    for c in cols_clean:
        cl = c.lower()
        if "electricity" in cl and not any(sub in cl for sub in ["chiller", "boiler", "fan", "pump", "coil"]):
            return c
    # Fallback
    for c in cols_clean:
        if "electricity" in c.lower():
            return c
    return None


# ---------------------------------------------------------------------------
# Tool 3: calculate_rmse
# ---------------------------------------------------------------------------

def calculate_rmse(
    target_csv_path: str,
    simulation_csv_path: str,
    column_name: str = "Whole Building:Facility Total Electricity Demand Rate [W](Hourly)",) -> dict:
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

    # Try exact match first, then priority facility meter match
    if column_name not in df_target.columns:
        found_col = _find_facility_electricity_col(df_target.columns)
        if found_col:
            column_name = found_col
        else:
            return {
                "status": "error",
                "message": f"Column not found in target CSV. Available: {list(df_target.columns)}",
            }

    sim_column_name = column_name
    if sim_column_name not in df_sim.columns:
        found_sim = _find_facility_electricity_col(df_sim.columns)
        if found_sim:
            sim_column_name = found_sim
        else:
            return {
                "status": "error",
                "message": f"Column not found in simulation CSV. Available: {list(df_sim.columns)}",
            }

    y_true = df_target[column_name].values.astype(float)
    y_pred = df_sim[sim_column_name].values.astype(float)

    if "[j]" in column_name.lower():
        y_true = y_true / 3600.0
    elif "[kwh]" in column_name.lower():
        y_true = y_true * 1000.0

    if "[j]" in sim_column_name.lower():
        y_pred = y_pred / 3600.0
    elif "[kwh]" in sim_column_name.lower():
        y_pred = y_pred * 1000.0

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
# ---------------------------------------------------------------------------
# Tool 4: calibrate_occupancy
# ---------------------------------------------------------------------------

def run_particle_swarm_local(obj_func, bounds, max_iter):
    lb, ub = bounds[0]
    num_particles = 5
    max_steps = max(1, max_iter // num_particles)
    positions = np.random.uniform(lb, ub, num_particles)
    velocities = np.zeros(num_particles)
    pbest_positions = copy.deepcopy(positions)
    pbest_scores = np.full(num_particles, np.inf)
    gbest_position = positions[0]
    gbest_score = np.inf
    w, c1, c2 = 0.5, 1.5, 1.5
    eval_counter = 0

    for step in range(max_steps):
        for i in range(num_particles):
            score = obj_func([positions[i]])
            eval_counter += 1
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i]
        r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_positions - positions) +
                      c2 * r2 * (gbest_position - positions))
        positions = np.clip(positions + velocities, lb, ub)

    return gbest_position, gbest_score, eval_counter, True, "PSO swarm execution completed."

def calibrate_occupancy(
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
    optimization_method: str = "Nelder-Mead",) -> dict:
    """
    Automated Nelder-Mead calibration of occupancy multiplier.

    Iteratively adjusts occupancy, runs EnergyPlus, and compares against
    a target CSV until RMSE converges or max_iterations is reached.

    Returns dict with optimal multiplier, final RMSE, iteration count.
    """
    _ensure_idd()
    
    # Enforce a hard upper limit of 20 iterations
    max_iterations = min(max_iterations, 20)

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

    # Load the target column using priority facility meter matching
    df_target = pd.read_csv(target_csv)
    df_target.columns = df_target.columns.str.strip()
    target_col_name = _find_facility_electricity_col(df_target.columns)
    if not target_col_name:
        return {"status": "error", "message": f"No facility electricity column in target CSV. Available: {list(df_target.columns)}"}
    
    target_series = df_target[target_col_name].values.astype(float)
    # Automatic unit conversion: Joules [J] -> Watts [W] (divide by 3600 seconds/hour)
    if "[j]" in target_col_name.lower():
        target_series = target_series / 3600.0
    elif "[kwh]" in target_col_name.lower():
        target_series = target_series * 1000.0

    target_values = target_series

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
            sim_idf.newidfobject(
                "OUTPUT:METER", Key_Name="Electricity:Facility",
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
            sim_col_name = _find_facility_electricity_col(df_sim.columns)
            if not sim_col_name:
                return 1e10

            y_pred = df_sim[sim_col_name].values.astype(float)
            if "[j]" in sim_col_name.lower():
                y_pred = y_pred / 3600.0
            elif "[kwh]" in sim_col_name.lower():
                y_pred = y_pred * 1000.0

            min_len = min(len(target_values), len(y_pred))
            rmse = float(np.sqrt(np.mean((target_values[:min_len] - y_pred[:min_len]) ** 2)))

            iteration_log.append({"iteration": iter_num, "multiplier": mult, "rmse": rmse})
            logger.info(f"Iteration {iter_num}: multiplier={mult:.4f}, RMSE={rmse:.2f}")
            if rmse <= rmse_tolerance:
                raise EarlyStoppingException()
            return rmse

        except EarlyStoppingException:
            raise
        except Exception as e:
            logger.error(f"Calibration iteration {iter_num} failed: {e}")
            return 1e10

    search_bounds = [(search_min, search_max)]
    
    class EarlyStoppingException(Exception): pass
    
    def run_optimization_block():
        if optimization_method == "Bayesian Optimization (Gaussian Process)":
            if not HAS_SKOPT:
                return {"status": "error", "message": "skopt library is missing. Cannot run Bayesian Optimization."}
            
            space = [Real(search_bounds[0][0], search_bounds[0][1], name='multiplier')]
            
            def black_box_wrapper(val):
                return objective(val)
                
            res = gp_minimize(
                func=black_box_wrapper,
                dimensions=space,
                n_calls=max_iterations,
                n_initial_points=max(3, min(5, max_iterations // 2)),
                random_state=42
            )
            best_x = res.x[0]
            best_f = res.fun
            total_evals = len(res.func_vals)
            success = True
            msg = "Bayesian convergence process completed."
            
        elif optimization_method == "Differential Evolution (Genetic Algorithm)":
            # DE evaluates the initial population (5 evaluations) + (generations * 5 evaluations)
            # We adjust max_generations so the total simulations never exceed max_iterations
            max_generations = max(1, (max_iterations - 5) // 5) if max_iterations > 5 else 1
            res = differential_evolution(
                func=objective,
                bounds=search_bounds,
                maxiter=max_generations,
                popsize=5,
                atol=rmse_tolerance,
                tol=multiplier_tolerance,
                seed=42,
                polish=False
            )
            best_x = res.x[0]
            best_f = res.fun
            total_evals = res.nfev
            success = res.success
            msg = res.message
            
        elif optimization_method == "Particle Swarm Optimization (PSO)":
            best_x, best_f, total_evals, success, msg = run_particle_swarm_local(
                objective, search_bounds, max_iterations
            )
            
        else:  # Default to Nelder-Mead
            res = minimize(
                objective,
                [initial_guess],
                method="Nelder-Mead",
                bounds=search_bounds,
                options={
                    "xatol": multiplier_tolerance,
                    "fatol": rmse_tolerance,
                    "maxiter": max_iterations,
                },
            )
            best_x = res.x[0]
            best_f = res.fun
            total_evals = res.nit
            success = res.success
            msg = res.message
        return best_x, best_f, total_evals, success, msg

    try:
        best_x, best_f, total_evals, success, msg = run_optimization_block()
    except EarlyStoppingException:
        logger.info("Optimizer stopped early due to RMSE reaching tolerance threshold.")
        best_x = iteration_log[-1]["multiplier"]
        best_f = iteration_log[-1]["rmse"]
        total_evals = len(iteration_log)
        success = True
        msg = "Converged by early stopping threshold"

    return {
        "status": "success" if success else "converged_by_threshold",
        "optimal_multiplier": round(float(best_x), 4),
        "final_rmse_watts": round(float(best_f), 2),
        "total_iterations": total_evals,
        "scipy_message": msg,
        "iteration_log": iteration_log,
        "optimization_method_used": optimization_method,
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
    idf_path: str = None,) -> dict:
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
    initial_guess: float = 0.0,) -> dict:
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
# Tool 6: calculate_gross_floor_area
# ---------------------------------------------------------------------------

def calculate_gross_floor_area(idf_path: str) -> dict:
    """
    Calculate total gross conditioned floor area (m²) from IDF geometry.

    Uses geomeppy surface geometry methods with multiple fallback strategies:
    1. geomeppy getsurfaces('floor') — preferred method
    2. Direct iteration over BuildingSurface:Detailed with Surface_Type='Floor'
    3. Zone volume heuristic if no floor surfaces resolve

    Args:
        idf_path: Path to the EnergyPlus IDF file.

    Returns:
        dict with total_gross_area_m2, method used, and per-zone breakdown.
    """
    _ensure_idd()

    idf_file = Path(idf_path)
    if not idf_file.exists():
        raise FileNotFoundError(f"IDF file not found: {idf_path}")

    idf = IDF(str(idf_file))
    total_area = 0.0
    method_used = "none"
    zone_breakdown = {}

    # Method 1: geomeppy getsurfaces('floor')
    try:
        floor_surfaces = idf.getsurfaces("floor")
        for s in floor_surfaces:
            if hasattr(s, "area") and s.area > 0:
                zone_name = getattr(s, "Zone_Name", "Unknown")
                area = float(s.area)
                total_area += area
                zone_breakdown[zone_name] = zone_breakdown.get(zone_name, 0.0) + area
        if total_area > 0:
            method_used = "geomeppy_getsurfaces"
    except Exception as e:
        logger.debug(f"getsurfaces('floor') fallback: {e}")

    # Method 2: Direct BuildingSurface:Detailed iteration
    if total_area == 0.0:
        surfaces = idf.idfobjects.get("BUILDINGSURFACE:DETAILED", [])
        for surface in surfaces:
            if getattr(surface, "Surface_Type", "").lower() == "floor":
                try:
                    area = float(surface.area)
                    zone_name = getattr(surface, "Zone_Name", "Unknown")
                    total_area += area
                    zone_breakdown[zone_name] = zone_breakdown.get(zone_name, 0.0) + area
                except Exception:
                    pass
        if total_area > 0:
            method_used = "buildingsurface_detailed"

    # Method 3: Fallback — estimate from Zone objects if available
    if total_area == 0.0:
        zones = idf.idfobjects.get("ZONE", [])
        if zones:
            logger.warning(
                "Could not calculate floor area from geometry. "
                f"Model has {len(zones)} zones but no resolvable floor surfaces."
            )
            method_used = "no_floor_geometry"

    # Round zone breakdown values
    zone_breakdown = {k: round(v, 2) for k, v in zone_breakdown.items()}

    return {
        "status": "success" if total_area > 0 else "warning",
        "total_gross_area_m2": round(total_area, 2),
        "method_used": method_used,
        "zone_count": len(zone_breakdown),
        "zone_breakdown_m2": zone_breakdown,
        "message": (
            f"Total gross conditioned floor area: {total_area:.2f} m² "
            f"across {len(zone_breakdown)} zones."
            if total_area > 0
            else "Could not determine floor area from IDF geometry. "
                 "Ensure the model has BuildingSurface:Detailed floor objects."
        ),
    }


# ---------------------------------------------------------------------------
# Tool 7: extract_annual_energy_kwh
# ---------------------------------------------------------------------------

def extract_annual_energy_kwh(output_directory: str) -> dict:
    """
    Parse EnergyPlus simulation output CSV to extract annual energy in kWh.

    Searches for meter output files (eplusmeter.csv, eplusout.csv) in the
    output directory. Converts values from Joules (EnergyPlus native unit)
    to kWh. Detects Electricity:Facility and NaturalGas:Facility columns.

    Args:
        output_directory: Path to the directory containing EnergyPlus CSV output.

    Returns:
        dict with electricity_kwh, gas_kwh, total_kwh, and source file info.
    """
    out_dir = Path(output_directory)
    if not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_directory}")

    # Try meter output first, then general output
    csv_file = None
    for candidate in ["eplusmeter.csv", "eplusout.csv"]:
        path = out_dir / candidate
        if path.exists():
            csv_file = path
            break

    if csv_file is None:
        # Search for any CSV with meter-like content
        csv_files = list(out_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV output files found in {output_directory}"
            )
        # Pick the largest CSV as a heuristic
        csv_file = max(csv_files, key=lambda f: f.stat().st_size)

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        # Fallback for ragged EnergyPlus CSVs (mismatched column lengths)
        with open(csv_file, 'r', encoding='utf-8-sig', errors='ignore') as f:
            max_cols = max(len(line.split(',')) for line in f)
        
        # Read with dynamic column range to prevent ParserError
        df = pd.read_csv(csv_file, names=range(max_cols), low_memory=False)
        
        # Promote first row to header
        df.columns = df.iloc[0].fillna('').astype(str).tolist()
        df = df[1:].reset_index(drop=True)

    df.columns = [str(c).strip() for c in df.columns]

    total_elec_kwh = 0.0
    total_gas_kwh = 0.0
    elec_col_used = None
    gas_col_used = None

    for col in df.columns:
        col_lower = col.lower()
        if "electricity:facility" in col_lower:
            raw_sum = df[col].sum()
            # EnergyPlus meters report in Joules; 1 kWh = 3,600,000 J
            total_elec_kwh = float(raw_sum / 3_600_000.0)
            elec_col_used = col
        elif "naturalgas:facility" in col_lower or "gas:facility" in col_lower:
            raw_sum = df[col].sum()
            total_gas_kwh = float(raw_sum / 3_600_000.0)
            gas_col_used = col

    # Fallback: if Electricity:Facility meter was not found, check demand rate columns
    if total_elec_kwh == 0.0:
        for col in df.columns:
            col_lower = col.lower()
            if "electricity demand rate" in col_lower or "electricity demand" in col_lower or ("electricity" in col_lower and "facility" in col_lower):
                raw_sum = df[col].sum()
                if "[w]" in col_lower or "rate" in col_lower:
                    # Watts (W) reported hourly -> kWh: sum(W * 1h) / 1000
                    total_elec_kwh = float(raw_sum / 1000.0)
                else:
                    # Joules (J) -> kWh: sum(J) / 3,600,000
                    total_elec_kwh = float(raw_sum / 3_600_000.0)
                elec_col_used = col
                break

    total_kwh = total_elec_kwh + total_gas_kwh

    return {
        "status": "success",
        "electricity_kwh": round(total_elec_kwh, 2),
        "gas_kwh": round(total_gas_kwh, 2),
        "total_kwh": round(total_kwh, 2),
        "source_file": str(csv_file),
        "electricity_column": elec_col_used,
        "gas_column": gas_col_used,
        "data_rows": len(df),
        "message": (
            f"Annual energy: {total_kwh:,.2f} kWh total "
            f"(Electricity: {total_elec_kwh:,.2f} kWh, "
            f"Gas: {total_gas_kwh:,.2f} kWh)"
        ),
    }


# ---------------------------------------------------------------------------
# Tool 8: get_epi_benchmark
# ---------------------------------------------------------------------------

# EPI benchmark table: building_type → climate_zone → kWh/m²/year
# Sources: ECBC (India), ASHRAE 90.1 reference baselines
_EPI_BENCHMARK_TABLE = {
    "apartment_highrise": {
        "cold": 120.0,
        "composite": 140.0,
        "hot_dry": 150.0,
        "warm_humid": 160.0,
        "temperate": 110.0,
    },
    "office": {
        "cold": 130.0,
        "composite": 160.0,
        "hot_dry": 170.0,
        "warm_humid": 180.0,
        "temperate": 120.0,
    },
    "hospital": {
        "cold": 250.0,
        "composite": 300.0,
        "hot_dry": 320.0,
        "warm_humid": 340.0,
        "temperate": 230.0,
    },
    "school": {
        "cold": 90.0,
        "composite": 110.0,
        "hot_dry": 120.0,
        "warm_humid": 130.0,
        "temperate": 85.0,
    },
    "retail": {
        "cold": 140.0,
        "composite": 170.0,
        "hot_dry": 180.0,
        "warm_humid": 190.0,
        "temperate": 130.0,
    },
    "hotel": {
        "cold": 170.0,
        "composite": 200.0,
        "hot_dry": 220.0,
        "warm_humid": 240.0,
        "temperate": 160.0,
    },
}

# Default EPI when building type or climate zone is unrecognized
_EPI_DEFAULT = 140.0


def get_epi_benchmark(building_type: str, climate_zone: str) -> dict:
    """
    Look up the baseline EPI benchmark (kWh/m²/year) for a building type
    and climate zone.

    Supported building types: apartment_highrise, office, hospital,
    school, retail, hotel.
    Supported climate zones: cold, composite, hot_dry, warm_humid, temperate.

    Args:
        building_type: Building type key (case-insensitive).
        climate_zone: Climate zone key (case-insensitive).

    Returns:
        dict with benchmark_epi, building_type, climate_zone, and available options.
    """
    b_type = building_type.strip().lower().replace(" ", "_").replace("-", "_")
    c_zone = climate_zone.strip().lower().replace(" ", "_").replace("-", "_")

    # Alias mappings for flexible natural language inputs
    building_aliases = {
        "hotels": "hotel",
        "offices": "office",
        "commercial_office": "office",
        "hospitals": "hospital",
        "schools": "school",
        "retails": "retail",
        "apartment": "apartment_highrise",
        "apartments": "apartment_highrise",
        "residential": "apartment_highrise",
    }
    climate_aliases = {
        "summer": "hot_dry",
        "hot": "hot_dry",
        "summer_hot": "hot_dry",
        "hot_and_dry": "hot_dry",
        "monsoon": "warm_humid",
        "humid": "warm_humid",
        "winter": "cold",
        "mild": "temperate",
    }

    b_type = building_aliases.get(b_type, b_type)
    c_zone = climate_aliases.get(c_zone, c_zone)

    type_data = _EPI_BENCHMARK_TABLE.get(b_type)
    used_default = False

    if type_data is None:
        benchmark = _EPI_DEFAULT
        used_default = True
    else:
        benchmark = type_data.get(c_zone)
        if benchmark is None:
            benchmark = _EPI_DEFAULT
            used_default = True

    res = {
        "status": "success",
        "benchmark_epi_kwh_m2_yr": float(benchmark),
        "building_type": b_type,
        "climate_zone": c_zone,
        "used_default": used_default,
        "message": (
            f"Benchmark EPI for {b_type} in {c_zone} climate: "
            f"{benchmark:.1f} kWh/m²/year"
            + (" (default — unrecognized type or zone)" if used_default else "")
        ),
    }

    if used_default:
        res["available_building_types"] = list(_EPI_BENCHMARK_TABLE.keys())
        res["available_climate_zones"] = ["cold", "composite", "hot_dry", "warm_humid", "temperate"]

    return res


# ---------------------------------------------------------------------------
# Tool 9: calculate_epi
# ---------------------------------------------------------------------------

def calculate_epi(
    idf_path: str,
    simulation_output_dir: str,
    building_type: str = "office",
    climate_zone: str = "composite",) -> dict:
    """
    Compute the Energy Performance Index (EPI) for a building.

    Combines:
    1. Floor area calculation from IDF geometry (via calculate_gross_floor_area)
    2. Annual energy extraction from simulation output (via extract_annual_energy_kwh)
    3. Benchmark lookup (via get_epi_benchmark)
    4. EPI ratio and code compliance evaluation

    Prerequisites: An EnergyPlus simulation must have already been run and
    the output CSV must exist in simulation_output_dir.

    Args:
        idf_path: Path to the IDF file (for floor area calculation).
        simulation_output_dir: Path to the simulation output directory
                               containing CSV results.
        building_type: Building type for benchmark (e.g. 'office', 'hospital').
        climate_zone: Climate zone for benchmark (e.g. 'composite', 'cold').

    Returns:
        dict with proposed EPI, benchmark EPI, EPI ratio, compliance status,
        floor area, energy breakdown, and performance summary.
    """
    # Step 1: Calculate floor area
    area_result = calculate_gross_floor_area(idf_path)
    gross_area = area_result["total_gross_area_m2"]

    if gross_area <= 0:
        return {
            "status": "error",
            "message": (
                "Cannot compute EPI: floor area is zero. "
                "Ensure the IDF model has resolvable floor geometry "
                "(BuildingSurface:Detailed with Surface_Type='Floor')."
            ),
            "floor_area_result": area_result,
        }

    # Step 2: Extract annual energy from simulation output
    energy_result = extract_annual_energy_kwh(simulation_output_dir)
    total_kwh = energy_result["total_kwh"]

    if total_kwh <= 0:
        return {
            "status": "error",
            "message": (
                "Cannot compute EPI: total annual energy is zero. "
                "Ensure the simulation completed and output meters "
                "(Electricity:Facility, NaturalGas:Facility) are configured."
            ),
            "energy_result": energy_result,
        }

    # Step 3: Look up benchmark
    benchmark_result = get_epi_benchmark(building_type, climate_zone)
    benchmark_epi = benchmark_result["benchmark_epi_kwh_m2_yr"]

    # Step 4: Compute EPI metrics
    proposed_epi = total_kwh / gross_area
    epi_ratio = proposed_epi / benchmark_epi
    is_compliant = bool(epi_ratio <= 1.0)

    pct_diff = abs(round((1 - epi_ratio) * 100, 1))
    perf_label = "better" if is_compliant else "worse"

    return {
        "status": "success",
        "gross_area_m2": round(gross_area, 2),
        "total_annual_kwh": round(total_kwh, 2),
        "electricity_kwh": energy_result["electricity_kwh"],
        "gas_kwh": energy_result["gas_kwh"],
        "proposed_epi_kwh_m2_yr": round(proposed_epi, 2),
        "benchmark_epi_kwh_m2_yr": benchmark_epi,
        "epi_ratio": round(epi_ratio, 3),
        "code_compliant": is_compliant,
        "building_type": building_type.strip().lower(),
        "climate_zone": climate_zone.strip().lower(),
        "performance_summary": (
            f"Building performance is {pct_diff}% {perf_label} "
            f"than the {building_type} code benchmark for {climate_zone} climate. "
            f"EPI Ratio: {epi_ratio:.3f} "
            f"({'COMPLIANT ✅' if is_compliant else 'NON-COMPLIANT ❌'})"
        ),
        "floor_area_method": area_result["method_used"],
        "energy_source_file": energy_result["source_file"],
    }


# ---------------------------------------------------------------------------
# Tool 10: inspect_and_visualize_ifc
# ---------------------------------------------------------------------------

def inspect_and_visualize_ifc(
    ifc_path: str,
    output_dir: str = "/workspace/outputs",
    output_html_name: str = "ifc_3d_visualization.html",) -> dict:
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
