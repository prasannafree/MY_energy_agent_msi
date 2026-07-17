import numpy as np
import pandas as pd
import copy
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import qmc
from geomeppy import IDF

# ==========================================
# USER-PROVIDED FUNCTIONS (INTEGRATED & PATCHED)
# ==========================================

def juzcsv(idf_object, weather_path: str, output_dir: str):
    """
    Executes EnergyPlus model and natively tracks all primary energy meters 
    into a unified eplusout.csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for obj_type in ['OUTPUT:TABLE:SUMMARYREPORTS', 'OUTPUT:VARIABLE', 'OUTPUT:METER']:
        if obj_type in idf_object.idfobjects:
            while len(idf_object.idfobjects[obj_type]) > 0:
                idf_object.removeidfobject(idf_object.idfobjects[obj_type][0])

    idf_object.newidfobject(
        'OUTPUT:VARIABLE', 
        Key_Value='*', 
        Variable_Name='Facility Total Electricity Demand Rate', 
        Reporting_Frequency='Hourly'
    )
    
    # Required patch for geomeppy execution stability within parallel threads
    idf_object.epw = str(weather_path)
    idf_object.run(
        weather=str(weather_path),
        output_directory=str(out_path),
        readvars=True 
    )

def altglo(idf_object, parameter_type: str, multiplier: float, save_directory: str, new_filename: str = "global_modified_model.idf"):
    """Intercepts an active IDF object, creates an independent copy, scales 
    the parameter value for ALL zones by a specified multiplier, 
    and writes out a brand new physical .idf file to disk.
    """
    modified_idf = copy.deepcopy(idf_object)
    param_choice = parameter_type.strip().lower()
    modified_count = 0

    if param_choice == 'people':
        people_objects = modified_idf.idfobjects['PEOPLE']
        for p in people_objects:
            method_value = str(getattr(p,"Number_of_People_Calculation_Method","")).strip().lower()
            try:
                if method_value == "people":
                    p.Number_of_People = float(p.Number_of_People) * multiplier
                    modified_count += 1
                elif method_value == "people/area":
                    p.People_per_Floor_Area = float(p.People_per_Floor_Area) * multiplier
                    modified_count += 1
                elif method_value == "area/person":
                    p.Floor_Area_per_Person = float(p.Floor_Area_per_Person) / multiplier
                    modified_count += 1
            except Exception as e:
                pass
    else:
        return None
        
    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)
    final_file_path = save_path / new_filename
    modified_idf.saveas(str(final_file_path))
    
    return modified_idf

# ==========================================
# OFFLINE PIPELINE: DISCRETE SIMULATIONS
# ==========================================

def run_pipeline_and_extract_data(idf_path, weather_path, run_id, multiplier):
    """
    Executes altglo and juzcsv dynamically by loading a clean local instance
    of the IDF file per process core.
    """
    temp_dir = Path(f"./eplus_runs/run_{run_id}")
    idf_save_dir = temp_dir / "idf"
    sim_out_dir = temp_dir / "output"
    
    # LOAD ITS OWN CLEAN CORE INSTANCE FROM DISK
    local_idf_instance = IDF(str(idf_path))
    
    # 1. Modify the IDF model based on the multiplier mapping
    scalar_multiplier = 1.0 + multiplier 
    modified_idf = altglo(local_idf_instance, 'people', scalar_multiplier, str(idf_save_dir))
    
    # 2. Run the EnergyPlus physics engine
    juzcsv(modified_idf, weather_path, str(sim_out_dir))
    
    # 3. Read generated eplusout.csv and extract the specific target column
    csv_file = sim_out_dir / "eplusout.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"EnergyPlus execution failed for run {run_id}. Output file not found.")
        
    df = pd.read_csv(csv_file)
    target_col = [col for col in df.columns if 'Facility Total Electricity Demand Rate' in col][0]
    
    return df[target_col].values
# ==========================================
# PARALLEL MULTIPROCESSING WORKER CONTAINER
# ==========================================

def parallel_worker(args):
    """
    Top-level unpacker function invoked across independent CPU cores.
    """
    idf_path, weather_file, idx, mult = args
    try:
        # Re-set the IDD path inside the new process core environment so geomeppy knows how to load the file
        eplus_idd_path = r"C:\Users\annaa\Downloads\IISC\sim2\Energy+.idd"
        IDF.setiddname(eplus_idd_path)
        
        print(f"🚀 Launching Run {idx+1} | Multiplier: {mult*100:.2f}%")
        hourly_demand = run_pipeline_and_extract_data(idf_path, weather_file, idx, mult)
        return [mult], hourly_demand
    except Exception as e:
        print(f"❌ Simulation Run {idx+1} failed due to error: {e}")
        return None
# ==========================================
# MAIN SURROGATE MODEL EXECUTION ENGINE
# ==========================================

if __name__ == "__main__":
    # --- PREPARATION CONFIGURATIONS ---
    eplus_idd_path = r"C:\Users\annaa\Downloads\IISC\sim2\Energy+.idd"
    IDF.setiddname(eplus_idd_path)
    idf_file_path = r"C:\Users\annaa\Downloads\IISC\sim2\ASHRAE901_OfficeLarge_STD2019_NewDelhi.idf"
    weather_file = r"C:\Users\annaa\Downloads\IISC\sim2\IND_DL_New.Delhi-Gandhi.Intl.AP.421810_TMYx.2009-2023.epw"
    
    # ... keep your Phase 1 space-filling generator ...
    print("--- Phase 1: Generating Latin Hypercube Space ---")
    sampler = qmc.LatinHypercube(d=1)
    sample_points = sampler.random(n=30)
    occupancy_multipliers = qmc.scale(sample_points, -1.0, 1.0).flatten()
    
    X_data = []
    y_data = []
    # Package tasks with strings rather than memory-sensitive object references
    worker_tasks = [
        (idf_file_path, weather_file, idx, mult) 
        for idx, mult in enumerate(occupancy_multipliers)
    ]    
    print("--- Phase 2: Building Master Simulation Dataset (Parallel Engine Activated) ---")
    
    # Detect your 20 logical processors and leave 1 thread free for system stability
    max_workers = max(1, os.cpu_count() - 1)
    print(f"💻 System detects 20 logical cores. Allocating {max_workers} threads to the task.")
    
    # Package parameters into tuples so they pass seamlessly across process boundaries
    
    # Fire off execution pools concurrently across cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(parallel_worker, worker_tasks))
        
    # Unpack valid processing sets into training arrays
    for res in results:
        if res is not None:
            X_data.append(res[0])
            y_data.append(res[1])

    X_data = np.array(X_data)
    y_data = np.array(y_data) 

    if len(X_data) == 0:
        raise ValueError("Critical Error: All parallel processing tasks failed. Check paths or files.")

    print("\n--- Phase 3: Machine Learning Training (Random Forest) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    surrogate = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    surrogate.fit(X_train, y_train)
    print(f"Training Complete. Validation Performance Score (R²): {surrogate.score(X_test, y_test):.4f}")

    # ==========================================
    # ONLINE PHASE: FAST STRIP COUPLING
    # ==========================================
    print("\n--- Phase 4: Running Sub-Second Calibration Loop ---")
    
    # Establish baseline target metrics from index 0 mapping
    real_world_data = y_data[0] + np.random.normal(0, 20, 8760)

    def surrogate_calibration_objective(x):
        current_mult = x[0]
        if current_mult < -1.0 or current_mult > 1.0:
            return 1e10 
            
        predicted_curve = surrogate.predict([[current_mult]])[0]
        rmse = np.sqrt(np.mean((real_world_data - predicted_curve) ** 2))
        return rmse

    # Run instant Nelder-Mead optimization against the trained surrogate model
    start = time.time()
    res = minimize(surrogate_calibration_objective, [0.0], method='Nelder-Mead')
    
    print(f"\nOptimization Convergence Complete:")
    print(f"  • Optimal Occupancy Change Discovered: {res.x[0]*100:.3f}%")
    print(f"  • Minimum Calibration RMSE achieved: {res.fun:.4f}")
    print(f"  • Total Convergence Latency: {time.time() - start:.4f} seconds!")