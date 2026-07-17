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
from sklearn.linear_model import Ridge

# ==========================================
# NEW: NATIVE EPW WEATHER PARSER FOR FEATURES
# ==========================================

def parse_epw_weather_features(epw_path: str):
    """
    Parses a standard EnergyPlus EPW weather file to extract continuous, 
    hourly environmental boundary conditions (8760 points each).
    Columns mapped: Dry Bulb Temp (Col 6), Direct Normal Radiation (Col 14).
    """
    # EPW files have 8 header lines before the actual weather data matrix starts
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    
    dry_bulb_temp = df[6].values          # Outdoor air dry bulb temperature [°C]
    direct_normal_rad = df[14].values     # Direct normal solar radiation [W/m²]
    hour_of_day = np.tile(np.arange(1, 25), 365) # Cyclical hourly indicator (1 to 24)
    
    print(f"🌦️ Weather Parser Success: Extracted {len(dry_bulb_temp)} hours of localized climate telemetry.")
    return dry_bulb_temp, direct_normal_rad, hour_of_day

# ==========================================
# USER-PROVIDED FUNCTIONS (INTEGRATED)
# ==========================================

def juzcsv(idf_object, weather_path: str, output_dir: str):
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
    
    idf_object.epw = str(weather_path)
    idf_object.run(
        weather=str(weather_path),
        output_directory=str(out_path),
        readvars=True 
    )

def altglo(idf_object, parameter_type: str, multiplier: float, save_directory: str, new_filename: str = "global_modified_model.idf"):
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
    return modified_idf

# ==========================================
# OFFLINE PIPELINE: DISCRETE SIMULATIONS
# ==========================================

def run_pipeline_and_extract_data(idf_path, weather_path, run_id, multiplier):
    temp_dir = Path(f"./eplus_runs/run_{run_id}")
    idf_save_dir = temp_dir / "idf"
    sim_out_dir = temp_dir / "output"
    
    local_idf_instance = IDF(str(idf_path))
    scalar_multiplier = 1.0 + multiplier 
    modified_idf = altglo(local_idf_instance, 'people', scalar_multiplier, str(idf_save_dir))
    
    juzcsv(modified_idf, weather_path, str(sim_out_dir))
    
    csv_file = sim_out_dir / "eplusout.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"EnergyPlus execution failed for run {run_id}. Output file not found.")
        
    df = pd.read_csv(csv_file)
    target_col = [col for col in df.columns if 'Facility Total Electricity Demand Rate' in col][0]
    
    return df[target_col].values

def parallel_worker(args):
    idf_path, weather_file, idx, mult = args
    try:
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
    # ==========================================
    # PATH CONFIGURATIONS
    # ==========================================
    eplus_idd_path = r"C:\Users\annaa\Downloads\IISC\sim2\Energy+.idd"
    IDF.setiddname(eplus_idd_path)
    idf_file_path = r"C:\Users\annaa\Downloads\IISC\sim2\ASHRAE901_OfficeLarge_STD2019_NewDelhi.idf"
    weather_file = r"C:\Users\annaa\Downloads\IISC\sim2\IND_DL_New.Delhi-Gandhi.Intl.AP.421810_TMYx.2009-2023.epw"
    cache_file = Path("./eplus_runs/simulation_cache.npz")
    
    # 1. Harvest environmental boundary features from your local weather matrix
    print("--- Phase 0: Harvesting Weather Features from EPW ---")
    dry_bulb, solar_rad, hours = parse_epw_weather_features(weather_file)
    
    # ==========================================
    # STRICT OFFLINE CACHE VERIFICATION LAYER
    # ==========================================
    if not cache_file.exists():
        print("\n❌ CRITICAL ERROR: Simulation cache binary not found on your hard drive!")
        print(f"Please place your 'simulation_cache.npz' file inside: {cache_file.parent.resolve()}")
        print("Aborting script execution to prevent unintended EnergyPlus generation loops.")
        import sys
        sys.exit(1)
        
    print("\n✨ Simulation cache confirmed on disk. Executing zero-simulation ML training pipeline...")
    cached_data = np.load(cache_file)
    raw_X_multipliers = cached_data['multipliers'].tolist()
    y_data = cached_data['y_data']
    print(f"✅ Successfully extracted data for {len(raw_X_multipliers)} discrete simulation profiles.")

    # ==========================================
    # PHASE 2.1: ESTABLISHING BASELINE PROFILE (0% SHIFT)
    # ==========================================
    print("\n--- Phase 2.1: Establishing Baseline Energy Profile (0% Shift) ---")
    # This remains as the single quick run needed to scale your delta equations
    base_hourly_demand = run_pipeline_and_extract_data(idf_file_path, weather_file, "baseline", 0.0)

    # ==========================================
    # PHASE 2.5: DELTA-RESIDUAL MATRIX FLATTENING
    # ==========================================
    print("\n--- Phase 2.5: Restructuring to Delta Target Matrix ---")
    X_flattened = []
    y_deltas = []
    
    for run_idx, mult in enumerate(raw_X_multipliers):
        for hour_idx in range(8760):
            feature_row = [
                mult, 
                dry_bulb[hour_idx], 
                solar_rad[hour_idx], 
                hours[hour_idx]
            ]
            X_flattened.append(feature_row)
            
            # Isolate occupancy effect by subtracting weather variance trends
            delta_w = y_data[run_idx][hour_idx] - base_hourly_demand[hour_idx]
            y_deltas.append(delta_w)
            
    X_flattened = np.array(X_flattened)  # Shape: (262800, 4)
    y_deltas = np.array(y_deltas)        # Shape: (262800,)

    # ==========================================
    # PHASE 3: CONTINUOUS REGRESSION SURROGATE
    # ==========================================
    print("\n--- Phase 3: Machine Learning Training (Continuous Ridge Surrogate) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_deltas, test_size=0.2, random_state=42)
    
    from sklearn.linear_model import Ridge
    surrogate = Ridge(alpha=1.0)
    surrogate.fit(X_train, y_train)
    print(f"🎯 Continuous Model Validation Performance Score (R²): {surrogate.score(X_test, y_test):.4f}")

    # ==========================================
    # PHASE 4: MATHEMATICAL NELDER-MEAD TRACKING
    # ==========================================
    print("\n--- Phase 4: Running Sub-Second Calibration Loop ---")
    
    # Setting an arbitrary validation profile to evaluate convergence slope handling
    test_target_idx = min(5, len(raw_X_multipliers) - 1)
    true_target_multiplier = raw_X_multipliers[test_target_idx]
    print(f"📍 Verification Goal: Can the optimizer discover the true multiplier of {true_target_multiplier*100:.3f}%?")
    
    real_world_data = y_data[test_target_idx] + np.random.normal(0, 5, 8760)

    def surrogate_calibration_objective(x):
        current_mult = x[0]
        if current_mult < -1.0 or current_mult > 1.0:
            return 1e10 
            
        # Efficient batch queries over 8,760 rows simultaneously
        query_matrix = np.zeros((8760, 4))
        query_matrix[:, 0] = current_mult
        query_matrix[:, 1] = dry_bulb
        query_matrix[:, 2] = solar_rad
        query_matrix[:, 3] = hours
        
        predicted_deltas = surrogate.predict(query_matrix)
        predicted_total_curve = base_hourly_demand + predicted_deltas
        
        rmse = np.sqrt(np.mean((real_world_data - predicted_total_curve) ** 2))
        return rmse

    start = time.time()
    # Initialize from the baseline reference coordinate
    res = minimize(surrogate_calibration_objective, [0.0], method='Nelder-Mead')
    
    print(f"\nOptimization Convergence Complete:")
    print(f"  • True Target Multiplier was:       {true_target_multiplier*100:.3f}%")
    print(f"  • Optimal Occupancy Discovered:     {res.x[0]*100:.3f}%")
    print(f"  • Minimum Calibration RMSE achieved: {res.fun:.4f}")
    print(f"  • Total Convergence Latency:        {time.time() - start:.4f} seconds!")