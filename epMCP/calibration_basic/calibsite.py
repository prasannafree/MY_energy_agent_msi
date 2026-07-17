import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from geomeppy import IDF
import streamlit as pd_st  # Standard streamlit mapping safely decoupled
import streamlit as st

# Import your custom modules
from altnew import altglo
from init import juzcsv, runperiod

# Set up the Streamlit page configuration
st.set_page_config(page_title="EnergyPlus Calibration Twin", layout="wide")
st.title("Automated Synthetic Twin Calibration Dashboard")
st.write("Upload your structural files, define evaluation constraints, and watch the calibration loop converge in real time.")

# =================================================================
# 1. SIDEBAR / INTERFACE INPUTS
# =================================================================
st.sidebar.header("📁 Path & Environment Settings")
output_dir_input = st.sidebar.text_input("Output Directory Paths", value=r"C:\Users\annaa\Downloads\IISC\sim1\code\temp_sim")
target_csv_input = st.sidebar.text_input("Ground Truth Target CSV Path", value=r"C:\Users\annaa\Downloads\abnormal\eplusout.csv")

st.sidebar.header("⚙️ Optimization Parameters")
max_iterations = st.sidebar.slider("Maximum Iterations (maxiter)", min_value=5, max_value=100, value=25)
x_tolerance = st.sidebar.number_input("Multiplier Tolerance (xatol)", value=0.01, format="%.3f")
f_tolerance = st.sidebar.number_input("RMSE Watt Tolerance (fatol)", value=5.0, format="%.1f")

st.header("📤 Step 1: Upload Simulation Artifacts")
col1, col2, col3 = st.columns(3)

with col1:
    uploaded_idf = st.file_uploader("Upload Baseline Model (.idf)", type=["idf"])
with col2:
    uploaded_epw = st.file_uploader("Upload Weather Profile (.epw)", type=["epw"])
with col3:
    uploaded_idd = st.file_uploader("Upload EnergyPlus Dictionary (.idd)", type=["idd"])

# =================================================================
# 2. RUNTIME SAVE HELPER
# =================================================================
def save_uploaded_file(uploaded_file, destination_dir):
    if uploaded_file is not None:
        dest_path = Path(destination_dir) / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(dest_path)
    return None

# =================================================================
# 3. MATHEMATICAL LOGIC & INTERFACE LOOP
# =================================================================
def calculate_rmse(target_csv_path, sim_csv_path):
    df_target = pd.read_csv(Path(target_csv_path))
    df_sim = pd.read_csv(Path(sim_csv_path))
    
    df_target.columns = df_target.columns.str.strip()
    df_sim.columns = df_sim.columns.str.strip()
    
    # Matching your modified request to isolate total electricity metric
    column_name = 'Whole Building:Facility Total Electricity Demand Rate [W](Hourly)'
    
    y_true = df_target[column_name].values
    y_pred = df_sim[column_name].values
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

col_sim_start, col_sim_end = st.columns(2)
with col_sim_start:
    sim_start = st.date_input("Simulation Start Date", value=None, key="sim_start")
with col_sim_end:
    sim_end = st.date_input("Simulation End Date", value=None, key="sim_end")


# Initialize session state lists to store data across the live loops
if 'history_multipliers' not in st.session_state:
    st.session_state.history_multipliers = []
if 'history_rmse' not in st.session_state:
    st.session_state.history_rmse = []
if 'iteration_counter' not in st.session_state:
    st.session_state.iteration_counter = 0

# The Objective function mapped inside the interface frame
def streamlit_calibration_loop(guess_multiplier, base_idf_object, weather_filepath, temp_dir, target_csv_filepath, chart_placeholder, status_placeholder):
    current_multiplier = guess_multiplier[0]
    st.session_state.iteration_counter += 1
    
    status_placeholder.markdown(f"**Iteration {st.session_state.iteration_counter}:** Testing Multiplier `x{current_multiplier:.4f}`...")
    
    try:
        # Step A: Alter parameter sets via your custom altglo logic
        altglo(
            idf_object=base_idf_object,
            parameter_type='people',
            multiplier=current_multiplier,
            save_directory=temp_dir,
            new_filename="temp_iteration.idf"
        )
        
        # Step B: Instantiate object architecture using verified file configurations
        temp_idf_file_path = Path(temp_dir) / "temp_iteration.idf"
        fresh_idf_object = IDF(str(temp_idf_file_path), str(weather_filepath))
        
        # Step C: Execute simulation via your updated juzcsv call
        runperiod(fresh_idf_object, sim_start, sim_end)
        juzcsv(
            idf_object=fresh_idf_object,
            weather_path=weather_filepath,
            output_dir=temp_dir
        )
        
        # Step D: Process outcomes and append validation vectors
        generated_csv = os.path.join(temp_dir, "eplusout.csv")
        error_score = calculate_rmse(target_csv_filepath, generated_csv)
        
        # Save values to history
        st.session_state.history_multipliers.append(current_multiplier)
        st.session_state.history_rmse.append(error_score)
        
        # Create a DataFrame and update the live dashboard line chart
        chart_data = pd.DataFrame({
            'Occupancy Multiplier': st.session_state.history_multipliers,
            'RMSE Error (Watts)': st.session_state.history_rmse
        }).sort_values(by='Occupancy Multiplier') # Sorting ensures a clean continuous line plot
        
        chart_placeholder.line_chart(chart_data, x='Occupancy Multiplier', y='RMSE Error (Watts)')
        
        return error_score

    except Exception as e:
        status_placeholder.error(f"❌ Iteration failed at scale x{current_multiplier:.4f}. Error details: {e}")
        return 99999999.0

# =================================================================
# 4. EXECUTION CONTROL
# =================================================================
st.write("---")
st.header("Step 2: Calibration Execution & Analytics")

if st.button("Start Automated Calibration Loop", type="primary"):
    # Verify inputs are loaded prior to processing
    if not (uploaded_idf and uploaded_epw and uploaded_idd):
        st.error("Missing files! Please upload all three files (IDF, EPW, IDD) before running.")
    else:
        # Clear out any stale history from a previous session run
        st.session_state.history_multipliers = []
        st.session_state.history_rmse = []
        st.session_state.iteration_counter = 0
        
        # Create output directories if missing
        os.makedirs(output_dir_input, exist_ok=True)
        
        # Write files out to disk so EnergyPlus can access them via string paths
        st.info("Writing files to local workspace...")
        saved_idd_path = save_uploaded_file(uploaded_idd, output_dir_input)
        saved_epw_path = save_uploaded_file(uploaded_epw, output_dir_input)
        saved_idf_path = save_uploaded_file(uploaded_idf, output_dir_input)
        
        # Globally bind the saved IDD file to geomeppy framework
        IDF.setiddname(saved_idd_path)
        base_model_object = IDF(saved_idf_path)
        
        st.success("Workspace loaded! Optimization engine initializing...")
        
        # Set up dynamic Streamlit layout elements
        status_box = st.empty()
        chart_box = st.empty()
        
        # Set boundaries and runtime guesses
        search_bounds = [(0.1, 5.0)]
        initial_guess = [1.0]
        
        # Execute Scipy Optimizer Loop
        optimization_result = minimize(
            streamlit_calibration_loop,
            initial_guess,
            args=(base_model_object, saved_epw_path, output_dir_input, target_csv_input, chart_box, status_box),
            method='Nelder-Mead',
            bounds=search_bounds,
            options={'xatol': x_tolerance, 'fatol': f_tolerance, 'maxiter': max_iterations}
        )
        
        # Display Final Calibration Conclusions
        status_box.empty()
        st.markdown("### Calibration Complete!")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric(label="Discovered Optimal Multiplier", value=f"x{optimization_result.x[0]:.4f}")
        with col_res2:
            st.metric(label="Final Converged RMSE Error", value=f"{optimization_result.fun:.2f} W")
        with col_res3:
            st.metric(label="Total Iterations", value=f"{optimization_result.nit}")
        if optimization_result.success:
            st.success(f"Convergence achieved successfully: {optimization_result.message}")
        else:
            st.warning(f"Loop finalized via constraint threshold: {optimization_result.message}")
