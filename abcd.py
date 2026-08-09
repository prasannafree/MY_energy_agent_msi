import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
from geomeppy import IDF
import streamlit as pd_st
import streamlit as st

# Optional external optimization library imports with runtime fallback handling
try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

# Import custom domain modules
from altnew import altglo
from init import juzcsv, runperiod

# =================================================================
# PAGE CONFIGURATION & HEADER
# =================================================================
st.set_page_config(page_title="EnergyPlus Calibration Twin", layout="wide")
st.title("Automated Synthetic Twin Calibration Dashboard")
st.write("Upload structural artifacts, select your optimization technique, and observe convergence dynamics in real time.")

# =================================================================
# 1. SIDEBAR & USER INTERFACE SELECTIONS
# =================================================================
st.sidebar.header("📁 Path & Environment Settings")
output_dir_input = st.sidebar.text_input("Output Directory Paths", value=r"C:\Users\annaa\Downloads\IISC\sim1\code\temp_sim")
target_csv_input = st.sidebar.text_input("Ground Truth Target CSV Path", value=r"C:\Users\annaa\Downloads\abnormal\eplusout.csv")

st.sidebar.header("⚙️ Optimization Engine Settings")
selected_method = st.sidebar.selectbox(
    "Select Optimization Algorithm",
    (
        "Nelder-Mead (Local Simplex)",
        "Bayesian Optimization (Gaussian Process)",
        "Differential Evolution (Genetic Algorithm)",
        "Particle Swarm Optimization (PSO)"
    )
)

max_iterations = st.sidebar.slider("Maximum Iterations / Calls", min_value=5, max_value=100, value=25)
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

col_sim_start, col_sim_end = st.columns(2)
import datetime
with col_sim_start:
    sim_start = st.date_input("Simulation Start Date", value=datetime.date(2026, 1, 1), key="sim_start")
with col_sim_end:
    sim_end = st.date_input("Simulation End Date", value=datetime.date(2026, 12, 31), key="sim_end")

# Initialize session state variables
if 'history_multipliers' not in st.session_state:
    st.session_state.history_multipliers = []
if 'history_rmse' not in st.session_state:
    st.session_state.history_rmse = []
if 'iteration_counter' not in st.session_state:
    st.session_state.iteration_counter = 0

# =================================================================
# 2. UTILITY & CORE EVALUATION FUNCTIONS
# =================================================================
def save_uploaded_file(uploaded_file, destination_dir):
    if uploaded_file is not None:
        dest_path = Path(destination_dir) / uploaded_file.name
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(dest_path)
    return None

def calculate_rmse(target_csv_path, sim_csv_path):
    df_target = pd.read_csv(Path(target_csv_path))
    df_sim = pd.read_csv(Path(sim_csv_path))
    
    df_target.columns = df_target.columns.str.strip()
    df_sim.columns = df_sim.columns.str.strip()
    
    column_name = 'Whole Building:Facility Total Electricity Demand Rate [W](Hourly)'
    
    y_true = df_target[column_name].values
    y_pred = df_sim[column_name].values
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def streamlit_calibration_loop(guess_multiplier, base_idf_object, weather_filepath, temp_dir, target_csv_filepath, chart_placeholder, status_placeholder):
    """Core black-box objective function evaluated across all optimizers."""
    # Handle single scalar inputs or vector wrapper formats passed by different solvers
    current_multiplier = float(guess_multiplier[0]) if isinstance(guess_multiplier, (list, np.ndarray)) else float(guess_multiplier)
    st.session_state.iteration_counter += 1
    
    status_placeholder.markdown(f"**Iteration {st.session_state.iteration_counter}:** Testing Multiplier `x{current_multiplier:.4f}`...")
    
    try:
        altglo(
            idf_object=base_idf_object,
            parameter_type='people',
            multiplier=current_multiplier,
            save_directory=temp_dir,
            new_filename="temp_iteration.idf"
        )
        
        temp_idf_file_path = Path(temp_dir) / "temp_iteration.idf"
        fresh_idf_object = IDF(str(temp_idf_file_path), str(weather_filepath))
        
        runperiod(fresh_idf_object, sim_start, sim_end)
        juzcsv(
            idf_object=fresh_idf_object,
            weather_path=weather_filepath,
            output_dir=temp_dir
        )
        
        generated_csv = os.path.join(temp_dir, "eplusout.csv")
        error_score = calculate_rmse(target_csv_filepath, generated_csv)
        
        st.session_state.history_multipliers.append(current_multiplier)
        st.session_state.history_rmse.append(error_score)
        
        chart_data = pd.DataFrame({
            'Occupancy Multiplier': st.session_state.history_multipliers,
            'RMSE Error (Watts)': st.session_state.history_rmse
        }).sort_values(by='Occupancy Multiplier')
        
        chart_placeholder.line_chart(chart_data, x='Occupancy Multiplier', y='RMSE Error (Watts)')
        
        return error_score

    except Exception as e:
        status_placeholder.error(f"❌ Iteration failed at scale x{current_multiplier:.4f}. Error details: {e}")
        return 99999999.0


# =================================================================
# 3. INDIVIDUAL OPTIMIZATION ALGORITHM FUNCTIONS
# =================================================================

def run_nelder_mead(obj_func, initial_guess, bounds, max_iter, xatol, fatol, func_args):
    """Local Simplex Optimization Method (Nelder-Mead)."""
    res = minimize(
        obj_func,
        initial_guess,
        args=func_args,
        method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': xatol, 'fatol': fatol, 'maxiter': max_iter}
    )
    return res.x[0], res.fun, res.nit, res.success, res.message


def run_bayesian_optimization(obj_func, bounds, max_iter, func_args):
    """Gaussian Process Surrogate-based Bayesian Optimization (BPS standard for expensive evaluations)."""
    if not HAS_SKOPT:
        st.error("`scikit-optimize` library is missing. Install via `pip install scikit-optimize` to run Bayesian Optimization.")
        return None, None, 0, False, "Missing skopt package"
        
    space = [Real(bounds[0][0], bounds[0][1], name='multiplier')]
    
    # Wrapper function for skopt argument signature
    def black_box_wrapper(val):
        return obj_func(val, *func_args)

    res = gp_minimize(
        func=black_box_wrapper,
        dimensions=space,
        n_calls=max_iter,
        n_initial_points=max(3, min(5, max_iter // 2)),
        random_state=42
    )
    return res.x[0], res.fun, len(res.func_vals), True, "Bayesian convergence process completed."


def run_differential_evolution(obj_func, bounds, max_iter, xatol, fatol, func_args):
    """Population-based Evolutionary Strategy (Genetic Algorithm variant for BPS global search)."""
    # Converting maximum iterations to generations with small popsize
    max_generations = max(1, max_iter // 5)
    
    res = differential_evolution(
        func=obj_func,
        bounds=bounds,
        args=func_args,
        maxiter=max_generations,
        popsize=5,
        atol=fatol,
        tol=xatol,
        seed=42
    )
    return res.x[0], res.fun, res.nfev, res.success, res.message


def run_particle_swarm(obj_func, bounds, max_iter, func_args):
    """Swarm Intelligence Optimization Strategy (PSO)."""
    lb, ub = bounds[0]
    num_particles = 5
    max_steps = max(1, max_iter // num_particles)
    
    # Vectorized initialization
    positions = np.random.uniform(lb, ub, num_particles)
    velocities = np.zeros(num_particles)
    
    pbest_positions = copy.deepcopy(positions)
    pbest_scores = np.full(num_particles, np.inf)
    
    gbest_position = positions[0]
    gbest_score = np.inf
    
    w, c1, c2 = 0.5, 1.5, 1.5  # Standard inertia and acceleration weights
    eval_counter = 0

    for step in range(max_steps):
        for i in range(num_particles):
            score = obj_func([positions[i]], *func_args)
            eval_counter += 1
            
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i]
                
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i]
                
        # Update velocities and positions
        r1, r2 = np.random.rand(num_particles), np.random.rand(num_particles)
        velocities = (w * velocities +
                      c1 * r1 * (pbest_positions - positions) +
                      c2 * r2 * (gbest_position - positions))
        positions = np.clip(positions + velocities, lb, ub)

    return gbest_position, gbest_score, eval_counter, True, "PSO swarm execution completed."


# =================================================================
# 4. EXECUTION CONTROLLER & ROUTER
# =================================================================
st.write("---")
st.header("Step 2: Calibration Execution & Analytics")

if st.button("Start Automated Calibration Loop", type="primary"):
    if not (uploaded_idf and uploaded_epw and uploaded_idd):
        st.error("Missing required files! Upload IDF, EPW, and IDD files before running execution.")
    else:
        st.session_state.history_multipliers = []
        st.session_state.history_rmse = []
        st.session_state.iteration_counter = 0
        
        os.makedirs(output_dir_input, exist_ok=True)
        
        st.info("Writing workspace files to local disk...")
        saved_idd_path = save_uploaded_file(uploaded_idd, output_dir_input)
        saved_epw_path = save_uploaded_file(uploaded_epw, output_dir_input)
        saved_idf_path = save_uploaded_file(uploaded_idf, output_dir_input)
        from eppy.modeleditor import IDDAlreadySetError

        # Safely set IDD name without crashing on Streamlit re-runs
        try:
            IDF.setiddname(saved_idd_path)
        except IDDAlreadySetError:
            pass  # IDD is already loaded in the current Python session
        base_model_object = IDF(saved_idf_path)
        
        st.success(f"Workspace initialized! Launching {selected_method}...")
        
        status_box = st.empty()
        chart_box = st.empty()
        
        search_bounds = [(0.1, 5.0)]
        initial_guess = [1.0]
        func_args = (base_model_object, saved_epw_path, output_dir_input, target_csv_input, chart_box, status_box)
        
        # Route selected solver method
        if selected_method == "Nelder-Mead (Local Simplex)":
            best_x, best_f, total_evals, is_success, msg = run_nelder_mead(
                streamlit_calibration_loop, initial_guess, search_bounds, max_iterations, x_tolerance, f_tolerance, func_args
            )
        elif selected_method == "Bayesian Optimization (Gaussian Process)":
            best_x, best_f, total_evals, is_success, msg = run_bayesian_optimization(
                streamlit_calibration_loop, search_bounds, max_iterations, func_args
            )
        elif selected_method == "Differential Evolution (Genetic Algorithm)":
            best_x, best_f, total_evals, is_success, msg = run_differential_evolution(
                streamlit_calibration_loop, search_bounds, max_iterations, x_tolerance, f_tolerance, func_args
            )
        elif selected_method == "Particle Swarm Optimization (PSO)":
            best_x, best_f, total_evals, is_success, msg = run_particle_swarm(
                streamlit_calibration_loop, search_bounds, max_iterations, func_args
            )

        # Output Summary Display
        status_box.empty()
        st.markdown("### Calibration Complete!")
        
        if best_x is not None:
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric(label="Optimal Multiplier", value=f"x{best_x:.4f}")
            with col_res2:
                st.metric(label="Converged RMSE Error", value=f"{best_f:.2f} W")
            with col_res3:
                st.metric(label="Total Iterations / Evaluations", value=f"{total_evals}")
            
            if is_success:
                st.success(f"Execution finished successfully: {msg}")
            else:
                st.warning(f"Optimization halted: {msg}")