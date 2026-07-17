# EnergyPlus Automated Calibration Dashboard

A Streamlit-based calibration framework for EnergyPlus models that automatically calibrates building occupancy parameters using numerical optimization. The application iteratively modifies an EnergyPlus IDF model, executes simulations, compares simulated outputs against a target dataset, and identifies the occupancy multiplier that minimizes the RMSE.

---

## Overview

This project automates the calibration of EnergyPlus building simulation models using a synthetic calibration workflow.

The application:

- Uploads an EnergyPlus model (`.idf`), weather file (`.epw`), and EnergyPlus dictionary (`.idd`)
- Generates modified IDF models by scaling occupancy values
- Runs EnergyPlus simulations
- Extracts hourly electricity demand
- Computes RMSE against a target CSV
- Uses the Nelder-Mead optimization algorithm to determine the optimal occupancy multiplier
- Displays optimization progress live using a Streamlit dashboard

---

## Project Structure

```
calibration/
│
├── calibsite.py          # Main Streamlit application
│
└── essentials/
    ├── init.py           # Simulation utilities
    └── altnew.py         # Occupancy modification utilities
```

---

## Module Description

### `calibsite.py`

Main Streamlit dashboard responsible for:

- User interface
- File uploads
- Simulation parameter selection
- Optimization control
- RMSE computation
- Live visualization of calibration progress
- Displaying final calibration results

This script orchestrates the complete calibration workflow by calling functions from the `essentials` package.

---

### `essentials/init.py`

Contains helper functions for EnergyPlus execution and preprocessing.

Major responsibilities include:

- Initializing EnergyPlus IDF models
- Running EnergyPlus simulations
- Updating simulation run periods
- Exporting hourly output variables to CSV
- Loading and cleaning simulation outputs
- Managing IDF version compatibility

Important functions:

- `init()`
- `juzcsv()`
- `runperiod()`
- `load_and_clean_data()`
- `versioncheck()`
- `update_idf_version_native()`

---

### `essentials/altnew.py`

Contains functions responsible for modifying EnergyPlus models before each simulation.

Current implementation supports:

- Global occupancy scaling
- Deep-copying the baseline model
- Preserving the original IDF
- Saving modified IDF files for simulation

Primary function:

- `altglo()`

---

## Calibration Workflow

```
Upload IDF + EPW + IDD
            │
            ▼
Load baseline EnergyPlus model
            │
            ▼
Generate modified IDF
(occupancy multiplier)
            │
            ▼
Run EnergyPlus simulation
            │
            ▼
Export hourly electricity demand
            │
            ▼
Compare with target CSV
            │
            ▼
Compute RMSE
            │
            ▼
Nelder-Mead Optimization
            │
            ▼
Repeat until convergence
            │
            ▼
Return optimal occupancy multiplier
```

---

## Optimization Method

The calibration uses SciPy's **Nelder-Mead** simplex optimization algorithm.

Objective:

- Minimize the Root Mean Squared Error (RMSE) between

```
Target Electricity Demand
```

and

```
Simulated Electricity Demand
```

The optimization searches for the occupancy multiplier that produces the closest match to the target profile.

---

## Required Inputs

The dashboard requires:

- EnergyPlus model (`.idf`)
- Weather file (`.epw`)
- EnergyPlus dictionary (`.idd`)
- Target output CSV containing hourly electricity demand

---

## Output

The application reports:

- Optimal occupancy multiplier
- Final RMSE
- Number of optimization iterations
- Live convergence plot
- Modified IDF models
- EnergyPlus simulation outputs

---

## Dependencies

Major Python packages:

- Streamlit
- Geomeppy
- EnergyPlus
- NumPy
- Pandas
- SciPy
- pathlib

Install dependencies using:

```bash
pip install streamlit geomeppy scipy pandas numpy
```

Ensure EnergyPlus is installed and properly configured on your system.

---

## Running the Application

Navigate to the calibration directory and execute:

```bash
streamlit run calibsite.py
```

The application will open in your default browser.

---

## Current Calibration Parameter

The current implementation calibrates:

- Occupancy (People) multiplier

The framework is modular and can be extended to calibrate additional parameters such as:

- Lighting loads
- Equipment loads
- HVAC schedules
- Infiltration rates
- Ventilation parameters

---

## Notes

- The original IDF remains unchanged throughout the optimization process.
- Each optimization iteration creates a temporary modified IDF before simulation.
- Electricity demand is currently used as the calibration objective, but additional variables can be incorporated with minor modifications.
