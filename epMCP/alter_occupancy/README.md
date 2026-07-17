# Alter Occupancy

A Streamlit-based interface for exploring, modifying, simulating, and comparing EnergyPlus building models. The application provides an end-to-end workflow for baseline simulation, occupancy parameter modification, performance visualization, and comparative analysis between baseline and altered building models.

---

## Overview

This module provides an interactive interface for EnergyPlus simulations by allowing users to:

- Upload EnergyPlus building models and weather files
- Generate baseline simulations
- Inspect building geometry and material properties
- Visualize annual and seasonal energy performance
- Modify occupancy parameters globally
- Run altered EnergyPlus simulations
- Compare baseline and modified simulation results

The application is intended for rapid experimentation with occupancy-driven energy analysis while providing detailed building diagnostics through an intuitive Streamlit interface.

---

## Folder Structure

```
alter_occupancy/
│
├── final.py                     # Main Streamlit application
│
└── essentials/
    ├── init.py                  # EnergyPlus initialization & simulation utilities
    ├── altnew.py                # Global occupancy modification engine
    ├── extractidf.py            # IDF metadata extraction utilities
    ├── appsupport.py            # Streamlit interface helpers
    ├── baseviz.py               # Baseline model visualization
    ├── weeksim.py               # Weekly load profile analytics
    ├── plotter.py               # Baseline vs altered comparison plots
```

---

## Main Components

### `final.py`

The primary application entry point.

Responsibilities include:

- Uploading IDF, IDD, and EPW files
- Initializing EnergyPlus models
- Executing baseline simulations
- Configuring simulation periods
- Visualizing baseline building performance
- Applying occupancy modifications
- Running altered simulations
- Comparing baseline and modified results
- Rendering interactive dashboards

---

### `init.py`

Provides core simulation utilities.

Functions include:

- Initializing EnergyPlus models
- Executing simulations
- Configuring simulation run periods
- Exporting EnergyPlus outputs to CSV
- Managing output variables
- IDF version handling

---

### `altnew.py`

Handles occupancy parameter modification.

Current implementation supports:

- Global scaling of PEOPLE objects
- Preservation of the original IDF model
- Generation of modified IDF files
- Support for different EnergyPlus occupancy definition methods

---

### `extractidf.py`

Extracts metadata directly from EnergyPlus IDF files.

Supported information includes:

- Thermal zones
- Building surfaces
- Windows
- Materials
- Occupancy objects

The extracted information is converted into Pandas DataFrames for downstream visualization.

---

### `appsupport.py`

Contains helper functions used by the Streamlit interface.

Responsibilities include:

- Application initialization
- File upload interface
- Path configuration
- Zone selection utilities

---

### `baseviz.py`

Generates baseline building analytics prior to any modifications.

Features include:

- Building geometry summary
- Floor area calculation
- Window-to-wall ratio
- Occupancy statistics
- Material thermal properties
- Building load composition
- Annual load duration curves

---

### `weeksim.py`

Provides seasonal and temporal performance analytics.

Includes:

- Typical weekly load profiles
- Peak summer week identification
- Lowest-demand week identification
- Annual load statistics
- Zone temperature heatmap generation

---

### `plotter.py`

Visualizes simulation outputs by comparing baseline and altered models.

Typical analyses include:

- Hourly electricity demand comparison
- Time-series visualization
- Baseline versus modified energy consumption

---

## Workflow

```
Upload IDF + EPW + IDD
            │
            ▼
Initialize EnergyPlus Model
            │
            ▼
Configure Simulation Period
            │
            ▼
Run Baseline Simulation
            │
            ▼
Generate Building Analytics
            │
            ▼
Modify Occupancy Parameters
            │
            ▼
Generate Modified IDF
            │
            ▼
Run Altered Simulation
            │
            ▼
Compare Baseline and Altered Performance
```

---

## Features

### Baseline Analysis

- Building geometry extraction
- Envelope statistics
- Material property inspection
- Window-to-wall ratio
- Peak occupancy estimation
- Load duration curves
- Energy breakdown visualization

### Occupancy Modification

- Global occupancy scaling
- Adjustable occupancy multiplier
- Automatic generation of modified IDF models
- Re-simulation using updated occupancy parameters

### Seasonal Analytics

- Typical weekly demand profile
- Peak summer week
- Lowest-demand week
- Annual minimum, average, and maximum demand
- Interactive zone temperature heatmaps

### Comparative Analysis

- Baseline vs altered hourly demand
- User-defined comparison period
- Interactive time-series visualization

---

## Required Inputs

The application requires:

- EnergyPlus model (`.idf`)
- EnergyPlus Data Dictionary (`.idd`)
- Weather file (`.epw`)

Users must also specify output directories for:

- Baseline simulation results
- Altered simulation results

---

## Outputs

The application generates:

- Baseline simulation outputs
- Modified IDF models
- Altered simulation outputs
- Hourly EnergyPlus CSV files
- Interactive visualizations
- Comparative energy performance analytics

---

## Dependencies

Major libraries used:

- Streamlit
- Geomeppy
- EnergyPlus
- Pandas
- NumPy
- Plotly
- Matplotlib

Install required packages using:

```bash
pip install streamlit geomeppy pandas numpy matplotlib plotly
```

Ensure EnergyPlus is installed and that the appropriate IDD file is available.

---

## Running the Application

Navigate to the project directory and execute:

```bash
streamlit run final.py
```

The Streamlit dashboard will launch in your default web browser.

---

## Current Supported Parameter

The current implementation supports global modification of:

- Occupancy (People)

The modular architecture allows additional building parameters such as lighting, equipment loads, HVAC settings, schedules, or infiltration rates to be incorporated with minimal changes.
