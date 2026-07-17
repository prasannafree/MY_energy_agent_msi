# epMCP

**epMCP** is a collection of Python tools for developing Model Context Protocol (MCP) capabilities around EnergyPlus. The repository provides modular applications for building energy simulation, occupancy modification, automated calibration, and surrogate-assisted optimization using EnergyPlus models.

The project is built around the idea of making EnergyPlus workflows more interactive, extensible, and automation-friendly through reusable Python modules and Streamlit applications.

---

## Features

- Interactive Streamlit dashboards for EnergyPlus workflows
- Occupancy modification of EnergyPlus building models
- Automated baseline and altered simulations
- Building metadata extraction and visualization
- Automated calibration using numerical optimization
- Experimental surrogate models for accelerating calibration
- Sample EnergyPlus models for testing and demonstration

---

## Repository Structure

```text
epMCP/
│
├── alter_occupancy/
│   ├── final.py
│   ├── README.md
│   └── essentials/
│
├── calibration_basic/
│   ├── calibsite.py
│   ├── README.md
│   └── essentials/
│
├── surrogateEP/
│   ├── dee.py
│   ├── surrogate.py
│   └── README.md
│
├── sample_models/
│   ├── README.md
│   ├── Energy+.idd
│   ├── DOE Large Office (Delhi)
│   ├── Apartment (Denver)
│   └── Weather Files
│
├── requirements.txt
└── README.md
```

---

## Modules

### Alter Occupancy

A Streamlit application for modifying occupancy parameters in EnergyPlus models and comparing baseline and altered simulations.

**Capabilities**

- Upload IDF, IDD, and EPW files
- Run baseline EnergyPlus simulations
- Modify occupancy parameters globally
- Generate altered simulations
- Compare energy demand profiles
- Visualize building geometry and simulation outputs

See `alter_occupancy/README.md` for detailed documentation.

---

### Calibration Basic

An automated calibration framework that estimates the optimal occupancy multiplier by minimizing the error between simulated and target electricity demand.

The application uses:

- EnergyPlus
- SciPy (Nelder-Mead Optimization)
- Streamlit

Current calibration parameter:

- Occupancy (People)

See `calibration_basic/README.md` for more information.

---

### surrogateEP

Experimental implementations investigating surrogate-assisted EnergyPlus calibration.

Included approaches:

- Random Forest surrogate modeling
- Feature-based surrogate regression
- Machine-learning-assisted optimization

These prototypes aim to reduce the number of computationally expensive EnergyPlus simulations required during calibration.

See `surrogateEP/README.md` for additional details.

---

### Sample Models

Contains example EnergyPlus models and supporting files used throughout the repository.

Included:

- DOE Large Office prototype with Delhi weather
- EnergyPlus Apartment example with Denver weather
- EnergyPlus IDD file

These files can be used directly with the applications provided in this repository.

---

## Typical Workflow

```text
Sample EnergyPlus Model
          │
          ▼
Alter Occupancy
          │
          ▼
Baseline & Modified Simulations
          │
          ▼
Calibration
          │
          ▼
Surrogate Model Development
```

Each module is independent and can be executed separately depending on the desired workflow.

---

## Requirements

- Python 3.11+
- EnergyPlus 26.1.0
- Streamlit
- Geomeppy
- NumPy
- Pandas
- SciPy
- scikit-learn
- Plotly
- Matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

Ensure that EnergyPlus 26.1.0 is installed and accessible from your system.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/<username>/epMCP.git
cd epMCP
```

Run any standalone application, for example:

```bash
cd alter_occupancy
streamlit run final.py
```

or

```bash
cd calibration_basic
streamlit run calibsite.py
```

---

## Applications

| Module | Purpose |
|---------|---------|
| Alter Occupancy | Occupancy modification and comparative EnergyPlus simulations |
| Calibration Basic | Automated occupancy calibration using numerical optimization |
| surrogateEP | Experimental surrogate-assisted calibration methods |
| Sample Models | Example EnergyPlus models and weather files |

---

## Technologies

- EnergyPlus 26.1.0
- Geomeppy
- Streamlit
- Python
- SciPy
- scikit-learn
- Pandas
- NumPy
- Plotly
- Matplotlib

---

## Future Work

Planned extensions include:

- Additional EnergyPlus parameter modification tools
- Multi-parameter calibration
- Extended MCP tool interfaces
- Physics-guided surrogate models
- AI agent integration for automated building analysis

---

## Acknowledgements

This project builds upon:

- EnergyPlus
- Geomeppy
- U.S. DOE Commercial Prototype Building Models
- EnergyPlus example building models
