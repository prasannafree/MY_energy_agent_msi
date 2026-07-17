# Sample Models

This directory contains the reference EnergyPlus models and supporting files used to test and demonstrate the applications in this repository.

These models serve as example inputs for occupancy modification, calibration, surrogate modeling, and visualization workflows.

---

## Contents

```
sample_models/
├── Energy+.idd
├── Delhi.epw
├── LargeOffice_Delhi.idf
├── Denver.epw
└── Apartment_Denver.idf
```

---

## Included Models

### 1. DOE Large Office Prototype (Delhi)

**Files**

- `LargeOffice_Delhi.idf`
- `Delhi.epw`

**Description**

A U.S. Department of Energy (DOE) Commercial Prototype Building Model representing a large office building. The model has been paired with a Delhi weather file to evaluate building performance under Indian climatic conditions.

**Primary Uses**

- Occupancy modification studies
- Building energy calibration
- Surrogate model development
- Comparative simulation analyses

---

### 2. EnergyPlus Apartment Example (Denver)

**Files**

- `Apartment_Denver.idf`
- `Denver.epw`

**Description**

An apartment building example distributed with the standard EnergyPlus installation. This model is used as a lightweight test case for validating simulation workflows and occupancy modification routines.

**Primary Uses**

- Workflow validation
- Rapid testing
- Demonstration of application features

---

## Weather Files

### Delhi.epw

Weather data representing Delhi, India, used with the DOE Large Office prototype.

### Denver.epw

Weather data representing Denver, Colorado, USA, used with the EnergyPlus apartment example.

---

## EnergyPlus Data Dictionary

### Energy+.idd

The EnergyPlus Input Data Dictionary (IDD) corresponding to the EnergyPlus version used in this project.

The IDD file defines the schema for EnergyPlus input files and is required by Geomeppy for parsing and editing IDF models.

---

## Usage

These files are intended as ready-to-use example inputs for the applications contained in this repository.

When running the Streamlit applications:

- Select the appropriate `Energy+.idd` file.
- Load one of the provided `.idf` models.
- Pair it with its corresponding `.epw` weather file.
- Specify output directories for simulation results.

---

## Sources

- **DOE Large Office Prototype:** U.S. Department of Energy (DOE) Commercial Prototype Building Models.
- **Apartment Example:** Distributed with the EnergyPlus example models.
- **Weather Files:** EnergyPlus Weather (EPW) format.
