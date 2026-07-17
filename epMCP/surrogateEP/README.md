## Folder Structure

```
surrogateEP/
├── surrogate.py
└── dee.py
```

---

## surrogate.py

A proof-of-concept implementation of surrogate-assisted calibration.

### Workflow

1. Generate occupancy multiplier samples using Latin Hypercube Sampling (LHS).
2. Execute multiple EnergyPlus simulations in parallel.
3. Collect hourly whole-building electricity demand profiles.
4. Train a Random Forest surrogate model to learn the relationship between occupancy multiplier and annual electricity demand.
5. Replace repeated EnergyPlus simulations with fast surrogate predictions during optimization.
6. Estimate the optimal occupancy multiplier using the Nelder-Mead optimization algorithm.

### Features

- Latin Hypercube sampling
- Parallel EnergyPlus simulations
- Random Forest regression
- Surrogate-assisted optimization
- Significant reduction in calibration runtime after model training

---

## dee.py

An improved surrogate modeling framework that extends the initial approach by incorporating environmental features and learning residual energy demand instead of absolute demand.

### Improvements over `surrogate.py`

- Extracts weather information directly from EPW files.
- Includes:
  - Outdoor dry-bulb temperature
  - Direct normal solar radiation
  - Hour of day
- Predicts the difference between baseline and modified simulations (delta learning).
- Uses Ridge Regression for continuous surrogate modeling.
- Converts annual simulation data into hourly training samples, substantially increasing the effective training dataset.

### Workflow

1. Load cached EnergyPlus simulation results.
2. Extract weather features from the EPW file.
3. Generate hourly feature vectors consisting of:
   - Occupancy multiplier
   - Outdoor temperature
   - Solar radiation
   - Hour of day
4. Train a continuous surrogate model on hourly residual energy demand.
5. Perform rapid calibration using the trained surrogate and Nelder-Mead optimization.

### Advantages

- Better generalization than direct annual-profile prediction.
- Incorporates environmental effects into the surrogate model.
- Larger effective training dataset through hourly decomposition.
- Faster optimization while maintaining calibration accuracy.

---

## Purpose

Running EnergyPlus repeatedly during calibration is computationally expensive. The scripts in this folder investigate surrogate-based alternatives that approximate simulation outputs with machine learning models, reducing optimization time from minutes per evaluation to milliseconds after training.

These implementations represent experimental research toward scalable, data-driven calibration methods for building energy models and provide a foundation for future physics-informed machine learning approaches.

---

## Dependencies

- Python 3.x
- NumPy
- Pandas
- SciPy
- scikit-learn
- Geomeppy
- EnergyPlus

---

## Status

This folder contains experimental research code and prototype implementations. The primary calibration workflow is available separately through the Streamlit application, while these scripts focus on investigating surrogate-based acceleration techniques.
