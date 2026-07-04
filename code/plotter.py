import pandas as pd
import streamlit as st
from pathlib import Path

def hourlycomplot(base_csv_path: Path, alt_csv_path: Path, start_day: int, end_day: int):
    """
    Ingests file paths for baseline and altered EnergyPlus CSV files, slices them
    dynamically by user-defined day inputs, and displays an interactive comparison graph.
    """
    try:
        # --- STEP 1: Load and Clean Both Datasets ---
        df_base = pd.read_csv(base_csv_path)
        df_base.columns = df_base.columns.str.strip()

        df_alt = pd.read_csv(alt_csv_path)
        df_alt.columns = df_alt.columns.str.strip()

        # Dynamically extract tracking columns to bypass version string mismatches
        energy_col_base = [col for col in df_base.columns if "Electricity Demand Rate" in col][0]
        energy_col_alt = [col for col in df_alt.columns if "Electricity Demand Rate" in col][0]

        # --- STEP 2: Establish Identical Chronological Metrics ---
        for df in [df_base, df_alt]:
            df['Day_of_Year'] = df.index // 24

        # --- STEP 3: Filter Both Datasets based on the runtime inputs ---
        # Note: If user enters identical days (e.g. 35 to 35), we force a minimum 1-day viewport
        if start_day == end_day:
            end_day += 1
        window_base = df_base[(df_base['Day_of_Year'] >= start_day) & (df_base['Day_of_Year'] <= end_day)].copy()
        window_alt = df_alt[(df_alt['Day_of_Year'] >= start_day) & (df_alt['Day_of_Year'] <= end_day)].copy()

        if window_base.empty or window_alt.empty:
            st.warning("The selected timeline window contains no valid database rows.")
            return

        # Synchronize chronological timelines sequentially starting from index hour zero
        window_base['Timeline_Hours'] = range(len(window_base))
        window_alt['Timeline_Hours'] = range(len(window_alt))

        # --- STEP 4: Build Metrics and Display Delta Cards ---
        avg_base = window_base[energy_col_base].mean()
        avg_alt = window_alt[energy_col_alt].mean()
        pct_diff = ((avg_alt - avg_base) / avg_base) * 100 if avg_base != 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Avg Baseline (Days {start_day}-{end_day})", f"{avg_base/1000:.2f} kW")
        m2.metric(f"Avg Altered (Days {start_day}-{end_day})", f"{avg_alt/1000:.2f} kW")
        m3.metric("Load Shift", f"{pct_diff:+.3f}%")

        st.divider()

        # --- STEP 5: Consolidate data structure for Streamlit Native Graph Interface ---
        # Using st.line_chart maps timestamps interactively, avoiding crowded static charts.
        st.write(f"#### Demand Profile Timeline Analysis (Day {start_day} to Day {end_day})")
        
        # Build a temporary charting dataframe
        chart_df = pd.DataFrame({
            "Continuous Timeline Hours": window_base['Timeline_Hours'],
            "Baseline Case Profile": window_base[energy_col_base].values,
            "Altered Case Profile": window_alt[energy_col_alt].values
        }).set_index("Continuous Timeline Hours")

        # Let user handle cluttered layout warning triggers
        days_selected = end_day - start_day + 1
        if days_selected > 30:
            st.warning(f"Displaying dense footprint: {days_selected} days ({len(chart_df)} hours) of simulation rows.")
        
        # Plot interactive frame line chart
        st.line_chart(chart_df, color=["#1f77b4", "#ff7f0e"])

    except IndexError:
        st.error("Pipeline Error: Could not locate electricity demand columns inside the target compilation spreadsheets.")
    except Exception as e:
        st.error(f"Visualization Component Failure: {e}")


def basecsv(base_dir_path: Path, day_of_year: int = 1):
    """
    Load baseline eplusout.csv from the provided base output directory and plot the selected
    day-of-year window using the baseline energy demand column.
    """
    base_csv_path = Path(base_dir_path) / "eplusout.csv"
    if not base_csv_path.exists():
        st.error(f"Baseline CSV file not found at '{base_csv_path}'")
        return

    try:
        df_base = pd.read_csv(base_csv_path)
        df_base.columns = df_base.columns.str.strip()

        # Prefer an explicit day-of-year column if it exists; otherwise infer from hourly rows.
        day_column = next(
            (
                col for col in df_base.columns
                if col.replace(" ", "").lower() in {"dayofyear", "day_of_year", "dayofyear"}
            ),
            None,
        )

        if day_column:
            df_base["Day_of_Year"] = df_base[day_column].astype(int)
        else:
            df_base["Day_of_Year"] = (df_base.index // 24) + 1

        selected_day_df = df_base[df_base["Day_of_Year"] == day_of_year].copy()
        if selected_day_df.empty:
            st.warning(f"Selected day {day_of_year} is not available in the baseline CSV dataset.")
            return

        energy_columns = [
            col for col in df_base.columns
            if "electricity" in col.lower() and "demand" in col.lower()
        ]
        if energy_columns:
            energy_col = energy_columns[0]
        else:
            numeric_columns = selected_day_df.select_dtypes(include="number").columns.tolist()
            energy_col = numeric_columns[0] if numeric_columns else None

        if energy_col is None:
            st.error("Unable to identify a numeric energy variable in the baseline CSV.")
            return

        selected_day_df["Hour"] = range(1, len(selected_day_df) + 1)
        selected_day_df[energy_col] = pd.to_numeric(selected_day_df[energy_col], errors="coerce")
        average_value = selected_day_df[energy_col].mean()

        st.metric(f"Baseline Day {day_of_year} Avg", f"{average_value/1000:.2f} kW")
        st.write(f"#### Baseline profile for Day {day_of_year}")

        chart_df = selected_day_df.set_index("Hour")[[energy_col]].copy()
        chart_df.columns = ["Baseline Electricity Demand [W]"]
        chart_df.index.name = "Hour"

        if len(chart_df) > 48:
            st.warning("Selected baseline window contains more than 48 rows. Confirm simulation timestep and data resolution.")

        st.line_chart(chart_df)

    except Exception as e:
        st.error(f"Baseline plotting failed: {e}")

