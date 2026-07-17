import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
import datetime

def hourlycomplot(base_csv_path: Path, alt_csv_path: Path, start_date: datetime.date, end_date: datetime.date):
    """
    Ingests file paths for baseline and altered EnergyPlus CSV files, slices them
    dynamically using calendar dates, and displays an interactive comparison graph with day dividers.
    """
    try:
        # --- STEP 1: Load and Clean Both Datasets ---
        df_base = pd.read_csv(base_csv_path)
        df_base.columns = df_base.columns.str.strip()

        df_alt = pd.read_csv(alt_csv_path)
        df_alt.columns = df_alt.columns.str.strip()

        # Dynamically extract tracking columns
        energy_col_base = [col for col in df_base.columns if "Electricity Demand Rate" in col][0]
        energy_col_alt = [col for col in df_alt.columns if "Electricity Demand Rate" in col][0]

        # --- STEP 2: Establish True Chronological Datetime Indices (Assuming Baseline Year 2026) ---
        for df in [df_base, df_alt]:
            df['Timestamp'] = pd.date_range(start='2026-01-01 01:00:00', periods=len(df), freq='h')

        # --- STEP 3: Filter Both Datasets Based on Calendar Date Objects ---
        # Convert date inputs to datetime boundaries for robust pandas slicing
        t_start = pd.to_datetime(start_date)
        t_end = pd.to_datetime(end_date) + pd.Timedelta(hours=23) # Capture the full final day

        window_base = df_base[(df_base['Timestamp'] >= t_start) & (df_base['Timestamp'] <= t_end)].copy()
        window_alt = df_alt[(df_alt['Timestamp'] >= t_start) & (df_alt['Timestamp'] <= t_end)].copy()

        if window_base.empty or window_alt.empty:
            st.warning("The selected calendar window contains no valid simulation rows.")
            return

        # --- STEP 4: Build Metrics and Display Delta Cards ---
        avg_base = window_base[energy_col_base].mean()
        avg_alt = window_alt[energy_col_alt].mean()
        pct_diff = ((avg_alt - avg_base) / avg_base) * 100 if avg_base != 0 else 0

        peak_base = window_base[energy_col_base].max()
        peak_alt = window_alt[energy_col_alt].max()
        peak_diff_kw = (peak_alt - peak_base) / 1000.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Baseline Case", f"{avg_base/1000:.2f} kW")
        m2.metric("Avg Altered Case", f"{avg_alt/1000:.2f} kW")
        m3.metric("Avg Load Shift", f"{pct_diff:+.2f}%")
        m4.metric("Peak Demand Delta", f"{peak_diff_kw:+.2f} kW", delta=f"{peak_diff_kw:.2f} kW", delta_color="inverse")

        st.divider()

        # --- STEP 5: Consolidate data structure for Altair Layered Chart ---
        start_str = start_date.strftime('%b %d, %Y')
        end_str = end_date.strftime('%b %d, %Y')
        st.write(f"#### Demand Profile Timeline Analysis ({start_str} to {end_str})")

        chart_df = pd.DataFrame({
            "Timestamp": window_base['Timestamp'],
            "Baseline Case": window_base[energy_col_base].values / 1000.0,
            "Altered Case": window_alt[energy_col_alt].values / 1000.0
        })

        melted_df = chart_df.melt(id_vars=["Timestamp"], value_vars=["Baseline Case", "Altered Case"],
                                  var_name="Simulation Case", value_name="Demand (kW)")

        # Base Line Curves
        base_lines = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('Timestamp:T', title='Timeline Horizon', axis=alt.Axis(format='%b %d %I %p', labelAngle=-45)),
            y=alt.Y('Demand (kW):Q', title='Operational Load Density (kW)'),
            color=alt.Color('Simulation Case:N', scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]))
        )

        # Generate Day Dividers at Midnight (Hour == 0)
        midnight_bounds = chart_df[
            (chart_df['Timestamp'].dt.hour == 0) & (chart_df['Timestamp'] != chart_df['Timestamp'].iloc[0])
        ]
        day_dividers = alt.Chart(midnight_bounds).mark_rule(
            color='#555555',
            strokeWidth=1.2,
            strokeDash=[4, 4]
        ).encode(x='Timestamp:T')

        # Change this configuration block at the bottom of hourlycomplot:
        final_chart = (base_lines + day_dividers).properties(
            height=400  # 🌟 REMOVE width="stretch" from here
        ).configure_axis(
            grid=False
        )

        # 🎯 THE FIX: Pass use_container_width=True directly into the Streamlit call
        st.altair_chart(final_chart, use_container_width=True)

    except Exception as e:
        st.error(f"Visualization Component Failure: {e}")


def basecsv(base_dir_path: Path, start_date: datetime.date, end_date: datetime.date):
    """
    Load baseline eplusout.csv from the provided base output directory and plot the selected
    calendar date range using the baseline energy demand column.
    """
    base_csv_path = Path(base_dir_path) / "eplusout.csv"
    if not base_csv_path.exists():
        st.error(f"Baseline CSV file not found at '{base_csv_path}'")
        return

    try:
        df_base = pd.read_csv(base_csv_path)
        df_base.columns = df_base.columns.str.strip()

        # Generate standard timestamp dimension
        df_base['Timestamp'] = pd.date_range(start='2026-01-01 01:00:00', periods=len(df_base), freq='h')

        t_start = pd.to_datetime(start_date)
        t_end = pd.to_datetime(end_date) + pd.Timedelta(hours=23)

        selected_df = df_base[(df_base['Timestamp'] >= t_start) & (df_base['Timestamp'] <= t_end)].copy()
        if selected_df.empty:
            st.warning(f"The selected timeframe window is outside the bounds of this simulation run dataset.")
            return

        energy_columns = [
            col for col in df_base.columns
            if "electricity" in col.lower() and "demand" in col.lower()
        ]
        energy_col = energy_columns[0] if energy_columns else selected_df.select_dtypes(include="number").columns[0]

        selected_df[energy_col] = pd.to_numeric(selected_df[energy_col], errors="coerce")
        average_value = selected_df[energy_col].mean()

        start_str = start_date.strftime('%b %d')
        end_str = end_date.strftime('%b %d')
        
        st.metric(f"Baseline Window Avg ({start_str}-{end_str})", f"{average_value/1000:.2f} kW")
        st.write(f"#### Baseline Profile Matrix Frame ({start_str} to {end_str})")

        chart_df = selected_df.set_index("Timestamp")[[energy_col]].copy()
        chart_df.columns = ["Baseline Electricity Demand (kW)"]
        chart_df["Baseline Electricity Demand (kW)"] /= 1000.0  # Keep chart cleanly in kW

        # Native Streamlit Chart (Interactive datetimes map cleanly automatically here too)
        st.line_chart(chart_df)

    except Exception as e:
        st.error(f"Baseline plotting failed: {e}")