import streamlit as st
import pandas as pd
from pathlib import Path

def weeksim(baseout_dir: str):
    st.subheader("Season-Scale Weekly Footprints")
    st.caption("Contrast structural performance across your most extreme systemic operational profiles.")

    base_csv_file = Path(baseout_dir) / "eplusout.csv" if baseout_dir else None

    if not base_csv_file or not base_csv_file.exists():
        st.warning("Baseline dataset missing. Finalize baseline model execution to unlock profiles.")
        return

    try:
        df_base = pd.read_csv(base_csv_file)
        df_base.columns = df_base.columns.str.strip()

        demand_col = [col for col in df_base.columns if 'Electricity Demand Rate' in col]
        if not demand_col:
            st.error("Could not find 'Electricity Demand Rate' column inside database arrays.")
            return
            
        col_name = demand_col[0]
        df_base['Demand_kW'] = pd.to_numeric(df_base[col_name], errors='coerce').dropna() / 1000.0

        # --- 1. TYPICAL WEEK LOGIC (168 Hours Array) ---
        df_base['Hour_of_Week'] = df_base.index % 168
        typical_week = df_base.groupby('Hour_of_Week')['Demand_kW'].mean().tolist()

        # --- 2. HOTTEST WEEK SEARCH LOGIC ---
        rolling_weekly_energy = df_base['Demand_kW'].rolling(window=168).sum()
        
        peak_week_end_hour = int(rolling_weekly_energy.idxmax())
        peak_week_start_hour = peak_week_end_hour - 167
        hottest_week = df_base['Demand_kW'].iloc[peak_week_start_hour:peak_week_end_hour + 1].tolist()

        # --- 3. COLDEST WEEK SEARCH LOGIC ---
        # Find the 168-hour window with the lowest continuous electrical power signature sum
        cold_week_end_hour = int(rolling_weekly_energy.idxmin())
        cold_week_start_hour = cold_week_end_hour - 167
        coldest_week = df_base['Demand_kW'].iloc[cold_week_start_hour:cold_week_end_hour + 1].tolist()

        # Padding checks for safe array sizes
        if len(hottest_week) < 168: hottest_week += [0] * (168 - len(hottest_week))
        if len(coldest_week) < 168: coldest_week += [0] * (168 - len(coldest_week))

        # --- 4. DATA CONSOLIDATION & RENDERING ---
        weekly_chart_df = pd.DataFrame({
            "Hour of the Week": list(range(1, 169)),
            "Coldest Winter Week Profile": coldest_week,
            "Typical/Average Week Baseline": typical_week,
            "Hottest Peak Summer Week": hottest_week
        }).set_index("Hour of the Week")

        # Summary Display Matrix Cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Annual Minimum Load", f"{df_base['Demand_kW'].min():.2f} kW")
        m2.metric("Annual Average Load", f"{df_base['Demand_kW'].mean():.2f} kW")
        m3.metric("Annual Maximum Load", f"{df_base['Demand_kW'].max():.2f} kW")

        # Insert this directly inside your render_weekly_stress_profiles function after finding index maximums:
        def get_calendar_date_range(start_hour):
            day_idx = start_hour // 24
            start_date = pd.to_datetime(day_idx, unit='D', origin='2026-01-01') # Adjust base year if necessary
            end_date = start_date + pd.Timedelta(days=6)
            return start_date.strftime('%b %d'), end_date.strftime('%b %d')

        hottest_start, hottest_end = get_calendar_date_range(peak_week_start_hour)
        coldest_start, coldest_end = get_calendar_date_range(cold_week_start_hour)

        m1, m2 = st.columns(2)
        m1.metric("Hottest Week Duration", f"{hottest_start} to {hottest_end}")
        m2.metric("Coldest Week Duration", f"{coldest_start} to {coldest_end}")

        st.caption("Interactive timeline mapping (Hour 1/Monday Morning through Hour 168/Sunday Night):")
        st.line_chart(weekly_chart_df, color=["#3182bd", "#2ca02c", "#d62728"])

    except Exception as e:
        st.error(f"Failed to generate weekly climatic stress curves: {e}")

# Code snippet to insert inside your heatmap module:
def heatmapzone(df, selected_zone):
    """
    Safely isolates the 8760 temperature data array for the targeted zone,
    handling any case-sensitivity variations gracefully.
    """
    # Clean up column names and search query to ensure a clean match
    selected_zone_clean = str(selected_zone).strip().upper()
    
    # 🛠️ Case-Insensitive Match Wrapper:
    match_col = [
        c for c in df.columns 
        if selected_zone_clean in c.upper() and 'ZONE AIR TEMPERATURE' in c.upper()
    ]
    
    if match_col:
        # Pull out the continuous 8,760 hourly data stream array
        flat_temps = pd.to_numeric(df[match_col[0]], errors='coerce').dropna().values
        
        # Check if we have exactly 8,760 hours before attempting to shape the grid matrix
        if len(flat_temps) == 8760:
            heatmap_matrix = flat_temps.reshape(365, 24)
            return heatmap_matrix
        else:
            # If simulation is partial, pad or crop safely to avoid reshaping crashes
            padded_temps = pd.Series(flat_temps).reindex(range(8760), fill_value=22.0).values
            return padded_temps.reshape(365, 24)
            
    return None