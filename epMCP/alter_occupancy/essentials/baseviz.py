import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from extractidf import surfaces, windows, occupancy, materials

def baseviz(idf_object, baseout_dir: str):
    """
    Parses structural metadata and baseline CSV data to render
    pre-modification building performance analytics.
    """
    st.header("Baseline Model Overview")
    st.caption("Review spatial profiles and energy baselines prior to modifying parameters.")
    
    # -------------------------------------------------------------------------
    # STEP 1: EXTRACT ARCHITECTURAL & ENVELOPE METRICS
    # -------------------------------------------------------------------------
    try:
        df_surf = surfaces(idf_object)
        df_wind = windows(idf_object)
        df_occ = occupancy(idf_object)
        df_mat = materials(idf_object)
        
        # Calculate areas cleanly via DataFrame operations
        df_floors = df_surf[df_surf['Surface_Type'].str.lower() == 'floor'] if not df_surf.empty else pd.DataFrame()
        total_floor_area = float(df_floors['Area_m2'].sum()) if not df_floors.empty else 0.0
        
        df_walls = df_surf[df_surf['Surface_Type'].str.lower() == 'wall'] if not df_surf.empty else pd.DataFrame()
        total_wall_area = float(df_walls['Area_m2'].sum()) if not df_walls.empty else 0.0
        total_win_area = float(df_wind['Area_m2'].sum()) if not df_wind.empty else 0.0
        
        # Calculate Window-to-Wall Ratio (WWR)
        wwr = (total_win_area / total_wall_area * 100) if total_wall_area > 0 else 0.0
        
        # Sum design human capacity safely if present
        total_people = 0
        if not df_occ.empty and 'Total_People_Count' in df_occ.columns:
            total_people = int(df_occ['Total_People_Count'].dropna().sum())

    except Exception as e:
        st.error(f"Error parsing structural IDF metadata: {e}")
        return
    base_csv_file = Path(baseout_dir) / "eplusout.csv" if baseout_dir else None

    # Define df_base as an empty DataFrame initially as a fallback safety mesh
    df_base = pd.DataFrame()

    # 2. Safely read the CSV file if it exists
    if base_csv_file and base_csv_file.exists():
        df_base = pd.read_csv(base_csv_file)
        df_base.columns = df_base.columns.str.strip()
    else:
        with diag_tab1:
            st.warning("Baseline dataset 'eplusout.csv' missing or not yet generated. Run the simulation first!")
        return

    # Render Summary Cards Dashboard
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Gross Floor Area", f"{total_floor_area:.2f} m²")
    m_col2.metric("Total Glazing Area", f"{total_win_area:.2f} m²")
    m_col3.metric("Window-to-Wall Ratio (WWR)", f"{wwr:.1f}%")
    m_col4.metric("Peak Design Occupancy", f"{total_people} People" if total_people > 0 else "Area-Based")

    st.divider()

    # -------------------------------------------------------------------------
    # STEP 2: RENDER ENVELOPE & ENERGY SIMULATION VISUALIZATIONS
    # -------------------------------------------------------------------------
    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
            "Envelope Spatial Layout", 
            "Peak Demand Profiler", 
            "Material Thermal Properties"
        ])
        
    # --- TAB 1: WALL ORIENTATIONS ---
    with diag_tab1:
        st.write("#### Hourly Facility Energy Breakdown")
        
        if not df_base.empty:
            # 1. Flexible string matching for columns (handling variations in names)
            total_col = [c for c in df_base.columns if 'Facility Total Electricity Demand Rate' in c]
            cooling_col = [c for c in df_base.columns if 'Cooling:Electricity' in c]
            humid_col = [c for c in df_base.columns if 'Humidifier:Electricity' in c]
            dehumid_col = [c for c in df_base.columns if 'Dehumidifier:Electricity' in c]
            heating_col = [c for c in df_base.columns if 'Heating:NaturalGas' in c]
            lights_col = [c for c in df_base.columns if 'InteriorLights:Electricity' in c]
            equip_col = [c for c in df_base.columns if 'InteriorEquipment:Electricity' in c]
            fans_col = [c for c in df_base.columns if 'Fans:Electricity' in c]
            pumps_col = [c for c in df_base.columns if 'Pumps:Electricity' in c]

            if not total_col:
                st.error("Missing master 'Facility Total Electricity Demand Rate' column to calculate proportions.")
                return

            # 2. Extract raw mean values across the dataset
            total_raw = float(df_base[total_col[0]].mean())
            
            # Convert Total Demand: Watts to Watt-hours (1 Watt for 1 hour = 1 Wh)
            total_wh = total_raw 

            # Convert Subloads: Joules to Watt-hours (Joules / 3600 = Wh)
            subloads = {
                "Cooling": float(df_base[cooling_col[0]].mean()) / 3600.0 if cooling_col else 0.0,
                "Humidification": float(df_base[humid_col[0]].mean()) / 3600.0 if humid_col else 0.0,
                "Dehumidification": float(df_base[dehumid_col[0]].mean()) / 3600.0 if dehumid_col else 0.0,
                "Heating": float(df_base[heating_col[0]].mean()) / 3600.0 if heating_col else 0.0,
                "Interior Lighting": float(df_base[lights_col[0]].mean()) / 3600.0 if lights_col else 0.0,
                "Interior Equipment": float(df_base[equip_col[0]].mean()) / 3600.0 if equip_col else 0.0,
                "Fans": float(df_base[fans_col[0]].mean()) / 3600.0 if fans_col else 0.0,
                "Pumps": float(df_base[pumps_col[0]].mean()) / 3600.0 if pumps_col else 0.0,
            }

            # Filter out any subloads that are completely zero or failed to parse
            active_subloads = {k: v for k, v in subloads.items() if v > 0}

            # 3. Calculate the Miscellaneous Disparity Delta
            sum_tracked_subloads = sum(active_subloads.values())
            misc_wh = total_wh - sum_tracked_subloads

            # If there is a positive gap, assign it to Miscellaneous
            if misc_wh > 0.1:
                active_subloads["Miscellaneous"] = misc_wh
            
            # 4. Format into a DataFrame for Native Streamlit Plotting
            pie_df = pd.DataFrame({
                "Load Component": active_subloads.keys(),
                "Energy Consumption [Wh]": active_subloads.values()
            })

            st.caption("Distribution of building sub-loads relative to total metered facility demand:")
            
            # 5. Render Native Interactive Pie/Donut Chart via st.plotly_chart 
            # (Using basic plotly express since it natively supports pie structures seamlessly in Streamlit)
            import plotly.express as px
            
            fig = px.pie(
                pie_df, 
                names="Load Component", 
                values="Energy Consumption [Wh]",
                hole=0.4,  # Makes it a clean donut chart layout
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Upload an IDF baseline simulation model run to populate the energy share analysis.")
            
    # --- TAB 2: DEMAND LOADS ---
    with diag_tab2:
        st.write("#### Interactive Annual Load Duration Profile")
        base_csv_file = Path(baseout_dir) / "eplusout.csv" if baseout_dir else None
        
        if base_csv_file and base_csv_file.exists():
            try:
                df_base = pd.read_csv(base_csv_file)
                df_base.columns = df_base.columns.str.strip()
                
                demand_col = [col for col in df_base.columns if 'Electricity Demand Rate' in col]
                
                if demand_col:
                    col_name = demand_col[0]
                    
                    # 1. Clean, sort, and parse values to numeric floats
                    clean_series = pd.to_numeric(df_base[col_name], errors='coerce').dropna()
                    sorted_demand = clean_series.sort_values(ascending=False).astype(float).tolist()
                    
                    if not sorted_demand:
                        st.error("No valid numeric energy values were extracted from the column.")
                    else:
                        # 2. Build a Clean 1-Column Series using the hours directly as the structural Index
                        # This prevents Streamlit from mistaking your X-axis labels for Y-axis line data data
                        chart_series = pd.Series(
                            data=sorted_demand,
                            index=range(1, len(sorted_demand) + 1)
                        )
                        
                        st.caption("Inspect sorted building load signatures over the total 8760 timeline grid:")
                        
                        # 3. Pass the Series straight through. Streamlit automatically maps 
                        # the Series index to X (Hours) and the values to Y (Demand [W]).
                        st.line_chart(chart_series, color="#ff7f0e")
                else:
                    st.info("Baseline CSV tracked successfully, but total electricity demand rates are missing. Add tracking rules to inspect metrics.")
            except Exception as e:
                st.error(f"Error compiling load duration datasets: {e}")
        else:
            st.warning("Baseline simulation dataset missing. Run baseline workflow initialization to unlock data streaming profiles.")
            
    # --- TAB 3: MATERIAL THERMAL VALUE COMPASS ---
    with diag_tab3:
        st.write("#### Material Envelope Thermal Specifications")
        if not df_mat.empty:
            st.caption("Review raw material profiles extracted from your layout construction layers:")
            
            # Build an interactive data preview layout framework
            df_thermal = df_mat.copy()
            
            # Engineering cross-calculation step: R-value indicator computation (Thickness / Conductivity)
            # convert thickness from mm back to meters dynamically for structural accurate R-value mapping
            df_thermal['Calculated_R_Value_m2K_W'] = (df_thermal['Thickness_mm'] / 1000.0) / df_thermal['Conductivity_W_mK']
            
            # Clean up display headers for readability
            df_thermal.columns = [
                "Material Reference ID", "Thickness [mm]", 
                "Conductivity [W/m·K]", "Density [kg/m³]", "Inferred R-Value [m²·K/W]"
            ]
            
            st.dataframe(df_thermal, use_container_width=True, hide_index=True)

'''
    st.subheader("Analysis & Load Diagnostics")
    geo_tab1, geo_tab2 = st.tabs(["Wall Orientations (Azimuth)", "Baseline Demand Profiles"])
    
    with geo_tab1:
        if not df_walls.empty:
            st.markdown("#### Spatial Envelope Orientation")
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.hist(df_walls['Azimuth_deg'], bins=12, color='#2b5c8f', edgecolor='w', alpha=0.85)
            ax.set_title("Wall Surface Area Distribution by Orientation", fontsize=10, weight='bold')
            ax.set_xlabel("Azimuth Angle (Degrees, North = 0° / 360°)", fontsize=8)
            ax.set_ylabel("Count of Wall Surfaces", fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig) # Prevent memory leaks
        else:
            st.info("No detailed wall orientation parameters available.")
            
    with geo_tab2:
        base_csv_file = Path(baseout_dir) / "eplusout.csv" if baseout_dir else None
        
        if base_csv_file and base_csv_file.exists():
            try:
                # Load baseline output cleanly
                df_base = pd.read_csv(base_csv_file)
                df_base.columns = df_base.columns.str.strip()
                
                # Dynamic matching for total facility electricity demand column
                demand_col = [col for col in df_base.columns if 'Facility Total Electricity Demand Rate' in col]
                
                if demand_col:
                    st.markdown("#### Peak Baseline Power Load Profile")
                    col_name = demand_col[0]
                    
                    # Generate a quick Load Duration Curve (Sorted Peak to Base)
                    sorted_demand = df_base[col_name].sort_values(ascending=False).values
                    
                    fig2, ax2 = plt.subplots(figsize=(7, 3))
                    ax2.plot(sorted_demand, color='#d95f02', lw=2, label='Baseline Demand')
                    ax2.set_title("Annual Load Duration Curve", fontsize=10, weight='bold')
                    ax2.set_xlabel("Sorted Operational Hours (1 to 8760)", fontsize=8)
                    ax2.set_ylabel("Demand Rate (W)", fontsize=8)
                    ax2.fill_between(range(len(sorted_demand)), sorted_demand, color='#d95f02', alpha=0.15)
                    ax2.grid(linestyle='--', alpha=0.5)
                    st.pyplot(fig2)
                    plt.close(fig2)
                else:
                    st.info("💡 CSV compiled successfully, but 'Facility Total Electricity Demand Rate' was not tracked. Add it to output variables to view curves.")
            except Exception as e:
                st.error(f"Failed to process baseline CSV metrics: {e}")
        else:
            st.warning("⚠️ Baseline simulation dataset not yet found on disk. Run initialization to unlock energy profile charts.")
'''