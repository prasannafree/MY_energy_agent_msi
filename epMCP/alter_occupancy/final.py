import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from extractidf import zones
from init import init, justcsv, juzcsv, runperiod
from appsupport import zonearray, appintro
from altnew import altglo
from plotter import hourlycomplot, basecsv
from baseviz import baseviz
from weeksim import weeksim, heatmapzone
import plotly.express as px
from datetime import date

idd_input, idf_input, epw_input, baseout, altout = appintro()

zone_options = []
initialized = False
idf_object = None

if idf_input:
    idd_path = Path(idd_input)
    idf_path = Path(idf_input)
    epw_path = Path(epw_input)
    
    if not idd_path.exists():
        st.error(f"Initialization Halted: IDD file not found at '{idd_path}'")
    elif not idf_path.exists():
        st.error(f"Initialization Halted: IDF file not found at '{idf_path}'")
    elif not epw_path.exists():
        st.error(f"Initialization Halted: EPW file not found at '{epw_path}'")
    else:
        try:
            # Safely parse file inside the error-handling boundary
            idf_object = init(idd_path, idf_path)
            zone_options = zonearray(idf_object)
            initialized = True

            st.subheader("Global Simulation Window Configuration")
            col_sim_start, col_sim_end = st.columns(2)
            with col_sim_start:
                sim_start = st.date_input("Simulation Start Date", value=None, key="sim_start")
            with col_sim_end:
                sim_end = st.date_input("Simulation End Date", value=None, key="sim_end")

            base_csv = Path(baseout)/"eplusout.csv"
            if not base_csv.exists():
                with st.spinner("Executing initial baseline simulation"):
                    idf_object.epw = str(epw_path)
                    runperiod(idf_object, sim_start, sim_end)
                    if sim_end:
                        juzcsv(idf_object, epw_path, baseout)
        except Exception as e:
            st.error(f"Pipeline Failure during parsing: {e}")

if initialized:
    st.success(f"Successfully initialized model! Found {len(zone_options)} active thermal zones.")
    st.divider()

    baseviz(idf_object=idf_object, baseout_dir=baseout)
    st.divider()
    weeksim(baseout)
    #st.subheader("Selection Framework")
    #selected_zones = st.multiselect(
        #"Select Target Thermal Zones to Modify:",
        #options=zone_options,
        #help="You can search and select multiple zones from your model data stream."
    #)
    st.markdown(f"### Spatial-Temporal Thermal Matrix")
    st.caption("Select an individual thermal zone to plot its hourly micro-climate variations across all 365 days of the year.")
    
    # User selects a single zone from the options extracted at initialization
    target_heatmap_zone = st.selectbox(
        "Select Zone for Heatmap Analysis:",
        options=zone_options,
        key="heatmap_zone_selector"
    )
    if target_heatmap_zone:
        base_csv_file = Path(baseout) / "eplusout.csv"
        
        if base_csv_file.exists():
            main_df = pd.read_csv(base_csv_file)
            main_df.columns = main_df.columns.str.strip()
            
            # 1. Pull the raw 365x24 numpy array from your baseviz function
            matrix_data = heatmapzone(main_df, target_heatmap_zone)
            
            if matrix_data is not None:
                st.markdown(f"#### 14-Day Micro-Climate Matrix: {target_heatmap_zone}")
                
                # 2. INTERACTIVE CALENDAR SELECTOR WIDGET
                # Anchor the selector to the simulation calendar timeline (Year 2017)
                selected_date = st.date_input(
                    "Select Entry Starting Date:",
                    value=date(2026, 7, 14),
                    min_value=date(2026, 1, 1),
                    max_value=date(2026, 12, 18),
                    key="heatmap_calendar_input"
                )
                
                # 3. CONVERT CALENDAR TIMESTAMP TO DAY NUMBER (1 to 365)
                # .timetuple().tm_yday extracts the absolute Julian day index automatically!
                start_day = pd.to_datetime(selected_date).timetuple().tm_yday
                
                # Calculate evaluation window end-bounds for visual display text
                base_date = pd.to_datetime(start_day - 1, unit='D', origin='2017-01-01')
                end_date = base_date + pd.Timedelta(days=13)
                
                st.info(f"📆 **Displaying Timeline Window:** {base_date.strftime('%B %d')} to {end_date.strftime('%B %d')} (Day Index: {start_day})")
                
                # 4. SLICE THE TARGET 2-WEEK MATRIX (Convert 1-based day input to 0-based index)
                start_idx = start_day - 1
                end_idx = start_idx + 14
                sliced_matrix = matrix_data[start_idx:end_idx, :]
                
                # 5. BUILD THE GRID DATAFRAME
                day_labels = [(base_date + pd.Timedelta(days=i)).strftime('%b %d') for i in range(14)]
                hour_labels = [f"{h:02d}:00" for h in range(24)]
                
                df_slice = pd.DataFrame(sliced_matrix, columns=hour_labels, index=day_labels)
                
                # 6. COMPACT NATIVE HEATMAP STYLE
                styled_slice = (
                    df_slice.style
                    .background_gradient(cmap="RdYlBu_r", axis=None)
                    .format(lambda x: f"{x:.1f}")  # Keep text clean to 1-decimal limit
                    .set_properties(**{
                        'width': '38px',     
                        'height': '22px',    
                        'text-align': 'center',
                        'font-size': '11px'   
                    })
                )
                
                # Render the final 14-day view block layout safely
                st.dataframe(styled_slice, width="stretch", height=400)
                
        else:
            st.error("Baseline eplusout.csv data file missing.")
        
    st.divider()
    
    # Track selection updates dynamically
    selected_zones = 1
    if selected_zones:

        st.divider()
        st.subheader("Occupancy Parameter Toggle")

        occupancy_percentage = st.slider(
            "Modify Occupancy Load for Selected Zones (%)",
            min_value = -100,
            max_value = 100,
            value = 50,
            step = 10,
            help = "Choose modification weight. +50% scales origional occupants by 1.5x."
        )

        multiplier_param = 1.0 + (occupancy_percentage / 100.0)
        st.info(f"Targeting: Original `Number of People` × **{multiplier_param:.2f}** for selected zones.")
        
        # 2. Trigger execution framework via an explicit click button
        if st.button("Run Altered Variant Simulation"):
            if not altout:
                st.error("Execution Terminated: Please supply an Altered Simulation Output Directory path.")
            else:
                try:
                    with st.spinner("Modifying model parameters globally and spinning up variant simulation..."):                    

                        modified_idf = altglo(
                            idf_object=idf_object,
                            parameter_type='people', # Hardcoded to 'people' to test your occupancy loop
                            multiplier=multiplier_param,
                            save_directory=altout,
                            new_filename="global_modified_model.idf"
                        )
                        st.write(modified_idf.idfobjects['PEOPLE'][0].Floor_Area_per_Person)
                        for p in modified_idf.idfobjects['PEOPLE'][:1]:
                            st.write("OBJECT NAME:", p.Name)
                            st.write("FIELDNAMES:")
                            st.write(p.fieldnames)

                            debug = {}

                            for field in p.fieldnames:
                                try:
                                    debug[field] = getattr(p, field)
                                except:
                                    debug[field] = "ERROR"

                            st.json(debug)
                        if modified_idf is not None:
                            # Bypass geomeppy internal attribute lookups cleanly
                            modified_idf.epw = str(epw_path)
                            runperiod(modified_idf, sim_start, sim_end)

                            # Fire simulation loop into the alternate directory sandbox
                            juzcsv(modified_idf, epw_path, altout)
                            st.success(f"Global Variant Run Successful! Output compiled cleanly inside: {altout}")
                        else:
                            st.error("Pipeline Failure: The global modification loop failed to update any tracking metrics.")
                            
                except Exception as e:
                    st.error(f"Pipeline Failure running altered iteration: {e}")    
    else:
        st.info("Please select one or more zones to proceed with parameter adjustments.")
elif idf_input == "":
    st.info("Provide a valid IDF file path above to extract and view the model's zone array.")


# --- TAB 3: HOURLY ANALYTICS ---
st.subheader("Hourly Performance Analytics Pane")
st.caption("Inspect and map operational energy load profiles across custom year-long timelines.")

st.write("BASEOUT:", baseout)
st.write("ALTOUT:", altout)


# Convert path directory inputs into strict system path instances
base_csv_file = Path(baseout) / "eplusout.csv" if baseout else None
alt_csv_file = Path(altout) / "eplusout.csv" if altout else None

if base_csv_file:
    st.write("Base CSV Path:", str(base_csv_file))
    st.write("Base Exists:", base_csv_file.exists())

if alt_csv_file:
    st.write("Alt CSV Path:", str(alt_csv_file))
    st.write("Alt Exists:", alt_csv_file.exists())

# Gatekeep execution loop until both source data files physically exist on disk
if base_csv_file and alt_csv_file and base_csv_file.exists() and alt_csv_file.exists():
    
    st.write("### Timeline Viewport Parameters")
    
    # 🛠️ THE CRITICAL MIGRATION FIX: Swapped out slider index integers for formal dual st.date_input parameters
    col_start, col_end = st.columns(2)
    
    with col_start:
        start_inp = st.date_input(
            "Evaluation Start Date", 
            value=date(2026, 7, 14),
            min_value=date(2026, 1, 1),
            max_value=date(2026, 12, 31),
            key="complot_start_date"
        )
        
    with col_end:
        end_inp = st.date_input(
            "Evaluation End Date", 
            value=date(2026, 7, 21),
            min_value=date(2026, 1, 1),
            max_value=date(2026, 12, 31),
            key="complot_end_date"
        )
    
    st.divider()

    if start_inp <= end_inp:
        # Clean handshake execution without parameter leaks or signature errors
        hourlycomplot(
            base_csv_path=base_csv_file,
            alt_csv_path=alt_csv_file,
            start_date=start_inp,  
            end_date=end_inp       
        ) 
       
else:
    st.info("Awaiting compilation datasets. Complete the baseline and variant simulation runs to update data visualization streams.")