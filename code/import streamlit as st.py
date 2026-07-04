import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from extractidf import zones
from init import init, justcsv
from appsupport import zonearray
from alteridf import altselect

st.set_page_config(page_title="IDF Zone Selector", layout="wide")
st.title("EnergyPlus IDF Zone Extraction Interface")
st.divider()

# --- STEP 1: FILE PATH INPUTS ---
st.subheader("Configure Model Environment Paths")
col1, col2, col3 = st.columns(3)

with col1:
    idd_input = st.text_input(
        "Data Dictionary (IDD) Absolute Path", 
        value=r"C:\EnergyPlusV26-1-1\Energy+.idd"
    )

with col2:
    idf_input = st.text_input(
        "Target Model (IDF) Absolute Path", 
        placeholder=r"C:\Users\annaa\Downloads\IISC\sim2\ASHRAE901_OfficeLarge_STD2019_NewDelhi.idf"
    )

with col3:
    epw_input = st.text_input(
        "Weather File (EPW) Absolute Path",
        placeholder=r"C:\Users\annaa\Downloads\IISC\sim2\IND_DL_New.Delhi-Gandhi.Intl.AP.421810_TMYx.2009-2023.epw"
    )

col4, col5 = st.columns(2)
with col4:
    baseout = st.text_input(
        "Baseline Simulation Output Directory",
        placeholder=r"C:\Users\...\sim_results"
    )
with col5:
    altout = st.text_input(
        "Altered Simulation Output Directory",
        placeholder=r"C:\Users\...\modidf"
    )


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

            base_csv = Path(baseout)/"eplusout.csv"
            if not base_csv.exists():
                with st.spinner("Executing initial baseline simulation"):
                    idf_object.epw = str(epw_path)
                    justcsv(idf_object, epw_path, baseout)
        except Exception as e:
            st.error(f"Pipeline Failure during parsing: {e}")

# --- STEP 3: INTERACTIVE DROPBOX (EXECUTED OUTSIDE TRY BLOCK) ---
if initialized:
    st.success(f"Successfully initialized model! Found {len(zone_options)} active thermal zones.")
    st.divider()
    
    st.subheader("Selection Framework")
    selected_zones = st.multiselect(
        "Select Target Thermal Zones to Modify:",
        options=zone_options,
        help="You can search and select multiple zones from your model data stream."
    )
    
    # Track selection updates dynamically
    if selected_zones:
        st.write(f"**Currently Selected ({len(selected_zones)}):**")
        st.json(selected_zones)

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
        if st.button("🔥 Run Altered Variant Simulation"):
            if not altout:
                st.error("Execution Terminated: Please supply an Altered Simulation Output Directory path.")
            else:
                try:
                    with st.spinner("Modifying target zones and spinning up altered simulation variant..."):
                        # Call your specialized clean altselect function using your loaded idf instance
                        modified_idf = altselect(
                            idf_object=idf_object,
                            selected_zones=selected_zones,
                            multiplier=multiplier_param,
                            save_directory=altout,
                            new_filename="modified_model.idf"
                        )
                        
                        if modified_idf is not None:
                            # Bypass geomeppy internal attribute lookups cleanly
                            modified_idf.epw = str(epw_path)
                            
                            # Fire simulation loop into the alternate directories sandbox
                            justcsv(modified_idf, epw_path, altout)
                            st.success(f"🎉 Variant Run Successful! Output compiled cleanly inside: {altout}")
                        else:
                            st.error("Pipeline Failure: The modification loop could not identify target zone entities.")
                            
                except Exception as e:
                    st.error(f"Pipeline Failure running altered iteration: {e}")    
    else:
        st.info("Please select one or more zones to proceed with parameter adjustments.")
elif idf_input == "":
    st.info("Provide a valid IDF file path above to extract and view the model's zone array.")


