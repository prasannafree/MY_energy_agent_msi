import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from extractidf import zones
from init import init, justcsv
from appsupport import zonearray, appintro
from altnew import altglo
from plotter import hourlycomplot, basecsv
from baseviz import baseviz

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

            base_csv = Path(baseout)/"eplusout.csv"
            if not base_csv.exists():
                with st.spinner("Executing initial baseline simulation"):
                    idf_object.epw = str(epw_path)
                    justcsv(idf_object, epw_path, baseout)
        except Exception as e:
            st.error(f"Pipeline Failure during parsing: {e}")

if initialized:
    st.success(f"Successfully initialized model! Found {len(zone_options)} active thermal zones.")
    st.divider()

    baseviz(idf_object=idf_object, baseout_dir=baseout)
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
# 2. Trigger execution framework via an explicit click button
        if st.button("🔥 Run Altered Variant Simulation"):
            if not altout:
                st.error("Execution Terminated: Please supply an Altered Simulation Output Directory path.")
            else:
                try:
                    with st.spinner("Modifying model parameters globally and spinning up variant simulation..."):
                        # Import the global modifier function cleanly
                        from alteridf import alteridf_global
                        
                        # Call the global modifier function instead of altselect
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
                            
                            # Fire simulation loop into the alternate directory sandbox
                            justcsv(modified_idf, epw_path, altout)
                            st.success(f"🎉 Global Variant Run Successful! Output compiled cleanly inside: {altout}")
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
    
    # Capture raw range integers from user input parameters
    start_day_input, end_day_input = st.slider(
        "Select timeline bounding window range (Days 1 to 365):",
        min_value=1,
        max_value=365,
        value=(195, 202), # Defaulting to your peak summer mid-July week parameter
        step=1
    )
    
    st.divider()

    
    # Call the standalone visualization component cleanly
    hourlycomplot(
        base_csv_path=base_csv_file,
        alt_csv_path=alt_csv_file,
        start_day=start_day_input,
        end_day=end_day_input
    )
    
else:
    st.info("Awaiting compilation datasets. Complete the baseline and variant simulation runs to update data visualization streams.")