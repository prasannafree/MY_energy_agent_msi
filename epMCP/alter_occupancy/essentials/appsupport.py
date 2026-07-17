import pandas as pd
from extractidf import zones
import streamlit as st
from pathlib import Path

def zonearray(idf_object) -> list:
    zones_df = zones(idf_object)
    zone_array = zones_df['Zone_Name'].tolist()

    return zone_array

import streamlit as st
from pathlib import Path

def appintro():

    st.set_page_config(page_title="EnergyPlus Backend Simulator", layout="wide")
    st.title("EnergyPlus User Interface")
    st.divider()

    st.subheader("Configure Model Environment Paths")
    col1, col2, col3 = st.columns(3)

    with col1:
        idd_input = st.text_input(
            "Data Dictionary (IDD) Absolute Path", 
            value=r"C:\EnergyPlusV26-1-1\Energy+.idd"
        )

    WORKING_DIR = Path("./active_model_files")
    WORKING_DIR.mkdir(exist_ok=True)
    
    idf_input = ""
    with col2:
        uploaded_idf = st.file_uploader("Upload Target Model (IDF)", type=["idf"])
        if uploaded_idf is not None:
            idf_bytes = uploaded_idf.getvalue()
            idf_path_target = WORKING_DIR / uploaded_idf.name
            idf_path_target.write_bytes(idf_bytes)
            idf_input = str(idf_path_target.resolve())
            st.caption(f"Loaded: `{uploaded_idf.name}`")

    epw_input = ""
    with col3:
        uploaded_epw = st.file_uploader("Upload Weather File (EPW)", type=["epw"])
        if uploaded_epw is not None:
            weather_bytes = uploaded_epw.getvalue()
            epw_path_target = WORKING_DIR / uploaded_epw.name
            epw_path_target.write_bytes(weather_bytes)
            epw_input = str(epw_path_target.resolve())
            st.caption(f"Loaded: `{uploaded_epw.name}`")

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
        
    # Return all variables so your main logic loop can use them
    return idd_input, idf_input, epw_input, baseout, altout