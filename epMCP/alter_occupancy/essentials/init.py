import inspect
from pathlib import Path
from geomeppy import IDF
import os
import pandas as pd

def init(idd_path: Path, idf_path: Path) -> IDF:
        if not idd_path.exists():
                raise FileNotFoundError(f"no idd")
        if not idf_path.exists():
                raise FileNotFoundError(f"no idf")
        
        IDF.setiddname(str(idd_path))
        idf_object = IDF(str(idf_path))
        return idf_object


def justcsv(idf_object, weather_path: str, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if 'OUTPUT:TABLE:SUMMARYREPORTS' in idf_object.idfobjects:
        while len(idf_object.idfobjects['OUTPUT:TABLE:SUMMARYREPORTS']) > 0:
            idf_object.removeidfobject(idf_object.idfobjects['OUTPUT:TABLE:SUMMARYREPORTS'][0])

    has_hourly = any(
        "Electricity Demand Rate" in getattr(v, 'Variable_Name', '')
        for v in idf_object.idfobjects.get('OUTPUT:VARIABLE', [])
    )

    if not has_hourly:
        idf_object.newidfobject(
            'OUTPUT:VARIABLE',
            Key_Value='*',
            Variable_Name='Facility Total Electricity Demand Rate',
            Reporting_Frequency='Hourly'
        )

    idf_object.newidfobject(
        'OUTPUT:VARIABLE',
        Key_Value='*',
        Variable_Name='Zone Air Temperature',
        Reporting_Frequency='Hourly'
    )

    idf_object.run(
        weather=str(weather_path),
        output_directory=str(out_path),
        readvars=True  # This is the vital switch that converts the binary data to CSV
    )

from pathlib import Path
import re

def versioncheck(file_path: Path): 
    if not file_path.exists():
        print(f"Error: Cannot find file at {file_path}")
        return

    # 2. Read lines sequentially until we catch the Version tag
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Look for lines containing "Version" followed by a comma and number
            match = re.search(r'Version\s*,\s*([\d\.]+)\s*;', line, re.IGNORECASE)
            if match:
                print(f"File Name: {file_path.name}")
                print(f"Current IDF Version: v{match.group(1)}")
                return
                
    print("Could not find a 'Version' object string in the first section of this file.")

def load_and_clean_data(csv_path: str):
    df = pd.read_csv(Path(csv_path))
    df.columns = df.columns.str.strip()
    # Establish time helper columns based on 8760 continuous index
    df['Hour'] = df.index % 24
    df['Day_of_Year'] = df.index // 24
    return df

import subprocess
from pathlib import Path

def juzcsv(idf_object, weather_path: str, output_dir: str):
    """
    Executes EnergyPlus model and natively tracks all primary energy meters 
    (including subload cooling, humidifiers, and fans) into a unified eplusout.csv.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Clear out pre-existing tracking configurations to prevent duplicate errors
    for obj_type in ['OUTPUT:TABLE:SUMMARYREPORTS', 'OUTPUT:VARIABLE', 'OUTPUT:METER']:
        if obj_type in idf_object.idfobjects:
            while len(idf_object.idfobjects[obj_type]) > 0:
                idf_object.removeidfobject(idf_object.idfobjects[obj_type][0])

    # 2. Add Complete Energy Subload METER Targets (Using OUTPUT:METER)
    
    #energy_meters = [
     #   'Cooling:Electricity',
      #  'Humidifier:Electricity',        # 💨 Tracks active steam/ultrasonic hum electrification
       # 'Dehumidifier:Electricity',      # 💧 Tracks dedicated desiccant/condenser moisture removal
        #'Heating:NaturalGas',
        #'InteriorLights:Electricity',
        #'InteriorEquipment:Electricity',
        #'Fans:Electricity',
        #'Pumps:Electricity'
    #]
    
    #for meter in energy_meters:
     #   idf_object.newidfobject(
      #      'OUTPUT:METER',
       #     Key_Name=meter,            # 🛠️ FIX: Correctly passes variables to object indices
        #    Reporting_Frequency='Hourly'
        #)

    # 3. Add Continuous Demand Rates and Thermal Variables (Using OUTPUT:VARIABLE)
    idf_object.newidfobject(
        'OUTPUT:VARIABLE', 
        Key_Value='*', 
        Variable_Name='Facility Total Electricity Demand Rate', 
        Reporting_Frequency='Hourly'
    )
    
    #idf_object.newidfobject(
     #   'OUTPUT:VARIABLE', 
      ## Variable_Name='Zone Air Temperature', 
        #Reporting_Frequency='Hourly'
    #)

    # 4. Fire execution engine with processing maps active
    idf_object.run(
        weather=str(weather_path),
        output_directory=str(out_path),
        readvars=True 
    )

def runperiod(idf_obj, start_date, end_date):
    """
    Dynamically overwrites the first RunPeriod object in the IDF 
    with the user's selected Streamlit dates.
    """
    if "RUNPERIOD" in idf_obj.idfobjects:
        # Grab the first RunPeriod object found in the IDF
        run_period = idf_obj.idfobjects["RUNPERIOD"][0]
        
        # Update fields dynamically
        run_period.Begin_Month = start_date.month
        run_period.Begin_Day_of_Month = start_date.day
        run_period.End_Month = end_date.month
        run_period.End_Day_of_Month = end_date.day
    else:
        # Fallback if no RunPeriod object exists, create one
        idf_obj.newidfobject(
            "RUNPERIOD",
            Name="Custom_Run_Period",
            Begin_Month=start_date.month,
            Begin_Day_of_Month=start_date.day,
            End_Month=end_date.month,
            End_Day_of_Month=end_date.day
        )

from pathlib import Path
from geomeppy import IDF

def update_idf_version_native(file_path: Path, new_version: str, idd_path: Path = None) -> bool:
    """
    Updates the IDF version using geomeppy's native object handling.
    """
    if not file_path.exists():
        print(f"Error: Cannot find file at {file_path}")
        return False

    # Optional: Set IDD if not already set globally in your script
    if idd_path and idd_path.exists():
        IDF.setiddname(str(idd_path))
    
    try:
        # Load the IDF file using geomeppy
        idf = IDF(str(file_path))
        
        # Pull the 'Version' object list (usually contains only one object)
        version_objects = idf.idfobjects['VERSION']
        
        if not version_objects:
            print(f"No Version object found in {file_path.name}.")
            return False
        
        # Access the first (and only) version object
        version_obj = version_objects[0]
        old_version = version_obj.Version_Identifier
        
        # Strip any leading 'v' to match standard EnergyPlus format (e.g., '24.1')
        clean_new_version = new_version.lstrip('vV')
        
        print(f"Updating {file_path.name} via geomeppy: v{old_version} -> v{clean_new_version}")
        
        # Update the field value natively
        version_obj.Version_Identifier = clean_new_version
        
        # Save the changes back to the file
        idf.save()
        print("File updated and saved successfully.")
        return True

    except Exception as e:
        print(f"An error occurred while processing the IDF: {e}")
        return False
    
update_idf_version_native(
    file_path=Path(r"C:\Users\annaa\Downloads\IISC\calibration\level4_ACH_par.idf"),
    new_version="26.1",
    idd_path=Path(r"C:\Users\annaa\Downloads\IISC\calibration\V9-4-0-Energy+.idd"))