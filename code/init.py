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