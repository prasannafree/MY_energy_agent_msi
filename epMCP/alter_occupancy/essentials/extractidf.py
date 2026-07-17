import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geomeppy import IDF

def zones(idf_object) -> pd.DataFrame:
        zones = idf_object.idfobjects['ZONE']
        zone_list = []
        for index, z in enumerate(zones):
                zone_data = {
                        'Zone_Index': index + 1,
                        'Zone_Name': z.Name,
                        'Direction_of_Relative_North': getattr(z, 'Direction_of_Relative_North', 0),
                        'Multiplier': getattr(z, 'Multiplier', 1),
                }
                zone_list.append(zone_data)

        zones_df = pd.DataFrame(zone_list)
        return zones_df



def surfaces(idf_object) -> pd.DataFrame:
        surfaces = idf_object.idfobjects['BUILDINGSURFACE:DETAILED']

        surface_list = []
        for s in surfaces:
                surface_list.append({
                        'Surface_Name': s.Name,
                        'Surface_Type': s.Surface_Type,
                        'Zone_Name': s.Zone_Name, 
                        'Construction_Name': s.Construction_Name,
                        'Area_m2': float(s.area),
                        'Azimuth_deg': float(s.azimuth),
                        'Tilt_deg': float(s.tilt),
                        'Whats_Out': s.Outside_Boundary_Condition
                })

        surfaces_df = pd.DataFrame(surface_list)
        return surfaces_df

import pandas as pd

def windows(idf_object) -> pd.DataFrame:
    """Extracts all windows from FENESTRATIONSURFACE:DETAILED, 
    ensuring no inline entries are missed.
    """
    # Safely check if the object class exists in the model keys
    if 'FENESTRATIONSURFACE:DETAILED' not in idf_object.idfobjects:
        return pd.DataFrame()  # Return a clean empty DataFrame if none exist
        
    fenestrations = idf_object.idfobjects['FENESTRATIONSURFACE:DETAILED']
    window_list = []

    for f in fenestrations:
        # Check 'Surface_Type' or look for 'Window' in the name string as a fallback
        surface_type = getattr(f, 'Surface_Type', '').lower()
        object_name = getattr(f, 'Name', '').lower()
        
        if surface_type == 'window' or 'window' in object_name:
            raw_multiplier = getattr(f, 'Multiplier', '1')
            if not raw_multiplier or str(raw_multiplier).strip() == '':
                raw_multiplier = '1'
            window_list.append({
                'Window_Name': f.Name,
                'Parent_Wall': getattr(f, 'Building_Surface_Name', 'Unknown'),
                'Construction': getattr(f, 'Construction_Name', 'Unknown'),
                'Area_m2': float(getattr(f, 'area', 0.0)) if hasattr(f, 'area') else 0.0,
                'Multiplier': int(float(raw_multiplier))
            })

    return pd.DataFrame(window_list)        
def materials(idf_object) -> tuple:
        materials = idf_object.idfobjects['MATERIAL']
        material_list = []
        for m in materials:
                material_list.append({
                        'Material_Name': m.Name,
                        'Thickness_mm': float(getattr(m, 'Thickness', 0))*1000,
                        'Conductivity_W_mK': float(getattr(m, 'Conductivity', np.nan)),
                        'Density_kg_m3': float(getattr(m, 'Density', np.nan))
                })
        materials_df =pd.DataFrame(material_list)

        return materials_df

def occupancy(idf_object) -> pd.DataFrame:
        people_objects = idf_object.idfobjects['PEOPLE']
        occupancy_list = []

        for p in people_objects:
                # Grabbing fields using safe getattr in case of version-matching structural quirks
                occupancy_list.append({
                'Occupancy_Name': p.Name,
                'Zone_or_Space_Name': getattr(p, 'Zone_or_ZoneList_or_Space_or_SpaceList_Name', 'Unknown'),
                'Number_of_People_Schedule': getattr(p, 'Number_of_People_Schedule_Name', 'Unknown'),
                'Calculation_Method': getattr(p, 'Number_of_People_Calculation_Method', 'Unknown'),
                
                'Total_People_Count': float(getattr(p, 'Number_of_People', 0)) if getattr(p, 'Number_of_People', None) else np.nan,
                'People_per_Zone_Floor_Area': float(getattr(p, 'People_per_Zone_Floor_Area', 0)) if getattr(p, 'People_per_Zone_Floor_Area', None) else np.nan,
                'Floor_Area_per_Person_m2': float(getattr(p, 'Floor_Area_per_Person', 0)) if getattr(p, 'Floor_Area_per_Person', None) else np.nan,
                
                'Activity_Level_Schedule': getattr(p, 'Activity_Level_Schedule_Name', 'Unknown'),
                'Fraction_Radiant': float(getattr(p, 'Fraction_Radiant', 0.3)) if getattr(p, 'Fraction_Radiant', None) else 0.3,
        })
        
        return pd.DataFrame(occupancy_list)