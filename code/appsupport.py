import pandas as pd
from extractidf import zones

def zonearray(idf_object) -> list:
    zones_df = zones(idf_object)
    zone_array = zones_df['Zone_Name'].tolist()

    return zone_array

