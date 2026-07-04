import inspect
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import iotools
from geomeppy import IDF

def weather(epw_path: Path) -> pd.DataFrame:
        weather_data, metadata = iotools.read_epw(epw_path)

        city_name = metadata.get('city', metadata.get('location', 'Unknown Location'))
        latitude = metadata.get('latitude', 'N/A')
        longitude = metadata.get('longitude', 'N/A')

        weather_df = weather_data[[
                'temp_air',
                'relative_humidity',
                'wind_speed',
                'ghi',
                'dni'
        ]].copy()

        weather_df.columns = ['Dry_Bulb_C', 'RH_pct', 'Wind_Speed_ms', 'GHI_Wm2', 'DNI_Wm2']

        monthly_temp = weather_df.groupby(weather_df.index.month)['Dry_Bulb_C'].mean()

        return weather_df