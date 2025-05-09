# %%
import os
import pathlib
import geopandas as gpd

import pandas as pd
import rioxarray as rxr

# Import packages
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns

# %%
%store -r curl_gdf

# %%
model_name = 'BNU-ESM'
variable_name = 'pr'
var_long_name = 'precipitation'
dir_path = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC/'

# %%
# build a directory for the macav2 data

curl_path = os.path.join(
    pathlib.Path.home(),
    'earth-analytics',
    'data',
    'habitat_suitability'
)
os.makedirs(curl_path, exist_ok=True)
# curl_gdf.to_file('habitat_suitability')
# curl_gdf = gpd.read_file(curl_path)

# Define the file path for the GeoDataFrame
curl_file = os.path.join(curl_path, 'curl_data.shp')

# Save the GeoDataFrame to the file
curl_gdf.to_file(curl_file)

# Read the GeoDataFrame back from the file
curl_gdf = gpd.read_file(curl_file)

# %%
def convert_longitude(longitude):
        """Convert longitude from 0-360 to -180-180"""
        return (longitude - 360) if longitude > 180 else longitude

maca_da_list = []
for site_name, site_gdf in {'curl': curl_gdf}.items():
        for variable in ["pr"]:
                for start_year in [2096]:
                        end_year = start_year + 3
                        maca_url = (
                                "http://thredds.northwestknowledge.net:8080/thredds"
                                f"/dodsC/MACAV2/BNU-ESM/macav2metdata_{variable}_BNU-ESM_"
                                f"r1i1p1_rcp85_{start_year}_{end_year}_CONUS_monthly.nc"
                        # "http://thredds/fileServer/MACAV2/BNU-ESM/macav2metdata"
                        # f"_{variable}_BNU-ESM_r1i1p1_rcp45_{start_year}"
                        # f"_{end_year}_CONUS_monthly.nc"
                        )
                        maca_da = xr.open_dataset(maca_url).squeeze().precipitation
                        bounds = site_gdf.to_crs(maca_da.rio.crs).total_bounds
                        maca_da = maca_da.assign_coords(
                                lon=("lon", [convert_longitude(l) for l in maca_da.lon.values])
                        )
                        maca_da = maca_da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
                        maca_da.rio.clip_box(*bounds)
                        maca_da_list.append(dict(
                                site_name=site_name,
                                variable=variable,
                                start_year=start_year,
                                da=maca_da))
pd.DataFrame(maca_da_list)

# %%
maca_df = pd.DataFrame(maca_da_list)
maca_df

# %%
%store maca_df

# %%
maca_da.rio.clip_box(*curl_gdf.to_crs(maca_da.rio.crs).total_bounds)
maca_da.rio.clip_box(*bounds)


