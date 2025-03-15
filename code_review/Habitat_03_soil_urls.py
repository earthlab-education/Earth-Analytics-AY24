# %%
# Import tools
import os
import pathlib #finds the home folder
import pandas as pd
import geopandas as gpd
import matplotlib
import hvplot.pandas
import zipfile
import numpy
import xarray as xr
import rioxarray as rxr
import cartopy as ccrs
import earthaccess
import glob
from glob import glob
from math import floor, ceil

# %%
# Load stored gdf's from the previous notebook
%store -r shey_gdf curl_gdf

# %%
# check shey_gdf bounds
shey_gdf.bounds

# %%
# define objects for each lat/long value. These can then be plugged into the soil_url_template for modularity.
minx_bg = shey_gdf.bounds["minx"].min()
maxx_bg = shey_gdf.bounds["maxx"].max()
miny_bg = shey_gdf.bounds["miny"].min()
maxy_bg = shey_gdf.bounds["maxy"].max()

# %%
minx_bg

# %%
curl_gdf.bounds

# %%
minx_lm = curl_gdf.bounds["minx"].min()
maxx_lm = curl_gdf.bounds["maxx"].max()
miny_lm = curl_gdf.bounds["miny"].min()
maxy_lm = curl_gdf.bounds["maxy"].max()

# %%
## Note: For subsequent runs of this code with different grasslands, input max and min lat and long. I haven't found a way to automatically input these yet.

# Define and format a url template for downloading shey soil tiles
p_soil_url_template = ("http://hydrology.cee.duke.edu/POLARIS/"
                     "PROPERTIES/v1.0/ph/mean/30_60/"
                     "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif")
p_soil_url=p_soil_url_template.format(
    min_lat=46, max_lat=47, min_lon=-97, max_lon=-96)
p_soil_url

# Define and format a url template for downloading curl soil tile
c_soil_url_template = ("http://hydrology.cee.duke.edu/POLARIS/"
                     "PROPERTIES/v1.0/ph/mean/30_60/"
                     "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif")
c_soil_url=p_soil_url_template.format(
    min_lat=42, max_lat=43, min_lon=-105, max_lon=-102)
c_soil_url

# %%
# Open the shey soil raster as a data array
soil_da = rxr.open_rasterio(
    p_soil_url,
    mask_and_scale=True
).squeeze()
soil_da

# %%
# Define the bounds of the shey and curl National Grasslands, in order to identify which tiles to download
p_bounds_min_lon, p_bounds_min_lat, p_bounds_max_lon, p_bounds_max_lat = (
    shey_gdf.total_bounds
)
c_bounds_min_lon, c_bounds_min_lat, c_bounds_max_lon, c_bounds_max_lat = (
    curl_gdf.total_bounds
)

# Create a list of soil urls needed for the shey gdf
p_soil_url_list=[]
for min_lon in range (floor(p_bounds_min_lon), ceil(p_bounds_max_lon)):
    for min_lat in range (floor(p_bounds_min_lat), ceil(p_bounds_max_lat)):
        soil_url = p_soil_url_template.format(
            min_lat = min_lat, max_lat = min_lat+1,
            min_lon=min_lon, max_lon = min_lon+1
        )
    p_soil_url_list.append(soil_url)
p_soil_url_list

# %%
# Create a list of soil urls needed for the curl gdf
c_soil_url_list=[]
for min_lon in range (floor(c_bounds_min_lon), ceil(c_bounds_max_lon)):
    for min_lat in range (floor(c_bounds_min_lat), ceil(c_bounds_max_lat)):
        soil_url = c_soil_url_template.format(
            min_lat = min_lat, max_lat = min_lat+1,
            min_lon=min_lon, max_lon = min_lon+1
        )
    c_soil_url_list.append(soil_url)
c_soil_url_list

# %%
%store p_soil_url_list
%store c_soil_url_list


