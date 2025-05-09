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
curl_gdf.bounds

# %%
minx_lm = curl_gdf.bounds["minx"].min()
maxx_lm = curl_gdf.bounds["maxx"].max()
miny_lm = curl_gdf.bounds["miny"].min()
maxy_lm = curl_gdf.bounds["maxy"].max()

# %%
# Define and format a url template for downloading shey soil tiles
pr_soil_url_template = ("http://hydrology.cee.duke.edu/POLARIS/"
                     "PROPERTIES/v1.0/pr/mean/30_60/"
                     "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif")
pr_soil_url=pr_soil_url_template.format(
    min_lat=43, max_lat=44, min_lon=-105, max_lon=-102)
pr_soil_url

# Define and format a url template for downloading curl soil tile
cr_soil_url_template = ("http://hydrology.cee.duke.edu/POLARIS/"
                     "PROPERTIES/v1.0/pr/mean/30_60/"
                     "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif")
cr_soil_url=cr_soil_url_template.format(
    min_lat=47, max_lat=49, min_lon=-105, max_lon=-102)
cr_soil_url

# %%
%store cr_soil_url pr_soil_url


