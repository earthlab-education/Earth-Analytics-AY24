# %%
### load packages

import os
from glob import glob
import pathlib

### reproducible file paths

### gbif packages
import pygbif.occurrences as occ
import pygbif.species as species
from getpass import getpass

### packages for unzipping and handling gbif data
import zipfile
import time

### deal with spatial data
import geopandas as gpd
import xrspatial

### deal with other types of data
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrm

### indalid geometries
from shapely.geometry import MultiPolygon, Polygon

### visualizing
import holoviews as hv
import hvplot.pandas
import hvplot.xarray

# %%
# ### Make reproducible file paths

# data_dir = os.path.join(
#     ### home directory
#     pathlib.Path.home(),
    
#     ### eda directory
#     'earth-analytics',
#     'data',

#     ### Project dir
#     'hab_suit'
# )

# ### make the dir
# os.makedirs(data_dir, exist_ok=True)

# %%
### Make reproducible file paths
data_dir = os.path.join(
	### home directory
	pathlib.Path.home(),
	
	### eda directory
	'earth-analytics',
	'data',

	### Project dir
	'hab_suit'
)

### make the dir
os.makedirs(data_dir, exist_ok=True)

# %%
%run make_directory.py # this isn't working... something about how I set up the .py function is wrong!

# %%
# Define info for grasslands download
grasslands_url = (
    "https://data.fs.usda.gov/geodata/edw/edw_resources/"
    "shp/S_USA.NationalGrassland.zip"
)
grasslands_dir = os.path.join(data_dir, 'grasslands')
os.makedirs(grasslands_dir, exist_ok=True)
grasslands_path = os.path.join(grasslands_dir, 'grasslands.shp')

# Only download once
if not os.path.exists (grasslands_path):
    grasslands_gdf = gpd.read_file(grasslands_url)
    grasslands_gdf.to_file(grasslands_path)

# Load from file
grasslands_gdf = gpd.read_file(grasslands_path)

# Check the data
grasslands_gdf.plot()

# %%
# Print the full grasslands_gdf. Note the NATIONALGR column (sort by grassland size)
grasslands_gdf.sort_values(by='GIS_ACRES', ascending=True, inplace=True)
grasslands_gdf


# %%

# from plot_grassland import plot_grassland
# %run plot_grassland.py
# plot = plot_grassland(grasslands_gdf=grasslands_gdf,  NATIONALGR=295518010328, title="Buffalo Gap National Grassland")
# hv.output(plot)

# %%
# Define and print the boundary for the shey National Grassland, with ESRI imagery as the background. 
# For future runs of this code on different study areas, simply replace the "NATIONALGR" value 
# on the first line below with the corresponding value for the new study area from the table above.

shey_gdf = grasslands_gdf[grasslands_gdf.NATIONALGR=='295509010328']
shey_gdf.dissolve().hvplot(
    geo=True, tiles='EsriImagery',
    title='Sheyenne National Grassland',
    fill_color=None, line_color='black', line_width=1.5,
    frame_width=800
)

# %%


# %%
%run plot_grassland.py

# %%
## Note: This code cell was successful test of my plot_grassland.py function. It works!!
# plot_grassland(shey_gdf, '295509010328', 'Sheyenne National Grassland')

# %%
# Define and print the curl National Grassland boundary, with ESRI imagery as the background.

curl_gdf = grasslands_gdf[grasslands_gdf.NATIONALGR=='295525010328']
curl_gdf.dissolve().hvplot(
    geo=True, tiles='EsriImagery',
    title='Curlew National Grassland',
    fill_color=None, line_color='black', line_width=1.5,
    frame_width=800
)

# %%
%store curl_gdf shey_gdf


