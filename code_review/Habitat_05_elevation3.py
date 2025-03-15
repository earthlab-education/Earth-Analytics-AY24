# %%
import os
import re
import pathlib
from glob import glob

import matplotlib.pyplot as plt
import earthaccess
import xrspatial
import geopandas as gpd
import rioxarray as rxr
import rioxarray.merge as rxrmerge

# %%
%store -r shey_gdf p_soil_url_list

# %%
# build project and elevation directories

data_dir = os.path.join(
    pathlib.Path.home(),
    'earth-analytics',
    'data'
)
project_dir = os.path.join(data_dir, 'habitat_suitability')
elevation_dir = os.path.join(data_dir, 'srtm')

os.makedirs(elevation_dir, exist_ok=True)

# %%
# login to earthaccess
earthaccess.login(strategy="interactive", persist=True)

# %%
# search for the appropriate DEM

datasets = earthaccess.search_datasets(keyword='SRTM DEM', count=11)
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
# Define a pattern to identify DEM tiles associated with Comanche National Grassland.
srtm_p_pattern = os.path.join(elevation_dir, 'N46*hgt.zip')
bounds_p = tuple(shey_gdf.total_bounds)
buffer = 0.25
xmin, ymin, xmax, ymax = bounds_p
bounds_buffer = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)
if not glob(srtm_p_pattern):
    srtm_p_results = earthaccess.search_data(
        short_name = "SRTMGL1",
        bounding_box=bounds_buffer    
    )
    srtm_p_results = earthaccess.download(srtm_p_results, elevation_dir)

# %%
import os
import zipfile
import rioxarray as rxr
import matplotlib.pyplot as plt
from glob import glob
import rioxarray.merge as rxrmerge

# Define patterns for the DEM tiles
srtm_p_pattern = glob(os.path.join(elevation_dir, 'N46*hgt.zip'))

# Ensure files are unzipped before processing
for zip_path in srtm_p_pattern:
    extract_path = elevation_dir  # Extract to the same directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Unzipped: {zip_path}")

# Now update the pattern to look for extracted .hgt files
srtm_p_pattern = glob(os.path.join(elevation_dir, 'N46*.hgt'))

srtm_p_da_list = []
for srtm_p_path in srtm_p_pattern:
    tile_da = rxr.open_rasterio(srtm_p_path, mask_and_scale=True).squeeze()
    cropped_da = tile_da.rio.clip_box(*bounds_buffer)
    srtm_p_da_list.append(cropped_da)

srtm_p_da = rxrmerge.merge_arrays(srtm_p_da_list)
srtm_p_da.plot(cmap='terrain')
shey_gdf.boundary.plot(ax=plt.gca(), color='white')


# %%
# Print the DEM for Comanche natl. Grassland
srtm_p_da_list=[]
for srtm_p_path in glob(srtm_p_pattern):
    tile_da = rxr.open_rasterio(srtm_p_path, mask_and_scale=True).squeeze()
    cropped_da = tile_da.rio.clip_box(*bounds_buffer)
    srtm_p_da_list.append(cropped_da)
    
srtm_p_da = rxrmerge.merge_arrays(srtm_p_da_list)
srtm_p_da.plot(cmap='terrain')
shey_gdf.boundary.plot(ax=plt.gca(), color='black')


