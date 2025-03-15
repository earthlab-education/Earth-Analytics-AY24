# %%
# import libraries
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
# load stored variables
%store -r shey_gdf shey_gdf c_soil_url_list p_soil_url_list

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
# shey_gdf.bounds
# bounds_buffer= (shey_gdf.total_bounds + 0.25)

# %%
# srtm_c_results = earthaccess.search_data(
#     short_name="SRTMGL1",
#     bounding_box=bounds_buffer    
# )
# print(srtm_c_results)  # See if any results are found


# %%
## Creating a pattern for selecting srtm tiles based on the grassland gdf bounds

# Extract bounding box from shey_gdf
xmin, ymin, xmax, ymax = shey_gdf.total_bounds

# Generate SRTM tile names based on integer degree tiles
latitudes = range(int(ymin), int(ymax) + 1)
longitudes = range(int(xmin), int(xmax) + 1)

# Create SRTM file patterns
srtm_c_pattern_list = []
for lat in latitudes:
    for lon in longitudes:
        lat_prefix = f"N{lat}" if lat >= 0 else f"S{abs(lat)}"
        lon_prefix = f"E{lon}" if lon >= 0 else f"W{abs(lon)}"
        srtm_c_pattern_list.append(os.path.join(elevation_dir, f"{lat_prefix}{lon_prefix}.hgt.zip"))

# # Use glob to find matching files
# srtm_c_pattern = [glob(pattern) for pattern in srtm_c_pattern_list]
# srtm_c_pattern = [item for sublist in srtm_c_pattern for item in sublist]  # Flatten list

# print("Matching SRTM files:", srtm_c_pattern)


# %%
srtm_c_pattern_list

# %%
print(os.listdir(elevation_dir))

# %%
# Define a pattern to identify DEM tiles associated with shey National Grassland.
srtm_c_pattern = [
    os.path.join(elevation_dir, 'N43*hgt.zip'),
]
bounds_c = tuple(shey_gdf.total_bounds)
buffer = 0.25
xmin, ymin, xmax, ymax = bounds_c
bounds_buffer = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

# Fix the first code block.
all_files = []
for pattern in srtm_c_pattern:
    all_files.extend(glob(pattern))

if not all_files:
    srtm_c_results = earthaccess.search_data(
        short_name="SRTMGL1",
        bounding_box=bounds_c
    )
    srtm_c_results = earthaccess.download(srtm_c_results, elevation_dir)

# if not glob(srtm_c_pattern):
#     srtm_c_results = earthaccess.search_data(
#         short_name = "SRTMGL1",
#         bounding_box=bounds_buffer    
#     )
#     srtm_c_results = earthaccess.download(srtm_c_results, elevation_dir)

# %%
import zipfile

# Path to the downloaded ZIP file
zip_path = strm_

# Folder where you want to extract the contents
extract_path = 'extracted_folder'

# Open the ZIP file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all contents to the specified folder
    zip_ref.extractall(extract_path)

print("Unzipping complete")

# %%
# # Print the DEM for shey natl. Grassland
# srtm_c_da_list=[]
# for srtm_c_path in glob(srtm_c_pattern):
#     tile_da = rxr.open_rasterio(srtm_c_path, mask_and_scale=True).squeeze()
#     cropped_da = tile_da.rio.clip_box(*bounds_buffer)
#     srtm_c_da_list.append(cropped_da)
    
# srtm_c_da = rxrmerge.merge_arrays(srtm_c_da_list)
# srtm_c_da.plot(cmap='terrain')
# shey_gdf.boundary.plot(ax=plt.gca(), color='black')

# Fix the second code block.
srtm_c_da_list = []
for pattern in srtm_c_pattern:
    for srtm_c_path in glob(pattern):
        tile_da = rxr.open_rasterio(srtm_c_path, mask_and_scale=True).squeeze()
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        srtm_c_da_list.append(cropped_da)

srtm_c_da = rxrmerge.merge_arrays(srtm_c_da_list)
srtm_c_da.plot(cmap='terrain')
shey_gdf.boundary.plot(ax=plt.gca(), color='white')


