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
%store -r curl_gdf shey_gdf c_soil_url_list p_soil_url_list

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
data_dir

# %%
# login to earthaccess
earthaccess.login(strategy="interactive", persist=True)

# %%
# search for the appropriate DEM

datasets = earthaccess.search_datasets(keyword='SRTM DEM', count=11)
for dataset in datasets:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %%
curl_gdf.bounds

# %%
shey_gdf.bounds

# %%
# xmin, ymin, xmax, ymax = shey_gdf.total_bounds
bounds = tuple(shey_gdf.total_bounds)
srtm_p_results = earthaccess.search_data(
    short_name = "SRTMGL1",
    bounding_box = bounds
)
print(srtm_p_results)  # See if any results are found

# %%
## Creating a pattern for selecting srtm tiles based on the grassland gdf bounds

# Extract bounding box from shey_gdf
xmin, ymin, xmax, ymax = shey_gdf.total_bounds

# Generate SRTM tile names based on integer degree tiles
latitudes = range(int(ymin), int(ymax) + 1)
longitudes = range(int(xmin), int(xmax) + 1)

# Create SRTM file patterns
srtm_p_pattern_list = []
for lat in latitudes:
    for lon in longitudes:
        lat_prefix = f"N{lat}" if lat >= 0 else f"S{abs(lat)}"
        lon_prefix = f"E{lon}" if lon >= 0 else f"W{abs(lon)}"
        srtm_p_pattern_list.append(os.path.join(elevation_dir, f"{lat_prefix}{lon_prefix}.hgt.zip"))

# # Use glob to find matching files
# srtm_p_pattern = [glob(pattern) for pattern in srtm_p_pattern_list]
# srtm_p_pattern = [item for sublist in srtm_p_pattern for item in sublist]  # Flatten list

# print("Matching SRTM files:", srtm_p_pattern)


# %%
srtm_p_pattern_list

# %%
print(os.listdir(elevation_dir))

# %%
## Note: Edits needed! Downloads a single zip file.

### zip path -- we know we want the "08" file because the watershed code starts with "08"
zip_path = os.path.join(elevation_dir, "C:\\Users\\moenc\\earth-analytics\\data\\srtm\\N46W97.hgt.zip")
# Download the zip file once
if not os.path.exists(zip_path):
    ### query with the url
    response = requests.get(shey_url)
    ### check if response was successful
    if response.status_code == 200:
        ### save the zip file
        with open(zip_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        exit()

# %%
# # Define a pattern to identify DEM tiles associated with curl National Grassland.
# srtm_p_pattern = [
#     os.path.join(elevation_dir, 'N46*hgt.zip'),
#     # os.path.join(elevation_dir, 'N42*hgt.zip'),
# ]
# bounds_p = tuple(shey_gdf.total_bounds)
# buffer = 0.25
# xmin, ymin, xmax, ymax = bounds_p
# bounds_buffer = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

# # compile srtm files into list
# all_files = []
# for pattern in srtm_p_pattern:
#     all_files.extend(glob(pattern))

# if not all_files:
#     srtm_p_results = earthaccess.search_data(
#         short_name="SRTMGL1",
#         bounding_box=bounds_p
#     )
#     srtm_p_results = earthaccess.download(srtm_p_results, elevation_dir)

# if not glob(srtm_p_pattern):
#     srtm_p_results = earthaccess.search_data(
#         short_name = "SRTMGL1",
#         bounding_box=bounds_buffer    
#     )
#     srtm_p_results = earthaccess.download(srtm_p_results, elevation_dir)

# %%
# all_files

# %%
import zipfile # extract the zipped files

# Path to the downloaded ZIP file
zip_path = 'C:/Users/moenc/earth-analytics/data/srtm/N46W96.hgt.zip'

# Folder where you want to extract the contents
extract_path = elevation_dir

# Open the ZIP file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all contents to the specified folder
    zip_ref.extractall(extract_path)

print("Unzipping complete")

# %%
# # Print the DEM for curl natl. Grassland
# srtm_p_da_list=[]
# for srtm_p_path in glob(srtm_c_pattern):
#     tile_da = rxr.open_rasterio(srtm_p_path, mask_and_scale=True).squeeze()
#     cropped_da = tile_da.rio.clip_box(*bounds_buffer)
#     srtm_p_da_list.append(cropped_da)
    
# srtm_p_da = rxrmerge.merge_arrays(srtm_p_da_list)
# srtm_p_da.plot(cmap='terrain')
# curl_gdf.boundary.plot(ax=plt.gca(), color='black')

# Fix the second code block.
srtm_p_da_list = []
for pattern in srtm_p_pattern:
    for srtm_p_path in glob(pattern):
        tile_da = rxr.open_rasterio(srtm_p_path, mask_and_scale=True).squeeze()
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        srtm_p_da_list.append(cropped_da)

srtm_p_da = rxrmerge.merge_arrays(srtm_p_da_list)
srtm_p_da.plot(cmap='terrain')
shey_gdf.boundary.plot(ax=plt.gca(), color='white')

# %%
%store srtm_p_da

# %%
srtm_p_results = earthaccess.search_data(
    short_name="SRTMGL1",
    bounding_box=bounds_buffer    
)
print(srtm_p_results)  # See if any results are found


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
# srtm_p_pattern = [glob(pattern) for pattern in srtm_p_pattern_list]
# srtm_p_pattern = [item for sublist in srtm_p_pattern for item in sublist]  # Flatten list

# print("Matching SRTM files:", srtm_p_pattern)

# %%
# define a pattern which identifies the DEM's belongning to shey National Grasslands

srtm_p_pattern = os.path.join(elevation_dir, 'N*hgt.zip')
bounds_p = tuple(shey_gdf.total_bounds)
if not glob(srtm_p_pattern):
    srtm_p_results = earthaccess.search_data(
        short_name = "SRTMGL1",
        bounding_box=bounds_p    
    )
    srtm_p_results = earthaccess.download(srtm_p_results, elevation_dir)

# %%
srtm_p_pattern

# %%
bounds_p

# %%
# # Print the DEM for shey natl. Grassland
# srtm_p_da_list=[]
# for srtm_p_path in glob(srtm_p_pattern):
#     tile_da = rxr.open_rasterio(srtm_p_path, mask_and_scale=True).squeeze()
#     try:
#         cropped_da = tile_da.rio.clip_box(*bounds_p)
#     except: 
#         continue
#     srtm_p_da_list.append(cropped_da)
    
# srtm_p_da = rxrmerge.merge_arrays(srtm_p_da_list)
# srtm_p_da.plot(cmap='terrain')
# shey_gdf.boundary.plot(ax=plt.gca(), color='black')

# %%
# reproject the curl DEM into utm 13 N crs
utm_13n_epsg = 32613
srtm_p_proj_da = srtm_p_da.rio.reproject(utm_13n_epsg)
srtm_p_proj_da = srtm_da.to_crs()
srtm_p_proj_da.plot()

# %%
# Reproject so units are in meters
utm13_epsg = 32613
srtm_p_proj_da = srtm_p_da.rio.reproject(utm13_epsg)
shey_proj_gdf = shey_gdf.to_crs(utm_13n_epsg)
bounds_proj = tuple(shey_proj_da.total_bounds)

# Calculate slope
slope_full_da = xrspatial.slope(srtm_p_proj_da)
# slope_da = slope_full_da.rio.clip_box(*bounds_proj)
slope_da = slope_full_da.rio.clip(shey_proj_gdf.geometry)

# Plot slope, with curl bounds overlay
slope_p_da.plot(cmap='terrain')
shey_proj_gdf.boundary.plot(ax=plt.gca(), color='white', linewidth=.5)
plt.show()


