# %%
# pip install pygbif

# %%
import os # Interoperable file paths
from glob import glob
import pathlib 
import requests

# Data Tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from math import floor, ceil

# Spatial tools
import earthaccess
import geopandas as gpd
import rioxarray as rxr
import rioxarray.merge as rxrm
import xarray as xr
import xrspatial

# GBIF packages
import pygbif.occurrences as occ
import pygbif.species as species
from getpass import getpass
import time
import zipfile

# Visualizing Tools
import holoviews as hv
import hvplot.pandas
import hvplot.xarray


# %%
%run scripts.py

# %%
# Define and create the project data directory
data_dir = os.path.join(
    # Home
    pathlib.Path.home(),
    # Earth Analytics
    'earth-analytics',
    'data',
    'habitat-suitability'
)
os.makedirs(data_dir, exist_ok=True)

# %% [markdown]
# # Habitat suitability under climate change
# 
# ## STEP 1: STUDY OVERVIEW

# %% [markdown]
# How will American Ginseng respond to different climate conditions for the rest of the century?

# %% [markdown]
# ### *Panax quinquefolius* (American Ginseng)

# %%
reset_credentials = False

# GBIF credentials
credentials = dict(
    GBIF_USER=(input, 'GBIF username:'),
    GBIF_PWD=(getpass, 'GBIF password:'),
    GBIF_EMAIL=(input, 'GBIF email:')
)
for env_variable, (prompt_func, prompt_text) in credentials.items():
    # Delete credential from environment if requested
    if reset_credentials and (env_variable in os.environ):
        os.environ.pop(env_variable)
    # Ask for credential and save to environment
    if not env_variable in os.environ:
        os.environ[env_variable] = prompt_func(prompt_text)

# %%
# Species Information
species_name = 'Panax quinquefolius'
species_info = species.name_lookup(species_name, rank='species')
species_result = species_info['results'][0]
species_key = species_result['nubKey']

species_result['species'], species_key

# %%
# GBIF species directory
gbif_dir = os.path.join(data_dir,
                        'GBIF',
                        species_result['species'])

# GBIF file path
gbif_pattern = os.path.join(gbif_dir, '*.csv')

# Only download once
if not glob(gbif_pattern):
    gbif_query = occ.download([
        f'speciesKey = {species_key}',
        'hasCoordinate = True'
    ])
    download_key = gbif_query[0]

    if not 'GBIF_DOWNLOAD_KEY' in os.environ:
        os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
    

    # Wait for download to build
    wait = occ.download_meta(download_key)['status']
    while not wait=='SUCCEEDED':
        wait = occ.download_meta(download_key)['status']
        time.sleep(5)
    
    # Download GBIF data
    download_info = occ.download_get(
        os.environ['GBIF_DOWNLOAD_KEY'],
        path=gbif_dir)

    # Unzip
    with zipfile.ZipFile(download_info['path']) as download_zip:
        download_zip.extractall(path=gbif_dir)
    
# Extracted .csv file path
gbif_path = glob(gbif_pattern)[0]


# %%
# Open GBIF csv as DataFrame
gbif_df = pd.read_csv(
    gbif_path,
    delimiter = '\t'
)
gbif_df.columns

# %%
gbif_gdf = (
    gpd.GeoDataFrame(
        gbif_df,
        geometry=gpd.points_from_xy(
            gbif_df.decimalLongitude,
            gbif_df.decimalLatitude
        ),
        crs='EPSG:4326'
    ))
gbif_gdf

# %%
# Plot
gbif_gdf.hvplot(
    geo=True, tiles='EsriImagery',
    title = 'American Ginseng occurrences in GBIF',
    fill_color = None, line_color = 'white',
    frame_width = 600
)

# %% [markdown]
# American Ginseng (*Panax quinquefolius*) is found in eastern forests of North America. In order to grow, it requires 70-80% shade, 40-50 inches of annual rainfall, and an average temperature of 50 degrees Fahrenheit. The soil also needs to be loamy, and at least 12 inches deep for the roots to grow, with a pH between five and six.

# %% [markdown]
# ### Sites
# 
# 

# %%
# Create site boundary data directory in the project folder
boundary_dir = os.path.join(data_dir, 'USFS')
os.makedirs(boundary_dir, exist_ok=True)

# Define info for USFS National Forests download
boundary_url = ("https://data-usfs.hub.arcgis.com/"
                "api/download/v1/items/3451bcca1dbc45168ed0b3f54c6098d3"
                "/shapefile?layers=0")
boundary_path = os.path.join(boundary_dir, 'USFS_boundary.shp')

# Only download once
if not os.path.exists(boundary_path):
    boundary_gdf = gpd.read_file(boundary_url)
    boundary_gdf.to_file(boundary_path)

# Load from file
else:
    boundary_gdf = gpd.read_file(boundary_path)

# Check the data
boundary_gdf

# %%
Panax_FS = gpd.overlay(gbif_gdf, boundary_gdf, how='intersection')


# %%
value_counts = Panax_FS['FORESTNAME'].value_counts()
value_counts

# %% [markdown]
# Since Shawnee National Forest and Nantahala National Forest have the highest recorded occurrences of American Ginseng, we will focus our analysis there.

# %%
# Select boundaries for the forests of interest
shawnee_gdf = boundary_gdf[boundary_gdf['FORESTNAME'] == 'Shawnee National Forest']
nantahala_gdf = boundary_gdf[boundary_gdf['FORESTNAME'] == 'Nantahala National Forest']

# Site list to loop through
site_list = [shawnee_gdf, nantahala_gdf]

# Concatenate GDFs to loop through later on
sites_gdf = gpd.GeoDataFrame(
    pd.concat([shawnee_gdf, nantahala_gdf], ignore_index=True))

sites_gdf.info()

# %%
# Plot boundary line of a gdf
def plot_site(gdf):
    gdf_plot = (
        gdf.dissolve().hvplot(
            geo=True, tiles='EsriImagery',
            title=gdf.FORESTNAME.values[0],
            fill_color=None, line_color='white', line_width=3,
            frame_width=600
        )
    )
    return gdf_plot

# %%
# Plot first site using script.py function
plot_site(shawnee_gdf)

# %%
# Second site
plot_site(nantahala_gdf)

# %% [markdown]
# YOUR SITE DESCRIPTION HERE

# %% [markdown]
# # Time Periods
# 
# This analysis will focus on a near future scenario from 2036 to 2066, and an end of century climate scenario from 2066 to 2096.

# %% [markdown]
# ### Climate models
# 
# Climate models were selected using the [Climate Toolbox: Future Climate Scatter tool](https://climatetoolbox.org/tool/Future-Climate-Scatter) to simulate changes in precipitation and temperature for the rest of the 21st century.
# 
# The climate models that will be used are:
# 
# -   Warm and wet HadGEM2-CC365
# -   Warm and dry CanESM2
# -   Cold and wet MIROC5
# -   Cold and dry MRI-CGCM3
# 

# %% [markdown]
# ## STEP 2: DATA ACCESS
# 
# ### Soil data
# 
# The [POLARIS dataset](http://hydrology.cee.duke.edu/POLARIS/) is a
# convenient way to uniformly access a variety of soil parameters such as
# pH and percent clay in the US. It is available for a range of depths (in
# cm) and split into 1x1 degree tiles.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download POLARIS data for a particular location, soil parameter,
# and soil depth. Your function should account for the situation where
# your site boundary crosses over multiple tiles, and merge the necessary
# data together.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include soil parameters that will help you to
# answer your scientific question. We recommend using a soil depth that
# best corresponds with the rooting depth of your species.</p></div></div>

# %%
def download_tif(url, save_path):
    """Function to download and save .tif files"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    else: 
        print(f"Failed to download {url}. Status code: {response.status_code}")

# %%
def polaris_soil_data(variable, statistic, depth, gdf, save_folder):
    
    # Ensure the save directory exists
    polaris_dir = os.path.join(data_dir, 'polaris/'+save_folder)
    os.makedirs(polaris_dir, exist_ok=True)

    # Create list of soil data URLs
    soil_url_template = ("http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
                "/{variable}"
                "/{statistic}"
                "/{depth}"
                "/lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif")

    bounds_min_lon, bounds_min_lat, bounds_max_lon, bounds_max_lat = (
        gdf.total_bounds)

    soil_url_list = []

    for lesser_lat in range(floor(bounds_min_lat), ceil(bounds_max_lat)):
        for lesser_lon in range(floor(bounds_min_lon), ceil(bounds_max_lon)):
            soil_url = soil_url_template.format(
                variable=variable,
                statistic=statistic,
                depth=depth,
                min_lat=lesser_lat, max_lat=lesser_lat+1, 
                min_lon=lesser_lon, max_lon=lesser_lon+1)
            soil_url_list.append(soil_url)

    # Download the files locally
    for url in soil_url_list:
        # Extract file name from url
        split_url = url.split("/")[-5:]
        filename = ''
        for i in split_url:
            filename+=i

        # Download the .tif file once
        save_path = os.path.join(polaris_dir, filename)
        if not os.path.exists(save_path):
            download_tif(url, save_path)

# %%
def set_buffer(boundary_gdf, buffer=0):
    """
    Increases the max bounds of a geo data frame by a set amount.
    Returns the max bounds as a tuple.
    """
    bounds = tuple(boundary_gdf.total_bounds)
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

    return bounds_buffer

# %%
def process_image(file_pattern, boundary_gdf, buffer=0):
    """Load image, crop to study boundary, merge assays"""
    
    # Set a buffer to crop images to
    bounds_buffer = set_buffer(boundary_gdf, buffer)
    
    # Open and crop the images
    da_list = []
    for file_path in glob(file_pattern):
        tile_da = (
            rxr.open_rasterio(file_path, mask_and_scale=True)
            .squeeze())
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        da_list.append(cropped_da)
    
    # Merge the list of cropped data arrays
    merged_da = rxrm.merge_arrays(da_list)
    
    # Returns a cropped and merged data array
    return merged_da

# %%
soil_ph_das = []

for site in site_list:
    site_soil_data = polaris_soil_data(
        'ph', 'p50', '15_30', site, site.FORESTNAME.values[0])
    site_soil_path = os.path.join(
        data_dir,
        'polaris',
        site.FORESTNAME.values[0],
        '*.tif')
    site_ph_da = process_image(site_soil_path, site)
    soil_ph_das.append(dict(
        site_name=site.FORESTNAME.values[0],
        soil_variable='ph',
        soil_da=site_ph_da))

soil_df = pd.DataFrame(soil_ph_das)
soil_df.info()

# %%
# Visualize the soil data

num_plots = len(soil_df.soil_da)

# Create subplots dynamically based on the number of plots
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6))

# In case of only one plot
if num_plots == 1:
    axes = [axes]

# Loop over soil DataArrays and site list
for i, (site_ph_da, site) in enumerate(zip(soil_df.soil_da, site_list)):

    # Display raster data
    im = axes[i].imshow(
        site_ph_da.values, 
        cmap='viridis', 
        interpolation='nearest', 
        extent=(
            site_ph_da.x.min(),
            site_ph_da.x.max(),
            site_ph_da.y.min(),
            site_ph_da.y.max()
        ))
    # Set title based on site name
    axes[i].set_title(f"Soil pH: {site.FORESTNAME.values[0]}")
    # Set site boundary line
    site.boundary.plot(ax=axes[i], color='black', linewidth=1)
    # Set colorbar
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Topographic data
# 
# One way to access reliable elevation data is from the [SRTM
# dataset](https://www.earthdata.nasa.gov/data/instruments/srtm),
# available through the [earthaccess
# API](https://earthaccess.readthedocs.io/en/latest/quick-start/).
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download SRTM elevation data for a particular location and
# calculate any additional topographic variables you need such as slope or
# aspect.</p>
# <p>Then, use loops to download and organize the rasters you will need to
# complete this section. Include topographic parameters that will help you
# to answer your scientific question.</p></div></div>
# 
# > **Warning**
# >
# > Be careful when computing the slope from elevation that the units of
# > elevation match the projection units (e.g. meters and meters, not
# > meters and degrees). You will need to project the SRTM data to
# > complete this calculation correctly.

# %%
# Download elevation data
def srtm_data_download(boundary_gdf, buffer=0.025):
    elevation_dir = os.path.join(data_dir, 
                                  'srtm',
                                  boundary_gdf.FORESTNAME.values[0])
    os.makedirs(elevation_dir, exist_ok=True)

    srtm_pattern = os.path.join(elevation_dir, '*.hgt.zip')

    bounds_buffer = set_buffer(boundary_gdf, buffer)

    if not glob(srtm_pattern):
        srtm_results = earthaccess.search_data(
            short_name='SRTMGL1',
            bounding_box=bounds_buffer
        )
        srtm_results = earthaccess.download(srtm_results, elevation_dir)
    
    srtm_da_list = []

    for srtm_path in glob(srtm_pattern):
        tile_da = rxr.open_rasterio(srtm_path, mask_and_scale=True).squeeze()
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        srtm_da_list.append(cropped_da)
    
    srtm_da =rxrm.merge_arrays(srtm_da_list)

    return srtm_da

# %%
def derive_slope(site_gdf, elevation_da):
    # Estimate the UTM CRS based on the bounds of the site
    epsg_utm = site_gdf.estimate_utm_crs()
    # Reproject boundary and elevation data so that units are in meters
    elevation_proj_da = elevation_da.rio.reproject(epsg_utm)
    site_proj_gdf = site_gdf.to_crs(epsg_utm)
    
    # Calculate slope using xrspatial
    slope_full_da = xrspatial.slope(elevation_proj_da)
    slope_da = slope_full_da.rio.clip(site_proj_gdf.geometry)

    return slope_da

# %%
def derive_aspect(site_gdf, elevation_da):
    # Estimate the UTM CRS based on the bounds of the site
    epsg_utm = site_gdf.estimate_utm_crs()
    # Reproject boundary and elevation data so that units are in meters
    elevation_proj_da = elevation_da.rio.reproject(epsg_utm)
    site_proj_gdf = site_gdf.to_crs(epsg_utm)
    
    # Calculate slope using xrspatial
    aspect_full_da = xrspatial.aspect(elevation_proj_da)
    aspect_da = aspect_full_da.rio.clip(site_proj_gdf.geometry)

    return aspect_da

# %%
elevation_das = []

for site in site_list:
    elevation_da = srtm_data_download(site)
    elevation_das.append(dict(
        site_name=site.FORESTNAME.values[0],
        elevation_da=elevation_da
    ))
topo_df = pd.DataFrame(elevation_das)

# %%
slope_das = []

for (site, elevation_da) in zip(site_list, topo_df.elevation_da):
    slope_da = derive_slope(site, elevation_da)
    slope_da = slope_da.rio.reproject_match(elevation_da)
    slope_das.append(slope_da)

topo_df['slope_da'] = slope_das

# %%
aspect_das = []

for (site, elevation_da) in zip(site_list, topo_df.elevation_da):
    aspect_da = derive_aspect(site, elevation_da)
    aspect_da = aspect_da.rio.reproject_match(elevation_da)
    aspect_das.append(aspect_da)

topo_df['aspect_da'] = aspect_das
    

# %%
topo_df.elevation_da[0].plot(cmap='terrain')
site_list[0].boundary.plot(ax=plt.gca(), color='black')

# %%
topo_df.elevation_da[1].plot(cmap='terrain')
site_list[1].boundary.plot(ax=plt.gca(), color='black')

# %%
# Visualize the elevation data

num_plots = len(topo_df.aspect_da)

# Create subplots dynamically based on the number of plots
fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6))

# In case of only one plot
if num_plots == 1:
    axes = [axes]

# Loop over soil DataArrays and site list
for i, (site_da, site) in enumerate(zip(topo_df.aspect_da, site_list)):

    # Display raster data
    im = axes[i].imshow(
        site_da.values, 
        cmap='terrain', 
        interpolation='nearest', 
        extent=(
            site_da.x.min(),
            site_da.x.max(),
            site_da.y.min(),
            site_da.y.max()
        ))
    # Set title based on site name
    axes[i].set_title(f"Aspect: {site.FORESTNAME.values[0]}")
    # Set site boundary line
    site.boundary.plot(ax=axes[i], color='white', linewidth=1)
    # Set colorbar
    fig.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Climate model data
# 
# You can use MACAv2 data for historical and future climate data. Be sure
# to compare at least two 30-year time periods (e.g. historical vs. 10
# years in the future) for at least four of the CMIP models. Overall, you
# should be downloading at least 8 climate rasters for each of your sites,
# for a total of 16. **You will *need* to use loops and/or functions to do
# this cleanly!**.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Write a <strong>function with a numpy-style docstring</strong> that
# will download MACAv2 data for a particular climate model, emissions
# scenario, spatial domain, and time frame. Then, use loops to download
# and organize the 16+ rasters you will need to complete this section. The
# <a
# href="http://thredds.northwestknowledge.net:8080/thredds/reacch_climate_CMIP5_macav2_catalog2.html">MACAv2
# dataset is accessible from their Thredds server</a>. Include an
# arrangement of sites, models, emissions scenarios, and time periods that
# will help you to answer your scientific question.</p></div></div>

# %%
def convert_longitude(longitude):
    """Convert longitude range from 0-360 to [-180:180]"""
    return (longitude - 360) if longitude > 180 else longitude

# %%
# Download climate data
def MACAv2_data(variable, model, scenario, start_year, site_gdf):
    """
    
    """
    end_year = 2099 if start_year == 2096 else (start_year + 4)
    maca_url = (
        'http://thredds.northwestknowledge.net:8080/thredds/dodsC/MACAV2'
        f'/{model}/macav2metdata_{variable}_{model}_r1i1p1_'
        f'{scenario}_{start_year}_{end_year}_CONUS_monthly.nc')
    maca_da = xr.open_dataset(maca_url).squeeze().precipitation
    bounds = site_gdf.to_crs(maca_da.rio.crs).total_bounds
    # Reassign coordinates to [-180:180]
    maca_da = maca_da.assign_coords(
        lon=("lon", [convert_longitude(l) for l in maca_da.lon.values])
    )
    maca_da = maca_da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    maca_da = maca_da.rio.clip_box(*bounds)


    return maca_da


# %%
# Define climate variables to loop through
variable_list = ['pr']
model_list = ['CanESM2', 'HadGEM2-CC365', 'MIROC5', 'MRI-CGCM3']
scenario_list = ['rcp85']
start_year_list = [2036, 2066]

maca_da_list = []
# Loop through all variables
for site in site_list:
    for variable in variable_list:
        for model in model_list:
            for scenario in scenario_list:
                for start_year in start_year_list:
                    year_list = list(
                        # Download 30-year period from start year
                        range(start_year, start_year+30, 5))
                    for year in year_list:
                        # Execute download and processing
                        maca_da = MACAv2_data(
                            variable, model, scenario, start_year, site)

                        # Structure dictionary with metadata
                        maca_da_list.append(dict(
                            site_name=site.FORESTNAME.values[0],
                            climate_model=model,
                            climate_variable=variable,
                            climate_scenario=scenario,
                            start_year=year,
                            climate_da=maca_da))

# Convert the DataArray list to a pandas DataFrame
MACAv2_df = pd.DataFrame(maca_da_list)


# %%
def compute_mean_of_group(group):
    # Stack the DataArrays in the group along a new dimension
    stacked_data = xr.concat(group, dim='concat_dim')
    # Return the mean along the 'concat_dim' axis
    return stacked_data.mean(dim='concat_dim')

# %%
# Split df by 30 year climate period
MACA_group_1 = MACAv2_df[MACAv2_df['start_year'] <= 2061]
MACA_group_2 = MACAv2_df[MACAv2_df['start_year'] > 2061]

# Group by sites, model, etc.
MACA_group_A = MACA_group_1.groupby(
    ['site_name', 'climate_model', 'climate_variable','climate_scenario'],
    as_index=False)['climate_da'].apply(compute_mean_of_group)
MACA_group_B = MACA_group_2.groupby(
    ['site_name', 'climate_model', 'climate_variable','climate_scenario'],
    as_index=False)['climate_da'].apply(compute_mean_of_group)

# %%
MACA_grouped_df = pd.concat([MACA_group_A, MACA_group_B],
                            ignore_index=True)

MACA_grouped_df.info()

# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Make sure to include a description of the climate data and how you
# selected your models. Include a citation of the MACAv2 data</p></div></div>
# 
# YOUR CLIMATE DATA DESCRIPTION AND CITATIONS HERE
# 
# ## STEP 3: HARMONIZE DATA
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Make sure that the grids for all your data match each other. Check
# out the <a
# href="https://corteva.github.io/rioxarray/stable/examples/reproject_match.html#Reproject-Match"><code>ds.rio.reproject_match()</code>
# method</a> from <code>rioxarray</code>. Make sure to use the data source
# that has the highest resolution as a template!</p></div></div>
# 
# > **Warning**
# >
# > If you are reprojecting data as you need to here, the order of
# > operations is important! Recall that reprojecting will typically tilt
# > your data, leaving narrow sections of the data at the edge blank.
# > However, to reproject efficiently it is best for the raster to be as
# > small as possible before performing the operation. We recommend the
# > following process:
# >
# >     1. Crop the data, leaving a buffer around the final boundary
# >     2. Reproject to match the template grid (this will also crop any leftovers off the image)

# %%
maca_da_repr_list = []

for row in range(len(MACA_grouped_df)):
    maca_da_orig = (
        MACA_grouped_df.iloc[row]['climate_da']
        .rio.write_crs(4326, inplace=False)
        .rio.set_spatial_dims('lat', 'lon')
        .groupby('time.year')
        .sum()
        .min('year')
    )
    maca_da_repr = maca_da_orig.rio.reproject_match(
        topo_df[topo_df['site_name'] == MACA_grouped_df.iloc[row]['site_name']]
        .elevation_da.item())
    maca_da_repr_list.append(maca_da_repr)

MACA_grouped_df['climate_da'] = maca_da_repr_list

# %%
soil_da_repr_list = []

for i in range(len(soil_df)):

    soil_da_orig = (
        soil_df.iloc[i]['soil_da'])
    soil_da_repr = soil_da_orig.rio.reproject_match(
        topo_df[topo_df['site_name'] == soil_df.iloc[i]['site_name']]
        .elevation_da.item())
    soil_da_repr_list.append(soil_da_repr)

soil_df['soil_da'] = soil_da_repr_list

# %%
aspect_da_repr_list = []

for i in range(len(topo_df)):

    aspect_da_orig = (
        topo_df.iloc[i]['aspect_da'])
    aspect_da_repr = aspect_da_orig.rio.reproject_match(
        topo_df.iloc[i].elevation_da)
    aspect_da_repr_list.append(aspect_da_repr)

topo_df['aspect_da'] = aspect_da_repr_list

# %%
# List of DataFrames to loop through
dfs = [soil_df, topo_df, MACA_grouped_df]

forest_df = dfs[0]

for merging_df in dfs[1:]:
    # Join DataFrames
    forest_df = pd.merge(forest_df, merging_df, on='site_name')

forest_df.info()

# %% [markdown]
# ## STEP 4: DEVELOP A FUZZY LOGIC MODEL
# 
# A fuzzy logic model is one that is built on expert knowledge rather than
# training data. You may wish to use the
# [`scikit-fuzzy`](https://pythonhosted.org/scikit-fuzzy/) library, which
# includes many utilities for building this sort of model. In particular,
# it contains a number of **membership functions** which can convert your
# data into values from 0 to 1 using information such as, for example, the
# maximum, minimum, and optimal values for soil pH.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>To train a fuzzy logic habitat suitability model:</p>
# <pre><code>1. Research S. nutans, and find out what optimal values are for each variable you are using (e.g. soil pH, slope, and current climatological annual precipitation). 
# 2. For each **digital number** in each raster, assign a **continuous** value from 0 to 1 for how close that grid square is to the optimum range (1=optimal, 0=incompatible). 
# 3. Combine your layers by multiplying them together. This will give you a single suitability number for each square.
# 4. Optionally, you may apply a suitability threshold to make the most suitable areas pop on your map.</code></pre></div></div>
# 
# > **Tip**
# >
# > If you use mathematical operators on a raster in Python, it will
# > automatically perform the operation for every number in the raster.
# > This type of operation is known as a **vectorized** function. **DO NOT
# > DO THIS WITH A LOOP!**. A vectorized function that operates on the
# > whole array at once will be much easier and faster.

# %%
def calculate_suitability_score(raster, optimal_value, tolerance_range):
    """
    Calculate a fuzzy suitability score (0-1) for each raster cell based on proximity to the optimal value.

    Args:
        raster (xarray.DataArray): Input raster layer.
        optimal_value (float): The optimal value for the variable.
        tolerance_range (float): The range within which values are considered suitable.

    Returns:
        xarray.DataArray: A raster of suitability scores (0-1).
    """
    # Calculate suitability scores using a fuzzy Gaussian function
    suitability = np.exp(-((raster - optimal_value) ** 2) / (2 * tolerance_range ** 2))
    return suitability


# %%
def build_habitat_suitability_model(input_rasters, optimal_values, tolerance_ranges,):
    """
    Build a habitat suitability model by combining fuzzy suitability scores for each variable.

    Args:
        input_rasters (list): List of paths to input raster files representing environmental variables.
        optimal_values (list): List of optimal values for each variable.
        tolerance_ranges (list): List of tolerance ranges for each variable.

    Returns:
        xarray.Datarray: A raster of combined suitability scores (0-1)
    """

    # Load and calculate suitability scores for each raster
    suitability_layers = []
    for raster, optimal_value, tolerance_range in zip(
        input_rasters, optimal_values, tolerance_ranges):
        suitability_layer = calculate_suitability_score(raster, optimal_value, tolerance_range)
        suitability_layers.append(suitability_layer)

    # Combine suitability scores by multiplying across all layers
    combined_suitability = suitability_layers[0]
    for layer in suitability_layers[1:]:
        combined_suitability *= layer

    return combined_suitability

# %%
optimal_values = [5.5, 45, 1143]
tolerance_ranges = [0.5, 180, 127]

# %%
# Create fuzzy logic suitability model
model_list = []

for i in range(len(forest_df)):
    rasters = [
        forest_df['soil_da'][i],
        forest_df['aspect_da'][i],
        forest_df['climate_da'][i]
    ]
    model = build_habitat_suitability_model(
        rasters,
        optimal_values,
        tolerance_ranges
    )
    model_list.append(model)

forest_df['fz_model'] = model_list

# %% [markdown]
# ## STEP 5: PRESENT YOUR RESULTS
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Generate some plots that show your key findings. Don’t forget to
# interpret your plots!</p></div></div>

# %%
(model_list[0].hvplot(title='Shawnee National Forest, Warm and Dry Climate 2036-2066',
                     cmap='greens')
+
model_list[1].hvplot(title='Shawnee National Forest, Warm and Wet Climate 2036-2066',
                     cmap='greens')
+
model_list[2].hvplot(title='Shawnee National Forest, Cool and Wet Climate 2036-2066',
                     cmap='greens')
+
model_list[3].hvplot(title='Shawnee National Forest, Cool and Dry Climate 2036-2066',
                     cmap='greens')
)

# %%
(model_list[4].hvplot(title='Shawnee National Forest, Warm and Dry Climate 2066-2096',
                     cmap='greens')
+
model_list[5].hvplot(title='Shawnee National Forest, Warm and Wet Climate 2066-2096',
                     cmap='greens')
+
model_list[6].hvplot(title='Shawnee National Forest, Cool and Wet Climate 2066-2096',
                     cmap='greens')
+
model_list[7].hvplot(title='Shawnee National Forest, Cool and Dry Climate 2066-2096',
                     cmap='greens')
)

# %% [markdown]
# At Shawnee National Forest, American Ginseng maximizes its suitable habitat under a cool and dry climate. It is possible that too much precipitation will cause the valuable roots to have fungal problems, reducing their habitat. This is especially apparent at the end of the century, when the suitable habitat for ginseng almost disappears under all but the cool and dry climate scenario.

# %%
(model_list[8].hvplot(title='Nantahala National Forest, Warm and Dry Climate 2036-2066',
                     cmap='blues')
+
model_list[9].hvplot(title='Nantahala National Forest, Warm and Wet Climate 2036-2066',
                     cmap='blues')
+
model_list[10].hvplot(title='Nantahala National Forest, Cool and Wet Climate 2036-2066',
                     cmap='blues')
+
model_list[11].hvplot(title='Nantahala National Forest, Cool and Dry Climate 2036-2066',
                     cmap='blues')
)

# %%
(model_list[12].hvplot(title='Nantahala National Forest, Warm and Dry Climate 2066-2096',
                     cmap='blues')
+
model_list[13].hvplot(title='Nantahala National Forest, Warm and Wet Climate 2066-2096',
                     cmap='blues')
+
model_list[14].hvplot(title='Nantahala National Forest, Cool and Wet Climate 2066-2096',
                     cmap='blues')
+
model_list[15].hvplot(title='Nantahala National Forest, Cool and Dry Climate 2066-2096',
                     cmap='blues')
)

# %% [markdown]
# Nantahala National Forest tells us a different story, where during the mid-century, American Ginseng does best under warm and dry climate conditions. However, at the end of the century, American ginseng's habitat is most suitable at high elevations under a warm and dry climate scenario, and low elevations under a cool and wet climate scenario.


