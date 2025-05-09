# %% [markdown]
# # Habitat suitability under climate change
# 
# [Our changing climate is changing where key grassland species can live,
# and grassland management and restoration practices will need to take
# this into
# account.](https://www.frontiersin.org/articles/10.3389/fpls.2017.00730/full)
# 
# In this coding challenge, you will create a habitat suitability model
# for a species of your choice that lives in the continental United States
# (CONUS). We have this limitation because the downscaled climate data we
# suggest, the [MACAv2 dataset](https://www.climatologylab.org/maca.html),
# is only available in the CONUS – if you find other downscaled climate
# data at an appropriate resolution you are welcome to choose a different
# study area. If you don’t have anything in mind, you can take a look at
# Sorghastrum nutans, a grass native to North America. [In the past 50
# years, its range has moved
# northward](https://www.gbif.org/species/2704414).
# 
# Your suitability assessment will be based on combining multiple data
# layers related to soil, topography, and climate. You will also need to
# create a **modular, reproducible, workflow** using functions and loops.
# To do this effectively, we recommend planning your code out in advance
# using a technique such as pseudocode outline or a flow diagram. We
# recommend planning each of the blocks below out into multiple steps. It
# is unnecessary to write a step for every line of code unles you find
# that useful. As a rule of thumb, aim for steps that cover the major
# structures of your code in 2-5 line chunks.
# 
# ## STEP 1: STUDY OVERVIEW
# 
# Before you begin coding, you will need to design your study.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>What question do you hope to answer about potential future changes in
# habitat suitability?</p></div></div>

# %% [markdown]
# How will potential climate changes in Northern and Southern Californian habitats affect suitability for the endemic blue oak?

# %% [markdown]
# ### Species
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Select the species you want to study, and research it’s habitat
# parameters in scientific studies or other reliable sources. You will
# want to look for reviews or overviews of the data, since an individual
# study may not have the breadth needed for this purpose. In the US, the
# National Resource Conservation Service can have helpful fact sheets
# about different species. University Extension programs are also good
# resources for summaries.</p>
# <p>Based on your research, select soil, topographic, and climate
# variables that you can use to determine if a particular location and
# time period is a suitable habitat for your species.</p></div></div>
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Write a description of your species. What habitat is it found in?
# What is its geographic range? What, if any, are conservation threats to
# the species? What data will shed the most light on habitat suitability
# for this species?</p></div></div>

# %% [markdown]
# The blue oak (*Quercus douglasii*) is a deciduous drought-tolerant tree endemic to California. 
# Blue oaks can be identified by their blue-green foliage, slightly to deeply lobed leaves, textured and pale gray bark, typical height between 20-60 ft., and natural settings of dry, rocky, and somewhat acidic to neutral soils. This species can be found among plant communities such as chaparral, foothill woodland, and oak woodland at elevations of 500-2000 ft. in the north and up to 5000 ft. in the south. The blue oak is dispersed throughout the state including in the central Sierra Nevada Eldorado National Forest and the central Coast and Transverse Ranges in Los Padres National Forest. Conservation threats facing the oak include escalation of pathogens and diminished regeneration. 

# %% [markdown]
# ### Sites
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Select at least two site to study, such as two of the U.S. National
# Grasslands. You can download the <a
# href="https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.NationalGrassland.zip">USFS
# National Grassland Units</a> and select your study sites. Generate a
# site map for each location.</p>
# <p>When selecting your sites, you might want to look for places that are
# marginally habitable for this species, since those locations will be
# most likely to show changes due to climate.</p></div></div>
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Write a site description for each of your sites, or for all of your
# sites as a group if you have chosen a large number of linked sites. What
# differences or trends do you expect to see among your sites?</p></div></div>

# %% [markdown]
# Los Padres National Forest covers around 1.75 million acres in two distinct land divisions among the central California and Transverse Ranges in Southern California. In the North, Eldorado National Forest is embedded in the central Sierra Nevada comprising nearly 800,000 acres. Differences in hydrologic regimes, terrain, and temperature are observed across the sites.

# %% [markdown]
# ### Time periods
# 
# In general when studying climate, we are interested in **climate
# normals**, which are typically calculated from 30 years of data so that
# they reflect the climate as a whole and not a single year which may be
# anomalous. So if you are interested in the climate around 2050, download
# at least data from 2035-2065.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Select at least two 30-year time periods to compare, such as
# historical and 30 years into the future. These time periods should help
# you to answer your scientific question.</p></div></div>

# %% [markdown]
# Time periods: 
# - Mid 21st Century (2036-2066)
# - Late 21st century (2066-2096)

# %% [markdown]
# ### Climate models
# 
# There is a great deal of uncertainty among the many global climate
# models available. One way to work with the variety is by using an
# **ensemble** of models to try to capture that uncertainty. This also
# gives you an idea of the range of possible values you might expect! To
# be most efficient with your time and computing resources, you can use a
# subset of all the climate models available to you. However, for each
# scenario, you should attempt to include models that are:
# 
# -   Warm and wet
# -   Warm and dry
# -   Cold and wet
# -   Cold and dry
# 
# for each of your sites.
# 
# To figure out which climate models to use, you will need to access
# summary data near your sites for each of the climate models. You can do
# this using the [Climate Futures Toolbox Future Climate Scatter
# tool](https://climatetoolbox.org/tool/Future-Climate-Scatter). There is
# no need to write code to select your climate models, since this choice
# is something that requires your judgement and only needs to be done
# once.
# 
# If your question requires it, you can also choose to include multiple
# climate variables, such as temperature and precipitation, and/or
# multiple emissions scenarios, such as RCP4.5 and RCP8.5.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Choose at least 4 climate models that cover the range of possible
# future climate variability at your sites. How did you choose?</p></div></div>
# 
# - Warm and wet (CanESM2)
# - Warm and dry (MIROC-ESM-CHEM)
# - Cool and wet (CNRM-CM5)
# - Cool and dry (GFDL-ESM2M)

# %% [markdown]
# ## Setup Analysis

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pathlib
import zipfile
from glob import glob
import tqdm as notebook_tqdm

import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import xarray as xr
import xrspatial

import matplotlib.pyplot as plt
import hvplot.pandas
import hvplot.xarray
import cartopy.crs as ccrs
import geoviews as gv

import earthaccess

# %%
from suitability_utils import *

# %% [markdown]
# Set Paths

# %%
# Plots
plots_dir = os.path.join(
    # Home directory
    pathlib.Path.home(),
    'Projects',
    # Project directory
    'habitat-suitability-lauren-alexandra',
    'plots'
)

# Datasets
datasets_dir = os.path.join(
    # Home directory
    pathlib.Path.home(),
    'Projects',
    # Project directory
    'habitat-suitability-lauren-alexandra',
    'datasets'
)

# Project data directory 
data_dir = os.path.join(
    # Home directory
    pathlib.Path.home(),
    'Projects',
    # Project directory
    'habitat-suitability-lauren-alexandra',
    'data'
)

# Define directories for data
land_units_dir = os.path.join(data_dir, 'usfs-national-lands')
eldorado_elevation_dir = os.path.join(data_dir, 'srtm', 'eldorado')
los_padres_elevation_dir = os.path.join(data_dir, 'srtm', 'los_padres')

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(eldorado_elevation_dir, exist_ok=True)
os.makedirs(los_padres_elevation_dir, exist_ok=True)

# %% [markdown]
# Load Sites

# %%
# Only extract once
usfs_pattern = os.path.join(land_units_dir, '*.shp')

if not glob(usfs_pattern):
    usfs_zip = f'{datasets_dir}/S_USA.NFSLandUnit.zip'

    # Unzip data
    with zipfile.ZipFile(usfs_zip, 'r') as zip:
        zip.extractall(path=land_units_dir)

# Find the extracted .shp file path
usfs_land_path = glob(usfs_pattern)[0]

# Load USFS land units from shapefile
usfs_land_units_gdf = (
    gpd.read_file(usfs_land_path)
)

# Obtain units with location name
valid_units = usfs_land_units_gdf.dropna(subset=['HQ_LOCATIO'])

# Select only CA units
all_ca_units = valid_units[valid_units['HQ_LOCATIO'].str.contains('CA')]

# %%
earthaccess.login(strategy="interactive", persist=True)

# %%
# Search for Digital Elevation Models

ea_dem = earthaccess.search_datasets(keyword='SRTM DEM', count=15)
for dataset in ea_dem:
    print(dataset['umm']['ShortName'], dataset['umm']['EntryTitle'])

# %% [markdown]
# Plot Sites

# %%
# Los Padres National Forest

los_padres_gdf = all_ca_units.loc[
    all_ca_units['NFSLANDU_2'] == 'Los Padres National Forest'
]
los_padres_gdf

# %%
# Create folium map instance
lp_interactive_site = los_padres_gdf.explore()

lp_interactive_site

# %%
# Eldorado National Forest

eldorado_gdf = all_ca_units.loc[
    all_ca_units['NFSLANDU_2'] == 'Eldorado National Forest'
]
eldorado_gdf

# %%
# Create folium map instance

eld_interactive_site = eldorado_gdf.explore()

eld_interactive_site

# %% [markdown]
# ## STEP 2: DATA ACCESS

# %% [markdown]
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
# Set site parameters

soil_prop = 'ph'
soil_prop_stat = 'mean'
# cm (the minimum depth for blue oaks is 1-2 ft)
soil_depth = '30_60'

# %%
eldorado_soil_da = download_polaris('eldorado', eldorado_gdf, soil_prop, 
                                    soil_prop_stat, soil_depth,
                                    'Eldorado-National-Forest',
                                    'Eldorado National Forest',
                                    data_dir, plots_dir)

# %%
los_padres_soil_da = download_polaris('los_padres', los_padres_gdf, soil_prop, 
                                      soil_prop_stat, soil_depth,
                                      'Los-Padres-National-Forest',
                                      'Los Padres National Forest',
                                      data_dir, plots_dir)

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
eldorado_elev_da = download_topography(
                        'eldorado', eldorado_gdf,
                        'Eldorado-National-Forest', 
                        'Eldorado National Forest',
                        eldorado_elevation_dir, data_dir, plots_dir
                    )

# %%
los_padres_elev_da = download_topography(
                        'los_padres', los_padres_gdf, 
                        'Los-Padres-National-Forest',
                        'Los Padres National Forest',
                        los_padres_elevation_dir, data_dir, plots_dir
                    )

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
# Set climate parameters

emissions_scenario = 'rcp45'
climate_models = ['CanESM2', 'MIROC-ESM-CHEM', 'CNRM-CM5', 'GFDL-ESM2M']
mid_century = [2036, 2041, 2046, 2051, 2056, 2061] # 2036-2066 
late_century = [2066, 2071, 2076, 2081, 2086, 2091] # 2066-2096 
time_periods = [mid_century, late_century]

# %% [markdown]
# Eldorado Projected Climate (RCP 4.5)

# %%
# Mid century 

download_climate('eldorado', eldorado_gdf,
                emissions_scenario, climate_models, mid_century,
                'eldorado_mid_century', data_dir)

# %%
# Late century

download_climate('eldorado', eldorado_gdf,
                emissions_scenario, climate_models, late_century,
                'eldorado_late_century', data_dir) 

# %% [markdown]
# Los Padres Projected Climate (RCP 4.5)

# %%
# Mid century 

download_climate('los padres', los_padres_gdf,
                emissions_scenario, climate_models, mid_century,
                'los_padres_mid_century', data_dir)

# %%
# Late century

download_climate('los padres', los_padres_gdf,
                emissions_scenario, climate_models, late_century,
                'los_padres_late_century', data_dir)

# %% [markdown]
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-respond"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Reflect and Respond</div></div><div class="callout-body-container callout-body"><p>Make sure to include a description of the climate data and how you
# selected your models. Include a citation of the MACAv2 data.</p></div></div>

# %% [markdown]
# Using the University of California Merced Future Climate Scenarios tool, I compared models under the RCP 4.5 emissions scenario (2050), looking at winter and summer mean temperatures (and change relative to historical by °F) as well as winter precipitation (and percent change relative to the historical value). 
# 
# - Warm and wet (CanESM2)
# - Warm and dry (MIROC-ESM-CHEM)
# - Cool and wet (CNRM-CM5)
# - Cool and dry (GFDL-ESM2M)
# 
# University of California Merced (n.d.). *Future Climate Scenarios*. The Climate Toolbox. https://climatetoolbox.org/tool/Future-Climate-Scenarios

# %% [markdown]
# [MACAV2-METDATA DAILY/MONTHLY](https://www.reacchpna.org/thredds/reacch_climate_CMIP5_macav2_catalog2.html) is a downscaled climate dataset that covers CONUS at 4 kilometers (1/24 degree) resolution. Temporal coverage encompasses historical model output (1950-2005) and projections (2006-2099) for climate normals represented as monthly averages of daily values. 
# 
# Abatzoglou, J. T. (2017). *REACCH Climatic Modeling CMIP5 MACAV2-METDATA DAILY/MONTHLY Catalog* [Data set]. https://www.reacchpna.org/thredds/reacch_climate_CMIP5_macav2_catalog2.html

# %% [markdown]
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

# %% [markdown]
# See harmonize_raster_layers() in utils.py.

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
# Optimal values for species for each variable
# Variables: elevation, temperature, aspect, soil pH
optimal_values = [1500.0, 90.0, 0.0, 6.75]

# Tolerance ranges (acceptable deviation) for each variable
tolerance_ranges = [1500.0, 20.0, 65.0, 0.75]

# %% [markdown]
# See build_habitat_suitability_model() in utils.py.

# %% [markdown]
# ## STEP 5: PRESENT YOUR RESULTS
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Generate some plots that show your key findings. Don’t forget to
# interpret your plots!</p></div></div>

# %% [markdown]
# #### Eldorado Suitability

# %%
# CanESM2 (Warm and wet)

build_habitat_suitability_model('eldorado', 'mid_century', 'CanESM2',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_mc_CanESM2_suitability')

build_habitat_suitability_model('eldorado', 'late_century', 'CanESM2',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_lc_CanESM2_suitability')

# MIROC-ESM-CHEM (Warm and dry)

build_habitat_suitability_model('eldorado', 'mid_century', 'MIROC-ESM-CHEM',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_mc_MIROC-ESM-CHEM_suitability')

build_habitat_suitability_model('eldorado', 'late_century', 'MIROC-ESM-CHEM',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_lc_MIROC-ESM-CHEM_suitability')

# CNRM-CM5 (Cool and wet)

build_habitat_suitability_model('eldorado', 'mid_century', 'CNRM-CM5',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_mc_CNRM-CM5_suitability')

build_habitat_suitability_model('eldorado', 'late_century', 'CNRM-CM5',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_lc_CNRM-CM5_suitability')

# GFDL-ESM2M (Cool and dry)

build_habitat_suitability_model('eldorado', 'mid_century', 'GFDL-ESM2M',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_mc_GFDL-ESM2M_suitability')

build_habitat_suitability_model('eldorado', 'late_century', 'GFDL-ESM2M',
                                optimal_values, tolerance_ranges, data_dir, 
                                'eld_lc_GFDL-ESM2M_suitability')

# %% [markdown]
# CanESM2 (Warm and wet)

# %%
plot_site(
    f"{data_dir}/eld_mc_CanESM2_suitability.tif", 
    eldorado_gdf, plots_dir,
    'Eldorado-National-Forest-Suitability-Mid-Century-CanESM2', 
    'Eldorado Mid Century Suitability: RCP 4.5 (CanESM2)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

plot_site(
    f"{data_dir}/eld_lc_CanESM2_suitability.tif", 
    eldorado_gdf, plots_dir,
    'Eldorado-National-Forest-Suitability-Late-Century-CanESM2', 
    'Eldorado Late Century Suitability: RCP 4.5 (CanESM2)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# MIROC-ESM-CHEM (Warm and dry)

# %%
plot_site(
    f"{data_dir}/eld_lc_MIROC-ESM-CHEM_suitability.tif",
    eldorado_gdf, plots_dir,
    'Eldorado-National-Forest-Suitability-Late-Century-MIROC-ESM-CHEM', 
    'Eldorado Late Century Suitability: RCP 4.5 (MIROC-ESM-CHEM)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# CNRM-CM5 (Cool and wet)

# %%
plot_site(
    f"{data_dir}/eld_lc_CNRM-CM5_suitability.tif",
    eldorado_gdf, plots_dir,
    'Eldorado-National-Forest-Suitability-Late-Century-CNRM-CM5', 
    'Eldorado Late Century Suitability: RCP 4.5 (CNRM-CM5)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# GFDL-ESM2M (Cool and dry)

# %%
plot_site(
    f"{data_dir}/eld_lc_GFDL-ESM2M_suitability.tif",
    eldorado_gdf, plots_dir,
    'Eldorado-National-Forest-Suitability-Late-Century-GFDL-ESM2M', 
    'Eldorado Late Century Suitability: RCP 4.5 (GFDL-ESM2M)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# #### Los Padres Suitability

# %%
# CanESM2 (Warm and wet)

build_habitat_suitability_model('los_padres', 'mid_century', 'CanESM2',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_mc_CanESM2_suitability')

build_habitat_suitability_model('los_padres', 'late_century', 'CanESM2',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_lc_CanESM2_suitability')

# MIROC-ESM-CHEM (Warm and dry)

build_habitat_suitability_model('los_padres', 'mid_century', 'MIROC-ESM-CHEM',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_mc_MIROC-ESM-CHEM_suitability')

build_habitat_suitability_model('los_padres', 'late_century', 'MIROC-ESM-CHEM',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_lc_MIROC-ESM-CHEM_suitability')

# CNRM-CM5 (Cool and wet)

build_habitat_suitability_model('los_padres', 'mid_century', 'CNRM-CM5',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_mc_CNRM-CM5_suitability')

build_habitat_suitability_model('los_padres', 'late_century', 'CNRM-CM5',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_lc_CNRM-CM5_suitability')

# GFDL-ESM2M (Cool and dry)

build_habitat_suitability_model('los_padres', 'mid_century', 'GFDL-ESM2M',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_mc_GFDL-ESM2M_suitability')

build_habitat_suitability_model('los_padres', 'late_century', 'GFDL-ESM2M',
                                optimal_values, tolerance_ranges, data_dir, 
                                'lp_lc_GFDL-ESM2M_suitability')

# %% [markdown]
# CanESM2 (Warm and wet)

# %%
plot_site(
    f"{data_dir}/lp_mc_CanESM2_suitability.tif", 
    los_padres_gdf, plots_dir,
    'Los-Padres-National-Forest-Suitability-Mid-Century-CanESM2', 
    'Los Padres Mid Century Suitability: RCP 4.5 (CanESM2)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

plot_site(
    f"{data_dir}/lp_lc_CanESM2_suitability.tif", 
    los_padres_gdf, plots_dir,
    'Los-Padres-National-Forest-Suitability-Late-Century-CanESM2', 
    'Los Padres Late Century Suitability: RCP 4.5 (CanESM2)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# MIROC-ESM-CHEM (Warm and dry)

# %%
plot_site(
    f"{data_dir}/lp_lc_MIROC-ESM-CHEM_suitability.tif", 
    los_padres_gdf, plots_dir,
    'Los-Padres-National-Forest-Suitability-Late-Century-MIROC-ESM-CHEM', 
    'Los Padres Late Century Suitability: RCP 4.5 (MIROC-ESM-CHEM)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# CNRM-CM5 (Cool and wet)

# %%
plot_site(
    f"{data_dir}/lp_lc_CNRM-CM5_suitability.tif", 
    los_padres_gdf, plots_dir,
    'Los-Padres-National-Forest-Suitability-Late-Century-CNRM-CM5', 
    'Los Padres Late Century Suitability: RCP 4.5 (CNRM-CM5)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# GFDL-ESM2M (Cool and dry)

# %%
plot_site(
    f"{data_dir}/lp_lc_GFDL-ESM2M_suitability.tif", 
    los_padres_gdf, plots_dir,
    'Los-Padres-National-Forest-Suitability-Late-Century-GFDL-ESM2M', 
    'Los Padres Late Century Suitability: RCP 4.5 (GFDL-ESM2M)', 
    'Suitability', 'YlGn', 'black', tif_file=True
)

# %% [markdown]
# #### Mid and Late 21st Century Suitability for the Endemic Blue Oak
# 
# Habitat suitability for mid and late 21st century was assessed for a medium emissions scenario (RCP 4.5) using four diverging climate models: CanESM2 (Warm/Wet), MIROC-ESM-CHEM (Warm/Dry), CNRM-CM5 (Cool/Wet), and GFDL-ESM2M (Cool/Dry). Species tolerance and optimal ranges for elevation (1,500 ± 1,500 meters), temperature (90 ± 20 °F), aspect (0 ± 65 degrees), and soil (6.75 ± 0.75 pH) were evaluated in site suitability scores. Given the oak's accommodating tolerance for maximum temperature fluctuations, model differences were minimal between plots and time periods. However, this flexibility does not indicate a robustness to extreme seasonal changes. Blue oaks are generally equipped to withstand drier conditions compared to other native trees, abundant in areas with less conducive soils where they often cohabitate with pines. Nonetheless, the onset of a [severe drought diminishes the threshold of site tolerance](https://oaks.cnr.berkeley.edu/wp-content/uploads/2020/04/Swiecki-Bernhardt-Oak-Mortality-4.2-2020.pdf). 
# 
# By the end of the century, the average annual maximum temperatures are [projected to grow by 7.5°F to 10.9°F](https://www.fs.usda.gov/Internet/FSE_DOCUMENTS/fseprd985002.pdf). Intense precipitation events are projected as well, with frequencies of [pronounced dry seasons and whiplash events](https://doi.org/10.1038/s41558-018-0140-y) expanding by over 50% across the state with a notable change in Southern California: a ~200% increase in extreme dry seasons, a ~150% increase in extreme wet seasons, and a ~75% increase in year-to-year whiplash. Historically blue oak mortality has been typically attributed to decay fungi such as canker rots and root-rotting decay fungi. Pathogens like *Phytophthora ramorum*, the cause for sudden oak death, respond faster than trees to drought followed by high precipitation events, accelerating disintegration. The combination of disease and long-term climate stress contributes to raised levels of mortality years after drought. Futhermore, although blue oaks have evolved in an environment where fires occur regularly and can survive low-to-moderate intensity fires, the threat of [conifer encroachment](https://oaks.cnr.berkeley.edu/conifer-encroachment/) subjects oaks to more canopy fire risk in a warmer climate.
# 
# **Sites**
# 
# Los Padres National Forest is [vulnerable to floods](https://www.fs.usda.gov/Internet/FSE_DOCUMENTS/fseprd497638.pdf) and debris flows after extreme precipitation events, in particular after recurrent wildfires, due to the forest’s position in the Coast and Transverse mountain ranges in relation to the coast. Sediment flow in Southern California has a history of high variability under El Niño Southern Oscillation (ENSO) events and has led to the destruction of oak riparian habitat. Eldorado National Forest is similarly projected to experience extreme climate variability between wet and dry years. Toward the end of the century at the lowest and highest elevations in the Sierra Nevada, warming temperatures will lead to greater evaporation and a [projected ~15% reduction in fuel and soil moisture](https://www.fs.usda.gov/Internet/FSE_DOCUMENTS/fseprd985002.pdf). This scenario indicates greater drought likelihood and impacts across native plant communities. Heightened fire weather conditions will generate [shifts in blue oak mortality rates](https://www.fs.usda.gov/Internet/FSE_DOCUMENTS/fseprd985002.pdf) as well as post-fire germination. Moreover a shift in oak mortality affects fire activity. Blue oak sites with [south-facing aspects](https://esajournals.onlinelibrary.wiley.com/doi/full/10.1002/ecs2.3558) in particular demonstrate lower potential for accumulated drainage and thus the highest level of mortality compared to other aspects. Sites like these are especially at risk for greater dead woody surface fuels, enhancing the probability of larger fires over longer periods.


