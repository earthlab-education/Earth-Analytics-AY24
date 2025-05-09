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
# Understanding grasslands is essential in understanding global carbon sinks. In particular, understanding the layer of soil that lies directly beneath the surface. The presence of Andropogon gerardii, or Big Bluestem, in grassland ecosystems has been significantly connected to the quantity of carbon stored in grassland ecosystems. Where there is more bluestem, there is more stored carbon. How has this changed in recent history - have there been any changes in Bluestem's range? How might Bluestem's range change in the future, given various climate scenarios? 
# 
# In this study, I will examine Bluestem's historical and future range across the Buffalo Gap and Little Missouri national grasslands. I will use several different variables to come up with a suitability raster, which will show areas in which the habitat is suitable for bluestem. I will look at elevation ranges as well as soil variables, and I will combine this with climate projection scenarios to determine ways in which the bluestem's habitat range might change.
# 

# %% [markdown]
# ### Pseudocode outline (by notebooks):
# 
# - Climate_01_gbif: This notebook accesses the species occurrence information for bluestem from the GBIF database.
# - Climate_02_grassland_boundaries: Accesses the grassland boundary polygons from the USFS website. It then selects the specific grasslands for the study using the NationalGR number, and prints the boundaries on top of ESRI map tiles.
# - Climate_03_soil_urls: Compiles a list of urls from the Polaris soil characteristics database. The code selects soil tiles based on the latitude and longitude boundaries of the grassland polygons. Currently this notebook is looking at soil pH values.
# - Climate_03_soil_urls2: This notebook is intended to look at a second set of soil characteristics (soil type).
# - Climate_04_soil_tiles: Prints out the Polaris soil tiles.
# - Climate_05_elevation1: Prints out an SRTM Digital Elevation Model for the Little Missouri national grassland.
# - Climate_05_elevation2: Prints out an SRTM Digital Elevation Model for the Buffalo Gap national grassland.
# - Climate_06_macav2: Runs the maca V2 climate projection scenario.
# - climate_07_fuzzy_logic: Contains the fuzzy logic model from last semester.
# - Climate_07_fuzzy_logic2: Contains the fuzzy logic model from this semester.
# - Climate_08_suitability: Uses raster multiplication to calculate a suitability score for bluestem.
# 

# %% [markdown]
# # Species Overview: Andropogon gerardii (Big Bluestem)
# 
# In terms of global carbon storage and sequestration, soil health is critical. Globally, soil organic carbon holds about 1550 Pg of carbon. Soil inorganic carbon holds about 950 Pg, which together make soil the third largest carbon pool on Earth after the geologic pool (4130 Pg, including fossil fuels) and the oceanic pool (38000 Pg)(Source 1). This figure represents around 80% of the carbon found in terrestrial ecosystems (Source 2).
# 
# In one study, the species Andropogon gerardii, commonly called Big Bluestem, was found to have a positive association with carbon storage, and was used as an indicator species (Source 3).
# 
# Andropogon gerardii habitat characteristics and tolerances:
# - Soil type: Lowland silt and clay loam
# - Soil horizon > 10in
# - elevation: 
#         CO     3,400 to 9,500 feet (1,036-2,896 m)
#         NM     3,500 to 9,000 feet (1,067-2,743 m)
#         MT     3,300 to 4,000 feet (1,006-1,219 m)
#         WY     3,600 to 9,000 feet (1,067-2,743 m)
# (Source 4)
# - Optimal soil temperature: 25C. "Soil temperatures below 20 °C caused significant reductions in foliar nitrogen (N) and phosphorus(P) concentration". "reduced photosynthetic rates observed at 5 and 10 °C Tsoil" "Net photosynthesis was < 12 /^mol m"^ s"' at 5 and 10 °C Tsoil and > 20 micromol\ m^-2 s^-1 at 15-40 °C.". Based on the graphs in this study, it seems that bluestem would be growth-limited above 30 and below 20 degrees C, and severely limited outside 15-35 degrees. (Source 5)
# - min soil temp -15C (source 8)
# - well drained or moist soil, full sun (Source 8)
# - Precip: 10-60mm / yr (source7)
# - pH range: 6.0-7.5 (source 10). Another source (11) says that it does fine with pH 5.5, so we'll use this as out lower-range tolerance. Optimal then is 6.5 (halfway)
# 
# Sources:
# 1) https://royalsocietypublishing.org/doi/pdf/10.1098/rstb.2007.2185
# 2) https://www.nature.com/scitable/knowledge/library/soil-carbon-storage-84223790/#:~:text=The%20amount%20of%20C%20in,in%20soil%20(Lal%202008).
# 3) https://pmc.ncbi.nlm.nih.gov/articles/PMC8367897/
# 4) https://www.fs.usda.gov/database/feis/plants/graminoid/andger/all.html
# 5) https://nph.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-8137.1992.tb01804.x
# 6) https://plants.usda.gov/DocumentLibrary/factsheet/pdf/fs_ange.pdf
# 7) https://plants.usda.gov/DocumentLibrary/plantguide/pdf/pg_scsc.pdf
# 8) https://pfaf.org/user/Plant.aspx?LatinName=Andropogon+gerardii 
# 9) https://link.springer.com/article/10.1023/A:1020320214750 
# 10) https://greatbasinseeds.com/product/big-bluestem/?srsltid=AfmBOoqeQLl_4uKA8wwv_4x4rFgQ7XERV07BlzAzxuMDXfQcyyM8s26m
# 11) https://halehabitat.com/products/big-bluestem

# %% [markdown]
# YOUR QUESTION HERE

# %%
pip install pygbif

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


# %% [markdown]
# Note: GBIF login is not working, however I kept the code in case it can be fixed!

# %%
### set gbif dir
gbif_dir = os.path.join(data_dir, 'gbif_ponderosa')


### access gbif
reset_credentials=True

### enter gbif username, password, email
credentials=dict(
	GBIF_USER=(input, 'GBIF username:'),
	GBIF_PWD=(getpass, 'GBIF password'),
	GBIF_EMAIL=(input, 'GBIF email')
)
for env_variable, (prompt_func, prompt_text) in credentials.items():

	### delete credential from the environment
	if reset_credentials and (env_variable in os.environ):
		os.environ.pop(env_variable)

	### ask for credential and save to environment
	if not env_variable in os.environ:
		os.environ[env_variable] = prompt_func(prompt_text)



# %%
### species names
species_name = 'Andropogon gerardii'

### species info for gbif
species_info = species.name_lookup(species_name,
					rank = 'SPECIES')

### grab the first resule
first_result = species_info['results'][0]

### get species key
species_key = first_result['nubKey']

### check on that
first_result['species'], species_key



# %%
### assign species code
species_key = 4932035

# %%
### set a file pattern
gbif_pattern = os.path.join(gbif_dir,
				'*.csv')

# %%
# ### download it once:
# if not glob(gbif_pattern):

# 	### submit my query to GBIF
# 	gbif_query = occ.download([
# 	f"speciesKey = {species_key}",
# 	"hasCoordinate = True",
# 	])


# ### only download once
# 	if not 'GBIF_DOWNLOAD_KEY' in os.environ:
# 		os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
# 		download_key=os.environ['GBIF_DOWNLOAD_KEY']

# 	### wait for download to build
# 		wait = occ.download_meta(download_key)['status']
# 		time.sleep(5)

# 	### download the data
# 	download_info = occ.download_get(
# 		os.environ['GBIF_DOWNLOAD_KEY'],
# 		path = data_dir
# 	)

# ### unzip it
# with zipfile.ZipFile(download_info['path']) as download_zip:
# 	download_zip.extractall(path = gbif_dir)


# ### find csv file path
# gbif_path = glob(gbif_pattern)[0]


# %%
# ### download it once:
# if not glob(gbif_pattern):

# 	### submit my query to GBIF
# 	gbif_query = occ.download([
# 	f"speciesKey = {species_key}",
# 	"hasCoordinate = True",
# 	])

# 	### only download once
# 	if not 'GBIF_DOWNLOAD_KEY' in os.environ:
# 		os.environ['GBIF_DOWNLOAD_KEY'] = gbif_query[0]
# 		download_key=os.environ['GBIF_DOWNLOAD_KEY']

# 		### wait for download to build
# 		wait = occ.download_meta(download_key)['status']
# 		time.sleep(5)

# 		### download the data
# 		download_info = occ.download_get(
# 			os.environ['GBIF_DOWNLOAD_KEY'],
# 			path = data_dir
# 	)

# ### unzip it
# with zipfile.ZipFile(download_info['path']) as download_zip:
# 	download_zip.extractall(path = gbif_dir)


# ### find csv file path
# gbif_path = glob(gbif_pattern)[0]


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
# YOUR SPECIES DESCRIPTION HERE

# %% [markdown]
# In terms of global carbon storage and sequestration, soil health is critical. Globally, soil organic carbon holds about 1550 Pg of carbon. Soil inorganic carbon holds about 950 Pg, which together make soil the third largest carbon pool on Earth after the geologic pool (4130 Pg, including fossil fuels) and the oceanic pool (38000 Pg)(Source 1). This figure represents around 80% of the carbon found in terrestrial ecosystems (Source 2).
# 
# In one study, the species Andropogon gerardii, commonly called Big Bluestem, was found to have a positive association with carbon storage, and was used as an indicator species (Source 3).
# 
# Andropogon gerardii habitat characteristics and tolerances:
# - Soil type: Lowland silt and clay loam
# - Soil horizon > 10in
# - elevation: 
#         CO     3,400 to 9,500 feet (1,036-2,896 m)
#         NM     3,500 to 9,000 feet (1,067-2,743 m)
#         MT     3,300 to 4,000 feet (1,006-1,219 m)
#         WY     3,600 to 9,000 feet (1,067-2,743 m)
# (Source 4)
# - Optimal soil temperature: 25C. "Soil temperatures below 20 °C caused significant reductions in foliar nitrogen (N) and phosphorus(P) concentration". "reduced photosynthetic rates observed at 5 and 10 °C Tsoil" "Net photosynthesis was < 12 /^mol m"^ s"' at 5 and 10 °C Tsoil and > 20 micromol\ m^-2 s^-1 at 15-40 °C.". Based on the graphs in this study, it seems that bluestem would be growth-limited above 30 and below 20 degrees C, and severely limited outside 15-35 degrees. (Source 5)
# - min soil temp -15C (source 8)
# - well drained or moist soil, full sun (Source 8)
# - Precip: 10-60mm / yr (source7)
# - pH 6.0-7.5 (source 10)
# 
# Sources:
# 1) https://royalsocietypublishing.org/doi/pdf/10.1098/rstb.2007.2185
# 2) https://www.nature.com/scitable/knowledge/library/soil-carbon-storage-84223790/#:~:text=The%20amount%20of%20C%20in,in%20soil%20(Lal%202008).
# 3) https://pmc.ncbi.nlm.nih.gov/articles/PMC8367897/
# 4) https://www.fs.usda.gov/database/feis/plants/graminoid/andger/all.html
# 5) https://nph.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-8137.1992.tb01804.x
# 6) https://plants.usda.gov/DocumentLibrary/factsheet/pdf/fs_ange.pdf
# 7) https://plants.usda.gov/DocumentLibrary/plantguide/pdf/pg_scsc.pdf
# 8) https://pfaf.org/user/Plant.aspx?LatinName=Andropogon+gerardii 
# 9) https://link.springer.com/article/10.1023/A:1020320214750 
# 10) https://greatbasinseeds.com/product/big-bluestem/?srsltid=AfmBOoqeQLl_4uKA8wwv_4x4rFgQ7XERV07BlzAzxuMDXfQcyyM8s26m

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
# YOUR SITE DESCRIPTION HERE

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
# YOUR TIME PERIODS HERE

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

# %% [markdown]
# LIST THE CLIMATE MODELS YOU SELECTED HERE AND CITE THE CLIMATE TOOLBOX

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
# Download soil data

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
# Download soil data

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
# Download climate data

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
# Download soil data

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
# Create fuzzy logic suitability model

# %% [markdown]
# ## STEP 5: PRESENT YOUR RESULTS
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Generate some plots that show your key findings. Don’t forget to
# interpret your plots!</p></div></div>

# %%
# Create plots

# %% [markdown]
# YOUR PLOT INTERPRETATION HERE


