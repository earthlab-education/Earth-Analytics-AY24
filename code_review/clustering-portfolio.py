# %% [markdown]
# # Land cover classification at the Lower Left Hand Creek Watershed in Colorado
# 
# In this notebook, you will use a k-means **unsupervised** clustering
# algorithm to group pixels by similar spectral signatures. **k-means** is
# an **exploratory** method for finding patterns in data. Because it is
# unsupervised, you don’t need any training data for the model. You also
# can’t measure how well it “performs” because the clusters will not
# correspond to any particular land cover class. However, we expect at
# least some of the clusters to be identifiable as different types of land
# cover.
# 
# You will use the [harmonized Sentinal/Landsat multispectral
# dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).
# You can access the data with an [Earthdata
# account](https://www.earthdata.nasa.gov/learn/get-started) and the
# [`earthaccess` library from
# NSIDC](https://github.com/nsidc/earthaccess):
# 
# ## STEP 1: SET UP
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Import all libraries you will need for this analysis</li>
# <li>Configure GDAL parameters to help avoid connection errors:
# <code>python      os.environ["GDAL_HTTP_MAX_RETRY"] = "5"      os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"</code></li>
# </ol></div></div>

# %%
import os
import pathlib
import pickle
import re
import warnings

import cartopy.crs as ccrs
import earthaccess
import earthpy as et
import geopandas as gpd
import geoviews as gv
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import numpy as np
import pandas as pd
import rioxarray as rxr
import rioxarray.merge as rxrmerge
from tqdm.notebook import tqdm
import xarray as xr
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score ### calculate silhouette score
import zipfile

os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"

warnings.simplefilter('ignore')

# %% [markdown]
# Below you can find code for a caching **decorator** which you can use in
# your code. To use the decorator:
# 
# ``` python
# @cached(key, override)
# def do_something(*args, **kwargs):
#     ...
#     return item_to_cache
# ```
# 
# This decorator will **pickle** the results of running the
# `do_something()` function, and only run the code if the results don’t
# already exist. To override the caching, for example temporarily after
# making changes to your code, set `override=True`. Note that to use the
# caching decorator, you must write your own function to perform each
# task!

# %%
def cached(func_key, override=True):
    """
    A decorator to cache function results
    
    Parameters
    ==========
    key: str
      File basename used to save pickled results
    override: bool
      When True, re-compute even if the results are already stored
    """
    def compute_and_cache_decorator(compute_function):
        """
        Wrap the caching function
        
        Parameters
        ==========
        compute_function: function
          The function to run and cache results
        """
        def compute_and_cache(*args, **kwargs):
            """
            Perform a computation and cache, or load cached result.
            
            Parameters
            ==========
            args
              Positional arguments for the compute function
            kwargs
              Keyword arguments for the compute function
            """
            # Add an identifier from the particular function call
            if 'cache_key' in kwargs:
                key = '_'.join((func_key, kwargs['cache_key']))
            else:
                key = func_key

            path = os.path.join(
                et.io.HOME, et.io.DATA_NAME, 'jars', f'{key}.pickle')
            
            # Check if the cache exists already or override caching
            if not os.path.exists(path) or override:
                # Make jars directory if needed
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                # Run the compute function as the user did
                result = compute_function(*args, **kwargs)
                
                # Pickle the object
                with open(path, 'wb') as file:
                    pickle.dump(result, file)
            else:
                # Unpickle the object
                with open(path, 'rb') as file:
                    result = pickle.load(file)
                    
            return result
        
        return compute_and_cache
    
    return compute_and_cache_decorator

# %% [markdown]
# ## STEP 2: STUDY SITE
# 
# For this analysis, you will use a watershed from the [Water Boundary
# Dataset](https://www.usgs.gov/national-hydrography/access-national-hydrography-products),
# HU12 watersheds (WBDHU12.shp).
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">You will:</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Download the Water Boundary Dataset for the Missouri region, region 10</li>
# <li>Select watershed 1101900050304</li>
# <li>Generate a site map of the watershed</li>
# </ol>
# <p>Try to use the <strong>caching decorator</strong></p></div></div>
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-response"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div></div><div class="callout-body-container callout-body"><p>Write a 2-3 sentence <strong>site description</strong> (with
# citations) of this area that helps to put your analysis in context.</p></div></div>

# %% [markdown]
# **SEE SITE DESCRIPTION BELOW SITE MAP**
# 
# * Helpful links:
# 
#     * How's My Waterway? for the Lower Left Hand Creek watershed: https://mywaterway.epa.gov/community/101900050304/overview
# 
#     * Breakdown of all watershed boundaries including boundary descriptions, names of regions, subretions, accounting units, and cataloging units: https://water.usgs.gov/GIS/huc_name.html
# 
#     * Documentation for et.data.get_data: https://earthpy.readthedocs.io/en/latest/gallery_vignettes/get_data.html
# 
#     * Documentation for .dissolve(): https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 2a. Download watershed boundary data for region 10
# 
# </span>

# %%
# call cached decorator
@cached('wbd-10-co')
# add function to apply decorator abilities to
def read_wbd_file(wbd_filename, huc_level, cache_key):
    # Define wbd_url
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Hydrography/WBD/HU2/Shape/"
        f"{wbd_filename}.zip")
    # et.data.get_data from earthpy library to download data from a zip file
    # can also download Earthpy Dataset Subsets w/ et.data.get_data 
    wbd_dir = et.data.get_data(url=wbd_url)
                  
    # Read desired data
    wbd_path = os.path.join(wbd_dir, 'Shape', f'WBDHU{huc_level}.shp')
    # engine is the library used to read the file
    wbd_gdf = gpd.read_file(wbd_path, engine='pyogrio')
    return wbd_gdf

# define huc_level
huc_level = 12

# create wbd_total_gdf 
co_wbd_total_gdf = read_wbd_file(
    # filename found from URL noted above code cell
    wbd_filename = 'WBD_10_HU2_Shape',
    # huc_level defined above
    huc_level = huc_level,
    # cache_key is a kwarg for cached decorator
    cache_key=f'hu{huc_level}'
)

# check wbd_total_gdf
co_wbd_total_gdf.head()


# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 2b. Select watershed of interest from region 10, 101900050304
# 
# </span>

# %%
# see all column names of the wbd_total_gdf
co_wbd_total_gdf.columns

# %%
# select the 101900050304 waterhsed rows from the wbd_total_gdf
wbd_gdf = (
    # from the huc12 column...
    co_wbd_total_gdf[co_wbd_total_gdf[f'huc{huc_level}']
    # select only the rows for 101900050304
    .isin(['101900050304'])]
    # dissolve all geometries to a single geometric feature
    # and aggreagate all rows of data in a group
    .dissolve()
)

# check wbd_gdf
wbd_gdf

# %%
print(f"Watershed {wbd_gdf['huc12']} is in {wbd_gdf['states']}")

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 2c. Create site map of watershed 101900050304 in Colorado, USA & describe watershed
# 
# </span>

# %%
# create site map with hvplot
(
    wbd_gdf
    .hvplot(
        # geographic plot, lat/lon axes
        geo=True,
        # overlay plot on EsriImagery tiles
        tiles='EsriImagery',
        # fill color
        color='white',
        # transparency of fill color
        alpha=0.4,
        # line color
        line_color='darkblue',
        line_width=4, 
        # define title
        title='Lower Left Hand Creek Watershed (101900050304) Boundary in Colorado, USA',
        # define size of plot
        frame_width=650
    )
)

# %% [markdown]
# <span style="color: maroon;">
# 
# **Site Description:**
# 
# Watershed 101900050304 is in Boulder County, Colorado. It is part of the Missouri Region which includes the United Staes drainage of the Missouri River and Saskatchewan River basins as well as some small closed basins. Nebraska is in region 10 and so are parts of Colorado, Iowa, Kansas, Minnesota, Missouri, Montana, North Dakota, South Dakota, and Wyoming. Watershed 101900050304 is in subregion 1019 which is the South Platte River Basin. Cataloging unit 10190005 is St. Vrain in Colorado.<sup>1</sup> Watershed 1019000503034 is called the Lower Left Hand Creek watershed and covers 9,496 acres. Look at the map of the watershed on How's My Waterway, we can see that part of Left Hand Creek runs west to east in this watershed. Left Hand Creek begins outside of the west side of watershed 101900050304, flows through the watershed, and then joins the St. Vrain Creek at the northeast tip of the watershed. There are is a lake in the Lower Left Hand Creek Watershed, Allens Lake, and a reservoir, Dodd Reservoir.<sup>2</sup>
# 
# **Data Description:**
# 
# The Watershed Boundary Dataset includes national hydrologic units. Hydrologic units are "an area of the landscape that drains to a portion of the stream network."<sup>3</sup>
# 
# **Citations:**
# 
# 1. “Boundary Descriptions and Names of Regions, Subregions, Accounting Units and Cataloging Units from the 1987 USGS Water-Supply Paper 2294.” USGS Water Resources: About USGS Water Resources, water.usgs.gov/GIS/huc_name.html. Accessed 2 Mar. 2025. 
# 
# 2. “How’s My Waterway?” EPA, Environmental Protection Agency, mywaterway.epa.gov/community/101900050304/overview. Accessed 2 Mar. 2025. 
# 
# 3. “Watershed Boundary Dataset.” USGS, National Hydrography, www.usgs.gov/national-hydrography/watershed-boundary-dataset. Accessed 2 Mar. 2025. 
# 
# 
# </span>

# %% [markdown]
# ## STEP 3: MULTISPECTRAL DATA
# 
# ### Search for data
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Log in to the <code>earthaccess</code> service using your Earthdata
# credentials:
# <code>python      earthaccess.login(persist=True)</code></li>
# <li>Modify the following sample code to search for granules of the
# HLSL30 product overlapping the watershed boundary from May to October
# 2023 (there should be 76 granules):
# <code>python      results = earthaccess.search_data(          short_name="...",          cloud_hosted=True,          bounding_box=tuple(gdf.total_bounds),          temporal=("...", "..."),      )</code></li>
# </ol></div></div>

# %% [markdown]
# <span style="color: maroon;">
# 
# 
# HLSL30 data description from the NASA worldview site:
# 
# "The Harmonized Landsat Sentinel-2 (HLS) project brings us 30 meter resolution true color surface reflectance imagery from the Operational Land Imager (OLI) instrument aboard the NASA/USGS Landsat 8 and 9 satellites, and the Multi-Spectral Instrument (MSI) aboard the European Space Agency (ESA) Sentinel-2A and Sentinel-2B satellites."
# 
# MSI data citation: 
# 
# Masek, J., Ju, J., Roger, J., Skakun, S., Vermote, E., Claverie, M., Dungan, J., Yin, Z., Freitag, B., Justice, C. (2021). <i>HLS Sentinel-2 Multi-spectral Instrument Surface Reflectance Daily Global 30m v2.0</i> [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed 2025-03-03 from https://doi.org/10.5067/HLS/HLSS30.002
# 
# OLI data citation: 
# 
# Masek, J., Ju, J., Roger, J., Skakun, S., Vermote, E., Claverie, M., Dungan, J., Yin, Z., Freitag, B., Justice, C. (2021). <i>HLS Operational Land Imager Surface Reflectance and TOA Brightness Daily Global 30m v2.0</i> [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed 2025-03-03 from https://doi.org/10.5067/HLS/HLSL30.002
# 
# </span>

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3a. Log in to EarthAccess and search for HLS tiles covering watershed 101900050304
# 
# </span>

# %%
# Log in to earthaccess
earthaccess.login(strategy="interactive", persist=True)

# %%
# Search for HLS tiles
wbd_hls_results = earthaccess.search_data(
    # dataset short name
    short_name='HLSL30',
    cloud_hosted=True,
    # the bounding box is the waterhsed boundary
    bounding_box=tuple(wbd_gdf.total_bounds),
    # temporal bounds from May - Oct 2023
    temporal=('2023-05-01', '2023-10-31')
)
wbd_hls_results


# %%
# add the wbd_hls_results go a geodataframe
wbd_hls_gdf = gpd.GeoDataFrame(wbd_hls_results)
wbd_hls_gdf

# %% [markdown]
# ### Compile information about each granule
# 
# I recommend building a GeoDataFrame, as this will allow you to plot the
# granules you are downloading and make sure they line up with your
# shapefile. You could also use a DataFrame, dictionary, or a custom
# object to store this information.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>For each search result:
# <ol type="1">
# <li>Get the following information (HINT: look at the [‘umm’] values for
# each search result):
# <ul>
# <li>granule id (UR)</li>
# <li>datetime</li>
# <li>geometry (HINT: check out the shapely.geometry.Polygon class to
# convert points to a Polygon)</li>
# </ul></li>
# <li>Open the granule files. I recomment opening one granule at a time,
# e.g. with (<code>earthaccess.open([result]</code>).</li>
# <li>For each file (band), get the following information:
# <ul>
# <li>file handler returned from <code>earthaccess.open()</code></li>
# <li>tile id</li>
# <li>band number</li>
# </ul></li>
# </ol></li>
# <li>Compile all the information you collected into a GeoDataFrame</li>
# </ol></div></div>

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3b. Explore HLS data before building granule metadata GeoDataFrame
# 
# </span>

# %%
# look at the 'umm' column for the first row
wbd_hls_gdf['umm'].loc[0]

# %%
# isolate the granule ID (UR) for the first row
row1 = list(wbd_hls_gdf['umm'].loc[0].values())
row1[1]

# %%
# isolate the datetime for the first row
row1[0]

# %%
# identify the keys for the first row
# the geometry, 'SpatialExtent' key is index 3
row1_keys = list(wbd_hls_gdf['umm'].loc[0].keys())
row1_keys

# %%
# isolate the SpatialExtent for the first row
# this is a dictionary
spatialextent = row1[3]
spatialextent

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3c. Use for loops to create a granule metadata GeoDataFrame. First explore the parts of the loop and then run one big loop.
# 
# </span>

# %%
# Break down the geometry part of the solution loop to identify the geometry of a granule
granule = wbd_hls_results[0]
# look at just the metadata, umm, of the first result
info_dict = granule['umm']
# create a list of all points, lat/lon, for the first result from the umm column of the first result
points = (
    info_dict
    # select the first thing in the GPolygons list
    ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
    ['Boundary']['Points'])
# create the geometry (Polygon) of the first result using Polygon from shapely
# loop through all the points in our points list and get the lat/long for each one to create the geometry
geometry = Polygon(
    [(point['Longitude'], point['Latitude']) for point in points])

print(points)
print(geometry)

# %%
# open test granule
opened_granule = earthaccess.open([granule])

# %%
# look at opened_granule
# 15 different .tif files, each for a different band
# All the bands for 1 granule have the same tile_id, geometry, and datetime
opened_granule

# %%
# What type of object is one element of the opened_granule?
type(opened_granule[0])

# %%
# get tile ids and band ids for the granule
# Compile a regular expression to search for metadata
# the r indicates this is a raw string and highlights all escaped characters (the dots), limits our need to escape special characters
uri_re = re.compile(
    r"HLS\.L30\.(?P<tile_id>T[0-9A-Z]+)"
    r"\.\d+T\d+\.v\d\.\d\."
    r"(?P<band_id>[A-Za-z0-9].+)\.tif"
)

#this yields a dictionary with the tile id and band id
uri_re.search(opened_granule[0].full_name).groupdict()

# %%
# Define a function to create a geodataframe with granule metadata
def create_granule_gdf(results):
    """
    Create a GeoDataFrame with HLS granule metadata: datetime, url, tile_id, band_id, and geometry.
    
    Parameters:
    results : list of HLS tiles from searching earthaccess
    
    Returns:
    file_gdf : GeoDataFrame w/ granule metadata
    """
    ## compile regular expression to search for metadata - tile_id and band_id
    url_re = re.compile(
        r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')

    # Loop through each granule
    # create an empty list for the metadata to be added to
    link_rows = []
    # For each granule in the HLS results...
    for granule in tqdm(results):
        # Identify granule metadata (umm column of HLS results)
        info_dict = granule['umm']
        # Identify granule datetime
        datetime = pd.to_datetime(
            info_dict
            ['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        # Identify the granule geometry by creating a list of all points,
        # lat/lon, for the first result from the umm column HLS result
        points = (
            info_dict
            # select the first thing in the GPolygons list
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
            ['Boundary']['Points'])
        # create geometry (Polygon) of the first result using Polygon from shapely
        # loop through all the points in our points list and get the lat/lon for each one to create the granule geometry
        geometry = Polygon(
            [(point['Longitude'], point['Latitude']) for point in points])
        
        # Get granule URL by opening each HLS result
        files = earthaccess.open([granule])

        # Build metadata DataFrame
        # for each URL in the files list...
        for file in files:
            # use a regular expression to search for the URL's metadata
            match = url_re.search(file.full_name)
            # if the regular expression search exists,
            if match is not None:
                # add a dictionary with the URL's metadata to the link_rows list
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group('tile_id')],
                            band=[match.group('band')],
                            url=[file],
                            geometry=[geometry]
                        ),
                        crs="EPSG:4326"
                    )
                )

    # Concatenate metadata DataFrame
    file_gdf = pd.concat(link_rows).reset_index(drop=True)
    return file_gdf

# %%
# use create_granule_gdf function to create a gdf with granule metadata
granule_gdf = create_granule_gdf(wbd_hls_results)

# %%
# check type of granule_gdf
type(granule_gdf)

# %%
# Check granule_gdf
# there are 675 rows - 45 granules, 15 bands per granule
granule_gdf

# %%
# how many tile_ids
granule_gdf.tile_id.unique()

# %%
# how many geometries
granule_gdf.geometry.unique()

# %% [markdown]
# ### Open, crop, and mask data
# 
# This will be the most resource-intensive step. I recommend caching your
# results using the `cached` decorator or by writing your own caching
# code. I also recommend testing this step with one or two dates before
# running the full computation.
# 
# This code should include at least one **function** including a
# numpy-style docstring. A good place to start would be a function for
# opening a single masked raster, applying the appropriate scale
# parameter, and cropping.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>For each granule:
# <ol type="1">
# <li><p>Open the Fmask band, crop, and compute a quality mask for the
# granule. You can use the following code as a starting point, making sure
# that <code>mask_bits</code> contains the quality bits you want to
# consider: ```python # Expand into a new dimension of binary bits bits =
# ( np.unpackbits(da.astype(np.uint8), bitorder=‘little’)
# .reshape(da.shape + (-1,)) )</p>
# <p># Select the required bits and check if any are flagged mask =
# np.prod(bits[…, mask_bits]==0, axis=-1) ```</p></li>
# <li><p>For each band that starts with ‘B’:</p>
# <ol type="1">
# <li>Open the band, crop, and apply the scale factor</li>
# <li>Name the DataArray after the band using the <code>.name</code>
# attribute</li>
# <li>Apply the cloud mask using the <code>.where()</code> method</li>
# <li>Store the DataArray in your data structure (e.g. adding a
# GeoDataFrame column with the DataArray in it. Note that you will need to
# remove the rows for unused bands)</li>
# </ol></li>
# </ol></li>
# </ol></div></div>

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3d. Create a DataFrame with reflectance information
# 
# </span>

# %%
@cached('llhc_reflectance_da_df', override=True)
def compute_reflectance_da(search_results, boundary_gdf):
    """
    Connect to files over VSI, crop, cloud mask, and wrangle
    
    Returns a single reflectance DataFrame 
    with all bands as columns and
    centroid coordinates and datetime as the index.
    
    Parameters
    ==========
    file_gdf (search_results) : pd.DataFrame
        File connection and metadata (datetime, tile_id, band, and url)
    boundary_gdf : gpd.GeoDataFrame
        Boundary use to crop the data
    """
    def open_dataarray(url, boundary_proj_gdf, scale=1, masked=True):
        # Open masked DataArray
        da = rxr.open_rasterio(url, masked=masked).squeeze() * scale
        
        # Reproject site boundary if needed
        if boundary_proj_gdf is None:
            ## reproject boundary site boundary_gdf to the same crs as the search_results crs
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)
            
        # Crop the DataArray with the opened urls to the reprojected site boundary
        cropped_da = da.rio.clip_box(*boundary_proj_gdf.total_bounds)
        return cropped_da
    
    def compute_quality_mask(da, mask_bits=[1, 2, 3]):
        """
        Mask out low quality data by bit
        
        Parameters
        ----------
        da : DataArray

        mask_bits : list
            bits to be masked
        
        Returns
        -------
        mask : numpy.ndarray
            quality mask
        """
        # Unpack bits into a new axis
        bits = (
            ## make each number in the dataarray binary
            np.unpackbits(
                ## change dataarray integer type to unsigned 8 bit integer type
                ## there will be 8 bits per number
                ## we do this because python will read the numbers as integers otherwise
                da.astype(np.uint8), bitorder='little'
            ).reshape(da.shape + (-1,))
        )

        # Select the required bits and check if any are flagged
        ## if the mask_bits equal 0, that means there isn't clouds, high aerosols, etc, so we only want to keep those bits.
        ## this will get rid of mask_bits that equal 1, aka where clouds, high aerosols, etc are.
        mask = np.prod(bits[..., mask_bits]==0,
                       ## add an axis so each number is converted to an array of 8 bits
                       ## each row has 1 number from original array in 8 bit format
                       axis=-1)
        return mask

    ## create a DataFrame with granule metadata
    file_gdf = create_granule_gdf(search_results)
    
    ## create empty list to gather bands that are cropped and have the cloud mask
    granule_da_rows= []
    ## set boundary_proj_gdf to None so that the site boundary_gdf gets reprojected to the same crs as the search_results crs
    boundary_proj_gdf = None

    # Loop through each image
    group_iter = file_gdf.groupby(['datetime', 'tile_id'])
    for (datetime, tile_id), granule_df in tqdm(group_iter):
        print(f'Processing granule {tile_id} {datetime}')
              
        # Open granule cloud cover
        cloud_mask_url = (
            # locate the Fmask band and corresponding url of each granule in the granule metadata DataFrame
            granule_df.loc[granule_df.band=='Fmask', 'url']
            ## remove the series wrapper of the column get the first value in the array
            .values[0])
        ## create a DataArray with the Fmask bands opened and cropped to the site bounds
        cloud_mask_cropped_da = open_dataarray(cloud_mask_url, boundary_proj_gdf, masked=False)

        # Compute cloud mask
        cloud_mask = compute_quality_mask(cloud_mask_cropped_da)

        # Loop through each spectral band
        da_list = []
        df_list = []
        for i, row in granule_df.iterrows():
            if row.band.startswith('B'):
                # Open, crop, and mask the band
                band_cropped = open_dataarray(
                    row.url, boundary_proj_gdf, scale=0.0001)
                band_cropped.name = row.band
                # Add the opened, cropped, and masked B band DataArray to the metadata DataFrame row
                ## also apply the cloud mask to the B band DataArray
                row['da'] = (band_cropped
                             ## for all the bits we don't want (clouds, cloud shadows, high aerosol, etc),
                             ## we'll have NaN values instead of the original spectral signature
                             .where(cloud_mask))
                ## add the metadata DataFrame row to the granule_da_rows list
                granule_da_rows.append(row
                                       ## first convert the row series to a DataFrame
                                       .to_frame()
                                       ## transpose the DataFrame
                                       .T)
    
    # Reassemble the metadata DataFrame
    return pd.concat(granule_da_rows)

# Use the above function to create a reflectance_da_df
llhc_reflectance_da_df = compute_reflectance_da(wbd_hls_results, wbd_gdf)

# %%
# Check the type of the reflectance_da_df
type(llhc_reflectance_da_df)

# %%
# Check the reflectance_da_df (this takes about a minute)
llhc_reflectance_da_df.head(3)

# %%
# look at dataarray for first row
llhc_reflectance_da_df.loc[0].da

# %% [markdown]
# ### Merge and Composite Data
# 
# You will notice for this watershed that: 1. The raster data for each
# date are spread across 4 granules 2. Any given image is incomplete
# because of clouds
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li><p>For each band:</p>
# <ol type="1">
# <li><p>For each date:</p>
# <ol type="1">
# <li>Merge all 4 granules</li>
# <li>Mask any negative values created by interpolating from the nodata
# value of -9999 (<code>rioxarray</code> should account for this, but
# doesn’t appear to when merging. If you leave these values in they will
# create problems down the line)</li>
# </ol></li>
# <li><p>Concatenate the merged DataArrays along a new date
# dimension</p></li>
# <li><p>Take the mean in the date dimension to create a composite image
# that fills cloud gaps</p></li>
# <li><p>Add the band as a dimension, and give the DataArray a
# name</p></li>
# </ol></li>
# <li><p>Concatenate along the band dimension</p></li>
# </ol></div></div>

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3e. Explore llhc_reflectance_da_df
# 
# </span>

# %%
## how many unique datetimes are there in the llhc_reflectance_da_df datetime column?
llhc_reflectance_da_df.datetime.unique()

# %%
## what is the name of the dataarray in the 0 index of the llhc_reflectance_da_df?
llhc_reflectance_da_df.loc[0].da.name

# %%
## how many unique tile_ids are there in the llhc_reflectance_da_df datetime column?
llhc_reflectance_da_df.tile_id.unique()

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 3f. Merge and composite granules of llhc_reflectance_da_df
# 
# </span>

# %%
@cached('llhc_reflectance_da')
def merge_and_composite_arrays(granule_da_df):
    # Merge and composite an image for each band
    df_list = []
    da_list = []
    for band, band_df in tqdm(granule_da_df.groupby('band')):
        merged_das = []
        for datetime, date_df in tqdm(band_df.groupby('datetime')):
            # Merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
            # Mask negative values
            merged_da = merged_da.where(merged_da>0)
            ## add merged dataarrays w/ negative values masked to the merged_das list
            merged_das.append(merged_da)
            
        # Composite images across dates
        ## concatenate merged dataarrays along the datetime dimension and
        ## reduce the dataarray's data by finding the median datetime
        composite_da = xr.concat(merged_das, dim='datetime').median('datetime')
        ## add a column to the composite_da called band
        composite_da['band'] = (
            ## [1:] will just look at elements from index 1 to the end (ignores index 0),
            ## int converts the value to an integer
            int(band[1:]))
        ## the name of the composite dataarray will be reflectance
        composite_da.name = 'reflectance'
        ## add the composite_da to the da_list
        da_list.append(composite_da)
        
    return xr.concat(da_list, dim='band')

llhc_reflectance_da = merge_and_composite_arrays(llhc_reflectance_da_df)
llhc_reflectance_da

# %% [markdown]
# llhc_reflectance_da hs 3 dimensions: band (10 spectral bands from HLS data), x, and y.
# 
# HLS UserGuide tells us the units and scale of each band (among other things). All bands except 10 and 11 have units of reflectance and a scale value of 0.0001. Bands 10 and 11 have units of degrees Celsius and scales of 0.01. Bands 10 and 11 are thermal bands. If we were to include bands 10 and 11 in our k-means clustering, we would need to change the scale to 0.0001. We'll drop bands 10 and 11 in this case since they're thermal bands.
# 
# Reflectance units are values between 0 and 1 that tells us what fraction of light is being reflected back to us.

# %% [markdown]
# ## STEP 4: K-MEANS
# 
# Cluster your data by spectral signature using the k-means algorithm.
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><ol type="1">
# <li>Convert your DataArray into a <strong>tidy</strong> DataFrame of
# reflectance values (hint: check out the <code>.to_dataframe()</code> and
# <code>.unstack()</code> methods)</li>
# <li>Filter out all rows with no data (all 0s or any N/A values)</li>
# <li>Fit a k-means model. You can experiment with the number of groups to
# find what works best.</li>
# </ol></div></div>

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 4a. Convert spectral DataArray to DataFrame
# 
# </span>

# %%
# Create a column for each of the bands
# yeilds a DataFrame with a column for each band and x and y columns (location of pixels)
model_df = (llhc_reflectance_da.to_dataframe()
            # unstack reflectance bands so they each get their own column
            .reflectance.unstack('band'))
model_df

# %%
# drop bands 10 and 11 since they're thermal bands and any rows w/ NaN values
model_df = model_df.drop(columns = [10, 11]).dropna()
model_df

# %%
# Check that the reflectance value ranges for each band
# make sense (should be from 0-1) by checking min and max values
min_values = model_df.min()
max_values = model_df.max()
print(min_values)
print(max_values)

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 4b. Execute k-means algorithm
# 
# </span>

# %%
# Initialize k means model
k_means = KMeans(n_clusters = 5)

# Fit model and predict
# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction = k_means.fit_predict(model_df.values)

# Add cluster labels to model_df as a column
# Add the predicted values back to the model DataFrame
model_df['clusters_5'] = prediction

# check new model_df
model_df

# %%
# Initialize k means model with 4 clusters
k_means_4 = KMeans(n_clusters = 4)

# Fit model and predict
# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction_4 = k_means_4.fit_predict(model_df.values)

# Add cluster labels to model_df as a column
# Add the predicted values back to the model DataFrame
model_df['clusters_4'] = prediction_4

# Initialize k means model with 3 clusters
k_means_3 = KMeans(n_clusters = 3)

# Fit model and predict
# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction_3 = k_means_3.fit_predict(model_df.values)

# Add cluster labels to model_df as a column
# Add the predicted values back to the model DataFrame
model_df['clusters_3'] = prediction_3

# check new model_df
model_df

# %% [markdown]
# ## STEP 5: PLOT
# 
# <link rel="stylesheet" type="text/css" href="./assets/styles.css"><div class="callout callout-style-default callout-titled callout-task"><div class="callout-header"><div class="callout-icon-container"><i class="callout-icon"></i></div><div class="callout-title-container flex-fill">Try It</div></div><div class="callout-body-container callout-body"><p>Create a plot that shows the k-means clusters next to an RGB image of
# the area. You may need to brighten your RGB image by multiplying it by
# 10. The code for reshaping and plotting the clusters is provided for you
# below, but you will have to create the RGB plot yourself!</p>
# <p>So, what is <code>.sortby(['x', 'y'])</code> doing for us? Try the
# code without it and find out.</p></div></div>

# %% [markdown]
# RGB Image (Red-Green-Blue Image): combine multispectral data for just the visual spectrum by stacking the red, green, and blue bands
# 
# To make RGB plot:
# 
# 1. Make DataArray w/ red (B04), green (B03), and blue (B02) bands

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 5a. Create RGB plot using llhc_reflectance_da
# 
# </span>

# %%
# make dataarray w/ bands to use for rgb plot
rgb = llhc_reflectance_da.sel(band=[4, 3, 2])

# hvplot has a rgb plot method. this will not give us a helpful plot b/c the values aren't what the rgb plot expects.
# the plot created by this is very dark b/c the numbers aren't in the right format.
# rgb plot method expects numbers to be pixel values, not integers.
(
    rgb.hvplot.rgb(
        # set the spatial components of how the pixels will be plotted
        y = 'y',
        x = 'x',
        bands = 'band',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None)
               )

# %%
# For the rgb plot method, we want image pixel values to range from 0-255
# Then convert to unsigned 8-bit integers (data type for images)
# don't include nan values in calculation
# This still yields a very dark plot
rgb_uint8 = ((
    # multiply all values in rgb dataarray by 255
    rgb * 255)
    # convert the type of the number to unsigned 8-bit integers
    .astype(np.uint8)
    # drop nan values, don't use them in the calculation
    .where(rgb!=np.nan))


# check plot of rgb_uint8
(
    rgb_uint8.hvplot.rgb(
        # set the spatial components of how the pixels will be plotted
        y = 'y',
        x = 'x',
        bands = 'band',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None)
               )

# %%
# We can increase the brightness of the image by multiplying all numbers by 10
# Even though we're changing the pixel values, it's just for display purposes so it's ok
# After plotting this we may get an error that the data values are too high.
rgb_uint8_bright = (rgb_uint8 * 10)

# check plot of rgb_uint8_bright
(
    rgb_uint8_bright.hvplot.rgb(
        # set the spatial components of how the pixels will be plotted
        y = 'y',
        x = 'x',
        bands = 'band',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None)
               )

# %%
# To fix the error, cap all pixel values at 255
# If value is 255 or less than 255, keep that value.
# if value is greater than 255, replace w/ 255
rgb_sat = (rgb_uint8_bright
           .where(rgb_uint8_bright < 255, 255))

# check plot of rgb_sat
(
    rgb_sat.hvplot.rgb(
        # set the spatial components of how the pixels will be plotted
        y = 'y',
        x = 'x',
        bands = 'band',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None)
               )

# %% [markdown]
# <span style="color: purple;">
# 
# #### STEP 5b. Create a plot of the RGB plot and k-means plot
# 
# </span>

# %%
# Plot the k-means clusters and rgb_sat plots next to each other
hv.Layout(
    rgb_sat.hvplot.rgb(
        # set the spatial components of how the pixels will be plotted
        y = 'y',
        x = 'x',
        bands = 'band',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None,
        title = 'RGB Plot of the Lower Left Hand Creek Watershed in Colorado',
        frame_width = 600)
    +
    (model_df.clusters_5
        # making it into an xarray
        .to_xarray()
        # making model_df.clusters into an xarray is causing important
        # spatial info to be lost. use sortby('x', 'y') to fix that.
        # w/o .sortby('x', 'y'), this plot is flipped compared to the rgb plot
        .sortby('x', 'y')
        .hvplot(
            x = 'x', y = 'y',
            # tell the plot that one unit vertically is the same as one unit horizontally
            data_aspect = 1,
            # drop axes
            xaxis = None, yaxis = None,
            # adjust coloring to a categorical color scheme
            cmap = 'Colorblind',
            title = 'Plot of K-means Clusters of Lower Left Hand Creek Watershed in Colorado (5 clusters)',
            frame_width = 600
        ))
    +
    (model_df.clusters_4
    # making it into an xarray
    .to_xarray()
    # making model_df.clusters into an xarray is causing important
    # spatial info to be lost. use sortby('x', 'y') to fix that.
    # w/o .sortby('x', 'y'), this plot is flipped compared to the rgb plot
    .sortby('x', 'y')
    .hvplot(
        x = 'x', y = 'y',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None,
        # adjust coloring to a categorical color scheme
        cmap = 'Colorblind',
        title = 'Plot of K-means Clusters of Lower Left Hand Creek Watershed in Colorado (4 clusters)',
        frame_width = 600
    ))
    +
    (model_df.clusters_3
    # making it into an xarray
    .to_xarray()
    # making model_df.clusters into an xarray is causing important
    # spatial info to be lost. use sortby('x', 'y') to fix that.
    # w/o .sortby('x', 'y'), this plot is flipped compared to the rgb plot
    .sortby('x', 'y')
    .hvplot(
        x = 'x', y = 'y',
        # tell the plot that one unit vertically is the same as one unit horizontally
        data_aspect = 1,
        # drop axes
        xaxis = None, yaxis = None,
        # adjust coloring to a categorical color scheme
        cmap = 'Colorblind',
        title = 'Plot of K-means Clusters of Lower Left Hand Creek Watershed in Colorado (3 clusters)',
        frame_width = 600
    ))
).cols(2)


# %% [markdown]
# **Plot Headline & Description:**
# The K-means algorithm does a decent job on classifying different land cover types in the Lower Left Hand Creek Watershed in Boulder County, Colorado. In the 5 cluster plot above, we can generally see that different clusters correlate with different types of land cover shown in the RGB plot. For example, there are buildings from neighborhoods identified with pink in the northeast corner of the plot 5 cluster plot and we can see those buildings in the northeast corner of the RGB plot. Also, in the 5 cluster plot, the black areas indicate bodies of water that are also seen in the RGB plot. Many of those water bodies are also shown in black in the 4 cluster plot, however they aren't identifiable in the 3 cluster plot. Also, the yellow areas in the 5 cluster plot seem to correlate pretty well with the brown and yellow areas in the RGB plot. 
# 
# If I were to do this analysis again, I would do a principal components analysis and/or find the silhouette scores for different amounts of clusters. In this analysis, I manually ran the k-means algorithm with 3 and 4 clusters and by comparing the plots, it looked like 5 clusters picked up on a decent amount of land cover variation in the watershed without picking up on too much. However, it'd be nice to have actual silhouette scores to decide with. 


