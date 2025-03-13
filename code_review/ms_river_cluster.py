# %% [markdown]
# # Land cover classification at the Mississppi Delta

# %% [markdown]
# ## Background Information
# ### This study will use the [harmonized Sentinal/Landsat multispectral dataset](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf) to look at patterns in vegetation data. The HUC region 8 watershed extends from Missouri to the Gulf of Mexico and the lower extent near New Orleans is the focus for this analysis. The EPA ecoregion designation is Mississippi Alluvial and SE Coastal Plains. According to a publication by the Louisiana Geological Survey, the area is comprised of "...a diversity of grasses, sedges, and rushes." However, there has been a tremendous amount of human engineering of this environment.
# 
# ## Sources
# * USDA (2012), Response to RFI for Long-Term Agro-ecosystem Research (LTAR) Network, available online at: https://www.ars.usda.gov/ARSUserFiles/np211/LMRBProposal.pdf.
# * John Snead, Richard P. McCulloh, and Paul V. Heinrich (2019) Landforms of the Louisiana Coastal Plain, Louisiana Geological Survey, available online at: https://www.lsu.edu/lgs/publications/products/landforms_book.pdf

# %%
# Import needed packages
# Standard libraries
import os # for operating system manipulation
import pathlib # for managing files and directories
import pickle # enables serialization of objects for saving
import re # pattern matching in strings
import warnings # control how warnings are displayed

# Geospatial Packages
import cartopy.crs as ccrs # coordinate reference system for maps
import earthaccess # simplifies access to nasa earthdata
import earthpy as et # functions that work with raster and vector data
import geopandas as gpd # read and manipulate geo dataframes
import geoviews as gv # geospatial visualization
import hvplot.pandas # plots pandas dataframes
import hvplot.xarray # plots data arrays
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np # numerican computation
import pandas as pd # tabular dataframes
import rioxarray as rxr # combine xarray with geospatial raster data
import rioxarray.merge as rxrmerge # merging multiple raster datasets
import seaborn as sns
from tqdm.notebook import tqdm # tracking processes with progress bar
import xarray as xr # gridded datasets
from shapely.geometry import Polygon # analyze geometric objects
from sklearn.cluster import KMeans # machine learning algorithm to group data
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

# Environmental Variables
os.environ["GDAL_HTTP_MAX_RETRY"] = "5" # geospatial data abstraction library for website query
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1" # combined lines try website 5 times with 1 second delay between

warnings.simplefilter('ignore') # suppress warnings


# %% [markdown]
# # Organize Information Downloads

# %%
# Setup data directories
data_dir = os.path.join(
    pathlib.Path.home(),
    'Documents',
    'eaclassprojects',
    'clusters',
    'land'
)
os.makedirs(data_dir, exist_ok=True)

data_dir

# %% [markdown]
# # Using a Decorator to Cache Results

# %%
# Define decorator
def cached(func_key, override=False):

    # save function results for retrieval and allow overwrite
    """
    A decorator to cache function results 
    
    Parameters
    ==========
    key: str
      File basename used to save pickled results
    override: bool
      When True, re-compute even if the results are already stored
    """
    # Wrap caching function
    def compute_and_cache_decorator(compute_function):
        """
        Wrap the caching function
        
        Parameters
        ==========
        compute_function: function
          The function to run and cache results
        """
        # detail usage of func_key
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
# # Study Site 

# %%
# Read watershed boundary dataset shapefile into a GeoDataFrame
@cached('wbd_08')
def read_wbd_file(wbd_filename, huc_level, cache_key):
    # Download and unzip
    wbd_url = (
        "https://prd-tnm.s3.amazonaws.com"
        "/StagedProducts/Hydrography/WBD/HU2/Shape/"
        f"{wbd_filename}.zip")
    wbd_dir = et.data.get_data(url=wbd_url)
                  
    # Read desired data
    wbd_path = os.path.join(wbd_dir, 'Shape', f'WBDHU{huc_level}.shp')
    wbd_gdf = gpd.read_file(wbd_path, engine='pyogrio')
    return wbd_gdf

huc_level = 12 # identify HUC level
wbd_gdf = read_wbd_file(
    "WBD_08_HU2_Shape", huc_level, cache_key=f'hu{huc_level}') # Call the function using cache

delta_gdf = (
    wbd_gdf[wbd_gdf[f'huc{huc_level}']
    .isin(['080902030506'])] # filter for specific river basin
    .dissolve() # create a single polygon
)

(
    delta_gdf.to_crs(ccrs.Mercator()) # Reproject to Mercator for web mapping
    .hvplot(
        alpha=.2, fill_color='white', # set styling
        tiles='EsriImagery', crs=ccrs.Mercator()) # add background map
    .opts(title='Mississippi River Watershed, Live Oak, LA', width=600, height=300) # set plot dimensions
)

# %%
wbd_gdf.head()

# %% [markdown]
# # Access Multispectral Data

# %%
# Log in to earthaccess
earthaccess.login(persist=True)
# Search for HLS tiles
results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(delta_gdf.total_bounds),
    temporal=("2023-05", "2023-09"),
)
# Confirm the contents
num_granules =len(results)
print(f"Number of granules found: {num_granules}")

print(results[0])

# %%
results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(delta_gdf.total_bounds),
    temporal=("2023-05", "2023-09"),
)

print(type(results))  # Prints the type of the 'results' object


# %% [markdown]
# # Compile Information about each Granule

# %%
# Extract and organize Earthaccess data
def get_earthaccess_links(results):
    url_re = re.compile(
        r'\.(?P<tile_id>\w+)\.\d+T\d+\.v\d\.\d\.(?P<band>[A-Za-z0-9]+)\.tif')

    # Loop through each granule
    link_rows = []  # Initialize a list to hold GeoDataFrames 
 
    url_dfs = [] # Initialize a list to hold individual file data
    for granule in tqdm(results):
        # Get granule metadata information
        info_dict = granule['umm']
        granule_id = info_dict['GranuleUR']
        datetime = pd.to_datetime(
            info_dict
            ['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        points = ( # Extract polygon coordinates
            info_dict
            ['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]
            ['Boundary']['Points'])
        geometry = Polygon( # Create a Shapely Polygon object
            [(point['Longitude'], point['Latitude']) for point in points])
        
        # Get data files associated with the granule
        files = earthaccess.open([granule])

        # Loop through each file within the granule
        for file in files:
            match = url_re.search(file.full_name)
            if match is not None:
                # Create a GeoDataFrame for the current file and append to list
                link_rows.append(
                    gpd.GeoDataFrame(
                        dict(
                            datetime=[datetime],
                            tile_id=[match.group('tile_id')], # Extract tile ID
                            band=[match.group('band')], # Extract band name
                            url=[file], # Store the file object
                            geometry=[geometry] # Store the geometry
                        ), 
                        crs="EPSG:4326" # set coordinate reference system
                    )
                )

    # Concatenate GeoDataFrames into a single gdf
    file_df = pd.concat(link_rows).reset_index(drop=True) # combine all rows
    return file_df

results = earthaccess.search_data(
    short_name="HLSL30",
    cloud_hosted=True,
    bounding_box=tuple(delta_gdf.total_bounds),
    temporal=("2023-05", "2023-09"),
)

if results:  # Check if results is not empty
    first_result = results[0]  # Get the first result
    file_df = get_earthaccess_links([first_result]) # Pass a list containing only the first result to the function
    print(file_df.head())

else:
    print("No results found.") 

# %% [markdown]
# # Open, Crop, and Mask Data using a Cached Decorator 

# %%
@cached('delta_reflectance_da_df') # Function output stored, same input gives cached output improving performance
def compute_reflectance_da(search_results, boundary_gdf):
    """
    Connect to files over VSI, crop, cloud mask, and wrangle
    
    Returns a single reflectance DataFrame 
    with all bands as columns and
    centroid coordinates and datetime as the index.
    
    Parameters
    ==========
    file_df : pd.DataFrame
        File connection and metadata (datetime, tile_id, band, and url)
    boundary_gdf : gpd.GeoDataFrame
        Boundary use to crop the data
    """
    def open_dataarray(url, boundary_proj_gdf, scale=1, masked=True):
        # Open masked DataArray
        da = rxr.open_rasterio(url, masked=masked).squeeze() * scale
        
        # Reproject boundary if needed
        if boundary_proj_gdf is None:
            boundary_proj_gdf = boundary_gdf.to_crs(da.rio.crs)
            
        # Crop
        cropped = da.rio.clip_box(*boundary_proj_gdf.total_bounds)
        return cropped
    
    def compute_quality_mask(da, mask_bits=[1, 2, 3]):
        """Mask out low quality data by bit"""
        # Unpack bits into a new axis
        bits = (
            np.unpackbits(
                da.astype(np.uint8), bitorder='little'
            ).reshape(da.shape + (-1,))
        )

        # Select the required bits and check if any are flagged
        mask = np.prod(bits[..., mask_bits]==0, axis=-1)
        return mask

    file_df = get_earthaccess_links(search_results)
    
    granule_da_rows= []
    boundary_proj_gdf = None

    # Loop through each image
    group_iter = file_df.groupby(['datetime', 'tile_id'])
    for (datetime, tile_id), granule_df in tqdm(group_iter):
        print(f'Processing granule {tile_id} {datetime}')
              
        # Open granule cloud cover
        cloud_mask_url = (
            granule_df.loc[granule_df.band=='Fmask', 'url']
            .values[0])
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
                # Add the DataArray to the metadata DataFrame row
                row['da'] = band_cropped.where(cloud_mask)
                granule_da_rows.append(row.to_frame().T)
    
    # Reassemble the metadata DataFrame
    return pd.concat(granule_da_rows)

reflectance_da_df = compute_reflectance_da(results, delta_gdf)

reflectance_da_df.head()

# %% [markdown]
# # Merge and Composite Data

# %%
@cached('delta_reflectance_da') # Function output stored, same input gives cached output improving performance
def merge_and_composite_arrays(granule_da_df):
    # Merge and composite and image for each band
    df_list = [] # List to store DataFrames
    da_list = [] # List to store the composited DataArrays for each band
    for band, band_df in tqdm(granule_da_df.groupby('band')): # Iterate through each band
        merged_das = [] # List to store merged DAs for each date within the current band

        for datetime, date_df in tqdm(band_df.groupby('datetime')): # Iterate through each date within the current band
            # Merge granules for each date
            merged_da = rxrmerge.merge_arrays(list(date_df.da))
            # Mask negative values
            merged_da = merged_da.where(merged_da>0)
            # Add the merged DA for the current date to the list
            merged_das.append(merged_da)
            
        # Composite images across dates with median reflectance value for each pixel reducing the effect of cloud cover.
        composite_da = xr.concat(merged_das, dim='datetime').median('datetime')
        composite_da['band'] = int(band[1:]) # Converts band name to integer 
        composite_da.name = 'reflectance' # Sets the name of the DataArray
        da_list.append(composite_da) # Add the composite DA for the current band to the list

    # Concatenate the composite DataArrays for all bands into a single DataArray   
    return xr.concat(da_list, dim='band') 

reflectance_da = merge_and_composite_arrays(reflectance_da_df) # Call the function to process the reflectance data
reflectance_da[0] # Display the first resulting reflectance DataArray

# %% [markdown]
# # Analyze using K-MEANS

# %%
### how many different types of vegetation are there?
# Convert spectral DataArray to DataFrame
model_df = reflectance_da.to_dataframe().reflectance.unstack('band')
model_df = model_df.drop(columns=[10, 11]).dropna()

model_df.head()

# %%
# Running the fit and predict functions at the same time.
# We can do this since we don't have target data.
prediction = KMeans(n_clusters=6).fit_predict(model_df.values)

# Add the predicted values back to the model DataFrame
model_df['clusters'] = prediction
model_df

# %% [markdown]
# # Plot Reflectance

# %%
# Select bands from reflectance data array
rgb = reflectance_da.sel(band=[4, 3, 2]) # band 4=red, 3=green and 2=blue
rgb_uint8 = (rgb * 255).astype(np.uint8).where(rgb!=np.nan) # convert to integers (0-255)
rgb_bright = rgb_uint8 * 10 # increase brightness 
rgb_sat = rgb_bright.where(rgb_bright < 255, 255) # no pixel exceeds 255

# Create a composite RGB image plot in square
(
    rgb_sat.hvplot.rgb( 
        x='x', y='y', bands='band',
        data_aspect=1,
        xaxis=None, yaxis=None)
    + # Overlay cluster data
    model_df.clusters.to_xarray().sortby(['x', 'y']).hvplot(
        cmap="accent", aspect='equal', xaxis=None, yaxis=None) 
)

# %% [markdown]
# # Landcover Analysis using K-Means Clusters from Sentinel/Landsat Multispectral Data

# %% [markdown]
# ## Land Cover Interpretation based on Spectral Data
# ### According to America’s Watershed Initiative, the wetlands in the lower Mississippi region being studied have been depleted annually since the 1930s and excess nutrient discharges have created "dead zones" or areas of low oxygen where life struggles to exist. [1] The K-means cluster analysis in the above images shows 6 clusters of land forms spread out over the region. Some, like clusters, 4 and 0 have areas where they are concentrated, but most landform clusters are highly dispersed. In a study by Roy et al, their " ...data suggests that single transition land loss is caused by wave-edge erosion, whereas multiple transition land loss is caused by subsidence." [2] Given that most of the landform clusters are fragmented, I would expect clusters 4 and 0 to be mostly aquatic and the remainder to be composed of grasses, sedges and rushes.
# 
# ### Source
# 1. America’s Watershed Initiative Report Card for the Mississippi River. Dec. 2015. Available online at: https://americaswatershed.org/wp-content/uploads/2015/12/Mississippi-River-Report-Card-Methods-v10.1.pdf
# 2. Samapriya Roy et al. 2020. Spatial and temporal patterns of land loss in the Lower Mississippi River Delta from 1983 to 2016. Remote Sensing of Environment 250 (2020) 112046. Available online at: https://www.sciencedirect.com/science/article/abs/pii/S0034425720304168


