import os
from glob import glob
from math import floor, ceil
import pandas as pd
import numpy as np
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import xarray as xr
import xrspatial
import matplotlib.pyplot as plt
import earthaccess

### Data Wrangling ###

def build_da(urls, bounds):
    """
    Build a DataArray from a list of urls.
    
    Args:
    urls (list): Input list of URLs.
    bounds (tuple): Site boundaries.

    Returns:
    xarray.DataArray: A merged DataArray.
    """
    
    all_das = []

    # Add buffer to bounds for plotting
    buffer = .025
    xmin, ymin, xmax, ymax = bounds
    bounds_buffer = (xmin-buffer, ymin-buffer, xmax+buffer, ymax+buffer)

    for url in urls:
        # Open data granule, mask missing data, scale data, 
        # and remove dimensions of length 1
        tile_da = rxr.open_rasterio(
                url,
                # For the fill/missing value
                mask_and_scale=True
            ).squeeze()
        # Unpack the bounds and crop tile
        cropped_da = tile_da.rio.clip_box(*bounds_buffer)
        all_das.append(cropped_da)

    merged = merge_arrays(all_das)
    return merged

def export_raster(da, raster_path, data_dir):
    """
    Export raster DataArray to a raster file.
    
    Args:
    raster (xarray.DataArray): Input raster layer.
    raster_path (str): Output raster directory.
    data_dir (str): Path of data directory.

    Returns: None
    """
    
    output_file = os.path.join(data_dir, os.path.basename(raster_path))
    da.rio.to_raster(output_file)

def harmonize_raster_layers(reference_raster, input_rasters, data_dir):
    """
    Harmonize raster layers to ensure consistent spatial resolution 
    and projection.

    Args:
    reference_raster (str): Path of raster to reference.
    input_rasters (list): List of rasters to harmonize.
    data_dir (str): Path of data directory.

    Returns:
    list: A list of harmonized rasters.
    """

    harmonized_files = []

    harmonized_files.append(reference_raster)
    # Load the reference raster
    ref_raster = rxr.open_rasterio(reference_raster, masked=True)
    # Use projection EPSG:3857 (WGS 84 with x/y coords)
    ref_raster = ref_raster.rio.write_crs(3857)

    for raster_path in input_rasters:
        # Load the input raster
        input_raster = rxr.open_rasterio(raster_path, masked=True)
        input_raster = input_raster.rio.write_crs(3857)

        # Reproject and align the input raster to match the reference raster

        # Only 2D/3D arrays with dimensions x/y are currently supported
        # by reproject_match()
        harmonized_raster = input_raster.rio.reproject_match(ref_raster)

        # Save the harmonized raster to the output directory
        output_file = os.path.join(data_dir, os.path.basename(raster_path))
        harmonized_raster.rio.to_raster(output_file)
        harmonized_files.append(output_file)

    return harmonized_files

### Calculations ###

def convert_longitude(longitude):
    """
    Convert longitude values from a range of 0 to 360 to -180 to 180.
    
    Args:
    longitude (float): Input longitude value.

    Returns:
    float: A value in the specified range.
    """
    
    return (longitude - 360) if longitude > 180 else longitude

def convert_temperature(temp):
    """
    Convert temperature from Kelvin to Fahrenheit.
    
    Args:
    temp (float): Input temperature value.

    Returns:
    float: A value in the Fahrenheit temperature scale.
    """

    return temp  * 1.8 - 459.67

def calculate_aspect(elev_da):
    """
    Create aspect DataArray from site elevation.
    
    Args:
    elev_da (xarray.DataArray): Input raster layer.

    Returns:
    xarray.DataArray: A raster of site aspect. 
    """

    # Calculate aspect (degrees)
    aspect_da = xrspatial.aspect(elev_da)
    aspect_da = aspect_da.where(aspect_da >= 0)

    return aspect_da

def calculate_suitability_score(raster, optimal_value, tolerance_range):
    """ 
    Calculate a fuzzy suitability score (0–1) for each raster cell based on 
    proximity to the optimal value.

    Args:
    raster (xarray.DataArray): Input raster layer. 
    optimal_value (float): Optimal value for the variable.
    tolerance_range (float): Suitable values range. 

    Returns:
    xarray.DataArray: A raster of suitability scores (0-1).
    """

    # Calculate using a fuzzy Gaussian function to assign scores 
    # between 0 and 1
    suitability = np.exp(
                    -((raster - optimal_value) ** 2) 
                    / (2 * tolerance_range ** 2)
                )

    # Suitability scores (0–1) 
    return suitability

### Visualization ###

def plot_site(site_da, site_gdf, plots_dir, site_fig_name, plot_title, 
              bar_label, plot_cmap, boundary_clr, tif_file=False):
    """
    Create custom site plot.
    
    Args:
    site_da (xarray.DataArray): Input site raster.
    site_gdf (geopandas.GeoDataFrame): Input site GeoDataFrame.
    plots_dir (str): Path of plots directory.
    site_fig_name (str): Site figure name.
    plot_title (str): Plot title. 
    bar_label (str): Plot bar variable name.
    plot_cmap (str): Plot colormap name.
    boundary_clr (str): Plot site boundary color.
    tif_file (bool): Indicates a site file.

    Returns:
    matplotlib.pyplot.plot: A plot of site values.
    """
    
    fig = plt.figure(figsize=(8, 6)) 
    ax = plt.axes()

    if tif_file:
        site_da = rxr.open_rasterio(site_da, masked=True)

    # Plot DataArray values
    site_plot = site_da.plot(
                            cmap=plot_cmap, 
                            cbar_kwargs={'label': bar_label}
                        )

    # Plot site boundary
    site_gdf.boundary.plot(ax=plt.gca(), color=boundary_clr)

    plt.title(f'{plot_title}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    fig.savefig(f"{plots_dir}/{site_fig_name}.png") 

    return site_plot

### Retrieve Data ###

def create_polaris_urls(soil_prop, stat, soil_depth, gdf_bounds):
    """
    Create POLARIS dataset URLs using site bounds.

    Args:
    soil_prop (str): Soil property.
    stat (str): Summary statistic. 
    soil_depth (str): Soil depth (cm).
    gdf_bounds (numpy.ndarray): Array of site boundaries.

    Returns:
    list: A list of POLARIS datset URLs. 
    """

    # Get latitude and longitude bounds from site
    min_lon, min_lat, max_lon, max_lat = gdf_bounds

    site_min_lon = floor(min_lon) 
    site_min_lat = floor(min_lat)  
    site_max_lon = ceil(max_lon)  
    site_max_lat = ceil(max_lat)

    all_soil_urls = []

    for lon in range(site_min_lon, site_max_lon): 
        for lat in range(site_min_lat, site_max_lat):
            current_max_lon = lon + 1
            current_max_lat = lat + 1

            soil_template = (
                "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/"
                "{soil_prop}/"
                "{stat}/"
                "{soil_depth}/"
                "lat{min_lat}{max_lat}_lon{min_lon}{max_lon}.tif"
            )

            soil_url = soil_template.format(
                soil_prop=soil_prop, stat=stat, soil_depth=soil_depth,
                min_lat=lat, max_lat=current_max_lat, 
                min_lon=lon, max_lon=current_max_lon
            )

            all_soil_urls.append(soil_url)

    return all_soil_urls

def download_polaris(site_name, site_gdf, soil_prop, stat, soil_depth, 
                     plot_path, plot_title, data_dir, plots_dir):
    """
    Retrieve POLARIS site data, build DataArray, plot site, and export raster.

    Args:
    site_name (str): Name of site.
    site_gdf (geopandas.GeoDataFrame): Site GeoDataFrame.
    soil_prop (str): Soil property.
    stat (str): Summary statistic. 
    soil_depth (str): Soil depth (cm).
    plot_path (str): Path of topographic plot.
    plot_title (str): Title of topographic plot.
    data_dir (str): Path of data directory.
    plots_dir (str): Path of plots directory.

    Returns:
    xarray.DataArray: A soil DataArray for a given location. 
    """
    
    # Collect site urls
    site_polaris_urls = create_polaris_urls(
                        soil_prop, stat, soil_depth, 
                        site_gdf.total_bounds
                    )
    
    # Gather site data into a single DataArray
    site_soil_da = build_da(site_polaris_urls, tuple(site_gdf.total_bounds))
    
    # Export soil data to raster
    export_raster(site_soil_da, f"{site_name}_soil_{soil_prop}.tif", data_dir)

    # Create site plot
    plot_site(
        site_soil_da, site_gdf, plots_dir,
        f'{plot_path}-Soil', f'{plot_title} Soil',
        'pH', 'viridis', 'lightblue'
    )

    return site_soil_da

def select_dem(bounds, site_gdf, download_dir):
    """
    Create elevation DataArray from NASA Shuttle Radar Topography Mission data.

    Args:
    bounds (tuple): Input site boundaries.
    site_gdf (geopandas.GeoDataFrame): Land unit GeoDataFrame.
    download_dir (str): Path of download directory.

    Returns:
    xarray.DataArray: A site elevation raster.
    """

    # Returns data granules for given bounds
    strm_granules = earthaccess.search_data(
        # SRTMGL1: NASA Shuttle Radar Topography Mission 
        # Global 1 arc second V003
        short_name="SRTMGL1",
        bounding_box=bounds
    )

    # Download data granules
    earthaccess.download(strm_granules, download_dir)

    # Set SRTM data dir. hgt = height 
    strm_pattern = os.path.join(download_dir, '*.hgt.zip')

    # Build merged elevation DataArray
    strm_da = build_da(glob(strm_pattern), tuple(site_gdf.total_bounds))

    return strm_da

def download_topography(site_name, site_gdf, plot_path, plot_title, 
                        elevation_dir, data_dir, plots_dir):
    """
    Retrieve topographic data, build DataArray, plot site, and export raster.

    Args:
    site_name (str): Name of site.
    site_gdf (geopandas.GeoDataFrame): Site GeoDataFrame.
    plot_path (str): Path of topographic plot.
    plot_title (str): Title of topographic plot.
    elevation_dir (str): Path of site elevation directory.
    data_dir (str): Path of data directory.
    plots_dir (str): Path of plots directory.

    Returns:
    xarray.DataArray: An elevation DataArray for a given location. 
    """
    
    # Produce Digital Elevation Model DataArray

    elev_da = select_dem(tuple(site_gdf.total_bounds), site_gdf, 
                        elevation_dir)
    export_raster(elev_da, f"{site_name}_elevation.tif", data_dir)
    plot_site(
        elev_da, site_gdf, plots_dir, f'{plot_path}-Elevation', 
        f'{plot_title} Elevation', 'Meters', 'terrain', 'black',
    )

    # Calculate aspect from elevation 

    aspect_da = calculate_aspect(elev_da)
    export_raster(aspect_da, f"{site_name}_aspect.tif", data_dir)
    plot_site(
        aspect_da, site_gdf, plots_dir, f'{plot_path}-Aspect', 
        f'{plot_title} Aspect', 'Degrees', 'terrain', 'black'
    )

    return elev_da

def get_projected_climate(site_name, site_gdf,
                          emissions_scenario, gcm, time_slices):
    """
    Create DataFrame of projected site climate for a given time period.
    
    Args:
    site_name (str): Site name.
    site_gdf (dict): Site GeoDataFrame.
    emissions_scenario (str): Climate scenario. 
    gcm (str): Global Climate Model. 
    time_slices (list): List of years indicating a time slice.

    Returns:
    pandas.DataFrame: A DataFrame of projected average max temperatures.
    """

    maca_das = []

    for start_year in time_slices:
        end_year = start_year + 4

        # Multivariate Adaptive Constructed Analogs (MACA)
        MACA_URL = (
            'http://thredds.northwestknowledge.net:8080/thredds/fileServer/' 
            f'MACAV2/{gcm}/macav2metdata_tasmax_{gcm}_r1i1p1_'
            f'{emissions_scenario}_{start_year}_{end_year}_CONUS_monthly.nc'
        )

        maca_da = xr.open_dataset(MACA_URL, engine='h5netcdf').squeeze()

        bounds = site_gdf.to_crs(maca_da.rio.crs).total_bounds

        # update coordinate range
        maca_da = maca_da.assign_coords(
            lon=("lon", [convert_longitude(l) for l in maca_da.lon.values])
        )

        maca_da = maca_da.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        maca_da = maca_da.rio.clip_box(*bounds)

        maca_das.append(dict(
                            site_name=site_name,
                            gcm=gcm,
                            start_year=start_year,
                            end_year=end_year,
                            da=maca_da))
    
    return pd.DataFrame(maca_das)

def download_climate(site_name, site_gdf, emissions_scenario, 
                     climate_models, time_slices, raster_path, data_dir):
    """
    Retrieve projected site climate for a given time period, 
    build composite DataArray, and export raster.
    
    Args:
    site_name (str): Site name.
    site_gdf (dict): Site GeoDataFrame.
    emissions_scenario (str): Climate scenario. 
    climate_models (list): Climate model names.
    time_slices (list): List of years indicating a time slice.
    raster_path (str): Path of site climate raster.
    data_dir (str): Path of data directory.

    Returns: None
    """

    for gcm in climate_models:
        print(f'Downloading climate data for {gcm}')

        # Retrieve MACA climate data
        site_proj_temp = get_projected_climate(
                                        site_name, site_gdf, 
                                        emissions_scenario, gcm, 
                                        time_slices
                                    )
        
        # Convert temperature to fahrenheit 
        projected_climates_df = site_proj_temp.assign(
                                    fahrenheit=lambda x: x.da.map(
                                        convert_temperature))

        # Create composite site projected climate
        site_clim_composite_ds = xr.concat(
            projected_climates_df.fahrenheit, dim='time').mean('time')

        site_clim_composite_da = site_clim_composite_ds.to_dataarray(
                                    dim='air_temperature', 
                                    name='air_temperature')

        export_raster(site_clim_composite_da, 
                    f"{raster_path}_{gcm}_max_temp.tif", data_dir)

### Create Suitability Model ###

def build_habitat_suitability_model(site_name, time_period, gcm,
                                    optimal_values, tolerance_ranges, 
                                    data_dir, raster_name):
    """
    Build a habitat suitability model by combining fuzzy suitability scores 
    for each variable. 

    Args:
    site_name (str): Name of site.
    time_period (str): Name of time period. 
    gcm (str): Global Climate Model.    
    optimal_values (list): List of optimal values for variables.
    tolerance_ranges (list): List of tolerance values for variables.
    data_dir (str): Path of data directory.
    raster_name (str): The name of model raster.

    Returns:
    str: The path of the suitability raster.
    """
    reference_raster = f"{data_dir}/{site_name}_elevation.tif" 

    input_rasters = [
        f"{data_dir}/{site_name}_{time_period}_{gcm}_max_temp.tif",
        f"{data_dir}/{site_name}_aspect.tif",
        f"{data_dir}/{site_name}_soil_ph.tif"
    ]

    harmonized_rasters = harmonize_raster_layers(reference_raster, 
                                                 input_rasters, data_dir)

    # Load and calculate suitability scores for each raster
    suitability_layers = []
    suit_zip = zip(harmonized_rasters, optimal_values, tolerance_ranges)
    for raster_path, optimal_value, tolerance_range in suit_zip:
        raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
        suitability_layer = calculate_suitability_score(
                                raster, optimal_value, tolerance_range
                            )
        suitability_layers.append(suitability_layer)

    # Combine suitability scores by multiplying across all layers
    combined_suitability = suitability_layers[0]
    for layer in suitability_layers[1:]:
        combined_suitability *= layer

    # Save the combined suitability raster
    output_file = os.path.join(data_dir, f"{raster_name}.tif")
    combined_suitability.rio.to_raster(output_file)
    print(f"Combined suitability raster saved to: {raster_name}.tif")

    # Path to the final combined suitability raster
    return output_file
