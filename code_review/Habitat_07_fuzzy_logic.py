# %%
# Imports
import os
import rioxarray as rxr
import xarray as xr
import numpy as np

# %%
%store -r c_soil_url_list

# %%
def calculate_suitability_score(raster, optimal_value, tolerance_range):
    """
    Calculate a fuzzy suitability score (0–1) for each raster cell based on proximity to the optimal value.

    Args:
        raster (xarray.DataArray): Input raster layer.
        optimal_value (float): The optimal value for the variable.
        tolerance_range (float): The range within which values are considered suitable.

    Returns:
        xarray.DataArray: A raster of suitability scores (0–1).
    """
    # Calculate suitability scores using a fuzzy Gaussian function
    suitability = np.exp(-((raster - optimal_value) ** 2) / (2 * tolerance_range ** 2))
    return suitability

raster = c_soil_url_list
optimal_value = 6.5
tolerance_range = 1

# %%
def build_habitat_suitability_model(
        input_rasters, optimal_values, tolerance_ranges, output_dir, threshold=None):
    """
    Build a habitat suitability model by combining fuzzy suitability scores for each variable.

    Args:
        input_rasters (list): List of paths to input raster files representing environmental variables.
        optimal_values (list): List of optimal values for each variable.
        tolerance_ranges (list): List of tolerance ranges for each variable.
        output_dir (str): Directory to save the combined suitability raster.
        threshold (float, optional): Threshold for highlighting highly suitable areas (default is None).

    Returns:
        str: Path to the final combined suitability raster.
    """
    input_rasters = [c_soil_url_list]
    optimal_values = [
        (ph >= 6.5 and ph <= 7), 
        (pr >= 32.5 and pr <= 37.5), 
        (temp >= 75 and temp <= 80), 
        (soil_type in ['silt', 'clay', 'loam'])
        ]
    tolerance_ranges = [
        (pH>=5.5 and pH<=8.0), 
        (pr>=10 and pr<=60), 
        (temp>=5 and temp<=95)
        ]
    output_dir = os.path.join('earth-analytics', 'data', 
                            'habitat_suitability', 'suitability_raster')
    threshold = 0.85

# %%
# This code cell is trying to fix an error "NameError: name input_rasters is not defined". I took the object definitions out of the function in case this is what's causing the error.
build_habitat_suitability_model

input_rasters = [c_soil_url_list]
optimal_values = [
    (ph >= 6.5 and ph <= 7), 
    (pr >= 32.5 and pr <= 37.5), 
    (temp >= 75 and temp <= 80), 
    (soil_type in ['sand', 'clay', 'loam'])
    ]
tolerance_ranges = [
    (ph>=5.5 and ph<=8.0), 
    (pr>=10 and pr<=60), 
    (temp>=5 and temp<=95)
    ]
output_dir = os.path.join('earth-analytics', 'data', 
                        'habitat_suitability', 'suitability_raster')
threshold = 0.85

# %%
# os.makedirs(output_dir, exist_ok=True)

# Load and calculate suitability scores for each raster
suitability_layers = []
for raster_path, optimal_value, tolerance_range in zip(input_rasters, optimal_values, tolerance_ranges):
    raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
    suitability_layer = calculate_suitability_score(raster, optimal_value, tolerance_range)
    suitability_layers.append(suitability_layer)

# Combine suitability scores by multiplying across all layers
combined_suitability = suitability_layers[0]
for layer in suitability_layers[1:]:
    combined_suitability *= layer

# Apply a threshold if provided
if threshold is not None:
    combined_suitability = xr.where(combined_suitability >= threshold, combined_suitability, 0)

# Save the combined suitability raster
output_file = os.path.join(hab_suit, "combined_suitability.tif")
combined_suitability.rio.to_raster(output_file)
print(f"Combined suitability raster saved to: {output_file}")

return output_file

# %%
# Example usage
if __name__ == "__main__":
    # Paths to input raster files (e.g., temperature, precipitation, soil pH)
    input_rasters = [
        "path_to_temperature_raster/temperature.tif",
        "path_to_precipitation_raster/precipitation.tif",
        "path_to_soil_ph_raster/soil_ph.tif"
    ]

    # Optimal values for Andropogon gerardii for each variable
    optimal_values = [25.0, 35.0, 6.5]  # Example: temperature in °C, precipitation in mm, soil pH

    # Tolerance ranges for each variable
    tolerance_ranges = [10.0, 25.0, 0.75]  # Example: acceptable deviation for each variable

    # Output directory to save the combined suitability raster
    output_dir = "path_to_output_directory"

    # Optional threshold to highlight highly suitable areas (e.g., 0.8)
    threshold = 0.85

    # Build the habitat suitability model
    combined_suitability_file = build_habitat_suitability_model(
        input_rasters, optimal_values, tolerance_ranges, output_dir, threshold
    )

    print("Habitat suitability model created:", combined_suitability_file)


