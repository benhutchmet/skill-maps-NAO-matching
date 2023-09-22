# Functions used for the lagging and variance adjustment and NAO matching techniques (Doug Smith's method)
# With help from Hazel Thornton's code

"""
functions.py
============

Usage:
    python functions.py <model> <variable> <region> <season> <forecast_range> 
                        <start_year> <end_year> <observations_path> <level>

    e.g. python functions.py HadGEM3-GC31-MM psl global DJFM 2-9 1960 2014 /home/users/benhutch/ERA5/*.nc None

Arguments:
    model : str
        Model name.
    variable : str
        Variable name.
    region : str
        Region name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    start_year : str
        Start year.
    end_year : str
        End year.
    observations_path : str
        Path to the observations.
    level : str
        Level name. The default is None.

"""

# Local imports
import os
import glob
import re
import sys
import argparse

# Third-party imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib
import cartopy.crs as ccrs
import iris
import iris.coord_categorisation as coord_cat
import iris.plot as iplt
from iris.util import unify_time_units, equalise_attributes
import scipy
import scipy.stats as stats
import pdb
import datetime
import iris.quickplot as qplt
from datetime import datetime

# Import CDO
from cdo import *
# cdo = Cdo()


# Import the dictionaries
import dictionaries as dic

# Define a function which regrids and selects the region according to the specified gridspec
# The same is done for the model data
def regrid_and_select_region(variable, region, observations_path, level=None):
    """
    Regrids and selects the region according to the specified gridspec.
    
    Parameters
    ----------
    variable : str
        Variable name.
    region : str
        Region name.
    observations_path : str
        Path to the observations.
    level : str, optional
        Level name. The default is None.
        
    Returns
    -------
    regrid_obs_path : str
        Path to the regridded observations.
    """

    # Check whether the gridspec path exists for the specified region
    gridspec_path = f"/home/users/benhutch/gridspec/gridspec-{region}.txt"

    if not os.path.exists(gridspec_path):
        print('The gridspec path does not exist for the specified region: ', region)
        sys.exit()

    # Form the wind speed variables list
    wind_speed_variables = ['ua', 'va', 'var131', 'var132']

    if variable in wind_speed_variables and level is None:
        print('The level must be specified for the wind speed variables')
        sys.exit()

    # Form the regrid sel region path accordingly
    if variable in wind_speed_variables:
        regrid_obs_path = f"/home/users/benhutch/ERA5/{region}_regrid_sel_region_{variable}_{level}.nc"
    else:
        regrid_obs_path = f"/home/users/benhutch/ERA5/{region}_regrid_sel_region.nc"
    
    # Check whether the regrid sel region path exists
    if not os.path.exists(regrid_obs_path):
        print('The regrid sel region path does not exist')
        print('Regridding and selecting the region')

        # Regrid and select the region using CDO
        cdo.remapbil(gridspec_path, input=observations_path, output=regrid_obs_path)

    return regrid_obs_path

# Define a function to load the obs data into Iris cubes
def load_obs(variable, regrid_obs_path):
    """
    Loads the obs data into Iris cubes.
    
    Parameters
    ----------
    variable : str
        Variable name.
    regrid_obs_path : str
        Path to the regridded observations.
        
    Returns
    -------
    obs : iris.cube.Cube
        Observations.
    """

    # Verify that the regrid obs path exists
    if not os.path.exists(regrid_obs_path):
        print('The regrid obs path does not exist')
        sys.exit()

    if variable not in dic.var_name_map:
        print('The variable is not in the dictionary')
        sys.exit()

    # Extract the variable name from the dictionary
    obs_variable = dic.var_name_map[variable]

    if obs_variable in dic.obs_ws_var_names:
        print('The obs variable is a wind speed variable')
        
        # Load the regrid obs file into an Iris cube
        obs = iris.load_cube(regrid_obs_path, obs_variable)
    else:
        # Load using xarray
        obs = xr.open_mfdataset(regrid_obs_path, combine='by_coords', parallel=True)[obs_variable]

        # Combine the two expver variables
        obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

        # Convert to an Iris cube
        obs = obs.to_iris()

        # if the type of obs is not a cube, then exit
        if type(obs) != iris.cube.Cube:
            print('The type of obs is not a cube')
            sys.exit()

    return obs

def process_data(datasets_by_model, variable):
    """Process the data.
    
    This function takes a dictionary of datasets grouped by models
    and processes the data for each dataset.
    
    Args:
        datasets_by_model: A dictionary of datasets grouped by models.
        variable: The variable to load, extracted from the command line.
        
    Returns:
        variable_data_by_model: the data extracted for the variable for each model.
        model_time_by_model: the model time extracted from each model for each model.
    """
    
    #print(f"Dataset type: {type(datasets_by_model)}")

    def process_model_dataset(dataset, variable, attributes):
        """Process a single dataset.
        
        This function takes a single dataset and processes the data.
        
        Args:
            dataset: A single dataset.
            variable: The variable to load, extracted from the command line.
            
        Returns:
            variable_data: the extracted variable data for a single model.
            model_time: the extracted time data for a single model.
        """
        
        if variable == "psl":
            # #print the variable data
            # #print("Variable data: ", variable_data)
            # # #print the variable data type
            # #print("Variable data type: ", type(variable_data))

            # # #print the len of the variable data dimensions
            # #print("Variable data dimensions: ", len(variable_data.dims))
            
            # Convert from Pa to hPa.
            # Using try and except to catch any errors.
            try:
                # Extract the variable.
                variable_data = dataset["psl"]

                # #print the values of the variable data
                # #print("Variable data values: ", variable_data.values)

            except:
                #print("Error converting from Pa to hPa")
                sys.exit()

        elif variable == "tas":
            # Extract the variable.
            variable_data = dataset["tas"]
        elif variable == "rsds":
            # Extract the variable.
            variable_data = dataset["rsds"]
        elif variable == "sfcWind":
            # Extract the variable.
            variable_data = dataset["sfcWind"]
        elif variable == "tos":
            # Extract the variable
            variable_data = dataset["tos"]
        elif variable == "ua":
            # Extract the variable
            variable_data = dataset["ua"]
        elif variable == "va":
            # Extract the variable
            variable_data = dataset["va"]
        else:
            #print("Variable " + variable + " not recognised")
            sys.exit()

        # If variable_data is empty, #print a warning and exit the program.
        if variable_data is None:
            #print("Variable " + variable + " not found in dataset")
            sys.exit()

        # Extract the time dimension.
        model_time = dataset["time"].values
        # Set the type for the time dimension.
        model_time = model_time.astype("datetime64[Y]")

        # If model_time is empty, #print a warning and exit the program.
        if model_time is None:
            #print("Time not found in dataset")
            sys.exit()

        # Set up the attributes for the variable.
        variable_data.attrs = attributes

        return variable_data, model_time
    
    # Create empty dictionaries to store the processed data.
    variable_data_by_model = {}
    model_time_by_model = {}
    for model, datasets in datasets_by_model.items():
        try:
            # Create empty lists to store the processed data.
            variable_data_by_model[model] = []
            model_time_by_model[model] = []
            # Loop over the datasets for this model.
            for dataset in datasets:
                
                # Extract the member name from the dataset.
                member = dataset.attrs["variant_label"]
                print("Processing dataset for model", model, "member", member)

                # Extract the attributes from the dataset.
                attributes = dataset.attrs

                # Process the dataset.
                variable_data, model_time = process_model_dataset(dataset, variable, attributes)
                # Append the processed data to the lists.
                variable_data_by_model[model].append(variable_data)
                model_time_by_model[model].append(model_time)
        except Exception as e:
            #print(f"Error processing dataset for model {model}: {e}")
            #print("Exiting the program")
            sys.exit()

    # Return the processed data.
    return variable_data_by_model, model_time_by_model

# We want to write a function which reads and processes the observations
# then returns the obs anomaly field
def read_obs(variable, region, forecast_range, season, observations_path, start_year, end_year, level=None):
    """
    Processes the observations to have the same grid as the model data
    using CDO. Then selects the region and season. Then calculates the
    anomaly field using the climatology. Then calculates the annual 
    mean of the anomaly field. Then selects the forecast range (e.g. 
    years 2-5). Then selects the season. Then returns the anomaly field.
    

    Parameters
    ----------
    variable : str
        Variable name.
    region : str
        Region name.
    forecast_range : str
        Forecast range.
    season : str
        Season name.
    observations_path : str
        Path to the observations.
    start_year : str
        Start year.
    end_year : str
        End year.
    level : str, optional
        Level name. The default is None.

    Returns
    -------
    obs_anomaly : iris.cube.Cube
        Anomaly field.

    """

    # First check that the obs_path exists
    if not os.path.exists(observations_path):
        print('The observations path does not exist')
        sys.exit()

    # Get the path to the regridded and selected region observations
    regrid_obs_path = regrid_and_select_region(variable, region, observations_path, level=level)

    # Load the obs data into Iris cubes
    obs = load_obs(variable, regrid_obs_path)

    # If the level is not None, then extract the level
    if level is not None:
        obs = obs.extract(iris.Constraint(air_pressure=level))

    # Select the season
    if season not in dic.season_month_map:
        raise ValueError('The season is not in the dictionary')
        sys.exit()
    
    # Extract the months corresponding to the season
    months = dic.season_month_map[season]

    # Set up the iris constraint for the start and end years
    # Create the date time objects
    start_date = datetime(int(start_year), 12, 1)
    end_date = datetime(int(end_year), 3, 31)
    iris_constraint = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
    # Apply the iris constraint to the cube
    obs = obs.extract(iris_constraint)

    # Set up the iris constraint
    iris_constraint = iris.Constraint(time=lambda cell: cell.point.month in months)
    # Apply the iris constraint to the cube
    obs = obs.extract(iris_constraint)

    # # Add a month coordinate to the cube
    # coord_cat.add_month(obs, 'time')

    # Calculate the seasonal climatology
    # First collapse the time dimension by taking the mean
    climatology = obs.collapsed('time', iris.analysis.MEAN)

    # Calculate the anomaly field
    obs_anomaly = obs - climatology

    # Calculate the annual mean anomalies
    obs_anomaly_annual = calculate_annual_mean_anomalies(obs_anomaly, season)

    # Select the forecast range
    obs_anomaly_annual_forecast_range = select_forecast_range(obs_anomaly_annual, forecast_range)

    # If the type of obs_anomaly_annual_forecast_range is not a cube, then convert to a cube
    # if type(obs_anomaly_annual_forecast_range) != iris.cube.Cube:
    #     obs_anomaly_annual_forecast_range = xr.DataArray.to_iris(obs_anomaly_annual_forecast_range)

    # Return the anomaly field
    return obs_anomaly_annual_forecast_range


def calculate_annual_mean_anomalies(obs_anomalies, season):
    """
    Calculates the annual mean anomalies for a given observation dataset and season.

    Parameters:
    obs_anomalies (xarray.Dataset): The observation dataset containing anomalies.
    season (str): The season for which to calculate the annual mean anomalies.

    Returns:
    xarray.Dataset: The annual mean anomalies for the given observation dataset and season.

    Raises:
    ValueError: If the input dataset is invalid.
    """

    # if the type of obs_anomalies is an iris cube, then convert to an xarray dataset
    if type(obs_anomalies) == iris.cube.Cube:
        obs_anomalies = xr.DataArray.from_iris(obs_anomalies)

    try:
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-3)
        elif season in ["DJF", "NDJF"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            obs_anomalies_shifted = obs_anomalies.shift(time=-1)
        else:
            obs_anomalies_shifted = obs_anomalies

        # Calculate the annual mean anomalies
        obs_anomalies_annual = obs_anomalies_shifted.resample(time="Y").mean("time")

        return obs_anomalies_annual
    except:
        print("Error shifting and calculating annual mean anomalies for observations")
        sys.exit()

def select_forecast_range(obs_anomalies_annual, forecast_range):
    """
    Selects the forecast range for a given observation dataset.

    Parameters:
    obs_anomalies_annual (xarray.Dataset): The observation dataset containing annual mean anomalies.
    forecast_range (str): The forecast range to select.

    Returns:
    xarray.Dataset: The observation dataset containing annual mean anomalies for the selected forecast range.

    Raises:
    ValueError: If the input dataset is invalid.
    """
    try:
        
        forecast_range_start, forecast_range_end = map(int, forecast_range.split("-"))
        #print("Forecast range:", forecast_range_start, "-", forecast_range_end)
        
        rolling_mean_range = forecast_range_end - forecast_range_start + 1
        #print("Rolling mean range:", rolling_mean_range)
        
        obs_anomalies_annual_forecast_range = obs_anomalies_annual.rolling(time=rolling_mean_range, center = True).mean()
        
        return obs_anomalies_annual_forecast_range
    except Exception as e:
        #print("Error selecting forecast range:", e)
        sys.exit()

# Load the data
def load_data(base_directory, models, variable, region, forecast_range, season, level=None):
    """Load the data from the base directory into a dictionary of datasets.
    
    This function takes a base directory and a list of models and loads
    all of the individual ensemble members into a dictionary of datasets
    grouped by models.
    
    Args:
        base_directory: The base directory where the data is stored.
        models: A list of models to load.
        variable: The variable to load, extracted from the command line.
        region: The region to load, extracted from the command line.
        forecast_range: The forecast range to load, extracted from the command line.
        season: The season to load, extracted from the command line.
        Level: The level to load, extracted from the command line. Default is None.
        
    Returns:
        A dictionary of datasets grouped by models.
    """
    
    # Create an empty dictionary to store the datasets.
    datasets_by_model = {}
    
    # Loop over the models.
    for model in models:
        
        # Create an empty list to store the datasets for this model.
        datasets_by_model[model] = []
        
        # If the level is not None, then we want to load the data for the specified level
        if level is not None:
            files_path = base_directory + "/" + variable + "/" + model + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + f"plev_{level}" + "/" + "outputs" + "/" + "mergetime" + "/" + "*.nc"
        else:
            # create the path to the files for this model
            files_path = base_directory + "/" + variable + "/" + model + "/" + region + "/" + f"years_{forecast_range}" + "/" + season + "/" + "outputs" + "/" + "mergetime" + "/" + "*.nc"

        # #print the path to the files
        #print("Searching for files in ", files_path)

        # Create a list of the files for this model.
        files = glob.glob(files_path)

        # if the list of files is empty, #print a warning and
        # exit the program
        if len(files) == 0:
            print("No files found for " + model)
            sys.exit()
        
        # #print the files to the screen.
        #print("Files for " + model + ":", files)

        # Loop over the files.
        for file in files:

            # #print the file to the screen.
            # print(file)

            # Conditional statement to ensure that models are common to all variables
            if model == "CMCC-CM2-SR5":
                # Don't use the files containing r11 and above or r2?i?
                if re.search(r"r1[1-9]", file) or re.search(r"r2.i.", file):
                    print("Skipping file", file)
                    continue
            elif model == "EC-Earth3":
                # Don't use the files containing r?i2 or r??i2
                if re.search(r"r.i2", file) or re.search(r"r..i2", file):
                    print("Skipping file", file)
                    continue
            elif model == "FGOALS-f3-L":
                # Don't use files containing r1-6i? or r??i?
                if any(re.search(fr"r{i}i.", file) for i in range(1, 7)) or re.search(r"r..i.", file):
                    print("Skipping file", file)
                    continue

            # check that the file exists
            # if it doesn't exist, #print a warning and
            # exit the program
            if not os.path.exists(file):
                #print("File " + file + " does not exist")
                sys.exit()

            # Load the dataset.
            dataset = xr.open_dataset(file, chunks = {"time":50, "lat":100, "lon":100})

            # Append the dataset to the list of datasets for this model.
            datasets_by_model[model].append(dataset)
            
    # Return the dictionary of datasets.
    return datasets_by_model

# Define a function to constrain the years to the years that are in all of the model members
def constrain_years(model_data, models):
    """
    Constrains the years to the years that are in all of the models.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    constrained_data (dict): The model data with years constrained to the years that are in all of the models.
    """
    
    # If the type of model_data is cube, then convert to a 
    
    # Initialize a list to store the years for each model
    years_list = []

    # #print the models being proces
    # #print("models:", models)
    
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # # print the model name
            # # #print("model name:", model)
            # print("years len:", len(years), "for model:", model)

            # if len years is less than 10
            # print the model name, member name, and len years
            if len(years) < 10:
                print("model name:", model)
                print("member name:", member)
                print("years len:", len(years))

            # Append the years to the list of years
            years_list.append(years)

    # # #print the years list for debugging
    # print("years list:", years_list)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))


    # # #print the common years for debugging
    # print("Common years:", common_years)
    # print("Common years type:", type(common_years))
    # print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # #print the years extracted from the model
            # #print('model years', years)
            # #print('model years shape', np.shape(years))
            
            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(years, common_years)

            # #print("years in both shape", np.shape(years_in_both))
            # #print("years in both", years_in_both)
            
            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_data:
                constrained_data[model] = []
            constrained_data[model].append(member)

    # # #print the constrained data for debugging
    # #print("Constrained data:", constrained_data)

    return constrained_data

# checking for Nans in observed data
def remove_years_with_nans(observed_data, model_data, models):
    """
    Removes years from the observed data that contain NaN values.

    Args:
        observed_data (xarray.Dataset): The observed data.
        model_data (dict): The model data.
        models (list): The list of models to be plotted.
        variable (str): the variable name.

    Returns:
        xarray.Dataset: The observed data with years containing NaN values removed.
    """

    # Check that there are no NaN values in the model data
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_by_model = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_by_model:
            
            # # Modify the time dimension
            # if type is not already datetime64
            # then convert the time type to datetime64
            if type(member.time.values[0]) != np.datetime64:
                member_time = member.time.astype('datetime64[ns]')

                # # Modify the time coordinate using the assign_coords() method
                member = member.assign_coords(time=member_time)
            
            
            # Extract the years
            model_years = member.time.dt.year.values

            # If the years has duplicate values
            if len(model_years) != len(set(model_years)):
                # Raise a value error
                raise ValueError("The models years has duplicate values for model " + model + "member " + member)

            # Only if there are no NaN values in the model data
            # Will we loop over the years
            if not np.isnan(member.values).any():
                print("No NaN values in the model data")
                # continue with the loop
                continue

            print("NaN values in the model data")
            print("Model:", model)
            print("Member:", member)
            print("Looping over the years")
            # Loop over the years
            for year in model_years:
                # Extract the data for the year
                data = member.sel(time=f"{year}")

                if np.isnan(data.values).any():
                    print("NaN values in the model data for this year")
                    print("Model:", model)
                    print("Year:", year)
                    if np.isnan(data.values).all():
                        print("All NaN values in the model data for this year")
                        print("Model:", model)
                        print("Year:", year)
                        # De-Select the year from the observed data
                        member = member.sel(time=member.time.dt.year != year)

                        print(year, "all NaN values for this year")
                else:
                    print(year, "no NaN values for this year")

    # Now check that there are no NaN values in the observed data
    for year in observed_data.time.dt.year.values:
        # Extract the data for the year
        data = observed_data.sel(time=f"{year}")

        # print("data type", (type(data)))
        # print("data vaues", data)
        # print("data shape", np.shape(data))

        # If there are any NaN values in the data
        if np.isnan(data.values).any():
            # If there are only NaN values in the data
            if np.isnan(data.values).all():
                # Select the year from the observed data
                observed_data = observed_data.sel(time=observed_data.time.dt.year != year)

                print(year, "all NaN values for this year")
        # if there are no NaN values in the data for a year
        # then #print the year
        # and "no nan for this year"
        # and continue the script
        else:
            print(year, "no NaN values for this year")

    # Set up the years to be returned
    obs_years = observed_data.time.dt.year.values

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # if obs years and model years are not the same
    if obs_years != model_years:
        print("obs years and model years are not the same")
        print("Aligning the years")

        # Find the years that are in both the model data and the common years
        years_in_both = np.intersect1d(obs_years, model_years)

        # Select only those years from the model data
        observed_data = observed_data.sel(time=observed_data.time.dt.year.isin(years_in_both))

        # for the model data
        for model in models:
            # Extract the model data
            model_data_by_model = model_data[model]

            # Loop over the ensemble members in the model data
            for member in model_data_by_model:
                # Extract the years
                model_years = member.time.dt.year.values

                # Select only those years from the model data
                member = member.sel(time=member.time.dt.year.isin(years_in_both))

                # Add the member to the constrained data dictionary
                if model not in constrained_data:
                    constrained_data[model] = []

                # Append the member to the constrained data dictionary
                constrained_data[model].append(member)

    return observed_data, constrained_data


# Calculate obs nao
def calculate_obs_nao(obs_anomaly, south_grid, north_grid):
    """
    Calculates the North Atlantic Oscillation (NAO) index for the given
    observations and gridboxes.

    Parameters
    ----------
    obs_anomaly : xarray.Dataset
        Anomaly field of the observations.
    south_grid : dict
        Dictionary containing the longitude and latitude values of the
        southern gridbox.
    north_grid : dict
        Dictionary containing the longitude and latitude values of the
        northern gridbox.

    Returns
    -------
    obs_nao : xarray.DataArray
        NAO index for the observations.

    """

    # Extract the lat and lon values
    # from the gridbox dictionary
    s_lon1, s_lon2 = south_grid["lon1"], south_grid["lon2"]
    s_lat1, s_lat2 = south_grid["lat1"], south_grid["lat2"]

    # second for the northern box
    n_lon1, n_lon2 = north_grid["lon1"], north_grid["lon2"]
    n_lat1, n_lat2 = north_grid["lat1"], north_grid["lat2"]

    # Take the mean over the lat and lon values
    south_grid_timeseries = obs_anomaly.sel(lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)).mean(dim=["lat", "lon"])
    north_grid_timeseries = obs_anomaly.sel(lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)).mean(dim=["lat", "lon"])

    # Calculate the NAO index for the observations
    obs_nao = south_grid_timeseries - north_grid_timeseries

    return obs_nao

# Write a function to calculate the NAO index
# For both the obs and model data
def calculate_nao_index_and_plot(obs_anomaly, model_anomaly, models, variable, season, forecast_range,
                                    output_dir, plot_graphics=False, azores_grid = dic.azores_grid, 
                                        iceland_grid = dic.iceland_grid, snao_south_grid = dic.snao_south_grid, 
                                            snao_north_grid = dic.snao_north_grid):
    """
    Calculates the NAO index for both the obs and model data.
    Then plots the NAO index for both the obs and model data if the plot_graphics flag is set to True.

    Parameters
    ----------
    obs_anomaly : xarray.Dataset
        Observations.
    model_anomaly : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets.
    models : list
        List of models to be plotted. Different models for each variable.
    variable : str
        Variable name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    plot_graphics : bool, optional
        Flag to plot the NAO index. The default is False.
    azores_grid : str, optional
        Azores grid. The default is dic.azores_grid.
    iceland_grid : str, optional
        Iceland grid. The default is dic.iceland_grid.
    snao_south_grid : str, optional
        SNAO south grid. The default is dic.snao_south_grid.
    snao_north_grid : str, optional
        SNAO north grid. The default is dic.snao_north_grid.

    Returns
    -------
    obs_nao: xarray.Dataset
        Observations. NAO index.
    model_nao: dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # If the variable is not psl, then exit
    if variable != 'psl':
        AssertionError('The variable is not psl')
        sys.exit()
    
    # if the season is JJA, use the summer definition of the NAO
    if season == "JJA":
        print("Calculating NAO index using summer definition")
        # Set up the dict for the southern box and northern box
        south_grid, north_grid = snao_south_grid, snao_north_grid
        # Set up the NAO type for the summer definition
        nao_type = "snao"
    else:
        print("Calculating NAO index using standard definition")
        # Set up the dict for the southern box and northern box
        south_grid, north_grid = azores_grid, iceland_grid
        # Set up the NAO type for the standard definition
        nao_type = "default"

    # Calculate the NAO index for the observations
    obs_nao = calculate_obs_nao(obs_anomaly, south_grid, north_grid)

    # Calculate the NAO index for the model data
    model_nao, years, \
    ensemble_members_count = calculate_model_nao_anoms(model_anomaly, models, azores_grid,
                                                                            iceland_grid, snao_south_grid, snao_north_grid,
                                                                                nao_type=nao_type)
    
    # If the plot_graphics flag is set to True
    if plot_graphics:
        # First calculate the ensemble mean NAO index
        ensemble_mean_nao, _ = calculate_ensemble_mean_nao_index(model_nao, models)

        # Calculate the correlation coefficients between the observed and model data
        r, p, _, _, _, _ = calculate_nao_correlations(obs_nao, ensemble_mean_nao, variable)

        # Plot the NAO index
        plot_nao_index(obs_nao, ensemble_mean_nao, variable, season, forecast_range, r, p, output_dir,
                            ensemble_members_count, nao_type=nao_type)
        
    return obs_nao, model_nao

# Write a function to rescale the NAO index
# We will only consider the non-lagged ensemble index for now
def rescale_nao(obs_nao, model_nao, models, season, forecast_range, output_dir, lagged = False):
    """
    Rescales the NAO index according to Doug Smith's (2020) method.
    
    Parameters
    ----------
    obs_nao : xarray.Dataset
        Observations.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets.
    models : list
        List of models to be plotted. Different models for each variable.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    lagged : bool, optional
        Flag to indicate whether the NAO index is lagged or not. The default is False.

    Returns
    -------
    rescaled_model_nao : numpy.ndarray
        Array contains the rescaled NAO index.
    ensemble_mean_nao : numpy.ndarray
        Ensemble mean NAO index. Not rescaled
    ensemble_members_nao : numpy.ndarray
        Ensemble members NAO index. Not rescaled
    """

    # First calculate the ensemble mean NAO index
    ensemble_mean_nao, ensemble_members_nao = calculate_ensemble_mean_nao_index(model_nao, models)

    # Extract the years from the ensemble members
    model_years = ensemble_mean_nao.time.dt.year.values
    # Extract the years from the obs
    obs_years = obs_nao.time.dt.year.values

    # If the two years arrays are not equal
    if not np.array_equal(model_years, obs_years):
        # Print a warning and exit the program
        print("The years for the ensemble members and the observations are not equal")
        sys.exit()

    # if the type of obs_nao is not a numpy array
    # Then convert to a numpy array
    if type(obs_nao) != np.ndarray:
        print("Converting obs_nao to a numpy array")
        obs_nao = obs_nao.values

    # Create an empty numpy array to store the rescaled NAO index
    rescaled_model_nao = np.empty((len(model_years)))

    # Loop over the years and perform the rescaling (including cross-validation)
    for i, year in enumerate(model_years):

        # Compute the rescaled NAO index for this year
        signal_adjusted_nao_index_year, _ = rescale_nao_by_year(year, obs_nao, ensemble_mean_nao, ensemble_members_nao, season,
                                                            forecast_range, output_dir, lagged=False, omit_no_either_side=1)

        # Append the rescaled NAO index to the list, along with the year
        rescaled_model_nao[i] = signal_adjusted_nao_index_year

    # Convert the list to an xarray DataArray
    # With the same coordinates as the ensemble mean NAO index
    rescaled_model_nao = xr.DataArray(rescaled_model_nao, coords=ensemble_mean_nao.coords, dims=ensemble_mean_nao.dims)

    # If the time type is not datetime64 for the rescaled model nao
    # Then convert the time type to datetime64
    if type(rescaled_model_nao.time.values[0]) != np.datetime64:
        rescaled_model_nao_time = rescaled_model_nao.time.astype('datetime64[ns]')

        # Modify the time coordinate using the assign_coords() method
        rescaled_model_nao = rescaled_model_nao.assign_coords(time=rescaled_model_nao_time)

    # Return the rescaled model NAO index
    return rescaled_model_nao, ensemble_mean_nao, ensemble_members_nao

# Define a new function to rescalse the NAO index for each year
def rescale_nao_by_year(year, obs_nao, ensemble_mean_nao, ensemble_members_nao, season,
                            forecast_range, output_dir, lagged=False, omit_no_either_side=1):
    """
    Rescales the observed and model NAO indices for a given year and season, and saves the results to disk.

    Parameters
    ----------
    year : int
        The year for which to rescale the NAO indices.
    obs_nao : pandas.DataFrame
        A DataFrame containing the observed NAO index values, with a DatetimeIndex.
    ensemble_mean_nao : pandas.DataFrame
        A DataFrame containing the ensemble mean NAO index values, with a DatetimeIndex.
    ensemble_members_nao : dict
        A dictionary containing the NAO index values for each ensemble member, with a DatetimeIndex.
    season : str
        The season for which to rescale the NAO indices. Must be one of 'DJF', 'MAM', 'JJA', or 'SON'.
    forecast_range : int
        The number of months to forecast ahead.
    output_dir : str
        The directory where to save the rescaled NAO indices.
    lagged : bool, optional
        Whether to use lagged NAO indices in the rescaling. Default is False.

    Returns
    -------
    None
    """

    # Print the year for which the NAO indices are being rescaled
    print(f"Rescaling NAO indices for {year}")

    # Extract the model years
    model_years = ensemble_mean_nao.time.dt.year.values

    # Ensure that the type of ensemble_mean_nao and ensemble_members_nao is a an array
    if type(ensemble_mean_nao) and type(ensemble_members_nao) != np.ndarray and type(obs_nao) != np.ndarray:
        AssertionError("The type of ensemble_mean_nao and ensemble_members_nao and obs_nao is not a numpy array")
        sys.exit()

    # If the year is not in the ensemble members years
    if year not in model_years:
        # Print a warning and exit the program
        print(f"Year {year} is not in the ensemble members years")
        sys.exit()

    # Extract the index for the year
    year_index = np.where(model_years == year)[0]

    # Extract the ensemble members for the year
    ensemble_members_nao_year = ensemble_members_nao[:, year_index]

    # Compute the ensemble mean NAO for this year
    ensemble_mean_nao_year = ensemble_members_nao_year.mean(axis=0)

    # Set up the indicies for the cross-validation
    # In the case of the first year
    if year == model_years[0]:
        print("Cross-validation case for the first year")
        print("Removing the first year and:", omit_no_either_side, "years forward")
        # Set up the indices to use for the cross-validation
        # Remove the first year and omit_no_either_side years forward
        cross_validation_indices = np.arange(0, omit_no_either_side + 1)
    # In the case of the last year
    elif year == model_years[-1]:
        print("Cross-validation case for the last year")
        print("Removing the last year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Remove the last year and omit_no_either_side years backward
        cross_validation_indices = np.arange(-1, -omit_no_either_side - 2, -1)
    # In the case of any other year
    else:
        # Omit the year and omit_no_either_side years forward and backward
        print("Cross-validation case for any other year")
        print("Removing the year and:", omit_no_either_side, "years backward")
        # Set up the indices to use for the cross-validation
        # Use the year index and omit_no_either_side years forward and backward
        cross_validation_indices = np.arange(year_index - omit_no_either_side, year_index + omit_no_either_side + 1)
    
    # Log which years are being used for the cross-validation
    print("Cross-validation indices:", cross_validation_indices)

    # Extract the ensemble members for the cross-validation
    # i.e. don't use the years given by the cross_validation_indices
    ensemble_members_nao_array_cross_val = np.delete(ensemble_members_nao, cross_validation_indices, axis=1)
    # Take the mean over the ensemble members
    # to get the ensemble mean nao for the cross-validation
    ensemble_mean_nao_cross_val = ensemble_members_nao_array_cross_val.mean(axis=0)

    # Remove the indicies from the obs_nao
    obs_nao_cross_val = np.delete(obs_nao, cross_validation_indices, axis=0)

    # Calculate the pearson correlation coefficient between the observed and model NAO indices
    acc_score, p_value = stats.pearsonr(obs_nao_cross_val, ensemble_mean_nao_cross_val)

    # Calculate the RPS score 
    rps_score = calculate_rps(acc_score, ensemble_members_nao_array_cross_val, obs_nao_cross_val)  

    # Compute the rescaled NAO index for the year
    signal_adjusted_nao_index = ensemble_mean_nao_year * rps_score

    return signal_adjusted_nao_index, ensemble_mean_nao_year


def calculate_rpc(acc_score, ensemble_members_array):
    """
    Calculates the RPC score. Ratio of predictable components.
    
    Parameters
    ----------
    acc_score : float
        The ACC score.
    ensemble_members_array : numpy.ndarray
        The ensemble members array.
        
    Returns
    -------
    rpc_score : float
        The RPC score.
    """

    # Calculate the ensemble mean over all members
    ensemble_mean = np.mean(ensemble_members_array, axis=0)

    # Calculate the standard deviation of the predictable signal for the forecasts (σfsig)
    sigma_fsig = np.std(ensemble_mean)

    # Calculate the total standard deviation of the forecasts (σftot)
    sigma_ftot = np.std(ensemble_members_array)

    # Calculate the RPC score
    rpc_score = acc_score / (sigma_fsig / sigma_ftot)

    return rpc_score

# Calculate the RPS score - ratio of predictable signals
def calculate_rps(acc_score, ensemble_members_array, obs_nao):
    """
    Calculates the RPS score. Ratio of predictable signals.
    
    Parameters
    ----------
    acc_score : float
        The ACC score.
    ensemble_members_array : numpy.ndarray
        The ensemble members array.
    obs_nao : numpy.ndarray
        The observed NAO index.
        
    Returns
    -------
    rps_score : float
        The RPS score.
    """

    # Calculate the ratio of predictable components (for the model)
    rpc = calculate_rpc(acc_score, ensemble_members_array)

    # Calculate the total standard deviation of the observations (σotot)
    obs_std = np.std(obs_nao)

    # Calculate the total standard deviation of the forecasts (σftot)
    model_std = np.std(ensemble_members_array)

    # Calculate the RPS score
    rps_score = rpc * (obs_std / model_std)

    return rps_score    


# Calculate the members which have the closest NAO index to the rescaled NAO index
def calculate_closest_members(year, rescaled_model_nao, model_nao, models, season, forecast_range, 
                                output_dir, lagged=False, no_subset_members=20):
    """
    Calculates the ensemble members (within model_nao) which have the closest NAO index to the rescaled NAO index.

    Parameters
    ----------
    year : int
        The year for which to rescale the NAO indices.
    rescaled_model_nao : xarray.DataArray
        Rescaled NAO index.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    models : list
        List of models to be plotted. Different models for each variable.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    output_dir : str
        Path to the output directory.
    lagged : bool, optional
        Flag to indicate whether the ensemble is lagged or not. The default is False.
    no_subset_members : int, optional
        Number of ensemble members to subset. The default is 20.
    
    Returns
    -------
    closest_nao_members : dict
        Dictionary containing the closest ensemble members for each model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Print the year which is being processed
    print(f"Calculating nearest members for year: {year}")

    # Extract the years for the rescaled NAO index and the model NAO index
    rescaled_model_nao_years = rescaled_model_nao.time.dt.year.values
    model_nao_years = model_nao[models[0]][0].time.dt.year.values

    # If the two years arrays are not equal
    if not np.array_equal(rescaled_model_nao_years, model_nao_years):
        # Print a warning and exit the program
        print("The years for the rescaled NAO index and the model NAO index are not equal")
        sys.exit()

    # Initialize a dictionary to store the smallest difference between the rescaled NAO index and the model NAO index
    smallest_diff = []

    # Extract the data for the year for the rescaled NAO index
    rescaled_model_nao_year = rescaled_model_nao.sel(time=f"{year}")

    # Form the list of ensemble members
    ensemble_members_list, ensemble_members_count = form_ensemble_members_list(model_nao, models)

    # Loop over the ensemble members
    for member in ensemble_members_list:
        # Extract the data for the year
        model_nao_year = member.sel(time=f"{year}")

        # Print the model name and the member name
        print("Model:", member.attrs["source_id"])
        print("Member:", member.attrs["variant_label"])

        # Calculate the difference between the rescaled NAO index and the model NAO index
        nao_diff = np.abs(rescaled_model_nao_year - model_nao_year)

        # Extract the attributes from the member
        member_attributes = member.attrs

        # Add the attributes to the diff
        nao_diff.attrs = member_attributes
        
        # Append the difference to the list
        smallest_diff.append(nao_diff)

        # Sort the list by the smallest difference
        smallest_diff.sort()

    # Select the first no_subset_members members, with the smallest differences
    smallest_diff = smallest_diff[:no_subset_members]

    # Loop over the members with the smallest differences
    for i, member in enumerate(smallest_diff):
        print("Smallest difference member:", i+1)
        # print the model name and the member name
        print("Model:", member.attrs["source_id"])
        print("Member:", member.attrs["variant_label"])

    return smallest_diff

# Write a function which performs the NAO matching
def nao_matching_other_var(rescaled_model_nao, model_nao, psl_models, match_variable, match_var_base_dir,
                            match_var_models, match_var_obs_path, region, season, forecast_range, 
                                start_year, end_year, output_dir, lagged=False, 
                                    no_subset_members=20, level = None):
    """
    Performs the NAO matching for the given variable. E.g. T2M.

    Parameters
    ----------
    rescaled_model_nao : xarray.DataArray
        Rescaled NAO index.
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    psl_models : list
        List of models to be plotted. Different models for each variable.
    match_variable : str
        Variable name for the variable which will undergo matching.
    match_var_base_dir : str
        Path to the base directory containing the variable data.
    match_var_models : list
        List of models to be plotted for the matched variable. Different models for each variable.
    region : str
        Region name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    start_year : int
        Start year.
    end_year : int
        End year.
    output_dir : str
        Path to the output directory.
    lagged : bool, optional
        Flag to indicate whether the ensemble is lagged or not. The default is False.
    no_subset_members : int, optional
        Number of ensemble members to subset. The default is 20.
    level : int, optional
        Pressure level. The default is None. For the matched variable.
    
    Returns
    -------
    None
    """

    # Print the variable which is being matched
    print(f"Performing NAO matching for {match_variable}")

    # Extract the obs data for the matched variable
    match_var_obs_anomalies = read_obs(match_variable, region, forecast_range,
                                        season, match_var_obs_path, start_year, end_year, level=level)
    
    # Extract the model data for the matched variable
    match_var_datasets = load_data(match_var_base_dir, match_var_models, match_variable, 
                                    region, forecast_range, season, level=level)
    
    # process the model data
    match_var_model_anomalies, _ = process_data(match_var_datasets, match_variable)

    # Make sure that each of the models have the same time period
    match_var_model_anomalies = constrain_years(match_var_model_anomalies, match_var_models)

    # Remove years containing NaN values from the obs and model data
    # and align the time periods
    match_var_obs_anomalies, match_var_model_anomalies = remove_years_with_nans(match_var_obs_anomalies,
                                                                                    match_var_model_anomalies,
                                                                                        match_var_models)

    # Now we want to make sure that the match_var_model_anomalies and the model_nao
    # have the same models
    model_nao_constrained, match_var_model_anomalies_constrained, \
    models_in_both = constrain_models_members(model_nao, psl_models, 
                                                match_var_model_anomalies, match_var_models)
    
    # Make sure that the years for rescaled_model_nao and model_nao 
    # and match_var_model_anomalies_constrained are the same
    rescaled_model_years = rescaled_model_nao.time.dt.year.values
    model_nao_years = model_nao_constrained[psl_models[0]][0].time.dt.year.values
    match_var_model_years = match_var_model_anomalies_constrained[match_var_models[0]][0].time.dt.year.values

    # If the years are not equal
    if not np.array_equal(rescaled_model_years, model_nao_years) or not np.array_equal(rescaled_model_years, match_var_model_years):
        # Print a warning and exit the program
        print("The years for the rescaled model NAO, the model NAO and the matched variable model anomalies are not equal")
        sys.exit()
    
    # Set up the years to loop over
    years = rescaled_model_years
                                                                                    
    # TODO: Loop over the years and perform the NAO matching
    for i, year in enumerate(years):
        print("Selecting members for year: ", year)

        # Extract the members with the closest NAO index to the rescaled NAO index
        # for the given year
        smallest_diff = calculate_closest_members(year, rescaled_model_nao, model_nao_constrained, models_in_both, 
                                                    season, forecast_range, output_dir, lagged=lagged,
                                                        no_subset_members=no_subset_members)  

        # Using the closest NAO index members, extract the same members
        # for the matched variable
        matched_var_members = extract_matched_var_members(match_var_model_anomalies_constrained, smallest_diff)


# Define a function which will extract the right model members for the matched variable
def extract_matched_var_members(match_var_model_anomalies_constrained, smallest_diff):
    """
    Extracts the right model members for the matched variable.
    These members have the correct magnitude of the NAO index.
    """

    # Create an empty list to store the matched variable members
    matched_var_members = []

    # Extract the models from the smallest_diff
    smallest_diff_models = [member.attrs["source_id"] for member in smallest_diff]

    # Loop over the models in the smallest_diff
    for model in smallest_diff_models:
        # Extract the model data for the model
        model_data = match_var_model_anomalies_constrained[model]

        # Extract the members for this model within the smallest_diff
        smallest_diff_members = [member.attrs["variant_label"] for member in smallest_diff if member.attrs["source_id"] == model]

        # Loop over the members in the model_data
        for member in model_data:
            # If the member is in the smallest_diff_members
            if member.attrs["variant_label"] in smallest_diff_members:
                # Append the member to the matched_var_members
                matched_var_members.append(member)
            else:
                continue

    # return the matched_var_members
    return matched_var_members


# Define a function which will make sure that the model_nao and the match_var_model_anomalies
# have the same models and members
def constrain_models_members(model_nao, psl_models, match_var_model_anomalies, match_var_models):
    """
    Makes sure that the model_nao and the match_var_model_anomalies have the same models and members.
    """

    # Set up dictionaries to store the models and members
    psl_models_dict = {}
    match_var_models_dict = {}

    # If the two models lists are not equal
    if not np.array_equal(psl_models, match_var_models):
        # Print a warning and exit the program
        print("The two models lists are not equal")
        print("Constraining the models")

        # Find the models that are in both the psl_models and the match_var_models
        models_in_both = np.intersect1d(psl_models, match_var_models)
    else:
        # Set the models_in_both to the psl_models
        print("The two models lists are equal")
        models_in_both = psl_models

    # Loop over the models in the model_nao
    for model in models_in_both:
        print("Model:", model)

        # Append the model to the psl_models_dict
        psl_models_dict[model] = []

        # Append the model to the match_var_models_dict
        match_var_models_dict[model] = []

        # Extract the NAO data for the model
        model_nao_by_model = model_nao[model]

        # Extract the match_var_model_anomalies for the model
        match_var_model_anomalies_by_model = match_var_model_anomalies[model]

        # Extract a list of the variant labels for the model
        variant_labels_psl = [member.attrs["variant_label"] for member in model_nao_by_model]
        print("Variant labels for the model psl:", variant_labels_psl)
        # Extract a list of the variant labels for the match_var_model_anomalies
        variant_labels_match_var = [member.attrs["variant_label"] for member in match_var_model_anomalies_by_model]
        print("Variant labels for the model match_var:", variant_labels_match_var)

        # If the two variant labels lists are not equal
        if not np.array_equal(variant_labels_psl, variant_labels_match_var):
            # Print a warning and exit the program
            print("The two variant labels lists are not equal")
            print("Constraining the variant labels")

            # Find the variant labels that are in both the variant_labels_psl and the variant_labels_match_var
            variant_labels_in_both = np.intersect1d(variant_labels_psl, variant_labels_match_var)

            # Now filter the model_nao data
            psl_models_dict[model] = filter_model_data_by_variant_labels(model_nao_by_model, variant_labels_in_both, psl_models_dict[model])
                
            # Now filter the match_var_model_anomalies data
            match_var_models_dict[model] = filter_model_data_by_variant_labels(match_var_model_anomalies_by_model, variant_labels_in_both, match_var_models_dict[model])

        else:
            print("The two variant labels lists are equal")
            # Loop over the members in the model_nao_by_model
            for member in model_nao_by_model:
                # Append the member to the psl_models_dict
                psl_models_dict[model].append(member)

            # Loop over the members in the match_var_model_anomalies_by_model
            for member in match_var_model_anomalies_by_model:
                # Append the member to the match_var_models_dict
                match_var_models_dict[model].append(member)

    return psl_models_dict, match_var_models_dict, models_in_both

def filter_model_data_by_variant_labels(model_data, variant_labels_in_both, model_dict):
    """
    Filters the model data to only include ensemble members with variant labels that are in both the model NAO data
    and the observed data.

    Parameters
    ----------
    model_data : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    variant_labels_in_both : list
        List of variant labels that are in both the model NAO data and the observed NAO data.
    model_dict : dict
        Dictionary containing the model names as keys and the variant labels as values.
        
    Returns
    -------
    psl_models_dict : dict
        Dictionary of filtered model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Loop over the members in the model_data
    for member in model_data:
        # Extract the variant label for the member
        variant_label = member.attrs["variant_label"]

        # Only if the variant label is in the variant_labels_in_both
        if variant_label in variant_labels_in_both:
            # Append the member to the model_dict
            model_dict.append(member)
        else:
            print("Variant label:", variant_label, "not in the variant_labels_in_both")

    return model_dict


# Define a new function to form the list of ensemble members
def form_ensemble_members_list(model_nao, models):
    """
    Forms a list of ensemble members, not a dictionary with model keys.
    Each xarray object should have the associated metadata stored in attributes.
    
    Parameters
    ----------
    model_nao : dict
        Dictionary of model data. Sorted by model.
        Each model contains a list of ensemble members, which are xarray datasets containing the NAO index.
    models : list
        List of models to be plotted. Different models for each variable.
    
    
    Returns
    -------
    ensemble_members_list : list
        List of ensemble members, which are xarray datasets containing the NAO index.
    """

    # Initialize a list to store the ensemble members
    ensemble_members_list = []

    # Initialize a dictionary to store the number of ensemble members for each model
    ensemble_members_count = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_nao_by_model = model_nao[model]

        # If the model is not in the ensemble_members_count dictionary
        if model not in ensemble_members_count:
            # Add the model to the ensemble_members_count dictionary
            ensemble_members_count[model] = 0

        # Loop over the ensemble members
        for member in model_nao_by_model:

            # if the type of time is not a datetime64
            if type(member.time.values[0]) != np.datetime64:
                # Extract the time values as a datetime64
                member_time = member.time.astype('datetime64[ns]')

                # Add the time values back to the member
                member = member.assign_coords(time=member_time)

            # If the years are not unique
            years = member.time.dt.year.values

            # Check that the years are unique
            if len(years) != len(set(years)):
                raise ValueError("Duplicate years in the member")
            
            # Add the member to the ensemble_members_list
            ensemble_members_list.append(member)

            # Add one to the ensemble_members_count dictionary
            ensemble_members_count[model] += 1

    return ensemble_members_list, ensemble_members_count

    


# Define a function for plotting the NAO index
def plot_nao_index(obs_nao, ensemble_mean_nao, variable, season, forecast_range, r, p, output_dir, 
                        ensemble_members_count, experiment = "dcppA-hindcast", nao_type="default"):
    """
    Plots the NAO index for both the observations and model data.
    
    Parameters
    ----------
    obs_nao : xarray.Dataset
        Observations.
    ensemble_mean_nao : xarray.Dataset
        Ensemble mean of the model data.
    variable : str
        Variable name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
    r : float
        Correlation coefficients between the observed and model data.
    p : float
        p-values for the correlation coefficients between the observed and model data.
    output_dir : str
        Path to the output directory.
    ensemble_members_count : dict
        Number of ensemble members for each model.
    experiment : str, optional
        Experiment name. The default is "dcppA-hindcast".
    nao_type : str, optional
        NAO type. The default is "default".    


    Returns
    -------
    None.

    """
    
    # Set the font size
    plt.rcParams.update({'font.size': 12})

    # Set up the figure
    fig = plt.figure(figsize=(10, 6))

    # Set up the title
    plot_name = f"{variable} {forecast_range} {season} {experiment} {nao_type} NAO index"

    # Process the obs and the model data
    # from Pa to hPa
    obs_nao = obs_nao / 100
    ensemble_mean_nao = ensemble_mean_nao / 100

    # Extract the years
    obs_years = obs_nao.time.dt.year.values
    model_years = ensemble_mean_nao.time.dt.year.values

    # If the obs years and model years are not the same
    if len(obs_years) != len(model_years):
        raise ValueError("Observed years and model years must be the same.")

    # Plot the obs and the model data
    plt.plot(obs_years, obs_nao, label="ERA5", color="black")

    # Plot the ensemble mean
    plt.plot(model_years, ensemble_mean_nao, label="dcppA", color="red")

    # Add a horizontal line at y=0
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    # Set the ylim
    plt.ylim(-10, 10)
    plt.ylabel("NAO index (hPa)")
    plt.xlabel("year")

    # Set up a textbox with the season name in the top left corner
    plt.text(0.05, 0.95, season, transform=fig.transFigure, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # If the nao_type is not default
    # then add a textbox with the nao_type in the top right corner
    if nao_type != "default":
        # nao type = summer nao
        # add a textbox with the nao_type in the top right corner
        plt.text(0.95, 0.95, nao_type, transform=fig.transFigure, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Set up the p value text box
    if p < 0.01:
        p_text = "< 0.01"
    elif p < 0.05:
        p_text = "< 0.05"
    else:
        p_text = f"= {p:.2f}"

    # Extract the ensemble members count
    no_ensemble_members = sum(ensemble_members_count.values())

    # Set up the title for the plot
    plt.title(f"ACC = {r:.2f}, p {p_text}, n = {no_ensemble_members}, years_{forecast_range}, {season}, {experiment}", fontsize=10)

    # Set up the figure name
    fig_name = f"{variable}_{forecast_range}_{season}_{experiment}_{nao_type}_NAO_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Save the figure
    plt.savefig(output_dir + "/" + fig_name, dpi=300, bbox_inches="tight")

    # Show the figure
    plt.show()




# Define a new function to calculate the correlations between the observed and model data
# for the NAO index time series
def calculate_nao_correlations(obs_nao, model_nao, variable):
    """
    Calculates the correlation coefficients between the observed North Atlantic Oscillation (NAO) index and the NAO indices
    of multiple climate models.

    Args:
        obs_nao (array-like): The observed NAO index values.
        model_nao (dict): A dictionary containing the NAO index values for each climate model.
        models (list): A list of strings representing the names of the climate models.

    Returns:
        A dictionary containing the correlation coefficients between the observed NAO index and the NAO indices of each
        climate model.
    """
    
    # First check the dimensions of the observed and model data
    print("observed data shape", np.shape(obs_nao))
    print("model data shape", np.shape(model_nao))

    # Find the years that are in both the observed and model data
    obs_years = obs_nao.time.dt.year.values
    model_years = model_nao.time.dt.year.values

    # print the years
    print("observed years", obs_years)
    print("model years", model_years)

    # If obs years and model years are not the same
    if len(obs_years) != len(model_years):
        print("obs years and model years are not the same")
        print("Aligning the years")

        # Find the years that are in both the observed and model data
        years_in_both = np.intersect1d(obs_years, model_years)

        # Select only the years that are in both the observed and model data
        obs_nao = obs_nao.sel(time=obs_nao.time.dt.year.isin(years_in_both))
        model_nao = model_nao.sel(time=model_nao.time.dt.year.isin(years_in_both))

        # Remove years with NaNs
        obs_nao, model_nao, obs_years, model_years = remove_years_with_nans(obs_nao, model_nao, variable)

    # Convert both the observed and model data to numpy arrays
    obs_nao_array = obs_nao.values
    model_nao_array = model_nao.values

    # Check that the observed data and ensemble mean have the same shape
    if obs_nao_array.shape != model_nao_array.shape:
        raise ValueError("Observed data and ensemble mean must have the same shape.")
    
    # Calculate the correlations between the observed and model data
    # Using the new function calculate_correlations_1D
    r, p = calculate_correlations_1D(obs_nao_array, model_nao_array)

    # Return the correlation coefficients and p-values
    return r, p, model_nao_array, obs_nao_array, model_years, obs_years

# Define a new function to calculate the one dimensional correlations
# between the observed and model data
def calculate_correlations_1D(observed_data, model_data):
    """
    Calculates the correlations between the observed and model data.
    
    Parameters:
    observed_data (numpy.ndarray): The processed observed data.
    model_data (numpy.ndarray): The processed model data.
    
    Returns:
    r (xarray.core.dataarray.DataArray): The spatial correlations between the observed and model data.
    p (xarray.core.dataarray.DataArray): The p-values for the spatial correlations between the observed and model data.
    """

    # Initialize empty arrays for the spatial correlations and p-values
    r = []
    p = []

    # Verify that the observed and model data have the same shape
    if observed_data.shape != model_data.shape:
        raise ValueError("Observed data and model data must have the same shape.")
    
    # Verify that they don't contain all NaN values
    if np.isnan(observed_data).all() or np.isnan(model_data).all():
        # #print a warning
        print("Warning: All NaN values detected in the data.")
        print("exiting the script")
        sys.exit()

    # Calculate the correlation coefficient and p-value
    r, p = stats.pearsonr(observed_data, model_data)

    # return the correlation coefficient and p-value
    return r, p

# Define a function to calculate the ensemble mean NAO index
def calculate_ensemble_mean_nao_index(model_nao, models):
    """
    Calculates the ensemble mean NAO index for the given model data.
    
    Parameters
    ----------
    model_nao (dict): The model data containing the NAO index for each ensemble member.
    models (list): The list of models to be plotted.
    
    Returns
    -------
    ensemble_mean_nao (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    """

    # Initialize a list for the ensemble members
    ensemble_members_nao = []

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_nao[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append the ensemble member to the list of ensemble members
            ensemble_members_nao.append(member)

    # Convert the list of ensemble members to a numpy array
    ensemble_members_nao = np.array(ensemble_members_nao)

    # Calculate the ensemble mean NAO index
    ensemble_mean_nao = np.mean(ensemble_members_nao, axis=0)

    # Convert the ensemble mean NAO index to an xarray DataArray
    ensemble_mean_nao = xr.DataArray(ensemble_mean_nao, coords=member.coords, dims=member.dims)

    return ensemble_mean_nao, ensemble_members_nao    



# Define a new function to calculate the model NAO index
# like process_model_data_for_plot_timeseries
# but for the NAO index
def calculate_model_nao_anoms(model_data, models, azores_grid, iceland_grid, 
                                snao_south_grid, snao_north_grid, nao_type="default"):
    """
    Calculates the model NAO index for each ensemble member and the ensemble mean.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.
    azores_grid (dict): Latitude and longitude coordinates of the Azores grid point.
    iceland_grid (dict): Latitude and longitude coordinates of the Iceland grid point.
    snao_south_grid (dict): Latitude and longitude coordinates of the southern SNAO grid point.
    snao_north_grid (dict): Latitude and longitude coordinates of the northern SNAO grid point.
    nao_type (str, optional): Type of NAO index to calculate, by default 'default'. Also supports 'snao'.

    Returns:
    ensemble_mean_nao_anoms (xarray.core.dataarray.DataArray): The equally weighted ensemble mean of the ensemble members.
    ensemble_members_nao_anoms (list): The NAO index anomalies for each ensemble member.
    years (numpy.ndarray): The years.
    ensemble_members_count (dict): The number of ensemble members for each model.
    """

    # Initialize a list for the ensemble members
    ensemble_members_nao_anoms = {}

    # Initialize a dictionary to store the number of ensemble members
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years(model_data, models)

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Set the ensemble members count to zero
        # if model is not in the ensemble members count dictionary
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # depending on the NAO type
            # set up the region grid
            if nao_type == "default":
                print("Calculating model NAO index using default definition")

                # Set up the dict for the southern box
                south_gridbox_dict = azores_grid
                # Set up the dict for the northern box
                north_gridbox_dict = iceland_grid
            elif nao_type == "snao":
                print("Calculating model NAO index using SNAO definition")

                # Set up the dict for the southern box
                south_gridbox_dict = snao_south_grid
                # Set up the dict for the northern box
                north_gridbox_dict = snao_north_grid
            else:
                print("Invalid NAO type")
                sys.exit()

            # Extract the attributes
            member_id = member.attrs["variant_label"]

            # Print the model and member id
            print("calculated NAO for model", model, "member", member_id)

            # Extract the attributes from the member
            attributes = member.attrs

            # Extract the lat and lon values
            # from the gridbox dictionary
            # first for the southern box
            s_lon1, s_lon2 = south_gridbox_dict["lon1"], south_gridbox_dict["lon2"]
            s_lat1, s_lat2 = south_gridbox_dict["lat1"], south_gridbox_dict["lat2"]

            # second for the northern box
            n_lon1, n_lon2 = north_gridbox_dict["lon1"], north_gridbox_dict["lon2"]
            n_lat1, n_lat2 = north_gridbox_dict["lat1"], north_gridbox_dict["lat2"]

            # Take the mean over the lat and lon values
            # for the southern box for the ensemble member
            try:
                south_gridbox_mean = member.sel(lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)).mean(dim=["lat", "lon"])
                north_gridbox_mean = member.sel(lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)).mean(dim=["lat", "lon"])
            except Exception as e:
                print(f"Error taking gridbox mean: {e}")
                sys.exit()

            # Calculate the NAO index for the ensemble member
            try:
                nao_index = south_gridbox_mean - north_gridbox_mean
            except Exception as e:
                print(f"Error calculating NAO index: {e}")
                sys.exit()

            # Extract the years
            years = nao_index.time.dt.year.values

            # Associate the attributes with the NAO index
            nao_index.attrs = attributes

            # If model is not in the ensemble_members_nao_anoms
            # then add it to the ensemble_members_nao_anoms
            if model not in ensemble_members_nao_anoms:
                ensemble_members_nao_anoms[model] = []

            # Append the ensemble member to the list of ensemble members
            ensemble_members_nao_anoms[model].append(nao_index)

            # Increment the count of ensemble members for the model
            ensemble_members_count[model] += 1

    return ensemble_members_nao_anoms, years, ensemble_members_count

# Write a function which reads in a cube of anomaly fields for the model data
# /home/users/benhutch/skill-maps-processed-data/psl/HadGEM3-GC31-MM/global/years_2-9/DJFM/outputs/mergetime
def load_model_cube(variable, region, season, forecast_range):
    """
    Loads the model cube of anomaly fields. 
    For all models, but a specific variable,
    region, season and forecast range.
    
    Parameters
    ----------
    variable : str
        Variable name.
    region : str
        Region name.
    season : str
        Season name.
    forecast_range : str
        Forecast range.
        
        Returns
        -------
    anom_mm: iris.cube.Cube
        Anomaly field. Multi-model ensemble.
    """

    # Form the path to the model cube
    model_path = f"/home/users/benhutch/skill-maps-processed-data/{variable}/*/{region}/years_{forecast_range}/{season}/outputs/mergetime/*.nc"

    # If there are no files which match the model path, then exit
    if len(glob.glob(model_path)) == 0:
        print('There are no files which match the model path')
        sys.exit()

    # Load the list of cubes into a single cube
    anom_cubes = iris.load(model_path)

    # Unify the time units
    unify_time_units(anom_cubes)

    # Remove attributes from the cubes
    removed_attributes = equalise_attributes(anom_cubes)

    # Merge the cubes into a single cube
    anom_mm = anom_cubes.merge_cube()

    return anom_mm

def main():
    """
    Main function. For testing purposes.
    """

    # Set up the variables for testing
    models = dic.psl_full_models
    output_dir = dic.plots_dir_canari

    # Extract the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('variable', type=str, help='Variable name')
    parser.add_argument('region', type=str, help='Region name')
    parser.add_argument('season', type=str, help='Season name')
    parser.add_argument('forecast_range', type=str, help='Forecast range')
    parser.add_argument('start_year', type=str, help='Start year')
    parser.add_argument('end_year', type=str, help='End year')
    parser.add_argument('observations_path', type=str, help='Path to the observations')
    parser.add_argument('level', type=str, help='Level name, if applicable')

    # Extract the arguments
    args = parser.parse_args()

    # If level is not numeric, then set to None
    if not args.level.isnumeric():
        args.level = None

    # Test the processing of the observations
    obs_anomaly = read_obs(args.variable, args.region, args.forecast_range, 
                            args.season, args.observations_path, args.start_year,
                                args.end_year, level=args.level)
    
    # Load and process the model data
    datasets = load_data(dic.base_dir_skm_pro, models, args.variable, args.region,
                            args.forecast_range, args.season)

    # Process the model data
    model_anomaly, _ = process_data(datasets, args.variable)

    # Make sure that the model and obs have the same time period
    model_anomaly = constrain_years(model_anomaly, models)

    # Remove years containing nans from the observations and align the time periods
    # for the observations and model
    obs_anomaly, model_anomaly = remove_years_with_nans(obs_anomaly, model_anomaly, models)

    # Calculate the NAO index
    obs_nao, model_nao = calculate_nao_index_and_plot(obs_anomaly, model_anomaly, models, args.variable, args.season,
                                                            args.forecast_range, output_dir, plot_graphics=False)

    # Test the NAO rescaling function
    rescaled_nao, ensemble_mean_nao, ensemble_members_nao = rescale_nao(obs_nao, model_nao, models, args.season,
                                                                        args.forecast_range, output_dir, lagged=False)
    
    # Calculate the closest members
    # test case set year
    year = 1966
    # Calculate the closest members
    smallest_diff = calculate_closest_members(year, rescaled_nao, model_nao, models, args.season, args.forecast_range, 
                                output_dir, lagged=False, no_subset_members=20)


if __name__ == '__main__':
    main()