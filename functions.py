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
import pdb
import datetime
import iris.quickplot as qplt

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

    def process_model_dataset(dataset, variable):
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
                # Process the dataset.
                variable_data, model_time = process_model_dataset(dataset, variable)
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
    start_date = datetime.datetime(int(start_year), 12, 1)
    end_date = datetime.datetime(int(end_year), 3, 31)
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
def remove_years_with_nans(observed_data, model_data, models, variable):
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
            # Extract the years
            model_years = member.time.dt.year.values

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

    return observed_data, model_data


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
    
    # Test the loading of the model cube
    anom_mm = load_model_cube(args.variable, args.region, args.season, args.forecast_range)


if __name__ == '__main__':
    main()