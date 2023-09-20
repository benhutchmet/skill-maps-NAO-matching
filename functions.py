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
    obs_anomaly = calculate_annual_mean_anomalies(obs_anomaly, season)

    # Return the anomaly field
    return obs_anomaly


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


if __name__ == '__main__':
    main()