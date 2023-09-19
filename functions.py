# Functions used for the lagging and variance adjustment and NAO matching techniques (Doug Smith's method)
# With help from Hazel Thornton's code

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
cdo = Cdo()


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
def read_obs(variable, region, forecast_range, season, observations_path, level=None):
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



def main():
    """
    Main function. For testing purposes.
    """

    # Extract the command line arguments


if __name__ == '__main__':
    main()