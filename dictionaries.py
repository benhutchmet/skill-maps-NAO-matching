# Dictionaries

# Define a dictionary to map the variable names to their corresponding names in the obs dataset
var_name_map = {
    "psl": "msl",
    "tas": "t2m",
    "sfcWind": "si10",
    "rsds": "ssrd",
    "tos": "sst",
    "ua": "var131",
    "va": "var132"
}

obs_ws_var_names = ["ua", "va", "var131", "var132"]

# Define a dictionary to map the season strings to their corresponding months
season_month_map = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JJAS": [6, 7, 8, 9],
    "SON": [9, 10, 11],
    "SOND": [9, 10, 11, 12],
    "NDJF": [11, 12, 1, 2],
    "DJFM": [12, 1, 2, 3],
    "djfm": [12, 1, 2, 3]
}

obs_path = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

psl_full_models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1" ]

base_dir_skm_pro = "/home/users/benhutch/skill-maps-processed-data"

save_dir = "/gws/nopw/j04/canari/users/benhutch/NAO-matching"

# Set up the tas models
tas_models = ["BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1"]

# Set up the sfcWind models
sfcWind_models = ["BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "HadGEM3-GC31-MM", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5"]

# Set up the rsds models
rsds_models = ["BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1"]

# Set up the grids for the skill maps
# Define the dimensions for the gridbox for the azores
azores_grid = {
    'lon1': -28,
    'lon2': -20,
    'lat1': 36,
    'lat2': 40
}

# Define the dimensions for the gridbox for the azores
iceland_grid = {
    'lon1': -25,
    'lon2': -16,
    'lat1': 63,
    'lat2': 70
}

# Define the dimensions for the summertime NAO (SNAO) southern pole
# As defined in Wang and Ting 2022
# This is the pointwise definition of the SNAO
# Which is well correlated with the EOF definition from Folland et al. 2009
snao_south_grid = {
    'lon1': -25, # degrees west
    'lon2': 5, # degrees east
    'lat1': 45,
    'lat2': 55
}

# Define the dimensions for the summertime NAO (SNAO) northern pole
# As defined in Wang and Ting 2022
snao_north_grid = {
    'lon1': -52, # degrees west
    'lon2': -22, # degrees west
    'lat1': 60,
    'lat2': 70
}

# Define the dimensions for the gridbox for the N-S UK index
# From thornton et al. 2019
uk_n_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 57,
    'lat2': 70
}

# And for the southern box
uk_s_box = {
    'lon1': 153,
    'lon2': 201,
    'lat1': 38,
    'lat2': 51
}


# plots directory canari
plots_dir_canari = "/gws/nopw/j04/canari/users/benhutch/plots"