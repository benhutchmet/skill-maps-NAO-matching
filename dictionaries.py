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