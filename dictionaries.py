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
    "DJFM": [12, 1, 2, 3]
}
