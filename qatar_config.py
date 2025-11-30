import numpy as np

QATAR_PARAMS = {
    'circuit_name': 'Lusail International Circuit',
    'gp_name': 'Qatar Grand Prix',
    'base_pace': 84.0,
    'num_laps': 57,
    'mandatory_pit_stops': 2,
    'max_stint_length': 25,
    'min_stint_length': 7,
    'min_compounds_required': 2,
    'rain_probability': 0.00,
    'sc_probability': 0.67,
    'vsc_probability': 0.67,
    'pit_time_loss': 26.3,
    'pit_time_std': 1.5,
    'fuel_load_kg': 110,
    'fuel_effect_per_kg': 0.035,
    'track_evolution_rate': -0.003,
    'dirty_air_effect': 0.35,
    'drs_zones': 2,
    'drs_effectiveness_factor': 1.0,
}

QATAR_PARAMS['fuel_consumption_rate'] = QATAR_PARAMS['fuel_load_kg'] / QATAR_PARAMS['num_laps']

COMPOUND_COLORS = {
    'HARD': '#808080',
    'MEDIUM': '#FFD700',
    'SOFT': '#FF0000'
}

GRID_POSITIONS = [1, 3, 5, 8, 10]

GRID_TO_DRIVER = {
    1: 'PIA',
    3: 'VER',
    5: 'ANT',
    8: 'ALO',
    10: 'LEC',
}

GRID_TO_TEAM_FACTOR = {
    1: 1.000,
    2: 1.000,
    3: 1.008,
    4: 1.006,
    5: 1.006,
    6: 1.015,
    7: 1.009,
    8: 1.016,
    9: 1.015,
    10: 1.009,
}

POSITION_PENALTIES = {
    1: 0.00, 2: 0.08, 3: 0.15, 4: 0.22, 5: 0.28,
    6: 0.33, 7: 0.38, 8: 0.42, 9: 0.46, 10: 0.50,
    11: 0.53, 12: 0.56, 13: 0.58, 14: 0.60, 15: 0.62,
    16: 0.63, 17: 0.64, 18: 0.65, 19: 0.66, 20: 0.67,
}

DRS_EFFECTIVENESS = {
    'mean_advantage': 0.40,
    'median_advantage': 0.35,
    'std_advantage': 0.12,
    'usage_probability': 0.35,
}

DRIVER_ERROR_RATE = 0.006

TIRE_DEFAULTS = {
    'SOFT': {'offset': 0.0, 'degradation_rate': 0.12},
    'MEDIUM': {'offset': 0.40, 'degradation_rate': 0.07},
    'HARD': {'offset': 0.80, 'degradation_rate': 0.04},
}

TIRE_ALLOCATION = {
    'PIA': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'NOR': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'VER': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'RUS': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'ANT': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'HAD': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'SAI': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'ALO': {'SOFT': 2, 'MEDIUM': 3, 'HARD': 1},
    'GAS': {'SOFT': 0, 'MEDIUM': 3, 'HARD': 1},  
    'LEC': {'SOFT': 0, 'MEDIUM': 3, 'HARD': 1},
    'HUL': {'SOFT': 3, 'MEDIUM': 3, 'HARD': 1},
    'LAW': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'BEA': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'BOR': {'SOFT': 2, 'MEDIUM': 3, 'HARD': 1},
    'ALB': {'SOFT': 2, 'MEDIUM': 3, 'HARD': 1},
    'TSU': {'SOFT': 5, 'MEDIUM': 3, 'HARD': 1},
    'OCO': {'SOFT': 2, 'MEDIUM': 3, 'HARD': 1},
    'HAM': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},
    'STR': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},  
    'COL': {'SOFT': 1, 'MEDIUM': 3, 'HARD': 1},  
}