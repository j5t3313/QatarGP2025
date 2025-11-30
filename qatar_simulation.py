import numpy as np
from itertools import product
from collections import Counter

from qatar_config import (
    QATAR_PARAMS, GRID_TO_DRIVER, GRID_TO_TEAM_FACTOR,
    POSITION_PENALTIES, DRS_EFFECTIVENESS, DRIVER_ERROR_RATE,
    TIRE_ALLOCATION
)
from qatar_tire_model import get_lap_time, get_lap_time_with_uncertainty


def generate_strategies(driver, sample_per_pattern=5):
    num_laps = QATAR_PARAMS['num_laps']
    max_stint = QATAR_PARAMS['max_stint_length']
    min_stint = QATAR_PARAMS['min_stint_length']
    
    allocation = TIRE_ALLOCATION.get(driver, {'SOFT': 2, 'MEDIUM': 3, 'HARD': 2})
    
    patterns = {}
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    
    for c1, c2, c3 in product(compounds, repeat=3):
        compound_counts = Counter([c1, c2, c3])
        
        valid_allocation = True
        for comp, count in compound_counts.items():
            if count > allocation.get(comp, 0):
                valid_allocation = False
                break
        
        if not valid_allocation:
            continue
        
        # Must use at least 2 different compounds
        if len(set([c1, c2, c3])) < 2:
            continue
        
        # Mandatory: must use both HARD and MEDIUM (C1 and C2)
        compounds_used = set([c1, c2, c3])
        if 'HARD' not in compounds_used or 'MEDIUM' not in compounds_used:
            continue
        
        pattern_key = f"{c1[0]}-{c2[0]}-{c3[0]}"
        if pattern_key not in patterns:
            patterns[pattern_key] = []
        
        for s1 in range(min_stint, max_stint + 1):
            remaining_after_s1 = num_laps - s1
            
            for s2 in range(min_stint, min(max_stint + 1, remaining_after_s1 - min_stint + 1)):
                s3 = num_laps - s1 - s2
                
                if s3 < min_stint or s3 > max_stint:
                    continue
                
                strategy = {
                    'name': f"{c1[0]}{s1}-{c2[0]}{s2}-{c3[0]}{s3}",
                    'stints': [
                        {'compound': c1, 'laps': s1},
                        {'compound': c2, 'laps': s2},
                        {'compound': c3, 'laps': s3},
                    ]
                }
                patterns[pattern_key].append(strategy)
    
    strategies = []
    for pattern_key, pattern_strategies in patterns.items():
        if len(pattern_strategies) <= sample_per_pattern:
            strategies.extend(pattern_strategies)
        else:
            indices = np.linspace(0, len(pattern_strategies) - 1, sample_per_pattern, dtype=int)
            for idx in indices:
                strategies.append(pattern_strategies[idx])
    
    return strategies


def generate_sc_periods(num_laps):
    sc_laps = set()
    vsc_laps = set()
    
    if np.random.rand() < QATAR_PARAMS['sc_probability']:
        sc_start = np.random.randint(1, num_laps - 5)
        sc_duration = np.random.randint(3, 6)
        sc_laps.update(range(sc_start, min(sc_start + sc_duration, num_laps + 1)))
    
    if np.random.rand() < QATAR_PARAMS['vsc_probability']:
        vsc_start = np.random.randint(1, num_laps - 3)
        if vsc_start not in sc_laps:
            vsc_duration = np.random.randint(2, 4)
            vsc_laps.update(range(vsc_start, min(vsc_start + vsc_duration, num_laps + 1)))
    
    return sc_laps, vsc_laps


def simulate_race(strategy, grid_position, compound_models):
    num_laps = QATAR_PARAMS['num_laps']
    pit_loss = QATAR_PARAMS['pit_time_loss']
    pit_std = QATAR_PARAMS['pit_time_std']
    fuel_load = QATAR_PARAMS['fuel_load_kg']
    fuel_rate = QATAR_PARAMS['fuel_consumption_rate']
    fuel_effect = QATAR_PARAMS['fuel_effect_per_kg']
    track_evolution = QATAR_PARAMS['track_evolution_rate']
    
    car_factor = GRID_TO_TEAM_FACTOR.get(grid_position, 1.01)
    
    sc_laps, vsc_laps = generate_sc_periods(num_laps)
    
    race_time = 0.0
    current_lap = 1
    current_position = grid_position
    
    for stint_idx, stint in enumerate(strategy['stints']):
        compound = stint['compound']
        stint_length = stint['laps']
        
        for lap_in_stint in range(1, stint_length + 1):
            if current_lap > num_laps:
                break
            
            lap_time = get_lap_time_with_uncertainty(compound, lap_in_stint, compound_models)
            
            lap_time *= car_factor
            
            lap_time += track_evolution * current_lap
            
            fuel_penalty = fuel_load * fuel_effect
            lap_time += fuel_penalty
            fuel_load = max(0, fuel_load - fuel_rate)
            
            pos_penalty = POSITION_PENALTIES.get(min(current_position, 20), 0.67)
            lap_time += pos_penalty
            
            if current_position > 1 and current_lap not in sc_laps and current_lap not in vsc_laps:
                if np.random.rand() < DRS_EFFECTIVENESS['usage_probability']:
                    drs_gain = np.random.normal(
                        DRS_EFFECTIVENESS['median_advantage'],
                        DRS_EFFECTIVENESS['std_advantage']
                    )
                    lap_time -= max(0.1, drs_gain)
            
            if compound == 'SOFT' and lap_in_stint > 12:
                lap_time += min(0.8, (lap_in_stint - 12) * 0.04)
            
            if compound == 'HARD' and lap_in_stint < 5:
                lap_time += max(0, (5 - lap_in_stint) * 0.08)
            
            if np.random.rand() < DRIVER_ERROR_RATE:
                lap_time += np.random.uniform(0.8, 2.5)
            
            if current_lap in sc_laps:
                lap_time *= 1.40
            elif current_lap in vsc_laps:
                lap_time *= 1.20
            
            race_time += lap_time
            current_lap += 1
        
        if stint_idx < len(strategy['stints']) - 1:
            pit_time = np.random.normal(pit_loss, pit_std)
            
            if any(lap in sc_laps for lap in range(max(1, current_lap - 2), current_lap + 1)):
                pit_time *= 0.15
            elif any(lap in vsc_laps for lap in range(max(1, current_lap - 2), current_lap + 1)):
                pit_time *= 0.50
            
            race_time += max(15.0, pit_time)
            
            position_change = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.25, 0.4, 0.2, 0.05])
            current_position = max(1, min(20, current_position + position_change))
    
    return race_time


def run_simulations(strategies, grid_position, compound_models, num_sims=500):
    results = {}
    
    for strategy in strategies:
        times = []
        for _ in range(num_sims):
            race_time = simulate_race(strategy, grid_position, compound_models)
            times.append(race_time)
        
        results[strategy['name']] = {
            'strategy': strategy,
            'times': times,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'median_time': np.median(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
        }
    
    return results


def rank_strategies(results):
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'])
    
    rankings = []
    for rank, (name, data) in enumerate(sorted_results, 1):
        rankings.append({
            'rank': rank,
            'name': name,
            'avg_time': data['avg_time'],
            'std_time': data['std_time'],
            'median_time': data['median_time'],
            'strategy': data['strategy'],
        })
    
    return rankings