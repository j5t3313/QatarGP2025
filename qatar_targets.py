import numpy as np
from qatar_config import QATAR_PARAMS
from qatar_tire_model import get_lap_time


def calculate_pit_threshold(current_compound, new_compound, laps_remaining, compound_models):
    pit_loss = QATAR_PARAMS['pit_time_loss']
    
    new_tire_lap1 = get_lap_time(new_compound, 1, compound_models)
    
    avg_new_tire_time = 0
    for lap in range(1, min(laps_remaining + 1, 26)):
        avg_new_tire_time += get_lap_time(new_compound, lap, compound_models)
    avg_new_tire_time /= min(laps_remaining, 25)
    
    pit_cost_per_lap = pit_loss / laps_remaining
    
    threshold = avg_new_tire_time + pit_cost_per_lap
    
    return {
        'threshold': threshold,
        'new_tire_lap1': new_tire_lap1,
        'avg_new_tire': avg_new_tire_time,
        'pit_cost_per_lap': pit_cost_per_lap,
    }


def generate_pit_thresholds(compound_models):
    laps_remaining_values = [5, 10, 15, 20, 25, 30, 35, 40]
    
    transitions = [
        ('SOFT', 'MEDIUM'),
        ('SOFT', 'HARD'),
        ('MEDIUM', 'HARD'),
        ('MEDIUM', 'SOFT'),
        ('HARD', 'MEDIUM'),
        ('HARD', 'SOFT'),
    ]
    
    all_thresholds = {}
    
    for from_comp, to_comp in transitions:
        key = f"{from_comp}->{to_comp}"
        all_thresholds[key] = {}
        
        for laps_remaining in laps_remaining_values:
            result = calculate_pit_threshold(from_comp, to_comp, laps_remaining, compound_models)
            all_thresholds[key][laps_remaining] = result['threshold']
    
    return all_thresholds


def print_pit_thresholds(thresholds):
    print("\n" + "=" * 70)
    print("TARGET LAP TIMES: Pit when slower than threshold")
    print("=" * 70)
    
    laps_values = sorted(list(list(thresholds.values())[0].keys()))
    
    header = f"{'Transition':<18}"
    for laps in laps_values:
        header += f"{laps:>7}"
    header += "  laps remaining"
    print(header)
    print("-" * 70)
    
    for transition, values in thresholds.items():
        row = f"{transition:<18}"
        for laps in laps_values:
            row += f"{values[laps]:>7.2f}"
        print(row)


def get_optimal_pit_lap(compound, stint_start_lap, compound_models, next_compound, max_stint=25):
    num_laps = QATAR_PARAMS['num_laps']
    
    best_pit_lap = stint_start_lap + max_stint
    best_total_time = float('inf')
    
    for pit_lap in range(stint_start_lap + 7, min(stint_start_lap + max_stint + 1, num_laps - 6)):
        stint1_time = 0
        for lap in range(1, pit_lap - stint_start_lap + 1):
            stint1_time += get_lap_time(compound, lap, compound_models)
        
        stint1_time += QATAR_PARAMS['pit_time_loss']
        
        remaining = num_laps - pit_lap
        stint2_time = 0
        for lap in range(1, remaining + 1):
            stint2_time += get_lap_time(next_compound, lap, compound_models)
        
        total = stint1_time + stint2_time
        
        if total < best_total_time:
            best_total_time = total
            best_pit_lap = pit_lap
    
    return best_pit_lap, best_total_time