import numpy as np
import pandas as pd
from tqdm import tqdm

from qatar_config import GRID_POSITIONS, GRID_TO_DRIVER, QATAR_PARAMS
from qatar_tire_model import build_tire_models
from qatar_simulation import generate_strategies, run_simulations, rank_strategies, simulate_race
from qatar_targets import generate_pit_thresholds, print_pit_thresholds
from qatar_visualizations import (
    plot_strategy_rankings, 
    plot_tire_degradation,
    plot_pit_thresholds,
    plot_strategy_distribution
)


def run_analysis(year=2025, num_sims=500):
    print("=" * 70)
    print(f"QATAR GP {year} STRATEGY ANALYSIS")
    print("=" * 70)
    
    print(f"\nCircuit: {QATAR_PARAMS['circuit_name']}")
    print(f"Race length: {QATAR_PARAMS['num_laps']} laps")
    print(f"Mandatory stops: {QATAR_PARAMS['mandatory_pit_stops']}")
    print(f"Max stint: {QATAR_PARAMS['max_stint_length']} laps")
    print(f"Pit loss: {QATAR_PARAMS['pit_time_loss']}s")
    
    print("\n" + "-" * 70)
    compound_models = build_tire_models(year)
    
    print("\n" + "-" * 70)
    print("TIRE MODEL SUMMARY")
    print("-" * 70)
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound in compound_models:
            m = compound_models[compound]
            print(f"  {compound}: base={m['alpha']:.2f}s, deg={m['beta']:.4f}s/lap")
    
    all_rankings = {}
    all_strategies = {}
    
    print("\n" + "-" * 70)
    print("RUNNING SIMULATIONS")
    print("-" * 70)
    
    for grid_pos in GRID_POSITIONS:
        driver = GRID_TO_DRIVER.get(grid_pos, 'UNKNOWN')
        print(f"\nP{grid_pos} ({driver}):")
        
        strategies = generate_strategies(driver)
        print(f"  Valid strategies: {len(strategies)}")
        all_strategies[grid_pos] = strategies
        
        print(f"  Running {num_sims} simulations per strategy...")
        results = {}
        for strategy in tqdm(strategies, desc=f"  P{grid_pos}", leave=False):
            times = []
            for _ in range(num_sims):
                race_time = simulate_race(strategy, grid_pos, compound_models)
                times.append(race_time)
            
            results[strategy['name']] = {
                'strategy': strategy,
                'times': times,
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'median_time': np.median(times),
            }
        
        rankings = rank_strategies(results)
        all_rankings[grid_pos] = rankings
        
        print(f"  Top 5 strategies:")
        for r in rankings[:5]:
            print(f"    {r['rank']}. {r['name']}: {r['avg_time']:.1f}s (±{r['std_time']:.1f}s)")
    
    print("\n" + "=" * 70)
    print("STRATEGY RANKINGS BY GRID POSITION")
    print("=" * 70)
    
    for grid_pos in GRID_POSITIONS:
        driver = GRID_TO_DRIVER.get(grid_pos, 'UNKNOWN')
        rankings = all_rankings[grid_pos]
        
        print(f"\nP{grid_pos} ({driver}) - Top 10:")
        print("-" * 50)
        
        fastest = rankings[0]['avg_time']
        for r in rankings[:10]:
            delta = r['avg_time'] - fastest
            print(f"  {r['rank']:2}. {r['name']:<20} {r['avg_time']:.1f}s (+{delta:.1f}s)")
    
    print("\n" + "=" * 70)
    thresholds = generate_pit_thresholds(compound_models)
    print_pit_thresholds(thresholds)
    
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATIONS")
    print("-" * 70)
    
    fig1 = plot_strategy_rankings(all_rankings)
    fig1.savefig('qatar_strategy_rankings.png', dpi=150, bbox_inches='tight')
    print("  Saved: qatar_strategy_rankings.png")
    
    fig2 = plot_tire_degradation(compound_models)
    fig2.savefig('qatar_tire_degradation.png', dpi=150, bbox_inches='tight')
    print("  Saved: qatar_tire_degradation.png")
    
    fig3 = plot_pit_thresholds(thresholds)
    fig3.savefig('qatar_pit_thresholds.png', dpi=150, bbox_inches='tight')
    print("  Saved: qatar_pit_thresholds.png")
    
    fig4 = plot_strategy_distribution(all_rankings)
    fig4.savefig('qatar_strategy_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved: qatar_strategy_patterns.png")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return {
        'compound_models': compound_models,
        'all_rankings': all_rankings,
        'all_strategies': all_strategies,
        'thresholds': thresholds,
    }


if __name__ == "__main__":
    results = run_analysis(year=2025, num_sims=500)