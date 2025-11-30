import numpy as np
import matplotlib.pyplot as plt
from qatar_config import COMPOUND_COLORS, GRID_POSITIONS


def plot_strategy_rankings(all_rankings, top_n=10):
    n_positions = len(all_rankings)
    fig, axes = plt.subplots(1, n_positions, figsize=(5 * n_positions, 8))
    
    if n_positions == 1:
        axes = [axes]
    
    for idx, (grid_pos, rankings) in enumerate(all_rankings.items()):
        ax = axes[idx]
        
        top = rankings[:top_n]
        names = [r['name'] for r in top]
        times = [r['avg_time'] for r in top]
        
        fastest = min(times)
        deltas = [t - fastest for t in times]
        
        colors = []
        for r in top:
            stints = r['strategy']['stints']
            first_compound = stints[0]['compound']
            colors.append(COMPOUND_COLORS.get(first_compound, '#999999'))
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, deltas, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Delta to fastest (s)')
        ax.set_title(f'Grid P{grid_pos}')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (delta, time) in enumerate(zip(deltas, times)):
            ax.text(delta + 0.1, i, f'+{delta:.1f}s', va='center', fontsize=8)
    
    plt.suptitle('Strategy Rankings by Grid Position', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_tire_degradation(compound_models, max_laps=25):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    laps = np.arange(1, max_laps + 1)
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound in compound_models:
            model = compound_models[compound]
            times = [model['alpha'] + model['beta'] * lap for lap in laps]
            ax.plot(laps, times, label=compound, color=COMPOUND_COLORS[compound], linewidth=2)
    
    ax.set_xlabel('Lap in Stint')
    ax.set_ylabel('Lap Time (s)')
    ax.set_title('Tire Degradation by Compound')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pit_thresholds(thresholds):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    transitions = list(thresholds.keys())
    laps_values = sorted(list(thresholds[transitions[0]].keys()))
    
    x = np.arange(len(laps_values))
    width = 0.12
    
    for i, transition in enumerate(transitions):
        values = [thresholds[transition][laps] for laps in laps_values]
        offset = (i - len(transitions) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=transition, alpha=0.8)
    
    ax.set_xlabel('Laps Remaining')
    ax.set_ylabel('Pit Threshold Lap Time (s)')
    ax.set_title('Pit When Slower Than Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(laps_values)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_strategy_distribution(all_rankings, top_n=20):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pattern_counts = {}
    
    for grid_pos, rankings in all_rankings.items():
        for r in rankings[:top_n]:
            stints = r['strategy']['stints']
            pattern = '-'.join([s['compound'][0] for s in stints])
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1
    
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])
    
    patterns = [p[0] for p in sorted_patterns]
    counts = [p[1] for p in sorted_patterns]
    
    ax.bar(patterns, counts, alpha=0.8)
    ax.set_xlabel('Strategy Pattern')
    ax.set_ylabel('Appearances in Top 20')
    ax.set_title('Most Common Strategies Across Grid Positions')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig