# Qatar GP 2025 Pit Strategy Model

Monte Carlo simulation for Formula 1 pit stop strategy optimization at the 2025 Qatar Grand Prix.

## Qatar-Specific Constraints

| Parameter | Value |
|-----------|-------|
| Race Length | 57 laps |
| Mandatory Stops | 2 (3 stints minimum) |
| Max Stint Length | 25 laps |
| Min Stint Length | 7 laps |
| Pit Time Loss | 26.3s |
| SC Probability | 67% |
| VSC Probability | 67% |

## Features

- Bayesian tire degradation modeling from FP1 and Sprint data
- Driver-specific strategy filtering based on remaining tire allocation
- Target lap time calculations for pit decision making
- Monte Carlo simulation with safety car modeling

## Installation

```bash
pip install fastf1 numpyro jax pandas numpy matplotlib tqdm
```

## Usage

```python
python qatar_main.py
```

## Configuration

Edit `qatar_config.py` to update:

- `GRID_TO_DRIVER`: Maps grid positions to driver codes
- `TIRE_ALLOCATION`: Available tire sets per driver after qualifying
- `GRID_TO_TEAM_FACTOR`: Car performance factors by grid position

## Outputs

### Strategy Rankings

Top 10 strategies ranked by average race time for each grid position (P1, P3, P5, P8, P10).

### Target Lap Times

Table showing pit thresholds by laps remaining:

| Transition | 5 laps | 10 laps | 15 laps | ... |
|------------|--------|---------|---------|-----|
| SOFT→MEDIUM | 87.2s | 86.1s | 85.7s | ... |
| MEDIUM→HARD | 86.8s | 85.9s | 85.5s | ... |

Pit when your current lap time exceeds the threshold for your remaining laps.

### Visualizations

- `qatar_strategy_rankings.png` - Strategy rankings by grid position
- `qatar_tire_degradation.png` - Tire degradation curves by compound
- `qatar_pit_thresholds.png` - Pit decision thresholds
- `qatar_strategy_patterns.png` - Most common strategy patterns

## Project Structure

```
qatar_config.py        - Circuit parameters, tire allocation, grid mapping
qatar_tire_model.py    - Bayesian tire degradation from practice data
qatar_simulation.py    - Strategy generation, race simulation
qatar_targets.py       - Pit threshold calculations
qatar_visualizations.py - Plotting functions
qatar_main.py          - Entry point
```

## Methodology

### Tire Model

Fits a linear degradation model per compound using FP1 and Sprint lap times:

```
lap_time = α + β × lap_in_stint
```

Uses truncated normal prior on β (degradation rate) to enforce non-negative values. Falls back to default parameters if insufficient data.

### Race Simulation

Each simulation calculates total race time including:

- Tire degradation per stint
- Fuel load effect (heavier = slower, burns off over race)
- Track evolution
- Position-based dirty air penalties
- DRS advantage probability
- Safety car and VSC periods
- Pit stop time with SC/VSC windows
- Driver error probability

### Strategy Generation

Enumerates valid 3-stint strategies respecting:

- Driver's available tire sets
- Minimum 2 different compounds
- 7-25 lap stint lengths
- Total = 57 laps

Samples representative strategies per compound pattern to limit simulation count.

### Target Lap Times

Calculates the lap time threshold where pitting becomes beneficial:

```
threshold = avg_new_tire_time + (pit_loss / laps_remaining)
```

If current lap time > threshold, pit now.

## Data Sources

- Tire data: FastF1 API (FP1, Sprint sessions)
- Circuit parameters: FIA race documents, Pirelli data
- Tire allocation: Post-qualifying reports
