# Qatar GP 2025 Pit Strategy Model

Monte Carlo simulation for Formula 1 pit stop strategy optimization at the 2025 Qatar Grand Prix.

## Qatar-Specific Constraints

| Parameter | Value |
|-----------|-------|
| Race Length | 57 laps |
| Mandatory Stops | 2 (3 stints minimum) |
| Max Stint Length | 25 laps |
| Min Stint Length | 7 laps |
| Mandatory Compounds | Hard (C1) AND Medium (C2) |
| Pit Time Loss | 26.3s |
| SC Probability | 67% |
| VSC Probability | 67% |

## Features

- Bayesian tire degradation modeling from FP1, Sprint Qualifying, Sprint, and Qualifying data
- Medium compound anchor with Pirelli offsets for soft/hard base pace
- Fuel-corrected lap times (0.035s/kg)
- Driver-specific strategy filtering based on remaining tire allocation
- Target lap time calculations for pit decision making
- Monte Carlo simulation with safety car modeling

## Installation

```bash
pip install fastf1 numpyro jax pandas numpy matplotlib tqdm
```

## Usage

```bash
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

| Transition | 10 laps | 20 laps | 30 laps |
|------------|---------|---------|---------|
| MEDIUM→HARD | 93.3s | 92.1s | 91.8s |

Pit when your current lap time exceeds the threshold for your remaining laps.

### Visualizations

- `qatar_strategy_rankings.png`
- `qatar_tire_degradation.png`
- `qatar_pit_thresholds.png`
- `qatar_strategy_patterns.png`

## Project Structure

```
qatar_config.py         Circuit parameters, tire allocation, grid mapping
qatar_tire_model.py     Bayesian tire degradation from practice data
qatar_simulation.py     Strategy generation, race simulation
qatar_targets.py        Pit threshold calculations
qatar_visualizations.py Plotting functions
qatar_main.py           Entry point
```

## Methodology

### Tire Model

Fits a linear degradation model per compound:

```
lap_time = α + β × lap_in_stint
```

Uses TruncatedNormal prior on β to enforce non-negative degradation. Medium compound is fitted directly from cross-session data; soft and hard base pace use Pirelli offsets anchored to medium, with degradation rates fitted from available data.

Final model (2025 Qatar):
- Soft: α = 89.40s, β = 0.150 s/lap
- Medium: α = 89.95s, β = 0.045 s/lap
- Hard: α = 90.55s, β = 0.025 s/lap

### Race Simulation

Each simulation calculates total race time including:

- Tire degradation sampled from Bayesian posterior
- Fuel load effect (110kg initial, 1.93kg/lap burn rate, 0.035s/kg)
- Position-based dirty air penalties (0.0s to 0.67s)
- DRS advantage (0.35s median, 35% probability)
- Driver error (0.6% probability, 0.8-2.5s)
- Safety car and VSC periods
- Pit stop time with SC/VSC windows (15% loss under SC)

### Strategy Generation

Enumerates valid 3-stint strategies:

- Must use both hard (C1) AND medium (C2) compounds
- Soft (C3) is optional
- 7-25 lap stint lengths
- Total = 57 laps
- Filtered by driver's remaining tire allocation

Samples 5 representative strategies per compound pattern.

### Target Lap Times

```
threshold = avg_new_tire_time + pit_loss / laps_remaining
```

If current lap time > threshold, pit now.

## Data Sources

- Lap times: FastF1 API (FP1, Sprint Qualifying, Sprint, Qualifying)
- Circuit parameters: FIA event documents
- Compound offsets: Pirelli preview documents
- Tire allocation: Post-qualifying reports
