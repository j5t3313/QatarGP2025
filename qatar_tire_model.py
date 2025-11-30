import fastf1
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

try:
    import jax.numpy as jnp
    import jax.random as random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

from qatar_config import QATAR_PARAMS, TIRE_DEFAULTS


def load_practice_data(year, gp_name):
    print(f"Loading sessions for {year} {gp_name}...")
    
    all_laps = []
    
    fuel_corrections = {
        'FP1': 80,
        'SQ': 15,
        'Sprint': 50,
        'Q': 10,
    }
    
    try:
        fp1 = fastf1.get_session(year, gp_name, 'FP1')
        fp1.load()
        fp1_laps = fp1.laps.copy()
        fp1_laps['Session'] = 'FP1'
        fp1_laps['FuelLoad'] = fuel_corrections['FP1']
        
        fp1_clean = fp1_laps[
            (fp1_laps['LapTime'].notna()) &
            (fp1_laps['Compound'].notna()) &
            (~fp1_laps['PitOutTime'].notna()) &
            (~fp1_laps['PitInTime'].notna()) &
            (fp1_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(fp1_clean) > 0:
            all_laps.append(fp1_clean)
            print(f"  FP1: {len(fp1_clean)} clean laps")
    except Exception as e:
        print(f"  FP1: Could not load - {e}")
    
    try:
        sprint_quali = fastf1.get_session(year, gp_name, 'SQ')
        sprint_quali.load()
        sq_laps = sprint_quali.laps.copy()
        sq_laps['Session'] = 'SQ'
        sq_laps['FuelLoad'] = fuel_corrections['SQ']
        
        sq_clean = sq_laps[
            (sq_laps['LapTime'].notna()) &
            (sq_laps['Compound'].notna()) &
            (~sq_laps['PitOutTime'].notna()) &
            (~sq_laps['PitInTime'].notna()) &
            (sq_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(sq_clean) > 0:
            all_laps.append(sq_clean)
            print(f"  Sprint Qualifying: {len(sq_clean)} clean laps")
    except Exception as e:
        print(f"  Sprint Qualifying: Could not load - {e}")
    
    try:
        sprint = fastf1.get_session(year, gp_name, 'Sprint')
        sprint.load()
        sprint_laps = sprint.laps.copy()
        sprint_laps['Session'] = 'Sprint'
        sprint_laps['FuelLoad'] = fuel_corrections['Sprint']
        
        sprint_clean = sprint_laps[
            (sprint_laps['LapTime'].notna()) &
            (sprint_laps['Compound'].notna()) &
            (~sprint_laps['PitOutTime'].notna()) &
            (~sprint_laps['PitInTime'].notna()) &
            (sprint_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(sprint_clean) > 0:
            all_laps.append(sprint_clean)
            print(f"  Sprint: {len(sprint_clean)} clean laps")
    except Exception as e:
        print(f"  Sprint: Could not load - {e}")
    
    try:
        quali = fastf1.get_session(year, gp_name, 'Q')
        quali.load()
        q_laps = quali.laps.copy()
        q_laps['Session'] = 'Q'
        q_laps['FuelLoad'] = fuel_corrections['Q']
        
        q_clean = q_laps[
            (q_laps['LapTime'].notna()) &
            (q_laps['Compound'].notna()) &
            (~q_laps['PitOutTime'].notna()) &
            (~q_laps['PitInTime'].notna()) &
            (q_laps['TrackStatus'] == '1')
        ].copy()
        
        if len(q_clean) > 0:
            all_laps.append(q_clean)
            print(f"  Qualifying: {len(q_clean)} clean laps")
    except Exception as e:
        print(f"  Qualifying: Could not load - {e}")
    
    if not all_laps:
        print("  No session data available")
        return pd.DataFrame()
    
    combined = pd.concat(all_laps, ignore_index=True)
    combined['StintLap'] = combined.groupby(['Driver', 'Session', 'Stint']).cumcount() + 1
    combined['LapTime_s'] = combined['LapTime'].dt.total_seconds()
    
    fuel_effect = QATAR_PARAMS['fuel_effect_per_kg']
    combined['LapTime_corrected'] = combined['LapTime_s'] - (combined['FuelLoad'] * fuel_effect)
    
    print(f"  Total: {len(combined)} clean laps (fuel-corrected)")
    return combined


def fit_bayesian_model(x, y, compound_name, base_pace):
    beta_priors = {
        'SOFT': (0.12, 0.05),
        'MEDIUM': (0.07, 0.03),
        'HARD': (0.04, 0.02),
    }
    beta_mean, beta_std = beta_priors.get(compound_name, (0.07, 0.03))
    
    def tire_model(x, y=None):
        alpha = numpyro.sample("alpha", dist.Normal(base_pace, 2.0))
        beta = numpyro.sample("beta", dist.TruncatedNormal(beta_mean, beta_std, low=0.0))
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        mu = alpha + beta * x
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    
    kernel = NUTS(tire_model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=False)
    mcmc.run(random.PRNGKey(42), x, y)
    
    return mcmc


def build_tire_models(year=2025):
    gp_name = QATAR_PARAMS['gp_name']
    base_pace = QATAR_PARAMS['base_pace']
    
    practice_data = load_practice_data(year, gp_name)
    
    compound_models = {}
    
    if practice_data.empty:
        print("Using default tire parameters")
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_models[compound] = {
                'type': 'default',
                'alpha': base_pace + TIRE_DEFAULTS[compound]['offset'],
                'beta': TIRE_DEFAULTS[compound]['degradation_rate'],
            }
        return compound_models
    
    print("\nBuilding tire models (medium compound as anchor)...")
    
    # Fit medium compound first - it has data across all session types
    medium_data = practice_data[practice_data['Compound'] == 'MEDIUM']
    
    if len(medium_data) < 10:
        print("  MEDIUM: Insufficient data, using defaults for all compounds")
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            compound_models[compound] = {
                'type': 'default',
                'alpha': base_pace + TIRE_DEFAULTS[compound]['offset'],
                'beta': TIRE_DEFAULTS[compound]['degradation_rate'],
            }
        return compound_models
    
    lap_times = medium_data['LapTime_corrected']
    mean_time = lap_times.mean()
    std_time = lap_times.std()
    
    clean_medium = medium_data[
        (lap_times >= mean_time - 2.5 * std_time) &
        (lap_times <= mean_time + 2.5 * std_time)
    ]
    
    x = clean_medium['StintLap'].values.astype(float)
    y = clean_medium['LapTime_corrected'].values.astype(float)
    
    medium_alpha = None
    medium_beta = None
    
    if NUMPYRO_AVAILABLE and len(x) >= 8:
        try:
            mcmc = fit_bayesian_model(x, y, 'MEDIUM', base_pace)
            samples = mcmc.get_samples()
            
            medium_alpha = np.mean(samples['alpha'])
            medium_beta = np.mean(samples['beta'])
            alpha_std = np.std(samples['alpha'])
            beta_std = np.std(samples['beta'])
            
            compound_models['MEDIUM'] = {
                'type': 'bayesian',
                'mcmc': mcmc,
                'alpha': medium_alpha,
                'beta': medium_beta,
                'alpha_std': alpha_std,
                'beta_std': beta_std,
                'samples': samples,
                'n_laps': len(clean_medium),
            }
            print(f"  MEDIUM: α={medium_alpha:.2f}s (±{alpha_std:.2f}), β={medium_beta:.4f}s/lap (±{beta_std:.4f}) [{len(clean_medium)} laps]")
            
        except Exception as e:
            print(f"  MEDIUM: Bayesian fit failed ({e}), using linear")
            mcmc = None
    
    if medium_alpha is None:
        coeffs = np.polyfit(x, y, 1)
        medium_beta = max(0, coeffs[0])
        medium_alpha = coeffs[1]
        compound_models['MEDIUM'] = {
            'type': 'linear',
            'alpha': medium_alpha,
            'beta': medium_beta,
            'n_laps': len(clean_medium),
        }
        print(f"  MEDIUM: α={medium_alpha:.2f}s, β={medium_beta:.4f}s/lap [{len(clean_medium)} laps]")
    
    # Soft: anchor to medium, fit degradation from available data
    soft_offset = -0.55  # Soft is typically 0.5-0.6s faster than medium
    soft_alpha = medium_alpha + soft_offset
    
    soft_data = practice_data[practice_data['Compound'] == 'SOFT']
    if len(soft_data) >= 10:
        soft_times = soft_data['LapTime_corrected']
        soft_mean = soft_times.mean()
        soft_std = soft_times.std()
        clean_soft = soft_data[
            (soft_times >= soft_mean - 2.5 * soft_std) &
            (soft_times <= soft_mean + 2.5 * soft_std)
        ]
        
        if len(clean_soft) >= 8:
            x_soft = clean_soft['StintLap'].values.astype(float)
            y_soft = clean_soft['LapTime_corrected'].values.astype(float)
            
            if NUMPYRO_AVAILABLE:
                try:
                    mcmc_soft = fit_bayesian_model(x_soft, y_soft, 'SOFT', soft_alpha)
                    samples_soft = mcmc_soft.get_samples()
                    soft_beta = np.mean(samples_soft['beta'])
                    beta_std = np.std(samples_soft['beta'])
                    
                    compound_models['SOFT'] = {
                        'type': 'bayesian_anchored',
                        'alpha': soft_alpha,
                        'beta': soft_beta,
                        'beta_std': beta_std,
                        'samples': samples_soft,
                        'n_laps': len(clean_soft),
                        'anchor_offset': soft_offset,
                    }
                    print(f"  SOFT: α={soft_alpha:.2f}s (anchored), β={soft_beta:.4f}s/lap (±{beta_std:.4f}) [{len(clean_soft)} laps]")
                except:
                    pass
    
    if 'SOFT' not in compound_models:
        soft_beta = TIRE_DEFAULTS['SOFT']['degradation_rate']
        compound_models['SOFT'] = {
            'type': 'anchored_default_deg',
            'alpha': soft_alpha,
            'beta': soft_beta,
            'anchor_offset': soft_offset,
        }
        print(f"  SOFT: α={soft_alpha:.2f}s (anchored), β={soft_beta:.4f}s/lap (default deg)")
    
    # Hard: anchor to medium, fit degradation from FP1 data
    hard_offset = 0.60  # Hard is typically 0.5-0.7s slower than medium
    hard_alpha = medium_alpha + hard_offset
    
    hard_data = practice_data[practice_data['Compound'] == 'HARD']
    if len(hard_data) >= 10:
        hard_times = hard_data['LapTime_corrected']
        hard_mean = hard_times.mean()
        hard_std = hard_times.std()
        clean_hard = hard_data[
            (hard_times >= hard_mean - 2.5 * hard_std) &
            (hard_times <= hard_mean + 2.5 * hard_std)
        ]
        
        if len(clean_hard) >= 8:
            x_hard = clean_hard['StintLap'].values.astype(float)
            y_hard = clean_hard['LapTime_corrected'].values.astype(float)
            
            if NUMPYRO_AVAILABLE:
                try:
                    mcmc_hard = fit_bayesian_model(x_hard, y_hard, 'HARD', hard_alpha)
                    samples_hard = mcmc_hard.get_samples()
                    hard_beta = np.mean(samples_hard['beta'])
                    beta_std = np.std(samples_hard['beta'])
                    
                    compound_models['HARD'] = {
                        'type': 'bayesian_anchored',
                        'alpha': hard_alpha,
                        'beta': hard_beta,
                        'beta_std': beta_std,
                        'samples': samples_hard,
                        'n_laps': len(clean_hard),
                        'anchor_offset': hard_offset,
                    }
                    print(f"  HARD: α={hard_alpha:.2f}s (anchored), β={hard_beta:.4f}s/lap (±{beta_std:.4f}) [{len(clean_hard)} laps]")
                except:
                    pass
    
    if 'HARD' not in compound_models:
        hard_beta = TIRE_DEFAULTS['HARD']['degradation_rate']
        compound_models['HARD'] = {
            'type': 'anchored_default_deg',
            'alpha': hard_alpha,
            'beta': hard_beta,
            'anchor_offset': hard_offset,
        }
        print(f"  HARD: α={hard_alpha:.2f}s (anchored), β={hard_beta:.4f}s/lap (default deg)")
    
    return compound_models


def get_lap_time(compound, lap_in_stint, compound_models):
    model = compound_models.get(compound, compound_models.get('MEDIUM'))
    alpha = model['alpha']
    beta = model['beta']
    return alpha + beta * lap_in_stint


def get_lap_time_with_uncertainty(compound, lap_in_stint, compound_models):
    model = compound_models.get(compound, compound_models.get('MEDIUM'))
    
    if model.get('type') == 'bayesian' and 'samples' in model:
        samples = model['samples']
        idx = np.random.randint(0, len(samples['alpha']))
        alpha = samples['alpha'][idx]
        beta = samples['beta'][idx]
        return alpha + beta * lap_in_stint
    
    alpha = model['alpha']
    beta = model['beta']
    return alpha + beta * lap_in_stint + np.random.normal(0, 0.25)