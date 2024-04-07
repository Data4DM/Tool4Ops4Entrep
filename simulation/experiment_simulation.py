import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from cmdstanpy import CmdStanModel

def sample_segment_belief_set_boundary(s, SMN):
    """
    Samples segment belief for a model and sets feedback boundaries.
    """
    if s== 0:
        SMN['a_b'][s] = np.random.normal(0, 1)  # Sample from a Normal distribution
        SMN['x_min'][s], SMN['x_max'][s] = -2, 2    # Set fixed feedback boundaries
    else:
        SMN['a_b'][s] = np.random.normal(SMN['a_mean_b'][s, 0].mean(), SMN['a_mean_b'][s, 0].std())
        SMN['x_min'][s], SMN['x_max'][s] = SMN['a_mean_b'][s, 0].mean() - 2 * SMN['a_mean_b'][s, 0].std(), SMN['a_mean_b'][s, 0].mean() + 2 * SMN['a_mean_b'][s, 0].std()
    return SMN


def sample_model_belief(s, m, SMN):
    """
    Uses cost to simulate model outcome based on segment belief. 
    how P(M = 1) increases with S in a logistic manner; with production feasibility as latent variable 
    introduces variability in the threshold at which M switches from 0 to 1
    D ~ Normal(μ, σ^2)
                |
                v
    F ~ Logistic(0, 1)  ----->  D + F > 0 ?
                                  |
                                  v
                              M ~ Bernoulli(expit(D))

    Probability P(M=1)= EM
                            ^
                        1.0 |                        
                            |                  ___---------
                            |            ___---^
                            |       ___--^
                            |  ___--^
                            | -^
                        0.5 |-^
                         _-^|
                    ---^    |
                 ___^       |
       ---------^           |
     ---------------------- 0.0 ---------------------> D
         -∞                  0                 +∞
    1. x-axis: values of the random variable S, which follows a normal distribution.
    2. y-axis:= P(M=1)=expit(S)
    3. As D increases, probability of M being 1 also increases, following the shape of the logistic sigmoid function (expit).
    4. When D = 0, the probability of M being 1 is 0.5, as expit(0) = 0.5, S -> -∞, P(M=1) -> 0, D -> +∞, P(M=1) -> 1
    
    The logistic sigmoid function (expit) maps the values of D to probabilities between 0 and 1, representing the probability of M being 1. 
    The latent variable F acts as a random threshold that determines whether M takes the value 1 or 0 based on the sum of D and F.
    """
    M = SMN.dims['M']
    # Sum of market acceptance in chosen segment and product feasibility in chosen specification as belief random variable S + business model belief (tech) T = M Convert a_b to probability and simulate model outcome
    
    SMN['x_b'][s, m] = np.random.binomial(SMN['SR'], expit(SMN['a_b'][s, m]))[0]
    return SMN


def sample_segment_real(s, m, SMN):
    """
    Samples new real segment beliefs and calculates feedback by combining 
    model belief outcome and real segment outcome.
    """
    if SMN['DR'] == 0:
        # REAL DESIRABILITY
        SMN['a_r'][s, m] = SMN['a_mean_r'].item(0)  # mean of np.random.normal(SMN['a_mean_r'][s, m], 1)
        SMN['x_r'][s, m] =  expit(SMN['a_mean_r'][0])
        return SMN

    elif SMN['DR'] == 1:
        #Sample a_r^s from a Normal distribution and simulate real segment outcome
        SMN['a_r'][s, m] = np.random.normal(np.mean(SMN['a_mean_b'][1,1]), 1)
        SMN['x_r'][s, m] = np.random.binomial(SMN['SR'], expit(SMN['a_r'][s,m]))[0]
    # determined during initialisation
    # SMN['a_r'][s, m] = SMN['a_mean_r'][0] # mean of np.random.normal(SMN['a_mean_r'][s, m], 1)
    # SMN['x_r'][s, m] = N * expit(SMN['a_r'][s, m]) / N  # mean of np.random.binomial(N, expit(SMN['a_r'][s, m])) / N
        return SMN

# Assuming you have a global cache for the last fitting
last_fit_params = {}
last_fit_result = None

def update_state(s, m, SMN, fit_dir='stan_fits'):
    """
    Updates state on prior belief on mean segment acceptance (a_mean_b) and cash

    Parameters:
    - s (int): Index of the current segment.
    - m (int): Index for the current model.
    - SMN (xarray.Dataset): Dataset containing the segment and production data.
    - model_file (str): Path to the Stan model file.

    Returns:
    - Updates the SMN dataset with new inferred values for a_b.
    - Saves the Stan fit object to a file.
    """
    global last_fit_params, last_fit_result

    sr = int(SMN.SR)
    K = SMN.dims['K']
    # use previous market prior for the next market
    stan_data = {
        'E': 1,
        'N': [sr],
        'G': [int(SMN['x_r'][s, m].item())],
        'segment': [1],
        'a_mean_b': np.mean(SMN['a_mean_b'][s,m,:]).item(), # posterior of the last seg becomes prior
        'DB': int(SMN.DB)
        }
    if s == 0:
        model_file = "stan/first.stan"
    else:
        model_file = "stan/market_pivot.stan"
    # # Check if we can reuse the last fit
    # if last_fit_params.get((s, m)) == stan_data:
    #     fit = last_fit_result
    # else:
    #     fit = load_fit(s, m, fit_dir)
    #     if fit is None:
    #         model_file = "stan/first.stan"
    model = CmdStanModel(stan_file=model_file)
    fit = model.sample(data=stan_data, show_console=False, show_progress=False, iter_sampling=int(K/4))
            # make_id_save(stan_data, fit, s, m, fit_dir)  # Save the new fit
        # Cache the current fit
        # last_fit_params[(s, m)] = stan_data
        # last_fit_result = fit
    
    prev_cash = SMN['cash_state'][s, m].item()

    s, m = determine_next_state(s, m, SMN)
    # Update the posterior into next prior
    if s < SMN.dims['S']:
        SMN['a_mean_b'][s, m, :] = fit.stan_variable('a').flatten()
        SMN['cash_state'][s, m] = prev_cash - SMN['unit_exp_cost'][0].item() 
    
    return SMN

def determine_next_action(s, m, SMN):
    """
    Determines the next action based on current state and thresholds.

    Inputs:
    - s (int): Current segment index.
    - m (int): Current model index.
    - SMN (xarray.Dataset): Dataset containing segment and model data.

    Output:
    - (str): The determined action (scale, fail, market_pivot, or product_pivot).
    """
    x_r_s_m = SMN['x_r'][s, m].item()
    c_s_m = SMN['cash_state'][s, m].item()
    c_s_0 = SMN['cash_state'][s, 0].item()
    x_min_s = SMN['x_min'][s].item()
    x_max_s = SMN['x_max'][s].item()

    if x_r_s_m > x_max_s:
        return "scale"
    elif c_s_m <= 0:
        return "fail"
    elif c_s_m < SMN['CT'][0].item() * c_s_0:
        return "market_pivot"
    elif x_min_s <= x_r_s_m <= x_max_s:
        if m == SMN.dims['M']-1:
            return "market_pivot"
        return "product_pivot"
    elif x_r_s_m < x_min_s:
        return "market_pivot"


def determine_next_state(s, m, SMN):
    """
    Determines the next state based on the current action.

    Inputs:
    - s (int): Current segment index.
    - m (int): Current model index.
    - SMN (xarray.Dataset): Dataset containing segment and model data.

    Output:
    - (tuple): The next state (segment index, model index).
    """
    next_action = determine_next_action(s, m, SMN)
    if next_action == "product_pivot":
        return s, m + 1
    elif next_action == "market_pivot":
        return s + 1, 0  # Reset model index

    return s, m  # Remain in current state if no action is determined


def load_fit(s, m, fit_dir='stan_fits'):
    """
    Loads a Stan fit object from file.

    Parameters:
    - s (int): Index of the model.
    - m (int): Index of the segment experiment.
    - fit_dir (str): Directory where fit files are stored.

    Returns:
    - The loaded Stan fit object.
    """
    fit_file = os.path.join(fit_dir, f'fit_s{s}_m{m}.csv')
    if os.path.exists(fit_file):
        fit = CmdStanModel(stan_file='path/to/stan/model')  # Ensure the correct path to your Stan model file is provided
        return fit.from_csv(fit_file)
    else:
        return None


def make_id_save(data, fit, s, m, fit_dir='stan_fits'):
    """
    Saves the data dictionary to a JSON file.

    Parameters:
    - data (dict): Data dictionary to save.
    - data_dir (str): Directory to save the data file.
    """
    data_dir = os.path.join(fit_dir, 'data')
    
    #Save data to a JSON file
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f'data_s{s}_m{m}.json')
    with open(data_file, 'w') as f:
        json.dump(data, f)

    fit_dir = os.path.join(fit_dir, f'fit_s{s}_m{m}')
    os.makedirs(fit_dir, exist_ok=True)
    fit.save_csvfiles(dir=fit_dir)
    return


def experiment_process(s, m, SMN):
    SMN = sample_model_belief(s, m, SMN)
    SMN = sample_segment_real(s, m, SMN)
    return SMN  # No specific action required


def start_experiment(SMN):
    S = SMN.dims['S']
    M = SMN.dims['M']
    
    # Read parameter values from SMN
    dr = int(SMN.DR)
    sr = int(SMN.SR)
    br = int(SMN.BR)
    db = int(SMN.DB)
    er = int(SMN.ER)
    ct = float(SMN.CT)
    
    # Create unique SMN name based on parameter values
    for s in range(S):
        action = None  # Initialize action variable
        SMN = sample_segment_belief_set_boundary(s, SMN)
        for m in range(M):
            SMN = experiment_process(s, m, SMN)
            action = determine_next_action(s, m, SMN)
            SMN['action'][s, m] = action
            if action in ["scale", "fail"]:
                SMN.to_netcdf(f"xarray_data/{SMN.SMN_name.values}.nc")
                return SMN
            if action in ["product_pivot", "market_pivot"]:
                SMN = update_state(s, m, SMN)
            else:
                break
            # if m == M-1:
            #     action = "scale" #pre-mature scale??
            #     break
    # plot_combined(SMN)
    SMN.to_netcdf(f"xarray_data/{SMN.SMN_name.values}.nc")
    SMN.close()
    return SMN


def calc_sim_summ_stat(DR, SR, BR, DB, ER, CT, S, M, K):
    # overconfidence, Belief (Diffuseness of prior belief), Real (Market Stability (0 or 1), Size of market segment, Experiment opportunity)
    for dr in DR:
        for sr in SR:
            for br in BR:
                for db in DB:
                    for er in ER:
                        for ct in CT:
                            # Perform calculations and store the result as SMN_{br}_{db}_{dr}_{sr}_{er}_{ct}
                            coords = {'H': np.arange(1), 'S': np.arange(S), 'M': np.arange(M), 'K': np.arange(K)}
                            SMN_name = f"DR{dr}_SR{sr}_BR{br}_DB{db}_ER{er}_CT{ct}_S{S}_M{M}_K{K}"
                            
                            data_vars = {
                                # HYPERPARAMS
                                'cash_state': (('S', 'M'), np.zeros((S, M))),
                                'unit_exp_cost': ('H', np.array([1])),
                                'DR': ('H', np.array([dr])),  # belief on real
                                'SR': ('H', np.array([sr])),  # size of segment
                                'BR': ('H', np.array([br])),  # belief on real
                                'DB': ('H', np.array([db])),  # belief on belief
                                'ER': ('H', np.array([er])),  # experiment opportunity
                                'CT': ('H', np.array([ct])),  # cash threshold
                                'SMN_name': ((), SMN_name),  # Add SMN_name as a 0-dimensional variable
                                
                                # BELIEF on DESIRABILITY and FEASIBILITY
                                'x_b': (('S', 'M'), np.zeros((S, M))),
                                'a_b': (('S', 'M'), np.zeros((S, M))),  # segment acceptance (belief) parameter for specific segment
                                'a_mean_b': (('S', 'M', 'K'), np.zeros((S, M, K))),  # SAMPLES from Posterior of average segment acceptance (across segments)
                                'x_min': ('S', np.zeros(S)),  # TODO: E , sigma of a_mean_b
                                'x_max': ('S', np.zeros(S)),
                                
                                # REAL DESIRABILITY
                                'x_r': (('S', 'M'), np.zeros((S, M))),
                                'a_r': (('S', 'M'), np.zeros((S, M))),  # segment acceptance (real) parameter for specific segment
                                'a_mean_r': (('H'), np.array([-br])),  # average segment acceptance (across segments)
                                'action': (('S', 'M'), np.full((S, M), '', dtype='object'))  # Agent with belief responding to Real segment feedback
                            }
                            SMN = xr.Dataset(data_vars=data_vars, coords=coords)
                            SMN['cash_state'].loc[dict(S=0, M=0)] = SMN.ER.item()  # initial cash
                            start_experiment(SMN)
    return