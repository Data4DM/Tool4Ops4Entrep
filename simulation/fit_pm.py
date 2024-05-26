# fit_pm.py
import numpy as np
import xarray as xr
import cmdstanpy
import os

global markets, products, e2p, e2m
markets = ["b2c", "b2b"]
products = ["man", "ai"]

e2p = {label: idx for idx, label in enumerate(products)}
e2m = {label: idx for idx, label in enumerate(markets)}

def sample_profit_obs(e, MPN):
    """Return the profit observation based on the experiment index and product-market pair."""
    p = (
        MPN['mu_a'][e].item() +
        pow(-1, e2p[MPN['product'][e].item()]) * MPN['mu_b_r'].item() +
        pow(-1, e2m[MPN['market'][e].item()]) * MPN['mu_c_r'].item()
    )
    return p

def decide_action(cash, profit_obs, low_profit_b, high_profit_b):
    """Decide the next action based on the current cash state and profit."""
    if cash <= 0:
        return "update_base"
    elif low_profit_b <= profit_obs <= high_profit_b:
        return "pivot_product"
    elif profit_obs < low_profit_b:
        return "pivot_market"
    elif profit_obs > high_profit_b:
        return "scale"

def predict_observe_update_belief_pm(e, MPN, k=2):
    """Observe signal in chosen product-market, update mus, make prediction on profit, low and high range, and decide action."""
    MPN['profit_b'][e] = MPN['mu_a'][e] + pow(-1, e2p[MPN['product'][e].item()]) * MPN['mu_b_b'][e] + pow(-1, e2m[MPN['market'][e].item()]) * MPN['mu_c_b'][e]
    MPN['low_profit_b'][e] = MPN['profit_b'][e] - k * MPN['sigma_mu'][e]
    MPN['high_profit_b'][e] = MPN['profit_b'][e] + k * MPN['sigma_mu'][e]
    
    action = decide_action(MPN['cash'][e].item(), sample_profit_obs(e, MPN), MPN['low_profit_b'][e], MPN['high_profit_b'][e])
    MPN['action'][e] = action
    
    pivot_product_model = cmdstanpy.CmdStanModel(stan_file='pivot_product.stan')
    pivot_market_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_market.stan')
    
    if action == "scale":
        return

    elif action == "pivot_product":
        data = {
        'profit_obs': sample_profit_obs(e, MPN),
        'market': e2m[MPN.market[e].item()] + 1,
        'product': e2p[MPN.product[e].item()] + 1,
        'mu_a_mean': MPN.mu_a[e].item(),  # Updated to use mu_a
        'mu_b_b_mean': MPN.mu_b_b[e].item(),
        'mu_c_b': MPN.mu_c_b[e].item(),
        }
        fit = pivot_product_model.sample(data=data, show_console=True)
        MPN['mu_b_b'][e+1] = fit.stan_variable('mu_b_b').mean()
        MPN['mu_c_b'][e+1] = MPN['mu_c_b'][e]
        MPN['mu_a'][e+1] = fit.stan_variable('mu_a').mean()  # Updated to use mu_a
        MPN['sigma_mu'][e+1] = fit.stan_variable('sigma_mu').mean()
        MPN['product'][e + 1] = 'man' if MPN['product'][e].item() == 'ai' else 'ai'
        MPN['market'][e + 1] = MPN['market'][e].item()
        MPN['cash'][e+1] = MPN['cash'][e].item() - 1

    elif action == "pivot_market":
        data = {
        'profit_obs': sample_profit_obs(e, MPN),
        'market': e2m[MPN.market[e].item()] + 1,
        'product': e2p[MPN.product[e].item()] + 1,
        'mu_a_mean': MPN.mu_a[e].item(),  # Updated to use mu_a
        'mu_c_b_mean': MPN.mu_c_b[e].item(),
        'mu_b_b': MPN.mu_b_b[e].item(),
        }
        fit = pivot_market_model.sample(data=data, show_console=True)
        MPN['mu_c_b'][e+1] = fit.stan_variable('mu_c_b').mean()
        MPN['mu_b_b'][e+1] = MPN['mu_b_b'][e]
        MPN['mu_a'][e+1] = fit.stan_variable('mu_a').mean()  # Updated to use mu_a
        MPN['sigma_mu'][e+1] = fit.stan_variable('sigma_mu').mean()
        MPN['market'][e + 1] = 'b2b' if MPN['market'][e].item() == 'b2c' else 'b2c'
        MPN['product'][e + 1] = MPN['product'][e].item()
        MPN['cash'][e+1] = MPN['cash'][e].item() - 1
    
    return MPN

def experiment(mu_b_d, mu_c_d, mu_a=.4, mu_b_r=.2, mu_c_r=.1, sigma_mu=.1,  cash=4, E=5, product='man', market='b2c'):
    """
    Record the expected reward from the experiment given initial parameters.
    """
    coords = {'H': np.arange(1), 'E': np.arange(E)}
    mu_b_b = mu_b_r + mu_b_d
    mu_c_b = mu_c_r + mu_c_d

    MPN_name = f"B{mu_b_d}_C{mu_c_d}_a{mu_a}_b{mu_b_r}_c{mu_c_r}_s{sigma_mu}_cash{cash}_E{E}_{product}_{market}"
    file_path = f"xarray_data/{MPN_name}.nc"

    if os.path.exists(file_path):
        MPN = xr.open_dataset(file_path)
        print(f"File {MPN_name} already exists. Skipping experiment.")
        return MPN

    data_vars = {
        'action': ('E', np.full(E, '', dtype='object')),
        'market': ('E', np.full(E, '', dtype='object')),
        'product': ('E', np.full(E, '', dtype='object')),

        'mu_a': ('E', np.zeros(E)),
        'mu_b_b': ('E', np.zeros(E)),
        'mu_c_b': ('E', np.zeros(E)),
        'sigma_mu': ('E', np.zeros(E)),

        'profit_b': ('E', np.zeros(E)),
        'low_profit_b': ('E', np.zeros(E)),
        'high_profit_b': ('E', np.zeros(E)),

        'mu_b_r': ('H', np.zeros(1)),
        'mu_c_r': ('H', np.zeros(1)),
        'profit_obs': ('E', np.zeros(E)),

        'k_sigma': ('H', np.zeros(1)),
        'cash': ('E', np.zeros(E, dtype=int)),
        'MPN_name': ((), MPN_name),
    }
    MPN = xr.Dataset(data_vars=data_vars, coords=coords)
    
    MPN['mu_a'][0] = mu_a 
    MPN['mu_b_r'][0] = mu_b_r
    MPN['mu_c_r'][0] = mu_c_r
    MPN['sigma_mu'][0] = sigma_mu

    MPN['mu_b_b'][0] = mu_b_b
    MPN['mu_c_b'][0] = mu_c_b

    MPN['cash'][0] = cash
    MPN['product'][0] = product
    MPN['market'][0] = market
        
    for e in range(E):
        MPN = predict_observe_update_belief_pm(e, MPN, k=2)

        if (e > 0 and MPN['action'][e].item() == "scale"):
            break
    MPN.to_netcdf(f"xarray_data/{MPN.MPN_name.values}.nc")
    return MPN

if __name__ == "__main__":
    # Start the experiment
    MPN = experiment(.1, 0, -.1,  .4, .2, .1,   .1,   4)