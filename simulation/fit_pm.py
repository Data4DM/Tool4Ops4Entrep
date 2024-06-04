# fit_pm.py
import numpy as np
import xarray as xr
import cmdstanpy
import os

os.chdir("/Users/hyunjimoon/Dropbox (MIT)/Ops Entreps cases xx/Ops4Entrep-backend/simulation")
global markets, products, e2p, e2m
markets = ["b2c", "b2b"]
products = ["man", "ai"]

e2p = {label: idx for idx, label in enumerate(products)}
e2m = {label: idx for idx, label in enumerate(markets)}

def sample_profit_obs(t, exPMN):
    """Return the profit observation based on the experiment index and product-market pair."""
    p = (
        pow(-1, e2p[exPMN['product'][t].item()]+1) * exPMN['mu_b_r'].item()/2 +
        pow(-1, e2m[exPMN['market'][t].item()]+1) * exPMN['mu_c_r'].item()/2
    )
    return p

def decide_action(profit_obs, low_profit_b, high_profit_b):
    """Decide the next action based on predicted and observed profit."""

    if low_profit_b <= profit_obs <= high_profit_b:
        return "pivot_product"
    elif profit_obs < low_profit_b:
        return "pivot_market"
    elif profit_obs > high_profit_b:
        return "scale"

def predict_observe_update_as_lbs(e, exPMN):
    """Predict, observe signal in chosen product-market, update mus, predict and observe profit, and decide action.
       LATENT BIT STATE (LBS): mu_b_b, mu_c_b, mu_a
       BIT STATE (BS): profit_b (predicted profit), low_profit_b, high_profit_b
       ATOM STATE (AS): product, market
    """
    # PREDICT: BS[e]| AS[e] profit in 
    for i, product in enumerate(products):
        for j, market in enumerate(markets):
            exPMN['profit_b'][i, j, e] = (
                exPMN['mu_a'][e] +
                pow(-1, e2p[product]+1) * exPMN['mu_b_b'][e]/2 +
                pow(-1, e2m[market]+1) * exPMN['mu_c_b'][e]/2
            )
    
    exPMN['low_profit_b'][e] = exPMN['profit_b'][e2p[exPMN['product'][e].item()], e2m[exPMN['market'][e].item()], e].item() - exPMN['k_sigma'].item() * exPMN['sigma_profit'][e].item()
    exPMN['high_profit_b'][e] = exPMN['profit_b'][e2p[exPMN['product'][e].item()], e2m[exPMN['market'][e].item()], e].item() + exPMN['k_sigma'].item() * exPMN['sigma_profit'][e].item()
    
    # OBSERVE signal[AS[e]]|AS[e]
    exPMN['profit_obs'][e] = sample_profit_obs(e, exPMN) 
    exPMN['action'][e] = decide_action(exPMN['profit_obs'][e], exPMN['low_profit_b'][e], exPMN['high_profit_b'][e])
    
    if exPMN['action'][e] == "scale":
        return exPMN
    
    elif e < exPMN.dims['ACT_PRED']:     
        pivot_product_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_product.stan')
        pivot_market_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_market.stan')
        
        if exPMN['action'][e] == "pivot_product":
            if e < exPMN.dims['ACT_PRED']-1:
                # UPDATE AS[e+1]| A[e], AS[e]
                exPMN['product'][e+1] = 'man' if exPMN['product'][e].item() == 'ai' else 'ai'
                exPMN['market'][e+1] = exPMN['market'][e].item()
            
            data = {
                'product': e2p[exPMN.product[e].item()] + 1,
                'market': e2m[exPMN.market[e].item()] + 1,
                'mu_b_b_mean': exPMN.mu_b_b[e].item(),
                'mu_c_b': exPMN.mu_c_b[e].item(),
                'mu_a_mean': exPMN.mu_a[e].item(),
                'profit_obs': exPMN['profit_obs'][e],
            }
            fit = pivot_product_model.sample(data=data)
            
            exPMN['mu_a_post'][e] = fit.stan_variable('mu_a')
            exPMN['mu_b_b_post'][e] = fit.stan_variable('mu_b_b')
            if e > 0:
                exPMN['mu_c_b_post'][e] = exPMN['mu_c_b_post'][e-1]
            else:
                exPMN['mu_c_b_post'][e] = exPMN['mu_c_b'][e]

            exPMN['profit_post'][e] = (exPMN['mu_a_post'][e] + 
                            pow(-1, e2p[exPMN['product'][e].item()]+1) * exPMN['mu_b_b_post'][e]/2 + 
                            pow(-1, e2m[exPMN['market'][e].item()]+1) * exPMN['mu_c_b_post'][e]/2
                            )

            # UPDATE LBS[e+1]|A[e], AS[e], LBS[e], signal[AS[e]]
            exPMN['mu_b_b'][e+1] = exPMN['mu_b_b_post'][e].mean()
            exPMN['mu_c_b'][e+1] = exPMN['mu_c_b'][e]
            exPMN['mu_a'][e+1] = fit.stan_variable('mu_a').mean() 
            # exPMN['sigma_obs'][e+1] = fit.stan_variable('sigma_obs').mean()

        elif exPMN['action'][e] == "pivot_market":
            if e < exPMN.dims['ACT_PRED']-1:
                # UPDATE AS[e+1]|AS[e], A[e]
                exPMN['market'][e+1] = 'b2b' if exPMN['market'][e].item() == 'b2c' else 'b2c'
                exPMN['product'][e+1] = exPMN['product'][e].item()

            data = {
                'product': e2p[exPMN.product[e].item()] + 1,
                'market': e2m[exPMN.market[e].item()] + 1,
                'mu_c_b_mean': exPMN.mu_c_b[e].item(),
                'mu_b_b': exPMN.mu_b_b[e].item(),
                'mu_a_mean': exPMN.mu_a[e].item(),  
                'profit_obs': exPMN['profit_obs'][e],
            }
            fit = pivot_market_model.sample(data=data)
            exPMN['profit_post'][e] = (fit.stan_variable('mu_a') + 
                                       pow(-1, e2p[exPMN['product'][e].item()]+1) * exPMN['mu_b_b'][e].item() + 
                                       pow(-1, e2m[exPMN['market'][e].item()]+1) * fit.stan_variable('mu_c_b')/2
                                       )

            exPMN['mu_a_post'][e] = fit.stan_variable('mu_a')
            exPMN['mu_c_b_post'][e] = fit.stan_variable('mu_c_b')
            if e > 0:
                exPMN['mu_b_b_post'][e] = exPMN['mu_b_b_post'][e-1]
            # UPDATE LBS[e+1]|A[e], AS[e], LBS[e], signal[AS[e]]
            exPMN['mu_b_b'][e+1] = exPMN['mu_b_b'][e]
            exPMN['mu_c_b'][e+1] = exPMN['mu_c_b_post'][e].mean()
            exPMN['mu_a'][e+1] = fit.stan_variable('mu_a').mean() 
            # exPMN['sigma_obs'][e+1] = fit.stan_variable('sigma_obs').mean()
        
        exPMN['sigma_profit'][e+1] = exPMN['profit_post'][e].std()
    return exPMN

def experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, mu_a, sigma_profit=.1, sigma_obs=.1, k=2, T=10, product='man', market='b2c'):
    """
    Record the expected reward from the experiment given initial parameters.
    mu_b_d = mu_b_b - mu_b_r  # belief and goal differ but we treat both as _b
    mu_c_d = mu_c_b - mu_c_r  
    """
    coords = {'HP': np.arange(1), 'P': np.arange(T+1), 'ACT_PRED': np.arange(T), 'ACT_PVT': np.arange(T), 'PD': np.arange(2), 'MK': np.arange(2), 'M': np.arange(4000)}
    exPMN_name = f"bB{mu_b_d}_cC{mu_c_d}_B{mu_b_r}_C{mu_c_r}_a{mu_a}_s{sigma_profit}_T{T}_{product}_{market}"
    file_path = f"data/experiment/{exPMN_name}.nc"

    if os.path.exists(file_path):
        exPMN = xr.open_dataset(file_path)
        print(f"File {exPMN_name} already exists.")
        return exPMN

    data_vars = {
        'mu_b_d': ('P', np.zeros(T+1)), 
        'mu_c_d': ('P', np.zeros(T+1)),
        'mu_b_b': ('P', np.zeros(T+1)),
        'mu_c_b': ('P', np.zeros(T+1)),
        'mu_a': ('P', np.zeros(T+1)),
        'sigma_obs': ('P', np.zeros(T+1)),
        'sigma_profit': ('P', np.zeros(T+1)),
        
        # PREDICT
        'market': ('ACT_PRED', np.full(T, '', dtype='object')), #entrant randomly choose (same Eprofit currently (belief comes from future))
        'product': ('ACT_PRED', np.full(T, '', dtype='object')),
        'profit_b': (('PD', 'MK', 'ACT_PRED'), np.zeros((2, 2, T))),
        
        'low_profit_b': ('ACT_PRED', np.zeros(T)),
        'high_profit_b': ('ACT_PRED', np.zeros(T)),
        # OBSERVE
        'profit_obs': ('ACT_PRED', np.zeros(T)),
        # UPDATED BIT STATE
        'profit_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_a_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_b_b_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_c_b_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        # UPDATE ATOM STATE
        'action': ('ACT_PVT', np.full(T, '', dtype='object')),

        'mu_b_r': ('HP', np.zeros(1)),
        'mu_c_r': ('HP', np.zeros(1)),
        'k_sigma': ('HP', np.zeros(1)),
        'exPMN_name': ((), exPMN_name),
    }
    exPMN = xr.Dataset(data_vars=data_vars, coords=coords)

    for t in range(exPMN.dims['P']):
        if t == 0:
            exPMN['mu_b_d'][0] = mu_b_d
            exPMN['mu_c_d'][0] = mu_c_d
            exPMN['mu_b_b'][0] =  mu_b_r + mu_b_d
            exPMN['mu_c_b'][0] = mu_c_r + mu_c_d 
            exPMN['mu_a'][0] = mu_a 
            exPMN['mu_b_r'][0] = mu_b_r
            exPMN['mu_c_r'][0] = mu_c_r
            exPMN['sigma_obs'][0] = sigma_obs
            exPMN['sigma_profit'][0] = sigma_profit

            exPMN['mu_a_post'][0] = mu_a
            exPMN['mu_b_b_post'][0] = exPMN['mu_b_b'][0]
            exPMN['mu_c_b_post'][0] = exPMN['mu_c_b'][0]
            
        elif t == 1:
            exPMN['product'][0] = product
            exPMN['market'][0] = market

            exPMN['k_sigma'][0] = k
            # exPMN['low_profit_b'][0] = mu_a - k * exPMN['sigma_obs'][0]
            # exPMN['high_profit_b'][0] = mu_a + k * exPMN['sigma_obs'][0]

            exPMN = predict_observe_update_as_lbs(t-1, exPMN) #time 1 is 0th experiment
        else:
            exPMN = predict_observe_update_as_lbs(t-1, exPMN)

        if exPMN['action'][t-1].item() == "scale":
            break

    exPMN.to_netcdf(f"data/experiment/{exPMN.exPMN_name.values}.nc")
    return exPMN

if __name__ == "__main__":
    # Start the experiment
    exPMN = experiment(mu_b_d= .1, mu_c_d= -.1, mu_b_r=.3, mu_c_r=.1, mu_a= .2, T=3)