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
        # exPMN['mu_a'][t].item() +
        pow(-1, e2p[exPMN['product'][t].item()]+1) * exPMN['mu_b_r'].item()/2+
        pow(-1, e2m[exPMN['market'][t].item()]+1) * exPMN['mu_c_r'].item()/2
    )
    return p

def decide_action(profit_obs, low_profit_b, high_profit_b):
    """Decide the next action based on predicted and obsereved profit."""

    if low_profit_b <= profit_obs <= high_profit_b:
        return "pivot_product"
    elif profit_obs < low_profit_b:
        return "pivot_market"
    elif profit_obs > high_profit_b:
        return "scale"

def predict_observe_update_as_lbs(t, exPMN):
    """predict Observe signal in chosen product-market, update mus, predict and observe profit, and decide action.
       LATENT BIT STATE (LBS): mu_b_b, mu_c_b, mu_a
       BIT STATE (BS): profit_b (predicted profit), low_profit_b, high_profit_b
       ATOM STATE (AS): product, market
    """
    #PREDICT: BS[t]| AS[t]
    exPMN['profit_b'][t] = (
        exPMN['mu_a'][t] + 
        pow(-1, e2p[exPMN['product'][t].item()]+1) * exPMN['mu_b_b'][t]/2 + 
        pow(-1, e2m[exPMN['market'][t].item()]+1) * exPMN['mu_c_b'][t]/2
        )
    exPMN['low_profit_b'][t] = exPMN['profit_b'][t].item() - exPMN['k_sigma'].item() * exPMN['sigma_mu'][t].item()
    exPMN['high_profit_b'][t] = exPMN['profit_b'][t].item() + exPMN['k_sigma'].item() * exPMN['sigma_mu'][t].item()
    
    #OBSERVE signal[AS[t]]|AS[t]
    exPMN['profit_obs'][t] = sample_profit_obs(t, exPMN) 
    exPMN['action'][t] = decide_action(exPMN['profit_obs'][t], exPMN['low_profit_b'][t], exPMN['high_profit_b'][t])
    
    if exPMN['action'][t] == "scale":
        return exPMN
    
    elif t < exPMN.dims['ACT_PRED']:     
        pivot_product_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_product.stan')
        pivot_market_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_market.stan')
        
        if exPMN['action'][t] == "pivot_product":
            if t < exPMN.dims['ACT_PRED']-1:
                #UPDATE AS[t+1]| A[t], AS[t]
                exPMN['product'][t+1] = 'man' if exPMN['product'][t].item() == 'ai' else 'ai'
                exPMN['market'][t+1] = exPMN['market'][t].item()
            
            data = {
            'product': e2p[exPMN.product[t].item()] + 1,
            'market': e2m[exPMN.market[t].item()] + 1,
            'mu_b_b_mean': exPMN.mu_b_b[t].item(),
            'mu_c_b': exPMN.mu_c_b[t].item(),
            'mu_a_mean': exPMN.mu_a[t].item(),
            'profit_obs': exPMN['profit_obs'][t],
            }
            fit = pivot_product_model.sample(data=data)
            exPMN['profit_post'][t] = fit.stan_variable('mu_b_b')/2 + exPMN['mu_c_b'][t].item()/2 + fit.stan_variable('mu_a')
            
            #UPDATE LBS[t+1]|A[t], AS[t],  LBS[t], signal[AS[t]]
            exPMN['mu_b_b'][t+1] = fit.stan_variable('mu_b_b').mean()
            exPMN['mu_c_b'][t+1] = exPMN['mu_c_b'][t]
            exPMN['mu_a'][t+1] = fit.stan_variable('mu_a').mean() 
            exPMN['sigma_mu'][t+1] = fit.stan_variable('sigma_mu').mean() 

        elif exPMN['action'][t] == "pivot_market":
            if t < exPMN.dims['ACT_PRED']-1:
                #UPDATE AS[t+1]|AS[t], A[t]
                exPMN['market'][t+1] = 'b2b' if exPMN['market'][t].item() == 'b2c' else 'b2c'
                exPMN['product'][t+1] = exPMN['product'][t].item()

            data = {
            'product': e2p[exPMN.product[t].item()] + 1,
            'market': e2m[exPMN.market[t].item()] + 1,
            'mu_c_b_mean': exPMN.mu_c_b[t].item(),
            'mu_b_b': exPMN.mu_b_b[t].item(),
            'mu_a_mean': exPMN.mu_a[t].item(),  
            'profit_obs': exPMN['profit_obs'][t],
            }
            fit = pivot_market_model.sample(data=data)
            exPMN['profit_post'][t] = exPMN['mu_b_b'][t].item() + fit.stan_variable('mu_c_b')/2 + fit.stan_variable('mu_a')
            
            #UPDATE LBS[t+1]|A[t], AS[t],  LBS[t], signal[AS[t]]
            exPMN['mu_b_b'][t+1] = exPMN['mu_b_b'][t]
            exPMN['mu_c_b'][t+1] = fit.stan_variable('mu_c_b').mean()
            exPMN['mu_a'][t+1] = fit.stan_variable('mu_a').mean() 
            exPMN['sigma_mu'][t+1] =  fit.stan_variable('sigma_mu').mean()
            
    return exPMN

def experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, mu_a, sigma_mu=.1, k=2, T=10, product='man', market='b2c'):
    """
    Record the expected reward from the experiment given initial parameters.
    mu_b_d = mu_b_b - mu_b_r  # belief and goal differ but we treat both as _b
    mu_c_d = mu_c_b - mu_c_r  
    """
    coords = {'HP': np.arange(1), 'P': np.arange(T+1), 'ACT_PRED': np.arange(T), 'ACT_PVT': np.arange(T), 'M': np.arange(4000)}
    exPMN_name = f"b{mu_b_d}_c{mu_c_d}_a{mu_a}_s{sigma_mu}_T{T}_{product}_{market}"
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
        'sigma_mu': ('P', np.zeros(T+1)),
        
        #PREDICT
        'market': ('ACT_PRED', np.full(T, '', dtype='object')), #entrant randomly choose (same Eprofit currently (belief comes from future))
        'product': ('ACT_PRED', np.full(T, '', dtype='object')),
        'profit_b': ('ACT_PRED', np.zeros(T)),
        
        'low_profit_b': ('ACT_PRED', np.zeros(T)),
        'high_profit_b': ('ACT_PRED', np.zeros(T)),
        #OBSERVE
        'profit_obs': ('ACT_PRED', np.zeros(T)),
        #UPDATED BIT STATE
        'profit_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        ##UPDATE ATOM STATE
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
            exPMN['sigma_mu'][0] = sigma_mu
            
        elif t == 1:
            # np.random.seed(42)
            # exPMN['product'][t] = np.random.choice(products)
            # exPMN['market'][t] = np.random.choice(markets)
            exPMN['product'][0] = product
            exPMN['market'][0] = market

            exPMN['k_sigma'][0] = k
            exPMN['low_profit_b'][0] = mu_a - k * exPMN['sigma_mu'][0]
            exPMN['high_profit_b'][0] = mu_a + k * exPMN['sigma_mu'][0]

            exPMN = predict_observe_update_as_lbs(t-1, exPMN) #time 1 is 0th experiment
        else:
            exPMN = predict_observe_update_as_lbs(t-1, exPMN)

        if exPMN['action'][t-1].item() == "scale":
            break

    exPMN.to_netcdf(f"data/experiment/{exPMN.exPMN_name.values}.nc")
    return exPMN

if __name__ == "__main__":
    # Start the experiment
    # exPMN = experiment(.1, -.1,   .2, .1,  .4, .1,   4)
    exPMN = experiment(mu_b_d= .2, mu_c_d= -.1,   mu_b_r=.3, mu_c_r=.1,  mu_a= .2, T = 3)