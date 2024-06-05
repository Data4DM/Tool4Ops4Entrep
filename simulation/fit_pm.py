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

def sample_profit_obs(t, em):
    """Return the profit observation based on the experiment index and product-market pair."""
    p = (
        pow(-1, e2p[em['product'][t].item()]+1) * em['mu_b_r'].item()/2 +
        pow(-1, e2m[em['market'][t].item()]+1) * em['mu_c_r'].item()/2
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

def predict_observe_update_as_lbs(e, em):
    """Predict, observe signal in chosen product-market, update mus, predict and observe profit, and decide action.
       LATENT BIT STATE (LBS): mu_b_b, mu_c_b, mu_a
       BIT STATE (BS): profit_b (predicted profit), low_profit_b, high_profit_b
       ATOM STATE (AS): product, market
    """
    # E step
    # PREDICT BAL: B[e]|A[e], L[e]
    for p in em.coords['PD'].values:
        for m in em.coords['MK'].values:
            em['profit_b'].loc[dict(PD=p, MK=m, ACT_PRED=e)] = mu2profit(em['mu_a'][e], em['mu_b_b'][e], em['mu_c_b'][e], p, m)
    
    em['low_profit_b'][e] = em['profit_b'][e2p[em['product'][e].item()], e2m[em['market'][e].item()], e].item() - em['k_sigma'].item() * em['sigma_profit'][e].item()
    em['high_profit_b'][e] = em['profit_b'][e2p[em['product'][e].item()], e2m[em['market'][e].item()], e].item() + em['k_sigma'].item() * em['sigma_profit'][e].item()
    
    # OBSERVE CA: C[A[e]]|A[e]
    em['profit_obs'][e] = sample_profit_obs(e, em) 
    em['action'][e] = decide_action(em['profit_obs'][e], em['low_profit_b'][e], em['high_profit_b'][e])
    
    # M step
    if em['action'][e] == "scale":
        return em
    
    elif e < em.dims['ACT_PRED']:     
        pivot_product_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_product.stan')
        pivot_market_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_market.stan')
        
        if em['action'][e] == "pivot_product":
            # UPDATE A ABC: A[e+1]| A[e], B[e], C[A[e]]            
            if e < em.dims['ACT_PRED']-1:
                em['product'][e+1] = 'man' if em['product'][e].item() == 'ai' else 'ai'
                em['market'][e+1] = em['market'][e].item()
            
            # UPDATE L LBC: L[e+1]| L[e], B[e], C[A[e]] (store posterior, set next stage prior)
            data = {
                'product': e2p[em.product[e].item()] + 1,
                'market': e2m[em.market[e].item()] + 1,
                'mu_b_b_mean': em.mu_b_b[e].item(),
                'mu_c_b': em.mu_c_b[e].item(),
                'mu_a_mean': em.mu_a[e].item(),
                'profit_obs': em['profit_obs'][e],
            }
            fit = pivot_product_model.sample(data=data)
            
            em['mu_a_post'][e] = fit.stan_variable('mu_a')
            em['mu_b_b_post'][e] = fit.stan_variable('mu_b_b')
            em['mu_c_b_post'][e] = em['mu_c_b_post'][e-1] if e > 0 else em['mu_c_b'][e]
            
            em['mu_b_b'][e+1] = em['mu_b_b_post'][e].mean()
            em['mu_c_b'][e+1] = em['mu_c_b'][e]
            em['mu_a'][e+1] = fit.stan_variable('mu_a').mean()          

        elif em['action'][e] == "pivot_market":
            # UPDATE A ABC: A[e+1]| A[e], B[e], C[A[e]]
            if e < em.dims['ACT_PRED']-1:
                em['market'][e+1] = 'b2b' if em['market'][e].item() == 'b2c' else 'b2c'
                em['product'][e+1] = em['product'][e].item()
            
            # UPDATE L LBC: L[e+1]| L[e], B[e], C[A[e]] (store posterior, set next stage prior)
            data = {
                'product': e2p[em.product[e].item()] + 1,
                'market': e2m[em.market[e].item()] + 1,
                'mu_c_b_mean': em.mu_c_b[e].item(),
                'mu_b_b': em.mu_b_b[e].item(),
                'mu_a_mean': em.mu_a[e].item(),  
                'profit_obs': em['profit_obs'][e],
            }
            fit = pivot_market_model.sample(data=data)

            em['mu_a_post'][e] = fit.stan_variable('mu_a')
            em['mu_c_b_post'][e] = fit.stan_variable('mu_c_b')
            em['mu_b_b_post'][e] = em['mu_b_b_post'][e-1] if e > 0 else em['mu_b_b'][e]
            
            em['mu_b_b'][e+1] = em['mu_b_b'][e]
            em['mu_c_b'][e+1] = em['mu_c_b_post'][e].mean()
            em['mu_a'][e+1] = fit.stan_variable('mu_a').mean()            
        
        # use A[e+1] reparameterization structure for L[e]
        em['profit_post'][e] = mu2profit(em['mu_a_post'][e], em['mu_b_b_post'][e], em['mu_c_b_post'][e], em['product'][e].item(), em['market'][e].item())
        if e == 0:
            em['profit_prior'][e] = np.random.normal(em['profit_b'].loc[dict(PD=em.product[e].item(), MK=em.market[e].item(), ACT_PRED=e)] , em['sigma_profit'][e], 4000)
        if e < em.dims['ACT_PRED']-1:
            em['profit_prior'][e+1] = mu2profit(em['mu_a_post'][e], em['mu_b_b_post'][e], em['mu_c_b_post'][e], em['product'][e+1].item(), em['market'][e+1].item())
            em['sigma_profit'][e+1] = em['profit_prior'][e+1].std()
    return em

def mu2profit(mu_a, mu_b, mu_c, product, market):
    """Return the profit based on the given parameters."""
    return mu_a + pow(-1, e2p[product]+1) * mu_b/2 + pow(-1, e2m[market]+1) * mu_c/2

def experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, mu_a, sigma_profit=.1, sigma_obs=.1, k=2, T=10, product='man', market='b2c'):
    """
    Record the expected reward from the experiment given initial parameters.
    mu_b_d = mu_b_b - mu_b_r  # belief and goal differ but we treat both as _b
    mu_c_d = mu_c_b - mu_c_r  
    """
    coords = {'HP': np.arange(1), 'P': np.arange(T+1), 'ACT_PRED': np.arange(T), 'ACT_PVT': np.arange(T), 'PD': ["man", "ai"], 'MK': ["b2c", "b2b"], 'M': np.arange(4000)}
    em_name = f"bB{np.round(mu_b_d,1)}_cC{np.round(mu_c_d,1)}_B{np.round(mu_b_r,1)}_C{np.round(mu_c_r,1)}_a{np.round(mu_a,1)}_s{np.round(sigma_profit,1)}_T{T}_{product}_{market}"
    file_path = f"data/experiment/{em_name}.nc"

    if os.path.exists(file_path):
        em = xr.open_dataset(file_path)
        print(f"File {em_name} already exists.")
        return em

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
        'profit_prior': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'profit_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_a_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_b_b_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        'mu_c_b_post': (('ACT_PRED', 'M'), np.zeros((T,4000))),
        # UPDATE ATOM STATE
        'action': ('ACT_PVT', np.full(T, '', dtype='object')),

        'mu_b_r': ('HP', np.zeros(1)),
        'mu_c_r': ('HP', np.zeros(1)),
        'k_sigma': ('HP', np.zeros(1)),
        'em_name': ((), em_name),
    }
    em = xr.Dataset(data_vars=data_vars, coords=coords)

    for t in range(em.dims['P']):
        if t == 0:
            em['mu_b_d'][0] = mu_b_d
            em['mu_c_d'][0] = mu_c_d
            em['mu_b_b'][0] =  mu_b_r + mu_b_d
            em['mu_c_b'][0] = mu_c_r + mu_c_d 
            em['mu_a'][0] = mu_a 
            em['mu_b_r'][0] = mu_b_r
            em['mu_c_r'][0] = mu_c_r
            em['sigma_obs'][0] = sigma_obs
            em['sigma_profit'][0] = sigma_profit

            em['mu_a_post'][0] = mu_a
            em['mu_b_b_post'][0] = em['mu_b_b'][0]
            em['mu_c_b_post'][0] = em['mu_c_b'][0]
            
        elif t == 1:
            em['product'][0] = product
            em['market'][0] = market

            em['k_sigma'][0] = k
            # em['low_profit_b'][0] = mu_a - k * em['sigma_obs'][0]
            # em['high_profit_b'][0] = mu_a + k * em['sigma_obs'][0]

            em = predict_observe_update_as_lbs(t-1, em) #time 1 is 0th experiment
        else:
            em = predict_observe_update_as_lbs(t-1, em)

        if em['action'][t-1].item() == "scale":
            break

    em.to_netcdf(f"data/experiment/{em.em_name.values}.nc")
    return em

if __name__ == "__main__":
    # Start the experiment
    em = experiment(mu_b_d= .1, mu_c_d= -.1, mu_b_r=.3, mu_c_r=.1, mu_a= .2, T=3)