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
    return mu2profit(em['mu_p_r'].item(), em['mu_m_r'].item(), em['product'][t].item(), em['market'][t].item())


def decide_action(profit_obs, low_profit_b, high_profit_b, CR):
    """Decide the next action based on predicted and observed profit and cost considerations."""

    if CR > 1:  # pivot_product cost > pivot_market cost (deeptech, 🐣 biotech with core technology)
        if profit_obs > high_profit_b:
            return "scale"
        elif low_profit_b <= profit_obs <= high_profit_b:
            return "pivot_market"
        else: 
            return "pivot_product" # angie won't abandon prob.comp unless sun rises from the west

    else:  # pivot_product cost <= pivot_market cost (it, 🦖big pharma with rigidity on distribution channel)
        if profit_obs > high_profit_b:
            return "scale" 
        elif low_profit_b <= profit_obs <= high_profit_b:
            return "pivot_product"
        else: 
            return "pivot_market"


def mu2profit(mu_p, mu_m, product, market):
    """Return the profit based on the given parameters."""
    return pow(-1, e2p[product]+1) * mu_p/2 + pow(-1, e2m[market]+1) * mu_m/2


def predict_observe_update_as_lbs(t, em):
    """Predict, observe signal in chosen product-market, update mus, predict and observe profit, and decide action.
       externally, TIME = 1 but e = 0
       LATENT BIT STATE (LBS): mu_p_b, mu_m_b
       BIT STATE (BS): profit_b (predicted profit), low_profit_b, high_profit_b
       ATOM STATE (AS): product, market
    """
    # E step
    # PREDICT BAL: B[t]|A[t], L[t]
    for p in em.coords['PD'].values:
        for m in em.coords['MK'].values:
            em['profit_b'].loc[dict(PD=p, MK=m, PRED=t)] = mu2profit(em['mu_p_b'][t], em['mu_m_b'][t], p, m)
    
    em['low_profit_b'][t] = em['profit_b'][e2p[em['product'][t].item()], e2m[em['market'][t].item()], t].item() - em['k_sigma'].item() * em['sigma_profit'][t].item()
    em['high_profit_b'][t] = em['profit_b'][e2p[em['product'][t].item()], e2m[em['market'][t].item()], t].item() + em['k_sigma'].item() * em['sigma_profit'][t].item()
    
    # OBSERVE CA: C[A[t]]|A[t]
    em['profit_obs'][t] = sample_profit_obs(t, em) 
    em['action'][t] = decide_action(em['profit_obs'][t], em['low_profit_b'][t], em['high_profit_b'][t], em['CR'][0])
    
    # M step
    if em['action'][t] == "scale":
        return em
    
    pivot_product_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_product.stan')
    pivot_market_model = cmdstanpy.CmdStanModel(stan_file='stan/pivot_market.stan')
    
    if em['action'][t] == "pivot_product":
        # UPDATE A ABC: A[t+1]| A[t], B[t], C[A[t]]            
        # if t < em.dims['PRED']-1:
        em['product'][t+1] = 'man' if em['product'][t].item() == 'ai' else 'ai'
        em['market'][t+1] = em['market'][t].item()
            
        # UPDATE L LBC: L[t+1]| L[t], B[t], C[A[t]] (store posterior, set next stage prior)
        data = {
            'product': e2p[em.product[t].item()] + 1,
            'market': e2m[em.market[t].item()] + 1,
            'mu_p_b_mean': em.mu_p_b[t].item(),
            'mu_m_b': em.mu_m_b[t].item(),
            'profit_obs': em['profit_obs'][t],
        }
        fit = pivot_product_model.sample(data=data)
        
        em['mu_p_b_prior'][t+1] = fit.stan_variable('mu_p_b') #posterior
        em['mu_m_b_prior'][t+1] = em['mu_m_b_prior'][t]

        em['mu_p_b'][t+1] = em['mu_p_b_prior'][t+1].mean()
        em['mu_m_b'][t+1] = em['mu_m_b'][t]    

        em['sigma_mu_p'][t+1] = em['mu_p_b_prior'][t+1].std()     
        em['sigma_mu_m'][t+1] = em['sigma_mu_m'][t]

    elif em['action'][t] == "pivot_market":
        # UPDATE A ABC: A[t+1]| A[t], B[t], C[A[t]]
        # if t < em.dims['PRED']-1:
        em['market'][t+1] = 'b2b' if em['market'][t].item() == 'b2c' else 'b2c'
        em['product'][t+1] = em['product'][t].item()
            
        # UPDATE L LBC: L[t+1]| L[t], B[t], C[A[t]] (store posterior, set next stage prior)
        data = {
            'product': e2p[em.product[t].item()] + 1,
            'market': e2m[em.market[t].item()] + 1,
            'mu_m_b_mean': em.mu_m_b[t].item(),
            'mu_p_b': em.mu_p_b[t].item(),
            'profit_obs': em['profit_obs'][t],
        }
        fit = pivot_market_model.sample(data=data)

        em['mu_m_b_prior'][t+1] = fit.stan_variable('mu_m_b') #posterior
        em['mu_p_b_prior'][t+1] = em['mu_p_b_prior'][t]

        em['mu_m_b'][t+1] = em['mu_m_b_prior'][t+1].mean()
        em['mu_p_b'][t+1] = em['mu_p_b'][t]

        em['sigma_mu_p'][t+1] = em['sigma_mu_p'][t]
        em['sigma_mu_m'][t+1] = em['mu_m_b_prior'][t+1].std()

    # use A[t+1] reparameterization structure for L[t]
    # em['profit_post'][t] = mu2profit(em['mu_m_b'][t+1], em['mu_p_b'][t+1], em['product'][t].item(), em['market'][t].item())
    em['profit_prior'][t+1] = mu2profit(em['mu_p_b_prior'][t+1], em['mu_m_b_prior'][t+1], em['product'][t+1].item(), em['market'][t+1].item())
    em['sigma_profit'][t+1] = em['profit_prior'][t+1].std() # CHECKED em['sigma_profit'][t+1]**2 (.67) ~  (em['sigma_mu_p'][t+1]**2 + em['sigma_mu_m'][t+1]**2)/4 (.69)
    
    return em


def experiment(mu_p2m, mu_sum, sigma_profit, k_sigma, T, product='man', market='b2c', CR = 2):
    """
    Record the expected reward from the experiment given initial parameters.
    mu_p_d = (mu_sum * mu_p2m)/ (mu_p2m + 1)
    mu_m_d = mu_sum/ (mu_p2m + 1)

    mu_p_d = mu_p_r - mu_p_b (=0) = mu_p_r # belief and goal differ but we treat both as _b
    mu_m_d = mu_m_r - mu_m_b (=0) = mu_m_r
    """
    coords = {'H': np.arange(1), 'B': np.arange(T+1), 'PRED': np.arange(T), 'ACT': np.arange(T), 'PD': ["man", "ai"], 'MK': ["b2c", "b2b"], 'M': np.arange(4000)}
    em_name = f"p2m-ratio{mu_p2m}_sum{mu_sum}_sigma{sigma_profit}_k-sigma{k_sigma}_exp{T}_CR{CR}_{product}_{market}"
    file_path = f"data/experiment/{em_name}.nc"

    # if os.path.exists(file_path):
    #     em = xr.open_dataset(file_path)
    #     print(f"File {em_name} already exists.")
    #     return em

    data_vars = {
        'mu_p_d': ('B', np.zeros(T+1)), 
        'mu_m_d': ('B', np.zeros(T+1)),
        'mu_p_b': ('B', np.zeros(T+1)),
        'mu_m_b': ('B', np.zeros(T+1)),
        'sigma_profit': ('B', np.zeros(T+1)),
        'sigma_mu_p':('B', np.zeros(T+1)), # belief before1, ..., beforeT (=T), predicted
        'sigma_mu_m':('B', np.zeros(T+1)),

        # PREDICT
        'market': ('B', np.full(T+1, '', dtype='object')), #entrant randomly choose (same Eprofit currently (belief comes from future))
        'product': ('B', np.full(T+1, '', dtype='object')),

        'profit_b': (('PD', 'MK', 'PRED'), np.zeros((2, 2, T))),
        'low_profit_b': ('PRED', np.zeros(T)),
        'high_profit_b': ('PRED', np.zeros(T)),

        # OBSERVE
        'profit_obs': ('PRED', np.zeros(T)),

        # UPDATED BIT STATE
        'profit_prior': (('B', 'M'), np.zeros((T+1,4000))), # even if no more experiment opp, it updates belief (parameter)
        # 'profit_post': (('PRED', 'M'), np.zeros((T,4000))),

        'mu_p_b_prior': (('B', 'M'), np.zeros((T+1,4000))), # updating belief is expensive - do only after observation
        'mu_m_b_prior': (('B', 'M'), np.zeros((T+1,4000))),

        # UPDATE ATOM STATE
        'action': ('ACT', np.full(T, '', dtype='object')),
        
        'mu_p_r': ('H', np.zeros(1)),
        'mu_m_r': ('H', np.zeros(1)), 
        'k_sigma': ('H', np.zeros(1)),
        'CR': ('H', np.zeros(1)),
        'em_name': ((), em_name),
    }
    em = xr.Dataset(data_vars=data_vars, coords=coords)
    em['profit_r'] = (('PD', 'MK'), np.zeros((len(em.coords['PD']), len(em.coords['MK']))))

    for t in range(em.dims['PRED']): # t=0,1,2
        if t == 0:
            em['mu_p_r'][0] = (mu_sum * mu_p2m)/ (mu_p2m + 1)
            em['mu_m_r'][0] = mu_sum/ (mu_p2m + 1)
            em['mu_p_b'][0] = 0 
            em['mu_m_b'][0] = 0 
            em['mu_p_d'][0] = em['mu_p_r'][0]
            em['mu_m_d'][0] = em['mu_m_r'][0]
            em['CR'][0] = CR #cost_pivot_product / cost_pivot_market; 🐣CR=2, 🦖CR=.5

            em['sigma_mu_p'][0] = sigma_profit * np.sqrt(2)
            em['sigma_mu_m'][0] = sigma_profit * np.sqrt(2)
            em['sigma_profit'][0] = sigma_profit
            
            em['mu_p_b_prior'][0] = np.random.normal(0, em['sigma_mu_p'][0], 4000)
            em['mu_m_b_prior'][0] = np.random.normal(0, em['sigma_mu_m'][0], 4000)
            em['profit_prior'][0] = np.random.normal(0, em['sigma_profit'][0], 4000)

            for p in em.coords['PD'].values:           
                for m in em.coords['MK'].values:
                    em['profit_r'].loc[dict(PD=p, MK=m)] = mu2profit(em['mu_p_r'].item(), em['mu_m_r'].item(), p, m)
            
            em['product'][0] = product
            em['market'][0] = market
            em['k_sigma'][0] = k_sigma

        em = predict_observe_update_as_lbs(t, em)

        if em['action'][t].item() == "scale":
            return em

    em.to_netcdf(f"data/experiment/{em.em_name.values}.nc")
    return em

if __name__ == "__main__":     
    em = experiment(mu_p2m = 3, mu_sum = 4, sigma_profit=1, k_sigma=3, T=2, product = 'man', market = 'b2c')