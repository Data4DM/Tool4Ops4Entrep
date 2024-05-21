import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmdstanpy

# Example labels for markets and products
global markets, products, e2p, e2m
markets = ["OM", "ENT"]
products = ["PC", "ECON"]

e2p = {label: idx for idx, label in enumerate(products)}
e2m = {label: idx for idx, label in enumerate(markets)}

def sample_rev_cost_signal_obs(e):
    """Return the counts for updating mu parameters based on the experiment index."""
    # if sigma_known:
    #     # Given posterior = prior + weight * (observation - prior)
    #     # reverse engineer observation:
    #     sigma_mu, sigma_obs = np.sqrt(0.01), np.sqrt(0.19)
    #     weight  = sigma_mu**2 / (sigma_mu**2 + sigma_obs**2)
    #     # $\mu_{\text {posterior }}=\mu_{\text {prior }}+\frac{\sigma^2}{\sigma^2+\sigma_o^2} \times\left(x-\mu_{\text {prior }}\right)$
    #     observation = (mu_posterior[e] - mu_prior[e]) / weight + mu_prior[e]
    #     return observation
    # else: 
    if e == 0:
        count_cp_pos, count_cp_neg = 0, 2
        count_ce_pos, count_ce_neg = 0, 0
        count_ro_pos, count_ro_neg = 0, 0
        count_re_pos, count_re_neg = 0, 2
        count_s_pos, count_s_neg = 0, 0
    elif e == 1:
        count_cp_pos, count_cp_neg = 1, 0
        count_ce_pos, count_ce_neg = 0, 0
        count_ro_pos, count_ro_neg = 0, 1
        count_re_pos, count_re_neg = 1, 0
        count_s_pos, count_s_neg = 0, 0
    elif e == 2:
        count_cp_pos, count_cp_neg = 0, 0
        count_ce_pos, count_ce_neg = 0, 0
        count_ro_pos, count_ro_neg = 0, 0
        count_re_pos, count_re_neg = 0, 0
        count_s_pos, count_s_neg = 0, 1
    elif e == 3:
        count_cp_pos, count_cp_neg = 1, 2
        count_ce_pos, count_ce_neg = 0, 0
        count_ro_pos, count_ro_neg = 1, 0
        count_re_pos, count_re_neg = 0, 3
        count_s_pos, count_s_neg = 0, 0
    else:
        count_cp_pos, count_cp_neg = 0, 0
        count_ce_pos, count_ce_neg = 0, 0
        count_ro_pos, count_ro_neg = 0, 0
        count_re_pos, count_re_neg = 0, 0
        count_s_pos, count_s_neg = 0, 0

    return [-(count_cp_pos-count_cp_neg)+(count_ce_pos-count_ce_neg), (count_ro_pos-count_ro_neg)- (count_re_pos-count_re_neg), count_s_pos-count_s_neg]

def sample_profit_obs(e, MPN):
    """Return the profit observation based on the experiment index and product-market pair."""
    
    p = (
        MPN['mu_dv_r'].item() +
        pow(-1, e2p[MPN['product'][e].item()]) * MPN['mu_p2e_r'].item() +
        pow(-1, e2m[MPN['market'][e].item()]) * MPN['mu_o2e_r'].item()
    )
    # if p < 0 or p > 1:
    #     warnings.warn(f"Invalid probability value: {p}. Clipping to the valid range (0, 1).", UserWarning)
    #     p = np.clip(p, 0, 1)

    # profit_obs = np.random.binomial(1, p)
    return p #profit_obs


def decide_action(cash, profit_obs, low_profit_b, high_profit_b):
    """Decide the next action based on the current cash state and profit."""
    if cash <= 0:
        return "pivot_self"
    elif low_profit_b <= profit_obs <= high_profit_b:
        return "pivot_product"
    elif profit_obs < low_profit_b:
        return "pivot_market"
    elif profit_obs > high_profit_b:
        return "pivot_self"
    
def observe_update_belief_predict_act_update_pm(e, MPN, sigma_known = False):
    """Observe signal in chosen product-market, update mus, make prediction on profit, low and high range, and decide action."""
    # Observe signal in chosen product-market
    if sigma_known:
        #sigma_mu, sigma_obs = np.sqrt(0.01), np.sqrt(0.19)
        MPN['sigma_mu'][e] = np.sqrt(0.01)
        MPN['sigma_obs'][e] = np.sqrt(0.19)
        MPN['mu_p2e_b'][e] = MPN['mu_p2e_b'][e-1].item() + .05 * sample_rev_cost_signal_obs(e)[0]
        MPN['mu_o2e_b'][e] = MPN['mu_p2e_b'][e-1].item() + .05 * sample_rev_cost_signal_obs(e)[1]
        MPN['mu_dv_b'][e] = MPN['mu_dv_b'][e-1].item() + .2 * sample_rev_cost_signal_obs(e)[2]   
        
    # else:
        # if e ==0:
        #     pass
        # else:
        #     weight  = MPN['sigma_mu'][e]**2 / (MPN['sigma_mu'][e]**2 + MPN['sigma_obs'][e]**2)
        #     MPN['mu_p2e_b'][e] = MPN['mu_p2e_b'][e-1].item() + weight * sample_rev_cost_signal_obs(e)[0]
        #     MPN['mu_o2e_b'][e] = MPN['mu_p2e_b'][e-1].item() + weight * sample_rev_cost_signal_obs(e)[1]
        #     MPN['mu_dv_b'][e] = MPN['mu_dv_b'][e-1].item() + weight * sample_rev_cost_signal_obs(e)[2]  
        
    # Make prediction on profit, low and high range with posterior
    MPN['profit_b'][e] = MPN['mu_dv_b'][e] + pow(-1, e2p[MPN['product'][e].item()]) * MPN['mu_o2e_b'][e] + pow(-1, e2m[MPN['market'][e].item()]) * MPN['mu_o2e_b'][e]
    MPN['low_profit_b'][e] = MPN['profit_b'][e] - 2 * MPN['sigma_mu'][e]
    MPN['high_profit_b'][e] = MPN['profit_b'][e] + 2 * MPN['sigma_mu'][e]

    # Decide action based on predicted and observed profit
    action = decide_action(MPN['cash'][e].item(), sample_profit_obs(e, MPN), MPN['low_profit_b'][e], MPN['high_profit_b'][e])
    MPN['action'][e] = action

    # Update pm
    pivot_product_model = cmdstanpy.CmdStanModel(stan_file='simulation/stan/pivot_product.stan')
    pivot_market_model = cmdstanpy.CmdStanModel(stan_file='simulation/stan/pivot_market.stan')
    pivot_self_model = cmdstanpy.CmdStanModel(stan_file='simulation/stan/pivot_self.stan')

    data = {
    'profit_obs': sample_profit_obs(e, MPN),
    'market': e2m[MPN.market[e].item()] + 1,
    'product': e2p[MPN.product[e].item()] + 1,
    
    'mu_p2e_b': MPN.mu_p2e_b[e].item(),
    'mu_dv_b': MPN.mu_dv_b[e].item(),
    'sigma_obs': MPN.sigma_obs.item(),
    'sigma_mu': MPN.sigma_mu[e].item()
    }
    
    if action == "pivot_self":
        
        data = {
            'profit_obs': sample_profit_obs(e, MPN),
            'market': e2m[MPN.market[e].item()] + 1,
            'product': e2p[MPN.product[e].item()] + 1,
            'mu_o2e_b': MPN.mu_o2e_b[e].item(),
            'mu_p2e_b': MPN.mu_p2e_b[e].item(),
            'mu_dv_b_mean': MPN.mu_dv_b[e].item(),
            'sigma_obs': MPN.sigma_obs.item(),
            # 'sigma_mu': MPN.sigma_mu[e].item()
        }
        fit = pivot_self_model.sample(data=data, show_console= True)
        MPN['mu_dv_b'][e] = fit.stan_variable('mu_dv_b').mean()
        MPN['product'][e + 1] = MPN['product'][e].item()
        MPN['market'][e + 1] = MPN['market'][e].item()
        MPN['cash'][e+1] = 2
        
    elif action == "pivot_product":
        
        data = {
        'profit_obs': sample_profit_obs(e, MPN),
        'market': e2m[MPN.market[e].item()] + 1,
        'product': e2p[MPN.product[e].item()] + 1,
        'mu_o2e_b': MPN.mu_o2e_b[e].item(),
        'mu_p2e_b_mean': MPN.mu_p2e_b[e].item(),
        'mu_dv_b': MPN.mu_dv_b[e].item(),
        'sigma_obs': MPN.sigma_obs.item(),
        # 'sigma_mu': MPN.sigma_mu[e].item()
        }
        fit = pivot_product_model.sample(data=data, show_console= True)
        MPN['mu_p2e_b'][e] = fit.stan_variable('mu_p2e_b').mean()
        MPN['product'][e + 1] = 'ECON' if MPN['product'][e].item() == 'PC' else 'PC'
        MPN['market'][e + 1] = MPN['market'][e].item()
        MPN['cash'][e+1] = MPN['cash'][e] - 1

    elif action == "pivot_market":
        
        data = {
        'profit_obs': sample_profit_obs(e, MPN),
        'market': e2m[MPN.market[e].item()] + 1,
        'product': e2p[MPN.product[e].item()] + 1,
        'mu_o2e_b_mean': MPN.mu_o2e_b[e].item(),
        'mu_p2e_b': MPN.mu_p2e_b[e].item(),
        'mu_dv_b': MPN.mu_dv_b[e].item(),
        'sigma_obs': MPN.sigma_obs.item(),
        # 'sigma_mu': MPN.sigma_mu[e].item()
        }
        
        fit = pivot_market_model.sample(data=data, show_console= True)
        MPN['mu_o2e_b'][e] = fit.stan_variable('mu_o2e_b').mean()
        MPN['market'][e + 1] = 'ENT' if MPN['market'][e].item() == 'OM' else 'OM'
        MPN['product'][e + 1] = MPN['product'][e].item()
        MPN['cash'][e+1] = MPN['cash'][e] - 1
    
    # MPN['sigma_mu'][e] = fit.stan_variable('sigma_mu').mean()
    # MPN['sigma_obs'][e] = fit.stan_variable('sigma_obs').mean()

    return MPN


def start_experiment(mu_dv_b=.7, mu_p2e_b=.1, mu_o2e_b=0, sigma_mu=np.sqrt(0.01), sigma_obs=np.sqrt(0.19), product='PC', market='OM', mu_dv_r=.5, mu_p2e_r=.2, mu_o2e_r=-.3, cash=2, E=5):
    """
    Start the experiment by processing actions for each experiment.

    Parameters:
    - num_experiments: The number of experiments to run.

    Returns:
    - The updated xarray.Dataset after processing all actions.
    """
    coords = {'H': np.arange(1), 'E': np.arange(E)}
    MPN_name = f"cash{cash}_mu_dv{mu_dv_b}_mu_p2e{mu_p2e_b}_mu_o2e{mu_o2e_b}_sigma_mu{sigma_mu}"
    data_vars = {
        'action': ('E', np.full(E, '', dtype='object')),
        'market': ('E', np.full(E, '', dtype='object')),
        'product': ('E', np.full(E, '', dtype='object')),

        'mu_dv_b': ('E', np.zeros(E)),
        'mu_p2e_b': ('E', np.zeros(E)),
        'mu_o2e_b': ('E', np.zeros(E)),
        'sigma_mu': ('E', np.zeros(E)),
        'profit_b': ('E', np.zeros(E)),
        'low_profit_b': ('E', np.zeros(E)),
        'high_profit_b': ('E', np.zeros(E)),

        'mu_dv_r': ('H', np.zeros(1)),
        'mu_p2e_r': ('H', np.zeros(1)),
        'mu_o2e_r': ('H', np.zeros(1)),
        'sigma_obs': ('H', np.zeros(1)),
        'profit_obs': ('E', np.zeros(E)),

        'cash': ('E', np.zeros(E, dtype=int)),
        'MPN_name': ((), MPN_name),
    }
    MPN = xr.Dataset(data_vars=data_vars, coords=coords)
    MPN['cash'][0] = cash
    MPN['product'][0] = product
    MPN['market'][0] = market
    MPN['mu_dv_b'][0] = mu_dv_b
    MPN['mu_p2e_b'][0] = mu_p2e_b
    MPN['mu_o2e_b'][0] = mu_o2e_b
    MPN['sigma_mu'][0] = sigma_mu
    MPN['sigma_obs'][0] = sigma_obs
    MPN['mu_dv_r'][0] = mu_dv_r
    MPN['mu_p2e_r'][0] = mu_p2e_r
    MPN['mu_o2e_r'][0] = mu_o2e_r
        
    for e in range(E):
        observe_update_belief_predict_act_update_pm(e, MPN)
    MPN.to_netcdf(f"xarray_data/{MPN.MPN_name.values}.nc")
    return MPN
if __name__ == "__main__":
    # Start the experiment
    MPN = start_experiment()