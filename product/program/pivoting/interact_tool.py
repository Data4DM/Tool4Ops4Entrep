import arviz as az
import matplotlib.patches as patches
from ipywidgets import Layout
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from fit_pm import experiment
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, FloatSlider, VBox, HBox, HTML, Output
from IPython.display import display
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

global markets, products, e2p, e2m
markets = ["b2c", "b2b"]
products = ["man", "ai"]

e2p = {label: idx for idx, label in enumerate(products)}
e2m = {label: idx for idx, label in enumerate(markets)}


def plot_layer0_belief(em):
    """
    Plots the EM algorithm process with belief updates and profit predictions using the em xarray dataset.
    
    Parameters:
    - em: xarray dataset containing the required variables
    """
    num_exp = np.where(em['action'].values == "scale")[0][0] + 1 if "scale" in em['action'].values else em['action'].shape[0]
    experiments = np.arange(1, num_exp+1)
    
    mu_p_b = em['mu_p_b'].values[:num_exp]
    sigma_mu_p = em['sigma_mu_p'].values[:num_exp]
    mu_m_b = em['mu_m_b'].values[:num_exp]
    sigma_mu_m = em['sigma_mu_m'].values[:num_exp]

    fig, axs = plt.subplots(1, 2, figsize=(32, 8), sharey=True)
    
    fig.suptitle(em['em_name'].item())

    axs[0].fill_between(experiments, mu_p_b - 2 * sigma_mu_p,  mu_p_b + 2 * sigma_mu_p, color='green', alpha=0.2, label='$\pm2\sigma$')
    axs[0].plot(experiments, mu_p_b, marker='>',linestyle='--',  label='Updated $\mu_{product}$', color='green')
    axs[0].axhline(y=em['mu_p_r'].values[0], color='green', linestyle='-', linewidth=5, label='Ground Truth $\mu_{product}$')
    axs[0].set_title('Belief in $\mu_{product}$ by Time')
    axs[0].set_xlabel('Experiment')
    axs[0].set_xticklabels([str(int(exp)) for exp in experiments])
    axs[0].set_xticks(experiments)
    axs[0].set_ylabel('Belief on $\mu_{product}$')
    axs[0].legend(loc='lower left')
    axs[0].grid(True)

    axs[1].fill_between(experiments, mu_m_b - 2 * sigma_mu_m,  mu_m_b + 2 * sigma_mu_m, color='purple', alpha=0.2, label='$\pm2\sigma$')
    axs[1].plot(experiments, mu_m_b, marker='>',linestyle='--', label='Updated $\mu_{market}$', color='purple')
    axs[1].axhline(y=em['mu_m_r'].values[0], color='purple', linestyle='-', linewidth=5, label='Ground Truth $\mu_{market}$')
    axs[1].set_title('Belief in $\mu_{market}$ by Time')
    axs[1].set_xlabel('Experiment')
    axs[1].set_xticks(experiments)
    axs[1].set_xticklabels([str(int(exp)) for exp in experiments])
    axs[1].set_ylabel('Belief on $\mu_{market}$')
    axs[1].legend(loc='lower left')
    axs[1].grid(True)

    plt.tight_layout()
    figure_title = em['em_name'].item() + "_L0.png"
    figure_path = os.path.join("data/figure/em/l0", figure_title)
    plt.savefig(figure_path)
    # plt.show()
    plt.close()

def plot_layer1_profit(em):
    cell_colors = {'ai-b2c': 'green', 'man-b2c': 'gray', 'man-b2b': 'purple', 'ai-b2b': 'orange'}
    ACT_markers = {'pivot_product': 'B', 'pivot_market': 'M', 'scale': 'S'}

    # Create subplots
    num_exp = np.where(em['action'].values == "scale")[0][0] + 1 if "scale" in em['action'].values else em['action'].shape[0]
    
    fig = plt.figure(figsize=(10 * num_exp, 15), dpi=300)
    fig.suptitle(em['em_name'].item())
    gs = fig.add_gridspec(2, num_exp * 2, height_ratios=[1, 1]) # Making the grid spec with more columns
    axs = np.empty((2, num_exp), dtype=object)
    
    for col in range(num_exp):
        axs[0, col] = fig.add_subplot(gs[0, col * 2:(col + 1) * 2]) # First row
        axs[1, col] = fig.add_subplot(gs[1, col * 2:(col + 1) * 2]) # Second row

    fig.suptitle(em['em_name'].item())

    low_profit_b = em['low_profit_b'].values[:num_exp]
    high_profit_b = em['high_profit_b'].values[:num_exp]

    for t in range(num_exp):
        p = em['product'][t].item()
        m = em['market'][t].item()
        pm_idx = f'{p}-{m}'
        ACT = em['action'][t].item()

        # Posterior distributions
        profit_b_prior = em['profit_prior'][t].values
        # profit_b_posterior = em['profit_post'][t].values
        
        axs[0, t].hist(profit_b_prior, bins=30, alpha=0.1, color=cell_colors[pm_idx], edgecolor='white')
        # axs[0, t].hist(profit_b_posterior, bins=30, alpha=.4, color=cell_colors[pm_idx], edgecolor='white')
        axs[0, t].axvline(x=em['profit_obs'][t].item(), color=cell_colors[pm_idx], linestyle='-', label='observed profit')
        axs[0, t].axvline(x=low_profit_b[t], color=cell_colors[pm_idx], linestyle='--', linewidth=4, label='Low profit bar')
        axs[0, t].axvline(x=high_profit_b[t], color=cell_colors[pm_idx], linestyle='--', linewidth=4, label='High profit bar')
        
        axs[0, t].set_title(f'EXPERIMENT {t+1}:\n ENV {pm_idx}, ACTION {ACT}      ->')
        axs[0, t].legend(prop={'size':15})
        axs[0, t].set_xlim(-3, 3)
        axs[0, t].set_ylim(0, 500)

    cell_colors = {'man-b2c': 'gray', 'ai-b2c': 'green', 'man-b2b': 'purple', 'ai-b2b': 'orange'}  # updated colors

    for t in range(num_exp):
        ax = axs[1, t]
        pd_mk_combinations = [(pd, mk) for pd in em.coords['PD'].values for mk in em.coords['MK'].values]
        coords = [(1, 2), (2, 1), (2, 3), (3, 2)]  # Adjusted coordinates for diamond shapes

        for idx, (pd, mk) in enumerate(pd_mk_combinations):
            profit_b = np.round(em['profit_b'].loc[dict(PD=pd, MK=mk, PRED=t)].item(), decimals=1)
            profit_r = np.round(em['profit_r'].loc[dict(PD=pd, MK=mk)].item(), decimals=1)
            color = cell_colors[f'{pd}-{mk}']
            x, y = coords[idx]

            ax.fill([x-1, x, x+1, x], [y, y+1, y, y-1], color=color)
            ax.text(x, y, f"({profit_b},{profit_r})", ha='center', va='center', fontsize=30, color='white')

        ax.set_aspect('equal')
        ax.axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    figure_title = em['em_name'].item() + "_L1.png"
    figure_path = os.path.join("data/figure/em/l1", figure_title)
    plt.savefig(figure_path)
    # plt.show()
    plt.close()


from ipywidgets import FloatSlider, IntSlider, Output
import matplotlib.pyplot as plt

def interact_tool():
    # Define sliders for the parameters mu_sum and mu_diff
    mu_p2m_slider = FloatSlider(min=.2, max=2, step=.2, value=.2, continuous_update=False, description="mu_p2m")
    mu_sum_slider = FloatSlider(min=-4, max=0, step=1, value=-2, continuous_update=False, description="mu_sum")
    sigma_slider = IntSlider(min=1, max=4, step=1, value=1, continuous_update=False, description="observation uc")
    k_slider = IntSlider(min=1, max=4, step=1, value=1, continuous_update=False, description="decision uc")
    t_slider = IntSlider(min=2, max=4, step=1, value=2, continuous_update=False, description="experiment#")
    output = Output()

    def func(mu_p2m, mu_sum, sigma_profit, k_sigma, T, product='man', market='b2c'):
        em = experiment(mu_p2m, mu_sum, sigma_profit, k_sigma, T=T, product=product, market=market)
        fig0 = plot_layer0_belief(em)
        fig1 = plot_layer1_profit(em)
        return fig0, fig1

    def on_value_change(change):
        fig0, fig1 = func(
            mu_p2m_slider.value, 
            mu_sum_slider.value,
            sigma_profit=sigma_slider.value,
            k_sigma=k_slider.value,
            T=t_slider.value  # Assuming T is a constant or you can add a slider for T if needed
        )
        with output:
            output.clear_output()
            plt.show(fig0)
            plt.show(fig1)

    # Link the sliders to the on_value_change function
    for slider in [mu_p2m_slider, mu_sum_slider, sigma_slider, k_slider, t_slider]:
        slider.observe(on_value_change, names='value')
    
    # Arrange sliders and output in a VBox layout
    interactive_simulation = VBox([
        HTML("<h1>Pivot Game</h1>"),
        HBox([
            VBox([HTML("<b>Parameters</b>"), mu_p2m_slider, mu_sum_slider, sigma_slider, k_slider,]),
        ]),
        output
    ])

    return interactive_simulation

if __name__ == "__main__":
    # func(mu_p_d= .2, mu_m_d= -.1, sigma_obs=.1, T=4, product='man', market='b2c')
    # th = xr.open_dataset("data/th/bB[-0.2  0.   0.2]_cC[-0.2  0.   0.2]_a[0.4]_b0.2_c0.1_s0.1_cash4_E5_man_b2c")
    # plot_th_given_experiment(th)
    
    em = experiment(mu_p2m = 1.65, mu_sum = 4, sigma_profit=2, k_sigma=.5, T=3, product = 'man', market = 'b2c')
    # em = xr.open_dataset("data/experiment/bB-0.2_cC-0.2_B0.3_C0.3_a0.1_s0.1_T10_man_b2c.nc")
    plot_layer0_belief(em)
    plot_layer1_profit(em)