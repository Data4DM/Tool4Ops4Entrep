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

global markets, products, e2p, e2m
markets = ["b2c", "b2b"]
products = ["man", "ai"]

e2p = {label: idx for idx, label in enumerate(products)}
e2m = {label: idx for idx, label in enumerate(markets)}

def plot_layer1_profit(em):
    P = em.dims['P']
    ACT_PRED = em.dims['ACT_PRED']
    ACT_PVT = em.dims['ACT_PVT']
    ACT_PVT_colors = {'scale': 'red', 'pivot_market': 'purple', 'pivot_product': 'green'}
    
    fig, axs = plt.subplots(1, ACT_PRED + 1, figsize=(4 * (ACT_PRED + 1), 3))

    profit_b = em['profit_b'].values
    low_profit_b = em['low_profit_b'].values
    high_profit_b = em['high_profit_b'].values

    profit_obs = em['profit_obs'].values
    actions = em['action'].values

    # Plot posterior distributions
    for t in range(ACT_PRED):
        profit_b_prior = em['profit_prior'][t].values
        profit_b_posterior = em['profit_post'][t].values
        
        axs[t].hist(profit_b_prior, bins=30, alpha=0.3, color='blue', edgecolor='white')
        axs[t].hist(profit_b_posterior, bins=30, alpha=0.3, color='skyblue', edgecolor='white')

        axs[t].axvline(x=em['profit_b'][e2p[em['product'][t].item()], e2m[em['market'][t].item()], t].item(), color='blue', linestyle=':', label='expected')
        axs[t].axvline(x=em['profit_obs'][t].item(), color=mcolors.to_rgba('purple', alpha=0.8), linestyle='-', label='observed')
        
        axs[t].axvline(x=profit_b_prior.mean(), color='skyblue', linestyle='--', label='prior mean')
        axs[t].axvline(x=profit_b_posterior.mean(), color='blue', linestyle='--', label='posterior mean')
        
        axs[t].set_title(f'time {t+1} profit experiment')
        axs[t].legend()
        axs[t].set_xlim(-.3, .6)
        axs[t].set_ylim(0, 500)

    # Initial plot for profit feedback with action dots
    
    predicted_profits = [profit_b[e2p[em['product'][0].item()], e2m[em['market'][0].item()], 0]]
    for t in range(ACT_PRED):
        predicted_profits.append(profit_b[e2p[em['product'][t].item()], e2m[em['market'][t].item()], t])
    
    axs[ACT_PRED].scatter(np.array(range(1, ACT_PRED + 1)), list(profit_obs), color=mcolors.to_rgba('purple', alpha=0.8), label='observed', marker='o')
    axs[ACT_PRED].plot(list(range(ACT_PRED + 1)), predicted_profits, color='blue', label='expected', linestyle=':')
    axs[ACT_PRED].fill_between(list(range(1, ACT_PRED + 1)), list(low_profit_b), list(high_profit_b), color='skyblue', alpha=0.3, label='low-high bar')
    
    ACT_PVT_markers = {'pivot_product': 'P', 'pivot_market': 'M', 'scale': 'S'}
    for i in range(ACT_PVT):
        markers = [ACT_PVT_markers.get(action, 'o') for action in actions]
        for j, marker in enumerate(markers):
            axs[ACT_PRED].scatter(j + 1.2, -0.25, color='red', s=50, marker=f'${marker}$')

    axs[ACT_PRED].set_title('EM alg. to max expected profit')
    axs[ACT_PRED].set_ylim(-.3, .6)
    axs[ACT_PRED].legend(loc='upper left')

    plt.tight_layout()
    figure_title = em['em_name'].item() + "_L1.png"
    figure_path = os.path.join("data/figure/interact", figure_title)
    plt.savefig(figure_path)
    plt.show()
    plt.close()




def plot_layer0_belief(em):
    """
    Plots the EM algorithm process with belief updates and profit predictions using the em xarray dataset.
    
    Parameters:
    - em: xarray dataset containing the required variables
    """
    num_time_points = em.dims['P']
    time_points = np.arange(1, num_time_points)
    products = ["man", "ai"]
    markets = ["b2c", "b2b"]

    # Extracting belief values from the em dataset
    mu_b_b = em['mu_b_b'].values
    mu_c_b = em['mu_c_b'].values
    mu_a = em['mu_a'].values
    
    # Extracting ground truth values from the em dataset
    mu_b_r = em['mu_b_r'].values[0]
    mu_c_r = em['mu_c_r'].values[0]
    
    # Extracting posterior samples from the em dataset
    mu_a_post = em['mu_a_post'].values
    mu_b_b_post = em['mu_b_b_post'].values
    mu_c_b_post = em['mu_c_b_post'].values

    # Extracting profit values from the em dataset
    profits_updated = em['profit_b'].values

    # Calculate credible intervals
    credible_interval = 0.94
    mu_a_hdi = np.array([az.hdi(mu_a_post[i], hdi_prob=credible_interval) for i in range(mu_a_post.shape[0])])
    mu_b_b_hdi = np.array([az.hdi(mu_b_b_post[i], hdi_prob=credible_interval) for i in range(mu_b_b_post.shape[0])])
    mu_c_b_hdi = np.array([az.hdi(mu_c_b_post[i], hdi_prob=credible_interval) for i in range(mu_c_b_post.shape[0])])

    # Create subplots to put the plots side by side
    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    # Plotting the parameter updates with ground truth on separate subplots
    az.plot_hdi(time_points, mu_b_b_post.T, hdi_prob=0.94, ax=axes[0], color='green', fill_kwargs={'alpha': 0.1})
    axes[0].plot(time_points, mu_b_b[1:], marker='o', label='Updated $\mu_{b}$ (production cost gap)', color='green')
    axes[0].fill_between(time_points, mu_b_b_hdi[:, 0], mu_b_b_hdi[:, 1], color='green', alpha=0.3)
    axes[0].axhline(y=mu_b_r, color='green', linestyle='--', label='Ground Truth $\mu_{b}$')
    axes[0].set_title('Belief in $\mu_{b}$ by Time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Belief on $\mu_{b}$')
    axes[0].legend(loc='lower left')
    axes[0].grid(True)

    az.plot_hdi(time_points, mu_c_b_post.T, hdi_prob=0.94, ax=axes[1], color='purple', fill_kwargs={'alpha': 0.1})
    axes[1].plot(time_points, mu_c_b[1:], marker='o', label='Updated $\mu_{c}$ (market revenue gap)', color='purple')
    axes[1].fill_between(time_points, mu_c_b_hdi[:, 0], mu_c_b_hdi[:, 1], color='purple', alpha=0.3)
    axes[1].axhline(y=mu_c_r, color='purple', linestyle='--', label='Ground Truth $\mu_{c}$')
    axes[1].set_title('Belief in $\mu_{c}$ by Time')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Belief on $\mu_{c}$')
    axes[1].legend(loc='lower left')
    axes[1].grid(True)

    az.plot_hdi(time_points, mu_a_post.T, hdi_prob=0.94, ax=axes[2], color='red', fill_kwargs={'alpha': 0.1})
    axes[2].plot(time_points, mu_a[1:], marker='o', label='Updated $\mu_{a}$ (baseline profit)', color='red')
    axes[2].fill_between(time_points, mu_a_hdi[:, 0], mu_a_hdi[:, 1], color='red', alpha=0.3)
    axes[2].set_title('Belief in $\mu_{a}$ by Time')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Belief on $\mu_{a}$')
    axes[2].legend(loc='lower left')
    axes[2].grid(True)

    bar_width = 0.2
    x = np.arange(len(time_points))  # the label locations

    profits_updated_reordered = np.zeros((4, len(time_points)))
    index = 0
    for p in em.coords['PD'].values:
        for m in em.coords['MK'].values:
            profits_updated_reordered[index] = em['profit_b'].loc[dict(PD=p, MK=m)].values
            index += 1

    axes[3].bar(x - 1.5*bar_width, profits_updated_reordered[0], bar_width, label='man-b2c', color='#006400')
    axes[3].bar(x - 0.5*bar_width, profits_updated_reordered[1], bar_width, label='ai-b2c', color='#32CD32')
    axes[3].bar(x + 0.5*bar_width, profits_updated_reordered[2], bar_width, label='man-b2b', color='#FFD700')
    axes[3].bar(x + 1.5*bar_width, profits_updated_reordered[3], bar_width, label='ai-b2b', color='#FFA500')

    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Belief on Profit')
    axes[3].set_title('BTS: Belief by Time and Space')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(time_points)
    axes[3].legend()

    plt.tight_layout()
    plt.show()
    figure_title = em['em_name'].item() + "_L0.png"
    figure_path = os.path.join("data/figure/interact", figure_title)
    plt.savefig(figure_path)

    # # EM Algorithm Process
    # steps = ['Predict (Expectation)', 'Observe (Expectation)', 'Update A (Maximization)', 'Update L (Maximization)']
    # arrow_positions = [(0.5, 0.85, 0.5, 0.75), (0.5, 0.65, 0.5, 0.55), (0.5, 0.45, 0.5, 0.35), (0.5, 0.25, 0.5, 0.15)]
    # arrow_colors = ['blue', 'green', 'red', 'purple']

    # for step, (x1, y1, x2, y2), color in zip(steps, arrow_positions, arrow_colors):
    #     axes[2].annotate('', xy=(x2, y2), xytext=(x1, y1),
    #                      arrowprops=dict(facecolor=color, edgecolor=color, shrink=0.05, width=2))
    #     axes[2].text(x2, (y1 + y2) / 2, step, ha='center', va='center', fontsize=10, color=color, weight='bold')

    # Layer labels
    # layers = ['Layer 3 (Market\'s signal)', 'Layer 2 (Product-Market)', 'Layer 1 (Product\'s expected profitability)', 'Layer 0 (Belief)']
    # y_positions = [0.9, 0.7, 0.5, 0.3]
    # colors = ['purple', 'red', 'blue', 'green']

    # for i, (layer, y, color) in enumerate(zip(layers, y_positions, colors)):
    #     axes[2].add_patch(patches.FancyBboxPatch((0.1, y - 0.05), 0.8, 0.1, boxstyle="round,pad=0.05", linewidth=.1, edgecolor='None', facecolor=color))
    #     axes[2].text(0.5, y, layer, ha='center', va='center', fontsize=12, weight='bold')

    # axes[2].axis('off')


def interact_tool():
    # Define sliders for the new parameters mu_a, mu_b_d, and mu_c_d
    mu_b_d_slider = FloatSlider(min=-0.3, max=0.3, step=0.1, value=-.3, continuous_update=False, description="mu_b_d = mu_b_b - mu_b_r")
    mu_c_d_slider = FloatSlider(min=-0.3, max=0.3, step=0.1, value=-.1, continuous_update=False, description="mu_c_d = mu_c_b - mu_c_r")
    mu_a_slider = FloatSlider(min=-.1, max=0.3, step=0.1, value=0.2, continuous_update=False, description="mu_a - mu_a_r(=0)")
    experiment_slider = IntSlider(min=1, max=4, step=1, value=3, continuous_update=False, description="T")

    output = Output()

    def func(mu_b_d, mu_c_d, mu_a, sigma_obs=.1, T=10, product='man', market='b2c'):
        mu_b_r = .3
        mu_c_r = .1
        em = experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, mu_a, sigma_obs, T=T, product=product, market=market)
        fig0 = plot_layer0_belief(em)
        fig1 = plot_layer1_profit(em)
        return fig0, fig1
    
    def on_value_change(change):
        fig0, fig1 = func(
            mu_b_d=mu_b_d_slider.value,
            mu_c_d=mu_c_d_slider.value,
            mu_a=mu_a_slider.value,
            T=experiment_slider.value
        )
        with output:
            output.clear_output()
            plt.show(fig0)
            plt.show(fig1)
    
    # Observe changes in slider values
    for slider in [mu_a_slider, mu_b_d_slider, mu_c_d_slider, experiment_slider]:
        slider.observe(on_value_change, names='value')

    # Arrange sliders and output in a VBox layout
    interactive_simulation = VBox([
        HTML("<h1>Pivot Game</h1>"),
        HBox([
            VBox([HTML("<b>Parameters</b>"), mu_a_slider, mu_b_d_slider, mu_c_d_slider, experiment_slider])
        ]),
        output
    ])

    return interactive_simulation

if __name__ == "__main__":
    func(mu_b_d= .2, mu_c_d= -.1, mu_a= .2, sigma_obs=.1, T=4, product='man', market='b2c')
    # theory = xr.open_dataset("data/theory/bB[-0.2  0.   0.2]_cC[-0.2  0.   0.2]_a[0.4]_b0.2_c0.1_s0.1_cash4_E5_man_b2c")
    # plot_theory_given_experiment(theory)
    em = xr.open_dataset("data/experiment/bB-0.2_cC-0.2_B0.3_C0.3_a0.1_s0.1_T10_man_b2c.nc")
    plot_layer0_belief(em)
    plot_layer1_profit(em)