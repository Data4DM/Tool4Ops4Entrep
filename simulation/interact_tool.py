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


def plot_layer0_belief(em):
    """
    Plots the EM algorithm process with belief updates and profit predictions using the em xarray dataset.
    
    Parameters:
    - em: xarray dataset containing the required variables
    """
    num_exp = np.where(em['action'].values == "scale")[0][0] + 1 if "scale" in em['action'].values else em['action'].shape[0]
    experiments = np.arange(1, num_exp + 1)

    # Extracting belief values from the em dataset
    mu_b_b = em['mu_b_b'].values[:num_exp]
    mu_c_b = em['mu_c_b'].values[:num_exp]
    
    # Extracting ground truth values from the em dataset
    mu_b_r = em['mu_b_r'].values[0]
    mu_c_r = em['mu_c_r'].values[0]
    
    # Extracting posterior samples from the em dataset
    mu_b_b_post = em['mu_b_b_post'].values[:num_exp]
    mu_c_b_post = em['mu_c_b_post'].values[:num_exp]

    # Create subplots to put the plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(32, 8), sharey=True)
    fig.suptitle(em['em_name'].item())

    # Plotting the parameter updates with ground truth on separate subplots
    az.plot_hdi(experiments, mu_b_b_post.T, hdi_prob=0.94, ax=axs[0], color='green', fill_kwargs={'alpha': 0.2})
    axs[0].plot(experiments, mu_b_b, marker='>',linestyle='--',  label='Updated $\mu_{b}$', color='green')
    axs[0].axhline(y=mu_b_r, color='green', linestyle='-', linewidth=5, label='Ground Truth $\mu_{b}$')
    axs[0].set_title('Belief in $\mu_{b}$ by Time')
    axs[0].set_xlabel('Experiment')
    axs[0].set_xticklabels([str(int(exp)) for exp in experiments])
    axs[0].set_xticks(experiments)
    axs[0].set_ylabel('Belief on $\mu_{b}$')
    axs[0].legend(loc='lower left')
    axs[0].grid(True)

    az.plot_hdi(experiments, mu_c_b_post.T, hdi_prob=0.94, ax=axs[1], color='purple', fill_kwargs={'alpha': 0.2})
    axs[1].plot(experiments, mu_c_b, marker='>',linestyle='--', label='Updated $\mu_{c}$', color='purple')
    axs[1].axhline(y=mu_c_r, color='purple', linestyle='-', linewidth=5, label='Ground Truth $\mu_{c}$')
    axs[1].set_title('Belief in $\mu_{c}$ by Time')
    axs[1].set_xlabel('Experiment')
    axs[1].set_xticks(experiments)
    axs[1].set_xticklabels([str(int(exp)) for exp in experiments])
    axs[1].set_ylabel('Belief on $\mu_{c}$')
    axs[1].legend(loc='lower left')
    axs[1].grid(True)


    plt.tight_layout()
    figure_title = em['em_name'].item() + "_L0.png"
    figure_path = os.path.join("data/figure/em/l0", figure_title)
    plt.savefig(figure_path)
    # plt.show()
    plt.close()

def plot_layer1_profit(em):
    P = em.dims['P']
    ACT_PRED = em.dims['ACT_PRED']
    ACT_PVT = em.dims['ACT_PVT']
    ACT_PVT_colors = {'scale': 'red', 'pivot_market': 'purple', 'pivot_product': 'green'}
    
    num_exp = np.where(em['action'].values == "scale")[0][0] + 1 if "scale" in em['action'].values else em['action'].shape[0]
    experiments = np.arange(1, num_exp+1)
    fig = plt.figure(figsize=(4 * num_exp, 9))
    fig.suptitle(em['em_name'].item())
    gs = fig.add_gridspec(3, num_exp * 2, height_ratios=[1, 1, 1])  # Making the grid spec with more columns

    profit_prior = em['profit_prior'][:num_exp, :4]  # Adjusting shape to be (num_exp, 4)
    low_profit_b = em['low_profit_b'].values[:num_exp]
    high_profit_b = em['high_profit_b'].values[:num_exp]

    profit_obs = em['profit_obs'].values[:num_exp]
    actions = em['action'].values[:num_exp]

    # Create subplots
    axs = np.empty((3, num_exp), dtype=object)
    axs[0, 0] = fig.add_subplot(gs[0, :])  # Span the first row across all columns
    for col in range(num_exp):
        axs[1, col] = fig.add_subplot(gs[1, col * 2:(col + 1) * 2])  # Second row
        axs[2, col] = fig.add_subplot(gs[2, col * 2:(col + 1) * 2], projection='3d')  # Third row
                
    # Initial plot for profit feedback with action dots in the first row
    predicted_profits = [em['profit_b'][e2p[em['product'][0].item()], e2m[em['market'][0].item()], 0]]
    for t in range(1, num_exp):
        predicted_profits.append(em['profit_b'][e2p[em['product'][t].item()], e2m[em['market'][t].item()], t])                                                        
    
    axs[0, 0].scatter(np.array(range(1, num_exp + 1)), list(profit_obs), color=mcolors.to_rgba('purple', alpha=0.8), label='observed', marker='o')
    axs[0, 0].plot(list(range(1, num_exp + 1)), predicted_profits, color='blue', label='expected', linestyle='--')
    axs[0, 0].fill_between(list(range(1, num_exp + 1)), list(low_profit_b), list(high_profit_b), color='skyblue', alpha=0.3, label='low-high bar')
    
    ACT_PVT_markers = {'pivot_product': 'P', 'pivot_market': 'M', 'scale': 'S'}
    for i in range(ACT_PVT):
        markers = [ACT_PVT_markers.get(action, 'o') for action in actions]
        for j, marker in enumerate(markers):
            axs[0, 0].scatter(j + 1.2, -0.25, color='red', s=50, marker=f'${marker}$')

    axs[0, 0].set_title('Expected Profit')
    axs[0, 0].set_ylim(-3, 3)
    axs[0, 0].legend(loc='upper left', prop={'size': 4})
    axs[0, 0].set_xlabel('Experiment')
    axs[0, 0].set_xticks(experiments)

    # Plot posterior distributions in the second row
    for t in experiments:
        profit_b_prior = em['profit_prior'][t-1].values
        profit_b_posterior = em['profit_post'][t-1].values
        
        axs[1, t-1].hist(profit_b_prior, bins=30, alpha=0.5, color='skyblue', edgecolor='white')
        axs[1, t-1].hist(profit_b_posterior, bins=30, alpha=0.3, color='#7B68EE', edgecolor='white')
        axs[1, t-1].axvline(x=em['profit_obs'][t-1].item(), color=mcolors.to_rgba('purple', alpha=0.8), linestyle='-', label='observed')
        
        axs[1, t-1].axvline(x=profit_b_prior.mean(), color='blue', linestyle='--', label='prior mean') # = em['profit_b'][e2p[em['product'][t-1].item()], e2m[em['market'][t-1].item()], t-1].item()
        axs[1, t-1].axvline(x=profit_b_posterior.mean(), color='#7B68EE', linestyle='--', label='posterior mean')
        
        axs[1, t-1].set_title(f'Time {t} Profit Experiment')
        axs[1, t-1].legend()
        axs[1, t-1].set_xlim(-3, 3)
        axs[1, t-1].set_ylim(0, 500)

    # Define colors for each cell
    colors = ['gray', 'green', 'purple', 'orange']
    labels = ['man-b2c', 'ai-b2c', 'man-b2b', 'ai-b2b']
    
    for t in range(num_exp):
        ax = axs[2, t]
        # Create 2x2 grids for each experiment
        x_positions = np.array([1, 0, 1, 0])
        y_positions = np.array([0, 0, 1, 1])
        dx = dy = 0.1  # Width and depth of the bars
        z_positions_b = profit_prior[t, :]
        z_positions_t = em['profit_r'].values.flatten()

        # Plot bars for beliefs
        ax.bar3d(x_positions, y_positions, np.zeros_like(x_positions), dx, dy, z_positions_b, color=colors, alpha=0.8, edgecolor='black')

        # Plot bars for truth
        ax.bar3d(x_positions, y_positions, np.zeros_like(x_positions), dx, dy, z_positions_t, color=colors, alpha=0.2, edgecolor='black')

        # Add text annotations for beliefs
        for i in range(4):
            ax.text(x_positions[i], y_positions[i], z_positions_b[i], f'{z_positions_b[i]:.2f}', color='black', ha='right', va='bottom', fontsize=10)

        # Add text annotations for truth
        for i in range(4):
            ax.text(x_positions[i], y_positions[i], z_positions_t[i], f'{z_positions_t[i]:.2f}', color='black', ha='right', va='top', fontsize=10)

        # Remove gray grid and set background to white
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_facecolor('white')

        # Remove grid lines
        ax.grid(False)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks(np.arange(-2, 3, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels(['', '', '', '', ''])
        
        ax.set_title(f'exp{t + 1}', fontsize=10, pad=20)

        # Add custom labels
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            ax.text(x, y, max(z_positions_b[i], z_positions_t[i]) + 0.3, f'{labels[i]}', color=colors[i], ha='center', va='bottom', fontsize=10)

    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    figure_title = em['em_name'].item() + "_L1.png"
    figure_path = os.path.join("data/figure/em/l1", figure_title)
    plt.savefig(figure_path)
    plt.show()
    plt.close()

from ipywidgets import FloatSlider, IntSlider, Output
import matplotlib.pyplot as plt

def interact_tool():
    # Define sliders for the parameters mu_sum and mu_diff
    mu_diff_slider = FloatSlider(min=-2, max=2, step=1, value=0, continuous_update=False, description="mu_diff")
    mu_sum_slider = FloatSlider(min=-4, max=0, step=1, value=-2, continuous_update=False, description="mu_sum")
    sigma_slider = IntSlider(min=1, max=4, step=1, value=1, continuous_update=False, description="sigma")
    k_slider = IntSlider(min=1, max=4, step=1, value=1, continuous_update=False, description="k")
    t_slider = IntSlider(min=2, max=4, step=1, value=2, continuous_update=False, description="experiment#")
    output = Output()

    def func(mu_b_d, mu_c_d, mu_b_r, mu_c_r, sigma_profit, k, T, product='man', market='b2c'):
        em = experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, sigma_profit, T=T, k=k, product=product, market=market)
        fig0 = plot_layer0_belief(em)
        fig1 = plot_layer1_profit(em)
        return fig0, fig1

    def on_value_change(change):
        mu_sum = mu_sum_slider.value
        mu_diff = mu_diff_slider.value
        mu_b_d = (mu_sum + mu_diff) / 2
        mu_c_d = (mu_sum - mu_diff) / 2

        fig0, fig1 = func(
            mu_b_d=mu_b_d,
            mu_c_d=mu_c_d,
            mu_b_r=3,
            mu_c_r=1,
            sigma_profit=sigma_slider.value,
            k=k_slider.value,
            T=t_slider.value  # Assuming T is a constant or you can add a slider for T if needed
        )
        with output:
            output.clear_output()
            plt.show(fig0)
            plt.show(fig1)

    # Link the sliders to the on_value_change function
    for slider in [mu_diff_slider, mu_sum_slider, sigma_slider, k_slider, t_slider]:
        slider.observe(on_value_change, names='value')
    
    # Arrange sliders and output in a VBox layout
    interactive_simulation = VBox([
        HTML("<h1>Pivot Game</h1>"),
        HBox([
            VBox([HTML("<b>Parameters</b>"), mu_diff_slider, mu_sum_slider, sigma_slider, k_slider,]),
        ]),
        output
    ])

    return interactive_simulation

if __name__ == "__main__":
    # func(mu_b_d= .2, mu_c_d= -.1, sigma_obs=.1, T=4, product='man', market='b2c')
    # th = xr.open_dataset("data/th/bB[-0.2  0.   0.2]_cC[-0.2  0.   0.2]_a[0.4]_b0.2_c0.1_s0.1_cash4_E5_man_b2c")
    # plot_th_given_experiment(th)
    em = experiment(mu_b_d= -3, mu_c_d= -1, mu_b_r=3, mu_c_r=1, sigma_profit=1, T=2, k=2, product = 'man', market = 'b2c')
    # em = xr.open_dataset("data/experiment/bB-0.2_cC-0.2_B0.3_C0.3_a0.1_s0.1_T10_man_b2c.nc")
    plot_layer0_belief(em)
    plot_layer1_profit(em)