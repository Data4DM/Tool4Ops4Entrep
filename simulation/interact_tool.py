
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


def plot_combined(exPMN):
    P = exPMN.dims['P']
    ACT_PRED = exPMN.dims['ACT_PRED']
    ACT_PVT = exPMN.dims['ACT_PVT']
    ACT_PVT_colors = {'scale': 'red', 'pivot_market': 'purple', 'pivot_product': 'green'}
    
    fig, axs = plt.subplots(1, P+1, figsize=(4 * (ACT_PRED+1), 3))

    profit_b = exPMN['profit_b'].values
    low_profit_b = exPMN['low_profit_b'].values
    high_profit_b = exPMN['high_profit_b'].values

    profit_obs = exPMN['profit_obs'].values
    actions = exPMN['action'].values

    # Initial belief plot with 4000 samples
    # initial_samples = np.random.normal(loc=profit_b[0], scale=exPMN['sigma_mu'][0], size=4000)
    # axs[1].hist(initial_samples, bins=30, alpha=0.5, color='skyblue', edgecolor='white')
    axs[0].axvline(x=profit_b[0], color='blue', linestyle='-', label='Initial Belief Mean')
    axs[0].set_title('Initial Belief')
    axs[0].legend()
    axs[0].set_xlim(-.3, .5)
    axs[0].set_ylim(0, 500)

    # Plot posterior distributions
    for t in range(ACT_PRED):
        profit_b_posterior = exPMN['profit_post'][t].values  # Correctly access the scalar value
        axs[t+1].hist(profit_b_posterior, bins=30, alpha=0.5, color='skyblue', edgecolor='white')
        axs[t+1].axvline(x=exPMN['profit_obs'][t].item(), color='red', linestyle='--', label='Observed Profit')
        axs[t+1].axvline(x=profit_b_posterior.mean(), color='blue', linestyle='-', label='Profit Posterior Mean')
        axs[t+1].set_title(f'profit @ time {t+1} pred, obs, posterior')
        axs[t+1].legend()
        axs[t+1].set_xlim(-.3, .5)
        axs[t+1].set_ylim(0, 500)

    # Initial plot for profit feedback with action dots
    axs[P].scatter(np.array(range(1,P)),list(profit_obs), color='red', label='Profit Observed', marker = 'x')
    axs[P].plot(list(range(P)),list(profit_b)+[profit_b[-1]], color='blue', label='Profit predicted')
    axs[P].fill_between(list(range(1,ACT_PRED+1)),  list(low_profit_b), list(high_profit_b), color='skyblue', alpha=0.5, label='Threshold band')
    # axs[0].scatter(np.array(range(ACT_PVT)) + 0.2, np.zeros(ACT_PVT), color=[ACT_PVT_colors.get(a, 'gray') for a in actions], s=100)
    axs[P].scatter(np.array(range(1, P)) + 0.2, np.full(ACT_PVT, -.25), color=[ACT_PVT_colors.get(a, 'gray') for a in actions], s=50)
    axs[P].set_title('Profit Feedback')
    axs[P].set_ylim(-.3, .5)
    axs[P].legend()

    plt.tight_layout()
    figure_title = exPMN['exPMN_name'].item() + ".png"
    figure_path = os.path.join("data/figure/interact", figure_title)
    plt.savefig(figure_path)
    plt.show()
    plt.close()
    
def func(mu_b_d, mu_c_d, mu_a, sigma_mu=.1, T=10, product='man', market='b2c'):
    mu_b_r = .3
    mu_c_r = .1
    exPMN = experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, mu_a, sigma_mu, T=T, product=product, market=market)
    fig = plot_combined(exPMN)

    return exPMN, fig



def interact_tool():
    # Define sliders for the new parameters mu_a, mu_b_d, and mu_c_d
    mu_b_d_slider = FloatSlider(min=-0.3, max=0.3, step=0.1, value=-.3, continuous_update=False, description="mu_b_d")
    mu_c_d_slider = FloatSlider(min=-0.3, max=0.3, step=0.1, value=-.1, continuous_update=False, description="mu_c_d")
    mu_a_slider = FloatSlider(min=-.1, max=0.3, step=0.1, value=0.2, continuous_update=False, description="mu_a")
    experiment_slider = IntSlider(min=1, max=4, step=1, value=3, continuous_update=False, description="T")

    output = Output()
    
    def on_value_change(change):
        exPMN, fig = func(
            mu_b_d=mu_b_d_slider.value,
            mu_c_d=mu_c_d_slider.value,
            mu_a=mu_a_slider.value,
            T=experiment_slider.value
        )
        with output:
            output.clear_output()
            plt.show(fig)
    
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
    func(mu_b_d= .2, mu_c_d= -.1, mu_a= .2, sigma_mu=.1, T=4, product='man', market='b2c')

# def interact_tool():
#     signal_market_params_children = [
#         IntSlider(min=0, max=1, step=1, value=1, continuous_update=False, description="m_dyn(DR)"),
#         IntSlider(min=1, max=2, step=1, value=1, continuous_update=False, description="m_size(SR)")
#     ]
#     prior_market_params_children = [
#         IntSlider(min=0, max=2, step=1, value=2, continuous_update=False, description="m_optimism(OB)"),
#         IntSlider(min=0, max=2, step=1, value=2, continuous_update=False, description="m_uncertainty(UB)")
#     ]

#     prior_pmf_params_children = [
#         FloatSlider(min=0.5, max=2, step=0.5, value=2, continuous_update=False, description="m_p_uc(CT)")
#     ]

#     prior_experiment_params_children = [
#         IntSlider(min=2, max=4, step=1, value=4, continuous_update=False, description="capital(ER)")
#     ]

#     output = Output()
#     def on_value_change(change):
#         exPMN, fig = func(
#             ob=prior_market_params_children[0].value,
#             ub=prior_market_params_children[1].value,
#             br=prior_pmf_params_children[0].value,
#             car=prior_experiment_params_children[0].value,
#             clr=signal_market_params_children[0].value,
#         )
#         with output:
#             output.clear_output()
#             plt.show(fig)

#     for child in signal_market_params_children + prior_market_params_children + prior_pmf_params_children + prior_experiment_params_children:
#         child.observe(on_value_change, names='value')

#     interactive_simulation = VBox([
#         HTML("<h1>Market-Product Experimentation Navigator</h1>"),
#         HTML("<h2>Real</h2>"),
#         HBox([
#             VBox([HTML("<b>Market Parameter</b>"), *signal_market_params_children]),
#             VBox([HTML("<b>Experiment Parameter</b>"), *prior_experiment_params_children])
#         ]),
#         HTML("<h2>Belief</h2>"),
#         HBox([
#             VBox([HTML("<b>Market Parameter</b>"), *prior_market_params_children]),
#             VBox([HTML("<b>Product Market Fit Parameter</b>"), *prior_pmf_params_children])
#         ]),
#         output
#     ])

#     return interactive_simulation

