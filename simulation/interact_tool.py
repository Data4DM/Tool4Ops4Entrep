from ipywidgets import Layout
from ipywidgets import IntSlider, FloatSlider, HBox, VBox, HTML, Output
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd
from experiment_simulation import sample_segment_belief_set_boundary, sample_model_belief, sample_segment_real, determine_next_action, update_state

def plot_combined(SMN):
    S = SMN.dims['S']
    M = SMN.dims['M']
    action_colors = {'scale': 'red', 'market_pivot': 'purple', 'product_pivot': 'green', 'fail': 'black'}

    # Create the subplots without sharing the y-axis
    fig, axs = plt.subplots(S, M+3, figsize=(4 * (M+3), 3 * S))

    # Share y-axis for the first column subplots
    for s in range(1, S):
        axs[s, 0].get_shared_y_axes().join(axs[s, 0], axs[0, 0])

    for s in range(S):
        x_r_s_m = SMN['x_r'][s, :].values
        x_min_s = SMN['x_min'][s].item()
        x_max_s = SMN['x_max'][s].item()
        c_s_m = SMN['cash_state'][s, :].values

        # Feedback and threshold in the first column
        axs[s, 0].plot(range(1, M+1), x_r_s_m, label='Feedback x_r')
        axs[s, 0].fill_between(range(1, M+1), x_min_s, x_max_s, color='gray', alpha=0.5, label='Threshold band')
        axs[s, 0].set_title(f'Segment {s} Feedback')
        axs[s, 0].legend()

        for m in range(1, M+1):
            a_mean_b_posterior = SMN['a_mean_b'][s, m-1, :].values
            if not np.all(a_mean_b_posterior == 0):
                axs[s, m].hist(a_mean_b_posterior, bins=30, alpha=0.5, color='skyblue', edgecolor='white')
                axs[s, m].axvline(x=SMN['a_mean_r'], color='green', linestyle='--', label='Ground Truth')
                axs[s, m].axvline(x= a_mean_b_posterior.mean(), color='blue', linestyle='-', label='Mean Posterior')
                axs[s, m].set_title(f'Exp {m-1} Alpha Update')
                axs[s, m].legend()
                axs[s, m].set_xlim(-5, 2)  # Set x-axis limits from -5 to 2
                axs[s, m].set_ylim(0, 500)  # Set y-axis limits from 0 to 500

        # Cash progress in the second-to-last column
        axs[s, M+1].plot(range(1, M+1), c_s_m, label='Cash State')
        axs[s, M+1].set_title(f'Segment {s} Cash State Progress')
        axs[s, M+1].legend()
        axs[s, M+1].set_ylim(0, 5)  # Set y-axis limits from 0 to 5
        axs[s, M+1].set_yticks(range(6))  # Set y-axis ticks to integer values
        axs[s, M+1].set_xticks(range(1, M+1))  # Set x-axis ticks to integer values

        # Actions in the last column
        actions = SMN['action'][s, :].values
        axs[s, M+2].scatter(range(1, M+1), [0.5] * M, color=[action_colors.get(a, 'gray') for a in actions])
        axs[s, M+2].set_title(f'Segment {s} Actions')
        axs[s, M+2].set_xticks(range(1, M+1))
        axs[s, M+2].axes.get_yaxis().set_visible(False)
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=act)
                    for act, color in action_colors.items()]
        axs[s, M+2].legend(handles=handles, loc='upper right', title="Actions", fontsize='small')

    figure_title = SMN['SMN_name'].item() + ".png"
    figure_path = os.path.join("figures", figure_title)
    plt.savefig(figure_path)
    plt.tight_layout()
    plt.show()
    # mean_series = pd.Series(SMN['a_mean_b'].mean(dim='K').values.flatten())
    # mean_series[mean_series != 0].plot(title="Lowering Overconfidence Gap by Mean of a (Market Acceptance)")

def plot_hypo_test(DR, SR, BR, DB, ER, CT, S, M, K):
    # Initialize variables to store the pivot counts
    db_market_pivots_low = 0
    db_market_pivots_high = 0
    db_product_pivots_low = 0
    db_product_pivots_high = 0

    dr_market_pivots_low = 0
    dr_market_pivots_high = 0
    dr_product_pivots_low = 0
    dr_product_pivots_high = 0

    sr_market_pivots_low = 0
    sr_market_pivots_high = 0
    sr_product_pivots_low = 0
    sr_product_pivots_high = 0

    br_market_pivots_low = 0
    br_market_pivots_high = 0
    br_product_pivots_low = 0
    br_product_pivots_high = 0

    er_market_pivots_low = 0
    er_market_pivots_high = 0
    er_product_pivots_low = 0
    er_product_pivots_high = 0

    ct_market_pivots_low = 0
    ct_market_pivots_high = 0
    ct_product_pivots_low = 0
    ct_product_pivots_high = 0

    # Iterate over all combinations of hyperparameters
    for dr in DR:
        for sr in SR:
            for br in BR:
                for db in DB:
                    for er in ER:
                        for ct in CT:
                            SMN_name = f"xarray_data/DR{dr}_SR{sr}_BR{br}_DB{db}_ER{er}_CT{ct}_S{S}_M{M}_K{K}"
                            SMN = xr.open_dataset(f"{SMN_name}.nc")

                            # Check if there are any 'scale' or 'fail' actions
                            act_seq = SMN.action.values.flatten()
                            exp_end = np.where((act_seq == 'scale') | (act_seq == 'fail'))[0][0] if (('scale' in act_seq) | ('fail' in act_seq)) else len(act_seq)
                            filtered_actions = act_seq[:exp_end]

                            # Count 'market_pivot' and 'product_pivot' for each hyperparameter value
                            if br == BR[0]:
                                br_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                br_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                br_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                br_product_pivots_high += (filtered_actions == 'product_pivot').sum()

                            if db == DB[0]:
                                db_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                db_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                db_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                db_product_pivots_high += (filtered_actions == 'product_pivot').sum()

                            if dr == DR[0]:
                                dr_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                dr_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                dr_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                dr_product_pivots_high += (filtered_actions == 'product_pivot').sum()

                            if sr == SR[0]:
                                sr_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                sr_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                sr_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                sr_product_pivots_high += (filtered_actions == 'product_pivot').sum()

                            if er == ER[0]:
                                er_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                er_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                er_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                er_product_pivots_high += (filtered_actions == 'product_pivot').sum()

                            if ct == CT[0]:
                                ct_market_pivots_low += (filtered_actions == 'market_pivot').sum()
                                ct_product_pivots_low += (filtered_actions == 'product_pivot').sum()
                            else:
                                ct_market_pivots_high += (filtered_actions == 'market_pivot').sum()
                                ct_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    # Calculate the ratios for each hyperparameter
    db_ratio_low = db_market_pivots_low / db_product_pivots_low if db_product_pivots_low > 0 else float('inf')
    db_ratio_high = db_market_pivots_high / db_product_pivots_high if db_product_pivots_high > 0 else float('inf')

    dr_ratio_low = dr_market_pivots_low / dr_product_pivots_low if dr_product_pivots_low > 0 else float('inf')
    dr_ratio_high = dr_market_pivots_high / dr_product_pivots_high if dr_product_pivots_high > 0 else float('inf')

    sr_ratio_low = sr_market_pivots_low / sr_product_pivots_low if sr_product_pivots_low > 0 else float('inf')
    sr_ratio_high = sr_market_pivots_high / sr_product_pivots_high if sr_product_pivots_high > 0 else float('inf')

    br_ratio_low = br_market_pivots_low / br_product_pivots_low if br_product_pivots_low > 0 else float('inf')
    br_ratio_high = br_market_pivots_high / br_product_pivots_high if br_product_pivots_high > 0 else float('inf')

    er_ratio_low = er_market_pivots_low / er_product_pivots_low if er_product_pivots_low > 0 else float('inf')
    er_ratio_high = er_market_pivots_high / er_product_pivots_high if er_product_pivots_high > 0 else float('inf')

    ct_ratio_low = ct_market_pivots_low / ct_product_pivots_low if ct_product_pivots_low > 0 else float('inf')
    ct_ratio_high = ct_market_pivots_high / ct_product_pivots_high if ct_product_pivots_high > 0 else float('inf')

    print("Ratios:")
    print("hypothesis: ratio/db>0, ratio/dr>0, ratio/sr>0, ratio/br>0, ratio/er>0, ratio/ct>0")
    
    print("DR Low: {:.2f}, DR High: {:.2f}".format(dr_ratio_low, dr_ratio_high))
    print("SR Low: {:.2f}, SR High: {:.2f}".format(sr_ratio_low, sr_ratio_high))
    print("BR Low: {:.2f}, BR High: {:.2f}".format(br_ratio_low, br_ratio_high))
    print("DB Low: {:.2f}, DB High: {:.2f}".format(db_ratio_low, db_ratio_high))
    print("ER Low: {:.2f}, ER High: {:.2f}".format(er_ratio_low, er_ratio_high))
    print("CT Low: {:.2f}, CT High: {:.2f}".format(ct_ratio_low, ct_ratio_high))

    # Calculate the difference in means for each hyperparameter
    db_mean_diff = db_ratio_high - db_ratio_low
    dr_mean_diff = dr_ratio_high - dr_ratio_low
    sr_mean_diff = sr_ratio_high - sr_ratio_low
    br_mean_diff = br_ratio_high - br_ratio_low
    er_mean_diff = er_ratio_high - er_ratio_low
    ct_mean_diff = ct_ratio_high - ct_ratio_low

    print("Difference in Means:")
    print("hypothesis: ratio/db>0, ratio/dr>0, ratio/sr>0, ratio/br>0, ratio/er>0, ratio/ct>0")
    
    print(f"DR: {dr_mean_diff:.2f}")
    print(f"SR: {sr_mean_diff:.2f}")
    print(f"BR: {br_mean_diff:.2f}")
    print(f"DB: {db_mean_diff:.2f}")
    print(f"ER: {er_mean_diff:.2f}")
    print(f"CT: {ct_mean_diff:.2f}")

    # Create a figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Plot DR ratios
    axs[0, 0].bar(['Low', 'High'], [dr_ratio_low, dr_ratio_high])
    axs[0, 0].set_title('DR')
    axs[0, 0].set_ylim(0, max(dr_ratio_low, dr_ratio_high) * 1.1)

    # Plot SR ratios
    axs[0, 1].bar(['Low', 'High'], [sr_ratio_low, sr_ratio_high])
    axs[0, 1].set_title('SR')
    axs[0, 1].set_ylim(0, max(sr_ratio_low, sr_ratio_high) * 1.1)

    # Plot BR ratios
    axs[0, 2].bar(['Low', 'High'], [br_ratio_low, br_ratio_high])
    axs[0, 2].set_title('BR')
    axs[0, 2].set_ylim(0, max(br_ratio_low, br_ratio_high) * 1.1)

    # Plot DB ratios
    axs[1, 0].bar(['Low', 'High'], [db_ratio_low, db_ratio_high])
    axs[1, 0].set_title('DB')
    axs[1, 0].set_ylim(0, max(db_ratio_low, db_ratio_high) * 1.1)  # Set y-axis limit with some padding

    # Plot ER ratios
    axs[1, 1].bar(['Low', 'High'], [er_ratio_low, er_ratio_high])
    axs[1, 1].set_title('ER')
    axs[1, 1].set_ylim(0, max(er_ratio_low, er_ratio_high) * 1.1)

    # Plot CT ratios
    axs[1, 2].bar(['Low', 'High'], [ct_ratio_low, ct_ratio_high])
    axs[1, 2].set_title('CT')
    axs[1, 2].set_ylim(0, max(ct_ratio_low, ct_ratio_high) * 1.1)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def func(dr, sr, br, db, er, ct, S, M, K=1000):

    # Create the dataset
    coords = {'H': np.arange(1), 'S': np.arange(S), 'M': np.arange(M), 'K': np.arange(K)}
    SMN_name = f"DR{dr}_SR{sr}_BR{br}_DB{db}_ER{er}_CT{ct}_S{S}_M{M}_K{K}"
    data_vars = {
        # HYPERPARAMS
        'cash_state': (('S', 'M'), np.zeros((S, M))),
        'unit_exp_cost': ('H', np.array([1])),
        'DR': ('H', np.array([dr])),  # belief on real
        'SR': ('H', np.array([sr])),  # size of segment
        'BR': ('H', np.array([br])),  # belief on real
        'DB': ('H', np.array([db])),  # belief on belief
        'ER': ('H', np.array([er])),  # experiment opportunity
        'CT': ('H', np.array([ct])),  # cash threshold
        'SMN_name': ((), SMN_name),  # Add SMN_name as a 0-dimensional variable
        
        # BELIEF on DESIRABILITY and FEASIBILITY
        'x_b': (('S', 'M'), np.zeros((S, M))),
        'a_b': (('S', 'M'), np.zeros((S, M))),  # segment acceptance (belief) parameter for specific segment
        'a_mean_b': (('S', 'M', 'K'), np.zeros((S, M, K))),  # SAMPLES from Posterior of average segment acceptance (across segments)
        'x_min': ('S', np.zeros(S)),  # TODO: E , sigma of a_mean_b
        'x_max': ('S', np.zeros(S)),
        
        # REAL DESIRABILITY
        'x_r': (('S', 'M'), np.zeros((S, M))),
        'a_r': (('S', 'M'), np.zeros((S, M))),  # segment acceptance (real) parameter for specific segment
        'a_mean_r': (('H'), np.array([-br])),  # average segment acceptance (across segments)
        'action': (('S', 'M'), np.full((S, M), '', dtype='object'))  # Agent with belief responding to Real segment feedback
    }
    SMN = xr.Dataset(data_vars=data_vars, coords=coords)
    SMN['cash_state'].loc[dict(S=0, M=0)] = er
    
    # Run the experiment
    for s in range(S):
        SMN = sample_segment_belief_set_boundary(s, SMN)
        for m in range(M):
            SMN = sample_model_belief(s, m, SMN)
            SMN = sample_segment_real(s, m, SMN)
            action = determine_next_action(s, m, SMN)
            SMN['action'][s, m] = action
            if action in ["scale", "fail"]:
                break
            if action in ["product_pivot", "market_pivot"]:
                SMN = update_state(s, m, SMN)
    fig = plot_combined(SMN)
    
    return SMN, fig

def interact_tool():
    signal_market_params_children = [
        IntSlider(min=0, max=1, step=1, value=1, continuous_update=False, description="DR"),
        IntSlider(min=1, max=1000, step=1, value=1, continuous_update=False, description="SR")
    ]

    prior_market_params_children = [
        FloatSlider(min=0.0, max=10.0, step=0.1, value=1.0, continuous_update=False, description="BR"),
        FloatSlider(min=0.0, max=10.0, step=0.1, value=1.0, continuous_update=False, description="DB")
    ]

    prior_pmf_params_children = [
        FloatSlider(min=0.0, max=1.0, step=0.1, value=0.3, continuous_update=False, description="CT")
    ]

    prior_experiment_params_children = [
        IntSlider(min=2, max=4, step=1, value=4, continuous_update=False, description="ER"),
        IntSlider(min=2, max=4, step=1, value=2, continuous_update=False, description="Market Candidate"),
        IntSlider(min=2, max=4, step=1, value=2, continuous_update=False, description="Product Candidate")
    ]

    output = Output()

    def on_value_change(change):
        SMN, fig = func(
            dr=signal_market_params_children[0].value,
            sr=signal_market_params_children[1].value,
            br=prior_market_params_children[0].value,
            db=prior_market_params_children[1].value,
            ct=prior_pmf_params_children[0].value,
            er=prior_experiment_params_children[0].value,
            S=prior_experiment_params_children[1].value,
            M=prior_experiment_params_children[2].value,
        )
        with output:
            output.clear_output()
            plt.show(fig)

    for child in signal_market_params_children + prior_market_params_children + prior_pmf_params_children + prior_experiment_params_children:
        child.observe(on_value_change, names='value')

    # Then include this in your main layout
    interactive_simulation = VBox([
        HTML("<h1>Market-Product Experimentation Navigator</h1>"),
        HTML("<h2>Real</h2>"),
        HBox([
            VBox([HTML("<b>Market Parameter</b>"), *signal_market_params_children]),
            VBox([HTML("<b>Experiment Parameter</b>"), *prior_experiment_params_children])
        ]),
        HTML("<h2>Belief</h2>"),
        HBox([
            VBox([HTML("<b>Market Parameter</b>"), *prior_market_params_children]),
            VBox([HTML("<b>Product Market Fit Parameter</b>"), *prior_pmf_params_children])
        ]),
        output
    ])

    # # Initial plot
    # SMN, fig = func(
    #     dr=signal_market_params_children[0].value,
    #     sr=signal_market_params_children[1].value,
    #     br=prior_market_params_children[0].value,
    #     db=prior_market_params_children[1].value,
    #     ct=prior_pmf_params_children[0].value,
    #     er=prior_experiment_params_children[0].value,
    #     S=prior_experiment_params_children[1].value,
    #     M=prior_experiment_params_children[2].value,
    # )
    # with output:
    #     plt.show(fig)
    return interactive_simulation
#signal_market_params_children, prior_experiment_params_children, prior_market_params_children, prior_pmf_params_children