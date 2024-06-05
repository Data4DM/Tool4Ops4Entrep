import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fit_pm import experiment
import os
import arviz as az
import matplotlib.patches as patches

#theory-A: higher mu_a -> faster scale
# in:  mu_a / (mu_b_r + mu_c_r)
# out: time to first scale

#theory-bBcC: higher mu_b_d - mu_c_d -> higher product pivot ratio
# in: mu_b_d - mu_c_d
# out: ratio of first pivot_product to first pivot_market

#theory-A|BC: scott's idea on higher mu_a -> (b, c) combination that exceeds lowbar
# in: mu_a
# TODO out: ratio of cells whose mu_b, mu_c exceed lowbar (b or r)?? 

def brute_force(mu_b_d_range = np.linspace(-0.2, 0.2, 3), mu_c_d_range = np.linspace(-0.2, 0.2, 3), T=4, sigma_profit = .1, product='man', market='b2c'):
    
    mu_a= [.1]
    mu_b_r=.3
    mu_c_r=.3
    theory_name = f"bB{mu_b_d_range}_cC{mu_c_d_range}_a{mu_a}_b{mu_b_r}_c{mu_c_r}_s{sigma_profit}_T{T}_{product}_{market}"
    file_path = f"data/theory/{theory_name}.nc"
    if os.path.exists(file_path):
        theory = xr.open_dataset(file_path)
        print(f"File {theory_name} already exists. Skipping experiment.")
        return theory
    
    theory = xr.Dataset(
        coords={
            'mu_a': mu_a,
            'mu_b_d': mu_b_d_range,
            'mu_c_d': mu_c_d_range
        },
        data_vars={
            'time_to_first_scale': (('mu_a', 'mu_b_d', 'mu_c_d'), np.full((len(mu_a), len(mu_b_d_range), len(mu_c_d_range)), np.nan)),
            'pivot_ratio': (('mu_a', 'mu_b_d', 'mu_c_d'), np.full((len(mu_a), len(mu_b_d_range), len(mu_c_d_range)), np.nan)),
            'experiments_above_lowbar': (('mu_a', 'mu_b_d', 'mu_c_d'), np.full((len(mu_a), len(mu_b_d_range), len(mu_c_d_range)), np.nan)),
            'theory_name': ((), theory_name),
        }
    )

    for mu_a in mu_a:
        for mu_b_d in mu_b_d_range:
            for mu_c_d in mu_c_d_range:

                em = experiment(np.round(mu_b_d,1),np.round(mu_c_d,1), mu_b_r, mu_c_r, mu_a)
                plot_layer0_belief(em)
                if em is not None:
                    theory['time_to_first_scale'].loc[dict(mu_a=mu_a, mu_b_d=mu_b_d, mu_c_d=mu_c_d)] = compute_first_scale_time(em)
                    theory['pivot_ratio'].loc[dict(mu_a=mu_a, mu_b_d=mu_b_d, mu_c_d=mu_c_d)] = compute_pivot_ratio(em)
                    theory['experiments_above_lowbar'].loc[dict(mu_a=mu_a, mu_b_d=mu_b_d, mu_c_d=mu_c_d)] = count_experiments_above_lowbar(em)
    theory.to_netcdf(f"data/theory/{theory.theory_name.values}.nc")
    return theory


def compute_first_scale_time(em):
    first_scale_time = None
    for t in range(em.dims['P']):
        if em['action'][t].item() == 'scale':
            first_scale_time = t
            break
    return first_scale_time


def compute_pivot_ratio(em):
    first_pivot_product = None
    first_pivot_market = None

    for e in range(em.dims['E']):
        if em['action'][e].item() == 'pivot_product' and first_pivot_product is None:
            first_pivot_product = e + 1
        if em['action'][e].item() == 'pivot_market' and first_pivot_market is None:
            first_pivot_market = e + 1

    if first_pivot_product is not None and first_pivot_market is not None:
        return first_pivot_product / first_pivot_market 
    else:
        return None


def count_experiments_above_lowbar(em, lowbar=0.2):
    count = 0
    for e in range(em.dims['E']):
        if em['profit_obs'][e].item() > lowbar:
            count += 1
    return count


def plot_theory_given_experiment(theory):
    # Plot for theory A
    for mu_a in theory['mu_a'].values:
        for mu_b_d in theory['mu_b_d'].values:
            for mu_c_d in theory['mu_c_d'].values:
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))

                # Plot for theory A
                ax[0].plot(theory['mu_b_d'].values, theory['time_to_first_scale'].sel(mu_a=mu_a, mu_c_d=mu_c_d), label=f'mu_a={mu_a}, mu_c_d={mu_c_d}')
                ax[0].set_xlabel('mu_b_d')
                ax[0].set_ylabel('Time to First Scale')
                ax[0].set_title('theory A: Time to First Scale vs. mu_b_d')
                ax[0].legend()

                # Plot for theory BC
                ax[1].plot(theory['mu_b_d'].values, theory['pivot_ratio'].sel(mu_a=mu_a, mu_c_d=mu_c_d), label=f'mu_a={mu_a}, mu_c_d={mu_c_d}')
                ax[1].set_xlabel('mu_b_d')
                ax[1].set_ylabel('Pivot Ratio')
                ax[1].set_title('theory BC: Pivot Ratio vs. mu_b_d')
                ax[1].legend()

                # Save the figure
                fig_name = f"data/figure/bB{np.round(mu_b_d,1)}_cC{mu_c_d}_a_{mu_a}.png"
                plt.savefig(fig_name)
                plt.close(fig)
    plot_actions_and_ratios_from_nc_files()

def extract_mu_values_from_filename(filename):
    parts = filename.split('_')
    mu_a_str = [p for p in parts if p.startswith('a')][0]
    mu_b_d_str = [p for p in parts if p.startswith('bB')][0]
    mu_c_d_str = [p for p in parts if p.startswith('cC')][0]
    mu_a = float(mu_a_str[1:])
    mu_b_d = float(mu_b_d_str[2:].strip('[]').split()[0])
    mu_c_d = float(mu_c_d_str[2:].strip('[]').split()[0])
    return mu_a, mu_b_d, mu_c_d

def plot_actions_and_ratios_from_nc_files(directory='data/experiment'):
    # Collect all .nc files in the directory
    nc_files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    
    action_colors = {'pivot_product': 'green', 'pivot_market': 'purple', 'scale': 'red'}
    mu_a_values = []
    mu_b_d_values = []
    mu_c_d_values = []
    actions = []

    # Read through each .nc file and extract mu_a, mu_b_d, mu_c_d and actions
    for nc_file in nc_files:
        file_path = os.path.join(directory, nc_file)
        ds = xr.open_dataset(file_path)
        
        mu_a, mu_b_d, mu_c_d = extract_mu_values_from_filename(nc_file)
        mu_a_values.append(mu_a)
        mu_b_d_values.append(mu_b_d)
        mu_c_d_values.append(mu_c_d)
        actions.append(ds['action'].values)
    
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # Plot for mu_a
    for i, mu_a in enumerate(mu_a_values):
        y_values = actions[i]
        x_values = np.full(len(y_values), mu_a)
        colors = [action_colors[action] for action in y_values if action in action_colors]
        ax[0].scatter(x_values[:len(colors)], np.arange(len(colors)), c=colors, label=f'mu_a={mu_a}')

    ax[0].set_xlabel('mu_a')
    ax[0].set_ylabel('Action Sequence')
    ax[0].set_title('Action Sequence vs mu_a')
    ax[0].legend()

    # Plot for mu_b_d and mu_c_d (Ratio of pivot_product to pivot_market)
    pivot_ratios = []
    for action_list in actions:
        pivot_product_count = sum(1 for action in action_list if action == 'pivot_product')
        pivot_market_count = sum(1 for action in action_list if action == 'pivot_market')
        if pivot_market_count > 0:
            pivot_ratio = pivot_product_count / pivot_market_count
        else:
            pivot_ratio = np.nan  # Avoid division by zero
        pivot_ratios.append(pivot_ratio)

    scatter = ax[1].scatter(mu_b_d_values, mu_c_d_values, c=pivot_ratios, cmap='viridis', edgecolor='k', s=100)
    ax[1].set_xlabel('mu_b_d')
    ax[1].set_ylabel('mu_c_d')
    ax[1].set_title('Ratio of pivot_product to pivot_market vs mu_b_d and mu_c_d')
    fig.colorbar(scatter, ax=ax[1], label='Pivot Product / Pivot Market Ratio')
    
    plt.tight_layout()
    fig_name = f"data/figure/aseq.png"
    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == "__main__":
    # em = xr.open_dataset("data/experiment/bB-0.3_cC-0.1_B0.3_C0.1_a0.2_s0.1_T2_man_b2c.nc")
    theory = brute_force()
    # # theory = xr.open_dataset("data/theory/bB[-0.2  0.   0.2]_cC[-0.2  0.   0.2]_a[0.4]_b0.2_c0.1_s0.1_cash4_E5_man_b2c")
    # plot_theory_given_experiment(theory)
    # plot_beliefs(em)
    plot_layer0_belief(em)
    # for mu_c_d in mu_c_d_range:

    # for mu_a_d in mu_a_d_range:

    #             act_seq = em.action.values.flatten()
    #             exp_end = np.where((act_seq == 'scale') | (act_seq == 'fail'))[0][0] if (('scale' in act_seq) | ('fail' in act_seq)) else len(act_seq)
    #             filtered_actions = act_seq[:exp_end]

    #             if ob == OB[0]:
    #                 ob_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 ob_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 ob_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 ob_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if ub == UB[0]:
    #                 ub_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 ub_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 ub_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 ub_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if br == BR[0]:
    #                 br_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 br_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 br_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 br_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if car == CAR[0]:
    #                 car_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 car_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 car_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 car_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if clr == CLR[0]:
    #                 clr_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 clr_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 clr_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 clr_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    # ub_ratio_low = ob_market_pivots_low / ob_product_pivots_low if ob_product_pivots_low > 0 else float('inf')
    # ub_ratio_high = ob_market_pivots_high / ob_product_pivots_high if ob_product_pivots_high > 0 else float('inf')

    # br_ratio_low = br_market_pivots_low / br_product_pivots_low if br_product_pivots_low > 0 else float('inf')
    # br_ratio_high = br_market_pivots_high / br_product_pivots_high if br_product_pivots_high > 0 else float('inf')

    # car_ratio_low = car_market_pivots_low / car_product_pivots_low if car_product_pivots_low > 0 else float('inf')
    # car_ratio_high = car_market_pivots_high / car_product_pivots_high if car_product_pivots_high > 0 else float('inf')

    # ob_ratio_low = ob_market_pivots_low / ob_product_pivots_low if ob_product_pivots_low > 0 else float('inf')
    # ob_ratio_high = ob_market_pivots_high / ob_product_pivots_high if ob_product_pivots_high > 0 else float('inf')

    # clr_ratio_low = clr_market_pivots_low / clr_product_pivots_low if clr_product_pivots_low > 0 else float('inf')
    # clr_ratio_high = clr_market_pivots_high / clr_product_pivots_high if clr_product_pivots_high > 0 else float('inf')

    # print("Ratios:")
    # print("hypothesis: ratio/ub>0, ratio/br>0, ratio/car>0, ratio/ob>0, ratio/clr>0")
    
    # print("OB Low: {:.2f}, OB High: {:.2f}".format(ob_ratio_low, ob_ratio_high))
    # print("UB Low: {:.2f}, UB High: {:.2f}".format(ub_ratio_low, ub_ratio_high))
    # print("BR Low: {:.2f}, BR High: {:.2f}".format(br_ratio_low, br_ratio_high))
    # print("CAR Low: {:.2f}, CAR High: {:.2f}".format(car_ratio_low, car_ratio_high))
    # print("CLR Low: {:.2f}, CLR High: {:.2f}".format(clr_ratio_low, clr_ratio_high))

    # ub_mean_diff = ub_ratio_high - ub_ratio_low
    # br_mean_diff = br_ratio_high - br_ratio_low
    # car_mean_diff = car_ratio_high - car_ratio_low
    # ob_mean_diff = ob_ratio_high - ob_ratio_low
    # clr_mean_diff = clr_ratio_high - clr_ratio_low

    # print("Difference in Means:")
    # print("hypothesis: ratio/ub>0, ratio/br>0, ratio/car>0, ratio/ob>0, ratio/clr>0")
    
    # print(f"OB: {ob_mean_diff:.2f}")
    # print(f"UB: {ub_mean_diff:.2f}")
    # print(f"BR: {br_mean_diff:.2f}")
    # print(f"CAR: {car_mean_diff:.2f}")
    # print(f"CLR: {clr_mean_diff:.2f}")

    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # axs[0, 0].bar(['Low', 'High'], [ob_ratio_low, ob_ratio_high])
    # axs[0, 0].set_title('OB')
    # axs[0, 0].set_ylim(0, max(ob_ratio_low, ob_ratio_high) * 1.1)

    # axs[0, 1].bar(['Low', 'High'], [ub_ratio_low, ub_ratio_high])
    # axs[0, 1].set_title('UB')
    # axs[0, 1].set_ylim(0, max(ub_ratio_low, ub_ratio_high) * 1.1)

    # axs[0, 2].bar(['Low', 'High'], [br_ratio_low, br_ratio_high])
    # axs[0, 2].set_title('BR')
    # axs[0, 2].set_ylim(0, max(br_ratio_low, br_ratio_high) * 1.1)

    # axs[1, 0].bar(['Low', 'High'], [car_ratio_low, car_ratio_high])
    # axs[1, 0].set_title('CAR')
    # axs[1, 0].set_ylim(0, max(car_ratio_low, car_ratio_high) * 1.1)

    # axs[1, 1].bar(['Low', 'High'], [clr_ratio_low, clr_ratio_high])
    # axs[1, 1].set_title('CLR')
    # axs[1, 1].set_ylim(0, max(clr_ratio_low, clr_ratio_high) * 1.1)

    # plt.tight_layout()

    # plt.show()
 


    #             act_seq = em.action.values.flatten()
    #             exp_end = np.where((act_seq == 'scale') | (act_seq == 'fail'))[0][0] if (('scale' in act_seq) | ('fail' in act_seq)) else len(act_seq)
    #             filtered_actions = act_seq[:exp_end]

    #             if ob == OB[0]:
    #                 ob_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 ob_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 ob_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 ob_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if ub == UB[0]:
    #                 ub_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 ub_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 ub_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 ub_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if br == BR[0]:
    #                 br_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 br_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 br_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 br_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if car == CAR[0]:
    #                 car_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 car_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 car_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 car_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    #             if clr == CLR[0]:
    #                 clr_market_pivots_low += (filtered_actions == 'market_pivot').sum()
    #                 clr_product_pivots_low += (filtered_actions == 'product_pivot').sum()
    #             else:
    #                 clr_market_pivots_high += (filtered_actions == 'market_pivot').sum()
    #                 clr_product_pivots_high += (filtered_actions == 'product_pivot').sum()

    # ub_ratio_low = ob_market_pivots_low / ob_product_pivots_low if ob_product_pivots_low > 0 else float('inf')
    # ub_ratio_high = ob_market_pivots_high / ob_product_pivots_high if ob_product_pivots_high > 0 else float('inf')

    # br_ratio_low = br_market_pivots_low / br_product_pivots_low if br_product_pivots_low > 0 else float('inf')
    # br_ratio_high = br_market_pivots_high / br_product_pivots_high if br_product_pivots_high > 0 else float('inf')

    # car_ratio_low = car_market_pivots_low / car_product_pivots_low if car_product_pivots_low > 0 else float('inf')
    # car_ratio_high = car_market_pivots_high / car_product_pivots_high if car_product_pivots_high > 0 else float('inf')

    # ob_ratio_low = ob_market_pivots_low / ob_product_pivots_low if ob_product_pivots_low > 0 else float('inf')
    # ob_ratio_high = ob_market_pivots_high / ob_product_pivots_high if ob_product_pivots_high > 0 else float('inf')

    # clr_ratio_low = clr_market_pivots_low / clr_product_pivots_low if clr_product_pivots_low > 0 else float('inf')
    # clr_ratio_high = clr_market_pivots_high / clr_product_pivots_high if clr_product_pivots_high > 0 else float('inf')

    # print("Ratios:")
    # print("hypothesis: ratio/ub>0, ratio/br>0, ratio/car>0, ratio/ob>0, ratio/clr>0")
    
    # print("OB Low: {:.2f}, OB High: {:.2f}".format(ob_ratio_low, ob_ratio_high))
    # print("UB Low: {:.2f}, UB High: {:.2f}".format(ub_ratio_low, ub_ratio_high))
    # print("BR Low: {:.2f}, BR High: {:.2f}".format(br_ratio_low, br_ratio_high))
    # print("CAR Low: {:.2f}, CAR High: {:.2f}".format(car_ratio_low, car_ratio_high))
    # print("CLR Low: {:.2f}, CLR High: {:.2f}".format(clr_ratio_low, clr_ratio_high))

    # ub_mean_diff = ub_ratio_high - ub_ratio_low
    # br_mean_diff = br_ratio_high - br_ratio_low
    # car_mean_diff = car_ratio_high - car_ratio_low
    # ob_mean_diff = ob_ratio_high - ob_ratio_low
    # clr_mean_diff = clr_ratio_high - clr_ratio_low

    # print("Difference in Means:")
    # print("hypothesis: ratio/ub>0, ratio/br>0, ratio/car>0, ratio/ob>0, ratio/clr>0")
    
    # print(f"OB: {ob_mean_diff:.2f}")
    # print(f"UB: {ub_mean_diff:.2f}")
    # print(f"BR: {br_mean_diff:.2f}")
    # print(f"CAR: {car_mean_diff:.2f}")
    # print(f"CLR: {clr_mean_diff:.2f}")

    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # axs[0, 0].bar(['Low', 'High'], [ob_ratio_low, ob_ratio_high])
    # axs[0, 0].set_title('OB')
    # axs[0, 0].set_ylim(0, max(ob_ratio_low, ob_ratio_high) * 1.1)

    # axs[0, 1].bar(['Low', 'High'], [ub_ratio_low, ub_ratio_high])
    # axs[0, 1].set_title('UB')
    # axs[0, 1].set_ylim(0, max(ub_ratio_low, ub_ratio_high) * 1.1)

    # axs[0, 2].bar(['Low', 'High'], [br_ratio_low, br_ratio_high])
    # axs[0, 2].set_title('BR')
    # axs[0, 2].set_ylim(0, max(br_ratio_low, br_ratio_high) * 1.1)

    # axs[1, 0].bar(['Low', 'High'], [car_ratio_low, car_ratio_high])
    # axs[1, 0].set_title('CAR')
    # axs[1, 0].set_ylim(0, max(car_ratio_low, car_ratio_high) * 1.1)

    # axs[1, 1].bar(['Low', 'High'], [clr_ratio_low, clr_ratio_high])
    # axs[1, 1].set_title('CLR')
    # axs[1, 1].set_ylim(0, max(clr_ratio_low, clr_ratio_high) * 1.1)

    # plt.tight_layout()

    # plt.show()