import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fit_pm import experiment
import os
import arviz as az
import matplotlib.patches as patches
from interact_tool import plot_layer0_belief, plot_layer1_profit

#th-A: higher mu_a -> faster scale
# in:  mu_a / (mu_b_r + mu_c_r)
# out: time to first scale

#th-bBcC: higher mu_b_d - mu_c_d -> higher product pivot ratio
# in: mu_b_d - mu_c_d
# out: ratio of first pivot_product to first pivot_market

#th-A|BC: scott's idea on higher mu_a -> (b, c) combination that exceeds lowbar
# in: mu_a
# TODO out: ratio of cells whose mu_b, mu_c exceed lowbar (b or r)??  

# def brute_force(mu_diff_range=np.linspace(-2, -2, 1), mu_sum_range=np.linspace(-4, 0, 2), sigma_profit_range = np.linspace(1, 2, 10), T=2,  product='man', market='b2c'):
def brute_force(mu_diff_range=np.linspace(-4, 4, 5), mu_sum_range=np.linspace(-4, 0, 5), sigma_profit_range = np.linspace(.5, 3, 5), k_range =np.linspace(.5, 3, 5), T=3,  product='man', market='b2c'):    
    mu_b_r=1
    mu_c_r=3
    th_name = f"mu_diff{mu_diff_range[0]}to{mu_diff_range[0]}l{len(mu_diff_range)}_mu_sum{mu_sum_range[0]}to{mu_sum_range[-1]}l{len(mu_sum_range)}_B{mu_b_r}_C{mu_c_r}_s{sigma_profit_range[0]}{sigma_profit_range[-1]}_k{k_range[0]}to{k_range[-1]}l{len(sigma_profit_range)}_T{T}_{product}_{market}"
    file_path = f"data/theory/{th_name}.nc"
    if os.path.exists(file_path):
        th = xr.open_dataset(file_path)
        print(f"File {th_name} already exists. Skipping experiment.")
        return th
    
    th = xr.Dataset(
        coords={
            'mu_diff': mu_diff_range,
            'mu_sum': mu_sum_range,
            'sigma_profit': sigma_profit_range,
            'k': k_range,
        },
        data_vars={
            'pivot_ratio': (('mu_diff', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_diff_range), len(mu_sum_range), len(sigma_profit_range), len(k_range)), np.nan)),
            'reach_optimality': (('mu_diff', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_diff_range), len(mu_sum_range), len(sigma_profit_range), len(k_range)), np.nan)),
            'time_to_reach_optimality': (('mu_diff', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_diff_range), len(mu_sum_range), len(sigma_profit_range), len(k_range)), np.nan)),
            'th_name': ((), th_name),
        }
    )
    
    for mu_diff in mu_diff_range:
        for mu_sum in mu_sum_range:  
            for sigma_profit in sigma_profit_range:
                for k in k_range:  
                    mu_b_d = (mu_sum + mu_diff) / 2
                    mu_c_d = (mu_sum - mu_diff) / 2
                    em = experiment(mu_b_d, mu_c_d, mu_b_r, mu_c_r, sigma_profit, k, T, product, market)
                    plot_layer0_belief(em)
                    plot_layer1_profit(em)
                    if em is not None:     
                        th['pivot_ratio'].loc[dict(mu_diff=mu_diff, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k)] = compute_pivot_ratio(em)
                        th['reach_optimality'].loc[dict(mu_diff=mu_diff, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k)] = compute_reach_optimality(em)
                        th['time_to_reach_optimality'].loc[dict(mu_diff=mu_diff, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k)] = compute_time_to_reach_optimality(em)
    th.to_netcdf(f"data/theory/{th.th_name.values}.nc")

    plot_theory_given_experiment(th)  # Call the plotting function after running the experiments
    return th

def compute_pivot_ratio(em):
    actions = em['action'].values
    pivot_product_count = 0
    pivot_market_count = 0

    for action in actions:
        if action == 'scale' or action == 'fail':
            break
        if action == 'pivot_product':
            pivot_product_count += 1
        if action == 'pivot_market':
            pivot_market_count += 1

    if pivot_market_count == 0:
        return em.dims['ACT_PRED']+1 # Avoid division by zero
    return pivot_product_count / pivot_market_count

def compute_reach_optimality(em):
    products = em['product'].values
    markets = em['market'].values

    for product, market in zip(products, markets):
        if product == 'ai' and market == 'b2b':
            return 1
    return 0

def compute_time_to_reach_optimality(em):
    products = em['product'].values
    markets = em['market'].values

    for idx, (product, market) in enumerate(zip(products, markets)):
        if product == 'ai' and market == 'b2b':
            return idx  # Return the first index where the condition is met

    return em.dims['ACT_PRED'] + 1  # Return this value if the condition is never met

def plot_theory_given_experiment(th):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # Changed to 1 row and 4 columns
    fig.suptitle(th['th_name'].item())

    def plot_metrics(ax, x_values, pivot_ratio, reach_optimality, time_to_reach_optimality, x_label, title):
        ax.plot(x_values, pivot_ratio, label='Pivot Ratio', color='blue')
        ax.plot(x_values, reach_optimality, label='Reach Optimality', color='green')
        ax.plot(x_values, time_to_reach_optimality, label='Time to Reach Optimality', color='red')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Metrics')
        ax.set_title(title)
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.legend()

    # Extracting values
    mu_diff = th['mu_diff'].values
    mu_sum = th['mu_sum'].values
    sigma_profit = th['sigma_profit'].values
    k = th['k'].values

    pivot_ratio_mu_diff = th['pivot_ratio'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values
    reach_optimality_mu_diff = th['reach_optimality'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values
    time_to_reach_optimality_mu_diff = th['time_to_reach_optimality'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values

    pivot_ratio_mu_sum = th['pivot_ratio'].mean(dim=['mu_diff', 'sigma_profit', 'k']).values
    reach_optimality_mu_sum = th['reach_optimality'].mean(dim=['mu_diff', 'sigma_profit', 'k']).values
    time_to_reach_optimality_mu_sum = th['time_to_reach_optimality'].mean(dim=['mu_diff', 'sigma_profit', 'k']).values

    pivot_ratio_sigma_profit = th['pivot_ratio'].mean(dim=['mu_diff', 'mu_sum', 'k']).values
    reach_optimality_sigma_profit = th['reach_optimality'].mean(dim=['mu_diff', 'mu_sum', 'k']).values
    time_to_reach_optimality_sigma_profit = th['time_to_reach_optimality'].mean(dim=['mu_diff', 'mu_sum', 'k']).values

    pivot_ratio_k = th['pivot_ratio'].mean(dim=['mu_diff', 'mu_sum', 'sigma_profit']).values
    reach_optimality_k = th['reach_optimality'].mean(dim=['mu_diff', 'mu_sum', 'sigma_profit']).values
    time_to_reach_optimality_k = th['time_to_reach_optimality'].mean(dim=['mu_diff', 'mu_sum', 'sigma_profit']).values

    plot_metrics(axs[0], mu_diff, pivot_ratio_mu_diff, reach_optimality_mu_diff, time_to_reach_optimality_mu_diff, 'mu_diff', 'Metrics by mu_diff')
    plot_metrics(axs[1], mu_sum, pivot_ratio_mu_sum, reach_optimality_mu_sum, time_to_reach_optimality_mu_sum, 'mu_sum', 'Metrics by mu_sum')
    plot_metrics(axs[2], sigma_profit, pivot_ratio_sigma_profit, reach_optimality_sigma_profit, time_to_reach_optimality_sigma_profit, 'sigma_profit', 'Metrics by sigma_profit')
    plot_metrics(axs[3], k, pivot_ratio_k, reach_optimality_k, time_to_reach_optimality_k, 'k', 'Metrics by k')

    plt.tight_layout()
    figure_title = th['th_name'].item() + ".png"
    figure_path = os.path.join("data/figure/th", figure_title)
    plt.savefig(figure_path)

    
def extract_mu_values_from_filename(filename):
    parts = filename.split('_')
    mu_b_d_str = [p for p in parts if p.startswith('bB')][0]
    mu_c_d_str = [p for p in parts if p.startswith('cC')][0]
    mu_b_d = float(mu_b_d_str[2:].strip('[]').split()[0])
    mu_c_d = float(mu_c_d_str[2:].strip('[]').split()[0])
    return mu_b_d, mu_c_d

def plot_actions_and_ratios_from_nc_files(directory='data/experiment'):
    # Collect all .nc files in the directory
    nc_files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    
    action_colors = {'pivot_product': 'green', 'pivot_market': 'purple', 'scale': 'red'}
    mu_b_d_values = []
    mu_c_d_values = []
    actions = []

    # Read through each .nc file and extract mu_b_d, mu_c_d and actions
    for nc_file in nc_files:
        file_path = os.path.join(directory, nc_file)
        ds = xr.open_dataset(file_path)
        
        mu_b_d, mu_c_d = extract_mu_values_from_filename(nc_file)
        mu_b_d_values.append(mu_b_d)
        mu_c_d_values.append(mu_c_d)
        actions.append(ds['action'].values)
    
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

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
    th = brute_force()
    # th = xr.open_dataset("data/theory/mu_diff-4.0to-4.0_mu_sum-4.0to0.0_B3_C1_s1.01.0_k1.0to1.0_T3_man_b2c.nc")
    # th = xr.open_dataset("data/theory/mu_diff-4.0to-4.0_mu_sum-4.0to0.0_B3_C1_s1.03.0_k1.0to3.0_T3_man_b2c.nc")
    plot_theory_given_experiment(th)
    # test_hypothesis_1
    
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