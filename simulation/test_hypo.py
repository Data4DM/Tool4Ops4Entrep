import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from fit_pm import experiment
import os
import arviz as az
import matplotlib.patches as patches
from interact_tool import plot_layer0_belief, plot_layer1_profit
from collections import Counter

# 1.1 ground truth diff (observed profitability boost) 
# - fig1: increase as mu_pc- increase (action follows hope)

# 1.2 belief on how noisy learning is
# - fig2: viscous prior with belief on noisy environment (P1|sigma=.1 vs P1|sigma=1 vs P1|sigma=3)

# 1.3 setting k
# - fig3: (size of org -> cost ratio of pivot ->) 1/k -> 1/R

# todo: express decision k as a function of (-1)^{cost_pivot_product < cost_pivot_market}
# 1.4 ground truth sum (observed profitability boost) 
# - fig4: increase as mu_pc+ increase (action speed follows 
# fixing mu_p and increasing mu_m VS fixing mu_m and increasing mu_p
# TODO out: ratio of cells whose mu_p, mu_m exceed lowbar (b or r)??  

# def brute_force(mu_p2m_range=np.linspace(-2, -2, 1), mu_sum_range=np.linspace(-4, 0, 2), sigma_profit_range = np.linspace(1, 2, 10), T=2,  product='man', market='b2c'):
def brute_force(mu_p2m_range=np.linspace(.3, 3, 5), mu_sum_range=np.linspace(1, 4, 5), sigma_profit_range = np.linspace(.5, 2, 3), k_sigma_range =np.linspace(.5, 2, 3), T=2,  product='man', market='b2c'):    
    th_name = f"p2m{mu_p2m_range[0]}to{mu_p2m_range[-1]}l{len(mu_p2m_range)}_sum{mu_sum_range[0]}to{mu_sum_range[-1]}l{len(mu_sum_range)}_s{sigma_profit_range[0]}{sigma_profit_range[-1]}_k{k_sigma_range[0]}to{k_sigma_range[-1]}l{len(sigma_profit_range)}_T{T}_{product}_{market}"
    file_path = f"data/theory/{th_name}.nc"
    if os.path.exists(file_path):
        th = xr.open_dataset(file_path)
        print(f"File {th_name} already exists. Skipping experiment.")
        return th
    
    th = xr.Dataset(
        coords={
            'mu_p2m': mu_p2m_range,
            'mu_sum': mu_sum_range,
            'sigma_profit': sigma_profit_range,
            'k': k_sigma_range,
        },
        data_vars={
            'pivot_ratio': (('mu_p2m', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_p2m_range), len(mu_sum_range), len(sigma_profit_range), len(k_sigma_range)), np.nan)),
            'reach_optimality': (('mu_p2m', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_p2m_range), len(mu_sum_range), len(sigma_profit_range), len(k_sigma_range)), np.nan)),
            'time_to_reach_optimality': (('mu_p2m', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_p2m_range), len(mu_sum_range), len(sigma_profit_range), len(k_sigma_range)), np.nan)),
            'act_seq': (('mu_p2m', 'mu_sum', 'sigma_profit', 'k'), np.full((len(mu_p2m_range), len(mu_sum_range), len(sigma_profit_range), len(k_sigma_range)),  '', dtype='object')), 
            'th_name': ((), th_name),
        }
    )
    
    for mu_p2m in mu_p2m_range:
        for mu_sum in mu_sum_range:  
            for sigma_profit in sigma_profit_range:
                for k_sigma in k_sigma_range:  

                    em = experiment(mu_p2m, mu_sum, sigma_profit, k_sigma, T, product, market) # assuming init mu are 0, mu_p_d = mu_p_r
                    plot_layer0_belief(em)
                    plot_layer1_profit(em)
                    if em is not None:     
                        th['pivot_ratio'].loc[dict(mu_p2m=mu_p2m, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k_sigma)] = compute_pivot_ratio(em)
                        th['reach_optimality'].loc[dict(mu_p2m=mu_p2m, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k_sigma)] = compute_reach_optimality(em)
                        th['time_to_reach_optimality'].loc[dict(mu_p2m=mu_p2m, mu_sum=mu_sum, sigma_profit = sigma_profit, k=k_sigma)] = compute_time_to_reach_optimality(em)
                        th['act_seq'].loc[dict(mu_p2m = mu_p2m, mu_sum = mu_sum, sigma_profit = sigma_profit, k=k_sigma)] = compute_sequence(em)
    th.to_netcdf(f"data/theory/{th.th_name.values}.nc")

    plot_theory_given_experiment(th)  # Call the plotting function after running the experiments
    return th

def compute_pivot_ratio(em):
    actions = em['action'].values
    pivot_product_count = 0
    pivot_market_count = 0

    for action in actions:
        if action == 'scale':
            break
        if action == 'pivot_product':
            pivot_product_count += 1
        if action == 'pivot_market':
            pivot_market_count += 1

    # if pivot_market_count == 0:
    #     return em.dims['ACT_PRED'] * 2 # Avoid division by zero
    return (pivot_product_count+1) / (pivot_market_count + 1)

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

    return em.dims['ACT_PRED'] * 2 # Return this value if the condition is never met


def compute_sequence(em, sequence_length=3):
    actions = em['action'].values
    actions = [a.replace('pivot_product', 'p') for a in actions]
    actions = [a.replace('pivot_market', 'm') for a in actions]
    actions = [a.replace('scale', 's') for a in actions]
    
    action_sequences = ''.join(actions) #[''.join(actions[i:i+sequence_length]) for i in range(len(actions) - sequence_length + 1)]
    # sequence_counts = Counter(action_sequences)
    
    # # Sort sequences by frequency
    # sequences = list(sequence_counts.keys())
    # frequencies = list(sequence_counts.values())
    
    # sorted_indices = np.argsort(frequencies)[::-1]
    # sorted_sequences = [sequences[i] for i in sorted_indices]
    # sorted_frequencies = [frequencies[i] for i in sorted_indices]
    
    return action_sequences #sorted_sequences, sorted_frequencies

def plot_theory_given_experiment(th):
    fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # Changed to 1 row and 5 columns
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
    mu_p2m = th['mu_p2m'].values
    mu_sum = th['mu_sum'].values
    sigma_profit = th['sigma_profit'].values
    k = th['k'].values

    pivot_ratio_mu_p2m = th['pivot_ratio'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values
    reach_optimality_mu_p2m = th['reach_optimality'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values
    time_to_reach_optimality_mu_p2m = th['time_to_reach_optimality'].mean(dim=['mu_sum', 'sigma_profit', 'k']).values

    pivot_ratio_mu_sum = th['pivot_ratio'].mean(dim=['mu_p2m', 'sigma_profit', 'k']).values
    reach_optimality_mu_sum = th['reach_optimality'].mean(dim=['mu_p2m', 'sigma_profit', 'k']).values
    time_to_reach_optimality_mu_sum = th['time_to_reach_optimality'].mean(dim=['mu_p2m', 'sigma_profit', 'k']).values

    pivot_ratio_sigma_profit = th['pivot_ratio'].mean(dim=['mu_p2m', 'mu_sum', 'k']).values
    reach_optimality_sigma_profit = th['reach_optimality'].mean(dim=['mu_p2m', 'mu_sum', 'k']).values
    time_to_reach_optimality_sigma_profit = th['time_to_reach_optimality'].mean(dim=['mu_p2m', 'mu_sum', 'k']).values

    pivot_ratio_k = th['pivot_ratio'].mean(dim=['mu_p2m', 'mu_sum', 'sigma_profit']).values
    reach_optimality_k = th['reach_optimality'].mean(dim=['mu_p2m', 'mu_sum', 'sigma_profit']).values
    time_to_reach_optimality_k = th['time_to_reach_optimality'].mean(dim=['mu_p2m', 'mu_sum', 'sigma_profit']).values

    plot_metrics(axs[0], mu_p2m, pivot_ratio_mu_p2m, reach_optimality_mu_p2m, time_to_reach_optimality_mu_p2m, 'mu_p2m', 'Metrics by mu_p2m')
    plot_metrics(axs[1], mu_sum, pivot_ratio_mu_sum, reach_optimality_mu_sum, time_to_reach_optimality_mu_sum, 'mu_sum', 'Metrics by mu_sum')
    plot_metrics(axs[2], sigma_profit, pivot_ratio_sigma_profit, reach_optimality_sigma_profit, time_to_reach_optimality_sigma_profit, 'sigma_profit', 'Metrics by sigma_profit')
    plot_metrics(axs[3], k, pivot_ratio_k, reach_optimality_k, time_to_reach_optimality_k, 'k', 'Metrics by k')

    # Plot histogram for action sequences
    act_seq = th['act_seq'].values.flatten()
    counter = Counter(act_seq)
    sorted_sequences = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_values = zip(*sorted_sequences)

    sorted_act_seq = []
    for label, value in zip(sorted_labels, sorted_values):
        sorted_act_seq.extend([label] * value)

    # Plotting using the sorted action sequences
    axs[4].hist(sorted_act_seq, bins=len(sorted_labels), color='gray', edgecolor='black')
    axs[4].set_xticks(range(len(sorted_labels)))
    axs[4].set_xlabel('Action Sequences')
    axs[4].set_ylabel('Frequency')
    axs[4].set_title('Histogram of Action Sequences')

    plt.tight_layout()
    figure_title = th['th_name'].item() + ".png"
    figure_path = os.path.join("data/figure/th", figure_title)
    plt.savefig(figure_path)

    
def extract_mu_values_from_filename(filename):
    parts = filename.split('_')
    mu_p_d_str = [p for p in parts if p.startswith('B')][0]
    mu_m_d_str = [p for p in parts if p.startswith('C')][0]
    mu_p_d = float(mu_p_d_str[2:].strip('[]').split()[0])
    mu_m_d = float(mu_m_d_str[2:].strip('[]').split()[0])
    return mu_p_d, mu_m_d

def plot_actions_and_ratios_from_nc_files(directory='data/experiment'):
    # Collect all .nc files in the directory
    nc_files = [f for f in os.listdir(directory) if f.endswith('.nc')]
    
    action_colors = {'pivot_product': 'green', 'pivot_market': 'purple', 'scale': 'red'}
    mu_p_d_values = []
    mu_m_d_values = []
    actions = []

    # Read through each .nc file and extract mu_p_d, mu_m_d and actions
    for nc_file in nc_files:
        file_path = os.path.join(directory, nc_file)
        ds = xr.open_dataset(file_path)
        
        mu_p_d, mu_m_d = extract_mu_values_from_filename(nc_file)
        mu_p_d_values.append(mu_p_d)
        mu_m_d_values.append(mu_m_d)
        actions.append(ds['action'].values)
    
    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    # Plot for mu_p_d and mu_m_d (Ratio of pivot_product to pivot_market)
    pivot_ratios = []
    for action_list in actions:
        pivot_product_count = sum(1 for action in action_list if action == 'pivot_product')
        pivot_market_count = sum(1 for action in action_list if action == 'pivot_market')
        if pivot_market_count > 0:
            pivot_ratio = pivot_product_count / pivot_market_count
        else:
            pivot_ratio = np.nan  # Avoid division by zero
        pivot_ratios.append(pivot_ratio)

    scatter = ax[1].scatter(mu_p_d_values, mu_m_d_values, c=pivot_ratios, cmap='viridis', edgecolor='k', s=100)
    ax[1].set_xlabel('mu_p_d')
    ax[1].set_ylabel('mu_m_d')
    ax[1].set_title('Ratio of pivot_product to pivot_market vs mu_p_d and mu_m_d')
    fig.colorbar(scatter, ax=ax[1], label='Pivot Product / Pivot Market Ratio')
    
    plt.tight_layout()
    fig_name = f"data/figure/aseq.png"
    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == "__main__":
    th = brute_force()
    # th = xr.open_dataset("data/theory/mu_p2m-4.0to-4.0_mu_sum-4.0to0.0_B3_C1_s1.01.0_k1.0to1.0_T3_man_b2c.nc")
    plot_theory_given_experiment(th)
    # test_hypothesis_1
    
    # for mu_m_d in mu_m_d_range:

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