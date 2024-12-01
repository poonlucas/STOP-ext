import pdb

from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
import torch

if __name__ == '__main__':
    data = {}
    t = [200 * i for i in range(10000)]
    t2 = [1 * i for i in range(2000000)]
    #dir = "results/queue/betas"
    #dir = "results/queue/betas_new/mdp2"
    dir = "results/hihi"
    beta = "0.8"
    # avg_backlog_lim = [1.7, 1.95]
    avg_backlog_lim = [10, 25]
    for basename in os.listdir(f'{dir}/0.9_{beta}'):
        if '.npy' not in basename:
            continue
        if '.npz' in basename:
            continue
        f_name = os.path.join(f'{dir}/0.9_{beta}', basename)
        # 'results_queue_main_mdp_2/env_queue_exp_main_algo_STOP-1.5_seed_553395_mdp-num_2_truncated-horizon_200_lr_0.0003_epochs_10_adam-beta_0.9.npy'
        names = f_name.split('_')
        summary = np.load(f_name, allow_pickle=True).item()
        results = summary['results']['STOP-2.5']

        # Loss
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Interaction Steps (x10)')
        ax1.set_ylabel('Avg. Queue Length', color=color)
        ax1.plot(t2, results['avg_backlog'], color=color)
        #ax1.set_ylim(avg_backlog_lim)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        ax2.set_ylabel('Loss (1e5)')
        color = 'tab:blue'
        ax2.plot(t, np.divide(results['total_losses'], 1e5), color=color, label='Total Loss')
        color = 'tab:green'
        ax2.plot(t, np.divide(results['value_losses'], 1e5), color=color, label='Value Loss')
        color = 'tab:orange'
        ax2.plot(t, np.divide(results['policy_losses'], 1e5), color=color, label='Policy Loss')
        color = 'tab:brown'
        ax2.plot(t, np.divide(results['entropy_losses'], 1e5), color=color, label='Entropy Loss')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()

        fig.tight_layout()
        plt.savefig(f'{dir}/{beta}_graphs/seed_{summary["seed"]}_loss.png')

        plt.close()

        # Dormant
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Interaction Steps (x10)')
        ax1.set_ylabel('Avg. Queue Length', color=color)
        ax1.plot(t2, results['avg_backlog'], color=color)
        ax1.set_ylim(avg_backlog_lim)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        ax2.set_ylabel('Dormant')
        color = 'tab:blue'
        actor_dormant = [i for i in results['actor_dormant']]
        ax2.plot(t, actor_dormant, color=color, label='Actor Dormant')
        color = 'tab:orange'
        critic_dormant = [i for i in results['critic_dormant']]
        ax2.plot(t, critic_dormant, color=color, label='Critic Dormant')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=0)
        ax2.legend()

        fig.tight_layout()
        plt.savefig(f'{dir}/{beta}_graphs/seed_{summary["seed"]}_dormant.png')

        plt.close()

        # Weight Norm
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Interaction Steps (x10)')
        ax1.set_ylabel('Avg. Queue Length', color=color)
        ax1.plot(t2, results['avg_backlog'], color=color)
        ax1.set_ylim(avg_backlog_lim)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        ax2.set_ylabel('Weight Norm')
        color = 'tab:blue'
        actor_weight_norm = [i for i in results['actor_weight_norm']]
        ax2.plot(t, actor_weight_norm, color=color, label='Actor')
        color = 'tab:orange'
        critic_weight_norm = [i for i in results['critic_weight_norm']]
        ax2.plot(t, critic_weight_norm, color=color, label='Critic')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()

        fig.tight_layout()
        plt.savefig(f'{dir}/{beta}_graphs/seed_{summary["seed"]}_weight_norm.png')

        plt.close()
