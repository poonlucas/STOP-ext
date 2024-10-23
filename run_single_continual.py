import numpy as np
# import matplotlib.pyplot as plt

import os
import shutil
import pdb
import argparse
import random
import copy

import torch

from policies import LQ, LSR, LCQ, LSCQ, MaxWeight, Random, \
    LASQ, LSQ, Threshold, LSQNModel, MWNModel, LQNModel, CleanRLPolicy, \
    CCMaxWeight, CCPriority1, CCPriority3, CCBackPressure
from server_allocation import SAQueue, SANetwork, NSSANetwork
from nmodel import NModelNetwork
from criss_cross import CrissCrossNetwork

from env_configs import queue_configs, ns_queue_configs, crisscross_configs

import utils


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# saving
parser.add_argument('--outfile', default=None)

# common setup
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--env_name', type=str, required=True)
parser.add_argument('--algo_name', type=str, default='all')
parser.add_argument('--mdp_num', default=0, type=int)
parser.add_argument('--gamma', default=0.999, type=float)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--act_function', default='relu', type=str)
parser.add_argument('--reward_function', default='opt', type=str)
parser.add_argument('--state_transformation', default='id', type=str)
parser.add_argument('--state_bound', default=np.inf, type=float)
parser.add_argument('--use_action_mask', default=False, type=str2bool)
parser.add_argument('--lyp_power', default=1., type=float)

parser.add_argument('--replay_epochs', default=10, type=int)
parser.add_argument('--adam_betas', nargs=2, default=[0.9, 0.9], type=float)

parser.add_argument('--truncated_horizon', default=2048, type=int)  # same as train_freq in DQN
parser.add_argument('--deployed_interaction_steps', default=250000, type=int)
parser.add_argument('--deployed_interaction_step_skip', default=10, type=int)

FLAGS = parser.parse_args()

assert (FLAGS.deployed_interaction_steps % FLAGS.deployed_interaction_step_skip == 0)

log_dir = 'temp_{}'.format(FLAGS.outfile)
os.makedirs(log_dir, exist_ok=True)


def get_env(mdp_num = 0):
    if FLAGS.env_name == 'queue':
        config = queue_configs[mdp_num]
        qs = []
        for idx, con in enumerate(config):
            info = {
                'arrival': {
                    'is_stationary': True,
                    'prob': con[0]
                },
                'service': {
                    'is_stationary': True,
                    'prob': con[1]
                },
                'connection': {
                    'is_stationary': True,
                    'prob': con[2],
                }
            }
            q = SAQueue(str(idx), info)
            qs.append(q)
        qs = np.array(qs)

    elif FLAGS.env_name == 'nsqueue':
        trans_config = ns_queue_configs[mdp_num]
        qs = []
        for config in trans_config:
            trans_qs = []
            for idx, con in enumerate(config):
                info = {
                    'arrival': {
                        'is_stationary': True,
                        'prob': con[0]
                    },
                    'service': {
                        'is_stationary': True,
                        'prob': con[1]
                    },
                    'connection': {
                        'is_stationary': True,
                        'prob': con[2],
                    }
                }
                q = SAQueue(str(idx), info)
                trans_qs.append(q)
            qs.append(trans_qs)
        qs = np.array(qs)

    elif FLAGS.env_name == 'nmodel':
        if FLAGS.mdp_num == 0:
            load = 0.99
            arrivals = [1.3 * load, 0.4 * load]
            holding_costs = [3., 1.]
            mus = [1., 1. / 2, 1.]
        elif FLAGS.mdp_num == 1:
            load = 0.99
            arrivals = [1.3 * load, 0.4 * load]
            holding_costs = [3., 1.]
            mus = [1., 1. / 2, 1.]
        elif FLAGS.mdp_num == 2:
            load = 0.95
            arrivals = [0.9, 0.8]
            holding_costs = [3., 1.]
            mus = [1., 0.9, 0.8]

    elif FLAGS.env_name == 'crisscross':
        arrivals, mus = crisscross_configs[mdp_num]
        env = CrissCrossNetwork(arrivals,
                                mus,
                                reward_func=FLAGS.reward_function,
                                state_trans=FLAGS.state_transformation,
                                state_bound=FLAGS.state_bound,
                                lyp_power=FLAGS.lyp_power)
        return env

    if FLAGS.env_name == 'gridworld' or FLAGS.env_name == 'queue':
        env = SANetwork(qs, reward_func=FLAGS.reward_function, \
                        state_trans=FLAGS.state_transformation,
                        gridworld=FLAGS.env_name == 'gridworld',
                        state_bound=FLAGS.state_bound,
                        lyp_power=FLAGS.lyp_power)
        print('stable policy exists {}'.format(env.is_stable()))

    if FLAGS.env_name == 'nsqueue':
        env = NSSANetwork(qs, reward_func=FLAGS.reward_function, \
                        state_trans=FLAGS.state_transformation,
                        gridworld=FLAGS.env_name == 'gridworld',
                        state_bound=FLAGS.state_bound,
                        lyp_power=FLAGS.lyp_power)
        print('stable policy exists {}'.format(env.is_stable()))

    elif FLAGS.env_name == 'nmodel':
        env = NModelNetwork(arrivals,
                            mus,
                            holding_costs,
                            reward_func=FLAGS.reward_function, \
                            state_trans=FLAGS.state_transformation,
                            state_bound=FLAGS.state_bound,
                            lyp_power=FLAGS.lyp_power)
    return env


def _train_RL(env, algo_name, variant=None, state_transformation=None, fname=None):
    print('training {} {}'.format(algo_name, variant))

    env.set_horizon(-1)

    pt_path = None

    env.set_use_mask(FLAGS.use_action_mask)
    pi = CleanRLPolicy(env,
                       learning_rate=FLAGS.lr,
                       gamma=FLAGS.gamma,
                       variant=variant,
                       num_steps=FLAGS.truncated_horizon,
                       update_epochs=FLAGS.replay_epochs,
                       use_action_mask=FLAGS.use_action_mask,
                       adam_betas=FLAGS.adam_betas
                       )

    # continuing task
    total_timesteps = FLAGS.deployed_interaction_steps

    pi.learn(total_timesteps=total_timesteps)
    stats = pi.get_stats(FLAGS.deployed_interaction_step_skip)
    return pi, stats


def run_experiment_algo(env, algo_name):
    fname = 'temp' if FLAGS.outfile is None else FLAGS.outfile
    print('executing {} policy'.format(algo_name))
    visited_native_states = None
    vis_ns = []
    pi_stats = None
    next_native_states = None
    if algo_name == 'PPO' or 'STOP' in algo_name:
        variant = 'zhang'
        rl_res = []
        print('executing {} {}'.format(algo_name, FLAGS.state_transformation))
        # env.set_state_transformation(FLAGS.state_transformation)
        # env.env_method('set_state_transformation', FLAGS.state_transformation)
        pi, pi_stats = _train_RL(env, algo_name=algo_name, state_transformation=FLAGS.state_transformation, fname=fname,
                                 variant=variant)
        rl_bl = pi_stats['backlog']
        rl_bl = rl_bl[:int(
            FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip)]  # remove extra fluff that sb3 has
        print(pi_stats)
        visited_native_states = pi_stats['visited_native_states']  # pi_stats['visited_native_states']
        vis_ns.append(visited_native_states)
        rl_res.append(rl_bl)
        # utils.plot_heatmap(pi, '{}_{}'.format(fname, 'zhang-' + t), transformation = t)
        backlogs = rl_res
    elif algo_name == 'LCQ' \
            or algo_name == 'MW' \
            or algo_name == 'LSCQ' \
            or algo_name == 'Rand' \
            or algo_name == 'Thresh' \
            or algo_name == 'MWN' \
            or algo_name == 'LSQN' \
            or algo_name == 'LQN' \
            or algo_name == 'CCMW' \
            or algo_name == 'CCP1' \
            or algo_name == 'CCP3' \
            or algo_name == 'CCBP':
        env.set_horizon(-1)
        if algo_name == 'LCQ':
            pi = LCQ(env)
        elif algo_name == 'MW':
            pi = MaxWeight(env)
        elif algo_name == 'LSCQ':
            pi = LSCQ(env)
        elif algo_name == 'Rand':
            pi = Random(env, smart=True)
        elif algo_name == 'Thresh':
            pi = Threshold(env)
        elif algo_name == 'MWN':
            pi = MWNModel(env)
        elif algo_name == 'LSQN':
            pi = LSQNModel(env)
        elif algo_name == 'LQN':
            pi = LQNModel(env)
        elif algo_name == 'CCMW':
            pi = CCMaxWeight(env)
        elif algo_name == 'CCP1':
            pi = CCPriority1(env)
        elif algo_name == 'CCP3':
            pi = CCPriority3(env)
        elif algo_name == 'CCBP':
            pi = CCBackPressure
        paths, _ = utils.collect_data(env, pi, 1, FLAGS.deployed_interaction_steps)
        bl = paths[0]['avg_backlog']
        bl = bl[:FLAGS.deployed_interaction_steps:FLAGS.deployed_interaction_step_skip]
        backlogs = [bl]
        ns = paths[0]['native_state']
        ns = ns[:FLAGS.deployed_interaction_steps:FLAGS.deployed_interaction_step_skip]
        vis_ns = [ns]
    return backlogs, vis_ns, pi_stats


def main():
    seed = FLAGS.seed
    utils.set_seed_everywhere(seed)

    env = get_env(FLAGS.mdp_num)
    denom = np.arange(1, FLAGS.deployed_interaction_steps / FLAGS.deployed_interaction_step_skip + 1)

    algo_name = FLAGS.algo_name
    algos = [algo_name]
    backlogs, visited_native_states, pi_stats = run_experiment_algo(env, algo_name)

    avg_backlogs = [np.divide(np.cumsum(backlog), denom) for backlog in backlogs]
    # avg_backlogs = [backlog for backlog in backlogs]

    summary = {
        'results': {},
        'seed': seed,
        'hp': {
            'truncated_horizon': FLAGS.truncated_horizon,
            'lr': FLAGS.lr,
            'gamma': FLAGS.gamma,
            'replay_epochs': FLAGS.replay_epochs,
        },
    }
    for idx, algo in enumerate(algos):
        summary['results'][algo] = {
            'avg_backlog': avg_backlogs[idx],
            # 'unstable_adv_mean': pi_stats['unstable_adv_mean'] if pi_stats else 0,
            # 'unstable_frac': pi_stats['unstable_frac'] if pi_stats else 0
            'visited_native_states': visited_native_states[idx]
            # 'avg_backlog_changes': backlog_changes[idx]
        }
        if 'STOP' in algo:
            summary['results'][algo].update({
                'actor_dormant': pi_stats['actor_dormant'],
                'critic_dormant': pi_stats['critic_dormant'],
                'actor_weight_norm': pi_stats['actor_weight_norm'],
                'critic_weight_norm': pi_stats['critic_weight_norm'],
                'total_losses': pi_stats['total_losses'],
                'value_losses': pi_stats['value_losses'],
                'policy_losses': pi_stats['policy_losses'],
                'entropy_losses': pi_stats['entropy_losses'],
                'old_approx_kls': pi_stats['old_approx_kls'],
                'approx_kls': pi_stats['approx_kls']
            })
    print(summary)
    np.save(FLAGS.outfile, summary)


if __name__ == '__main__':
    main()
