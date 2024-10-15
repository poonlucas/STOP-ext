import gymnasium as gym
import numpy as np
import pdb

import torch

from utils import powerset, symlog, symsqrt, sigmoid, tanh
import math, random
import copy

TIME_SMOOTHER = 10000


class CrissCrossNetwork(gym.Env):
    def __init__(self, arrivals, mus, network_name='criss-cross', reward_func='opt',
                 state_trans='id', reward_transformation='id', use_mask=False, state_bound=np.inf, lyp_power = 1.):
        self.arrivals = arrivals  # arrival rates [lambda_1, lambda_2 (always 0), lambda_3]
        self.mus = mus  # service rates [mu_1, mu_2, mu_3]
        self.uniform_rate = np.sum(arrivals) + np.sum(mus)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.arrivals, self.uniform_rate)  # normalized arrival rates
        self.p_compl = np.divide(self.mus, self.uniform_rate)  # normalized service rates
        self.cumsum_rates = np.unique(np.cumsum(np.concatenate([self.p_arriving, self.p_compl])))  # prob dist

        self.reward_func = reward_func

        self.action_space = gym.spaces.MultiDiscrete([3, 2])

        self.dim = len(mus)

        self.lower_state_bound = np.zeros(self.dim)
        self.upper_state_bound = state_bound
        self.observation_space = gym.spaces.Box(low=self.lower_state_bound,
                                                high=self.upper_state_bound,
                                                shape=(self.dim,),
                                                dtype=float)

        self.horizon = -1
        self.state_trans = state_trans
        self.reward_transformation = reward_transformation
        self.use_mask = use_mask
        self.t = 0
        self.network_name = network_name
        self.lyp_power = lyp_power

    def set_horizon(self, horizon):
        self.horizon = horizon

    def set_state_transformation(self, trans):
        self.state_trans = trans

    def is_stable(self):
        return True

    def transform_state(self, state):
        lens = state[:self.dim]
        if self.state_trans == 'symloge':
            adj_lens = symlog(lens, base='e')
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'symlog10':
            adj_lens = symlog(lens, base='10')
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'symsqrt':
            adj_lens = symsqrt(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'sigmoid':
            adj_lens = sigmoid(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'tanh':
            adj_lens = tanh(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'id':
            return np.array(state)

    def reset(self, seed=None, options=None):
        lens = []
        for q in range(self.dim):
            lens.append(0)
        lens = np.array(lens)

        self.native_state = np.array(lens)
        self.native_prev_state = np.array(lens)
        self.state = self.transform_state(self.native_state)
        self.prev_state = self.transform_state(self.native_prev_state)
        self.goal = np.array([0 for _ in range(self.dim)])
        print('init state: {}, goal state: {}'.format(self.native_state, self.goal))
        return self.state, {}

    def _total_queue_length(self, state):
        return np.sum(state)

    def _backlog_change(self, state, next_state):
        prev_lengths = np.mean(np.abs(state[:self.dim] - self.goal))
        curr_lengths = np.mean(np.abs(next_state[:self.dim] - self.goal))  # Manhatten distance
        metric = curr_lengths - prev_lengths
        return metric

    def reward_function(self, state=None, action=None, next_state=None):

        total_q_len = np.sum(np.abs(next_state[:self.dim] - self.goal))

        if self.reward_func == 'opt':
            reward = -1 * total_q_len
        elif 'stab-pow' in self.reward_func:
            opt_rew = -total_q_len
            if np.abs(self.lyp_power - 1) <= 1e-5:
                opt_rew = 1. / (total_q_len + 1)
            prev_lens = np.power(np.sum(state[:self.dim] - self.goal), self.lyp_power)
            curr_lens = np.power(np.sum(next_state[:self.dim] - self.goal), self.lyp_power)
            reward = -1 * (curr_lens - prev_lens) + opt_rew
        elif 'stab' in self.reward_func:
            opt_rew = -total_q_len
            if np.abs(self.lyp_power - 1) <= 1e-5:
                opt_rew = 1. / (total_q_len + 1)
            prev_lens = np.sum(np.power(state[:self.dim] - self.goal, self.lyp_power))
            curr_lens = np.sum(np.power(next_state[:self.dim] - self.goal, self.lyp_power))
            reward = -1 * (curr_lens - prev_lens) + opt_rew
        return reward

    def next_state_N1(self, state, action):
        """
        :param state: current state
        :param action: action
        :return: next state
        """

        w = np.random.random()
        wi = 0
        while w > self.cumsum_rates[wi]:
            wi += 1

        if wi == 0:  # We get class 1 job
            state_next = state + np.asarray([1, 0, 0])
        elif wi == 1:  # We get class 3 job
            state_next = state + np.asarray([0, 0, 1])
        elif wi == 2 and (action[0] == 1) and (state[0] > 0):  # Serviced class 1
            state_next = state - np.asarray([1, -1, 0])  # class 1 becomes class 2
        elif wi == 3 and (action[1] == 1) and (state[1] > 0):  # Serviced class 2
            state_next = state - np.asarray([0, 1, 0])
        elif wi == 4 and (action[0] == 2) and (state[2] > 0):  # Serviced class 3
            state_next = state - np.asarray([0, 0, 1])
        else:
            state_next = state

        return state_next

    def step(self, a):
        a = a.reshape(-1)
        assert self.action_space.contains(a)
        # assert self.observation_space.contains(self.native_state)

        # record previous state
        self.prev_state = self.state
        self.native_prev_state = self.native_state
        next_state = self.next_state_N1(self.native_state, a)

        next_state = np.array(next_state)
        next_state = np.clip(next_state, self.lower_state_bound, self.upper_state_bound)

        # for idx, q in enumerate(self.qs):
        #     q.num_jobs = next_state[idx]            

        self.native_state = next_state
        self.state = self.transform_state(self.native_state)

        backlog = self._total_queue_length(state = self.native_state)

        # if based on change, then need updated state to compute reward
        reward = self.reward_function(self.native_prev_state, a, self.native_state)

        self.t += 1
        done = False
        if self.horizon != -1:
            done = True if self.t >= self.horizon else False

        # cost may be either of metrics
        info = {
            'backlog': backlog,
            # 'backlog_change': backlog_change,
            'time': self.t,
            'native_state': self.native_prev_state,
            'next_native_state': self.native_state,
            'action': a
        }
        # assert not done
        return self.state, reward, done, False, info

    def set_use_mask(self, flag):
        self.use_mask = flag

    def set_state_bound(self, bound):
        self.lower_state_bound = 0
        self.upper_state_bound = bound

    def mask_extractor(self, obs):
        # iterate for each queue but vectorized across examples
        obs_dim = self.observation_space.shape[0]  # [0,0,0]
        masks = np.zeros((obs.shape[0], self.action_space.nvec[0]) * self.action_space.nvec[1]).astype(bool)
        if self.use_mask:
            obs = obs[:, -obs_dim:]
            lens = obs[:, :self.dim]
            cons = obs[:, self.dim: 2 * self.dim]
            for i in range(self.dim):
                masks[:, i] = np.logical_or(lens[:, i] == 0, cons[:, i] == 0) # if either empty or disconnected set mask on
                #masks[:, i] = (np.logical_or(lens[:, i] == 0, cons[:, 2 * i + 1] == 0)) # if either empty or disconnected set mask on
        return masks

    def critical_state_check(self, obs, action):
        return False, False
        obs_dim = self.observation_space.shape[0]
        obs = obs[:, -obs_dim:]
        lens = obs[:, :self.dim]
        cons = obs[:, self.dim: 2 * self.dim]
        critical = np.zeros((obs.shape[0], self.action_space.n)).astype(bool)

        for i in range(self.dim):
            critical[:, i] = np.logical_or(lens[:, i] == 0,
                                           cons[:, i] == 0)  # if either empty or disconnected set mask on

        # at least one queue empty, but excluding all queues empty
        check = not np.all(critical, axis=1) and np.any(critical, axis=1)

        def indices_for_true_values(row):
            return np.where(row)[0]

        unstable_action_taken = False
        if check:
            empty_qs = np.apply_along_axis(indices_for_true_values, axis=1, arr=critical)
            unstable_action_taken = np.isin(action, empty_qs, assume_unique=True)
            # print (obs, empty_qs, action, unstable_action_taken)
        return check, unstable_action_taken