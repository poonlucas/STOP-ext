import argparse
import pdb

from numba import jit
import numpy as np
import os
import time
import itertools
from itertools import product

from env_configs import queue_configs
from server_allocation import SAQueue, SANetwork
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# common setup
parser.add_argument('--outfile', default=None)

parser.add_argument('--env_name', type=str, required=True)
parser.add_argument('--algo_name', type=str, default='all')
parser.add_argument('--mdp_num', default=0, type=int)
parser.add_argument('--state_bound', default=np.inf, type=float)
parser.add_argument('--reward_function', default='opt', type=str)
parser.add_argument('--state_transformation', default='id', type=str)
parser.add_argument('--lyp_power', default=1., type=float)

parser.add_argument('--compare', default=None, type=str)
parser.add_argument('--normalize', default=False, type=str2bool)
parser.add_argument('--model', default=False, type=str2bool)

FLAGS = parser.parse_args()

out_dir = os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', f'{FLAGS.outfile}')
os.makedirs(out_dir, exist_ok=True)


# For Numba
@jit(nopython=True)
def opt_value_iteration(iterations, Q, bound, gamma, probs):
    for it in range(iterations):
        q = np.copy(Q)
        for i in range(bound):
            for j in range(bound):
                a_i = (i + 1) if (i + 1) < bound else i
                a_j = (j + 1) if (j + 1) < bound else j
                s_i = (i - 1) if i > 0 else i
                s_j = (j - 1) if i > 0 else j

                # [0, 0]
                curr_q = -((i + j) / 2.) + (gamma * np.max(Q[i, j]))
                # [1, 1]
                a_both = -((a_i + a_j) / 2.) + (gamma * np.max(Q[a_i, a_j]))
                # [1, 0]
                a_1 = -((a_i + j) / 2.) + (gamma * np.max(Q[a_i, j]))
                # [0, 1]
                a_2 = -((i + a_j) / 2.) + (gamma * np.max(Q[i, a_j]))
                # [-1, 0]
                s_1 = -((s_i + j) / 2.) + (gamma * np.max(Q[s_i, j]))
                # [-1, 1]
                sa_12 = -((s_i + a_j) / 2.) + (gamma * np.max(Q[s_i, a_j]))
                # [0, -1]
                s_2 = -((i + s_j) / 2.) + (gamma * np.max(Q[i, s_j]))
                # [1, -1]
                as_12 = -((a_i + s_j) / 2.) + (gamma * np.max(Q[a_i, s_j]))

                values = np.asarray([[s_1, curr_q, sa_12, a_2, a_1, a_both],
                                     [s_2, curr_q, a_2, as_12, a_1, a_both]])

                q_vals = [np.dot(probs[0], values[0].T), np.dot(probs[1], values[1].T)]
                q[i, j, :] = q_vals
        Q = q
        print(f'Iteration {it}')
    return Q


class SALyapunov:
    def __init__(self, env):
        self.env = env
        self.bound = 30
        self.ma = np.zeros((self.bound, self.bound))
        for i in range(self.bound):
            for j in range(self.bound):
                self.ma[i, j] = -(i ** self.env.lyp_power + j ** self.env.lyp_power) / 2

    def get_ma(self):
        return self.ma


class SALyapunovPow:
    def __init__(self, env):
        self.env = env
        self.bound = 30
        self.ma = np.zeros((self.bound, self.bound))
        for i in range(self.bound):
            for j in range(self.bound):
                self.ma[i, j] = -((i + j) / 2) ** self.env.lyp_power

    def get_ma(self):
        return self.ma


class SALyapunovPiecewise:
    def __init__(self, env):
        self.env = env
        self.bound = 30
        self.ma = np.zeros((self.bound, self.bound))
        for i in range(self.bound):
            for j in range(self.bound):
                if i == 0:
                    self.ma[i, j] = -((i + j) / 2) ** 2.5
                else:
                    self.ma[i, j] = -((i + j) / 2)

    def get_ma(self):
        return self.ma


class SAOptimal:  # Value iteration bounded by state space of 60
    def __init__(self, env):
        self.env = env
        self.bound = 120  # 60, 120
        self.iterations = 3000
        self.gamma = 0.99

        arrivals = np.array(list(product([0, 1], repeat=len(env.qs))))
        services = np.array(list(product([-1, 0], repeat=len(env.qs))))
        arrival_probs = {}
        service_probs = {}
        for i in range(len(arrivals)):
            arr_prob = 1.
            ser_prob = 1.
            for q in range(len(env.qs)):
                arr_prob *= env.qs[q].get_arrival_prob(0) if arrivals[i][q] else (1 - env.qs[q].get_arrival_prob(0))
                ser_prob *= (env.qs[q].get_service_prob(0) * env.qs[q].get_connection_prob(0)) if services[i][q] else \
                    (1 - env.qs[q].get_service_prob(0) * env.qs[q].get_connection_prob(0))
            arrival_probs[tuple(arrivals[i])] = arr_prob
            service_probs[tuple(services[i])] = ser_prob
        combined = [arrivals, services]
        outcomes = list(itertools.product(*combined))
        all_probs = {}
        for a in range(len(env.qs)):
            for out in outcomes:
                arr = out[0]
                ser = out[1]
                temp_ser = np.full_like(ser, 0)
                temp_ser[a] = ser[a]
                combined_out = arr + temp_ser
                combined_out_w_act = tuple(combined_out) + (a,)
                if combined_out_w_act not in all_probs:
                    all_probs[combined_out_w_act] = 0
                all_probs[combined_out_w_act] += (arrival_probs[tuple(arr)] * service_probs[tuple(ser)])

        print(all_probs)

        a_1 = []
        a_2 = []

        for key, value in all_probs.items():
            if key[2] == 0:
                a_1.append(value)
            elif key[2] == 1:
                a_2.append(value)

        self.probs = np.stack((np.array(a_1), np.array(a_1)))

        self.Q = np.zeros((self.bound, self.bound, 2))
        if os.path.isfile(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', 'opt', 'optimal_q',
                                       f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                       f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                       f'_bound_{self.bound}.npy')):
            self.Q = np.load(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', 'opt', 'optimal_q',
                                          f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                          f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                          f'_bound_{self.bound}.npy'))
        else:
            # For numba
            self.Q = opt_value_iteration(self.iterations, self.Q, self.bound, self.gamma, self.probs)
            save_dir = os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', 'opt', 'optimal_q')
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', 'opt', 'optimal_q',
                                 f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                 f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                 f'_bound_{self.bound}'), self.Q)

    def get_ma(self):
        # 30
        ma = np.zeros((30, 30))
        for q1 in range(30):
            for q2 in range(30):
                ma[q1, q2] = np.max(self.Q[int(q1), int(q2)])
        return ma


@jit(nopython=True)
def policy_value_iteration(iterations, Q, bound, gamma, probs, policy, mus):
    for it in range(iterations):
        q = np.copy(Q)
        for i in range(bound):
            for j in range(bound):
                a_i = (i + 1) if (i + 1) < bound else i
                a_j = (j + 1) if (j + 1) < bound else j
                s_i = (i - 1) if i > 0 else i
                s_j = (j - 1) if i > 0 else j

                # [0, 0]
                curr_q = -((i + j) / 2.) + (gamma * Q[i, j])
                # [1, 1]
                a_both = -((a_i + a_j) / 2.) + (gamma * Q[a_i, a_j])
                # [1, 0]
                a_1 = -((a_i + j) / 2.) + (gamma * Q[a_i, j])
                # [0, 1]
                a_2 = -((i + a_j) / 2.) + (gamma * Q[i, a_j])
                # [-1, 0]
                s_1 = -((s_i + j) / 2.) + (gamma * Q[s_i, j])
                # [-1, 1]
                sa_12 = -((s_i + a_j) / 2.) + (gamma * Q[s_i, a_j])
                # [0, -1]
                s_2 = -((i + s_j) / 2.) + (gamma * Q[i, s_j])
                # [1, -1]
                as_12 = -((a_i + s_j) / 2.) + (gamma * Q[a_i, s_j])

                values = np.asarray([[s_1, curr_q, sa_12, a_2, a_1, a_both],
                                     [s_2, curr_q, a_2, as_12, a_1, a_both]])

                a = 0
                if policy == "random":
                    pot_queues = np.arange(2)
                    a = np.random.choice(pot_queues)
                elif policy == "lcq":
                    if i <= j:
                        a = 1
                elif policy == "maxweight":
                    if j * mus[1] >= i * mus[0]:
                        a = 1
                elif policy == "lscq":
                    if mus[0] > mus[1]:
                        if i > 0:
                            a = 0
                        else:
                            a = 1
                    else:
                        if j > 0:
                            a = 1
                        else:
                            a = 0

                q_val = np.dot(probs[a], values[a].T)
                q[i, j] = q_val
        Q = q
        print(f'Iteration {it}')
    return Q


class SAPolicy:
    def __init__(self, env, policy):
        self.env = env
        self.bound = 60  # 60
        self.iterations = 3000  # 3000
        self.gamma = 0.99
        self.policy = policy

        arrivals = np.array(list(product([0, 1], repeat=len(env.qs))))
        services = np.array(list(product([-1, 0], repeat=len(env.qs))))
        arrival_probs = {}
        service_probs = {}
        for i in range(len(arrivals)):
            arr_prob = 1.
            ser_prob = 1.
            for q in range(len(env.qs)):
                arr_prob *= env.qs[q].get_arrival_prob(0) if arrivals[i][q] else (1 - env.qs[q].get_arrival_prob(0))
                ser_prob *= env.qs[q].get_service_prob(0) if services[i][q] else (1 - env.qs[q].get_service_prob(0))
            arrival_probs[tuple(arrivals[i])] = arr_prob
            service_probs[tuple(services[i])] = ser_prob
        combined = [arrivals, services]
        outcomes = list(itertools.product(*combined))
        all_probs = {}
        for a in range(len(env.qs)):
            for out in outcomes:
                arr = out[0]
                ser = out[1]
                temp_ser = np.full_like(ser, 0)
                temp_ser[a] = ser[a]
                combined_out = arr + temp_ser
                combined_out_w_act = tuple(combined_out) + (a,)
                if combined_out_w_act not in all_probs:
                    all_probs[combined_out_w_act] = 0
                all_probs[combined_out_w_act] += (arrival_probs[tuple(arr)] * service_probs[tuple(ser)])

        a_1 = []
        a_2 = []

        for key, value in all_probs.items():
            if key[2] == 0:
                a_1.append(value)
            elif key[2] == 1:
                a_2.append(value)

        self.probs = np.stack((np.array(a_1), np.array(a_1)))

        self.Q = np.zeros((self.bound, self.bound))
        if os.path.isfile(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', f'{FLAGS.outfile}',
                                       f'{self.policy}_q',
                                       f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                       f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                       f'_bound_{self.bound}.npy')):
            self.Q = np.load(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', f'{FLAGS.outfile}',
                                          f'{self.policy}_q',
                                          f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                          f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                          f'_bound_{self.bound}.npy'))
        else:
            # For numba
            start = time.time()
            mus = np.array([env.qs[0].get_service_prob(0), env.qs[1].get_service_prob(0)])
            self.Q = policy_value_iteration(self.iterations, self.Q, self.bound, self.gamma, self.probs, self.policy,
                                            mus)
            end = time.time()
            print(start - end)
            np.save(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', f'{FLAGS.outfile}',
                                 f'{self.policy}_q',
                                 f'q_{self.env.qs[0].get_arrival_prob(0)}_{self.env.qs[1].get_arrival_prob(0)}_'
                                 f'{self.env.qs[0].get_service_prob(0)}_{self.env.qs[1].get_service_prob(0)}'
                                 f'_bound_{self.bound}'), self.Q)

    def get_ma(self):
        ma = np.zeros((30, 30))
        for q1 in range(30):
            for q2 in range(30):
                ma[q1, q2] = self.Q[int(q1), int(q2)]
        return ma


def get_env():
    if FLAGS.env_name == 'queue':
        config = queue_configs[FLAGS.mdp_num]
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
        env = SANetwork(qs, reward_func=FLAGS.reward_function,
                        state_trans=FLAGS.state_transformation,
                        gridworld=FLAGS.env_name == 'gridworld',
                        state_bound=FLAGS.state_bound,
                        lyp_power=FLAGS.lyp_power)
        print('stable policy exists {}'.format(env.is_stable()))
        return env


def main():
    env = get_env()

    pi = None
    if FLAGS.algo_name == "lyp":
        pi = SALyapunov(env)
    elif FLAGS.algo_name == "lyp-pow":
        pi = SALyapunovPow(env)
    elif FLAGS.algo_name == "lyp-piecewise":
        pi = SALyapunovPiecewise(env)
    elif FLAGS.algo_name == "opt":
        pi = SAOptimal(env)
    elif FLAGS.algo_name == "random" or "lcq" or "mw" or "lscq":
        pi = SAPolicy(env, FLAGS.algo_name)
    ma = pi.get_ma()

    if FLAGS.model:
        ma_shape = ma.shape
        X1 = []  # ax^2 + by^2 + cx + dy
        X2 = []  # ax^2 + by^2 + cxy
        X3 = []  # ax^2 + by^2 + 2abxy
        X4 = []  # ax + by
        Y = []
        for i in range(1, ma_shape[0]):
            for j in range(1, ma_shape[1]):
                X1.append([i ** 2, j ** 2, i, j])
                X2.append([i ** 2, j ** 2, i * j])
                X4.append([i, j])
                Y.append(ma[i, j])

        # for i in range(ma_shape[0]):
        #     X1.append([i ** 2, i])
        #     X2.append([i ** 2])
        #     # X4.append([i])
        #     Y.append(ma[i, 0])

        X1 = np.array(X1)
        X2 = np.array(X2)
        X4 = np.array(X4)
        theta1 = np.dot(np.dot(np.linalg.inv(np.dot(X1.T, X1)), X1.T), Y)
        theta2 = np.dot(np.dot(np.linalg.inv(np.dot(X2.T, X2)), X2.T), Y)
        theta4 = np.dot(np.dot(np.linalg.inv(np.dot(X4.T, X4)), X4.T), Y)

        print(f"Option 1: {theta1[0]} q1^2 + {theta1[1]} q2^2 + {theta1[2]} q1 + {theta1[3]} q2")
        print(f"Option 2: {theta2[0]} q1^2 + {theta2[1]} q2^2 + {theta2[2]} q1q2")
        print(f"Option 3: {theta4[0]} q1 + {theta4[1]} q2")

        # print(f"Option 1: {theta1[0]} q1^2 + {theta1[1]} q1")
        # print(f"Option 2: {theta2[0]} q1^2")

        # model = []
        # for i in range(ma_shape[0]):
        #     temp = []
        #     for j in range(ma_shape[1]):
        #         temp.append(np.sum(np.dot(theta, np.asarray([i ** 2, j ** 2, i, j]))))
        #     model.append(temp)

    if FLAGS.normalize:
        min_ma = np.min(ma)
        max_ma = np.max(ma)
        ma = (np.array(ma) - min_ma) / (max_ma - min_ma)
        np.save(os.path.join('heatmaps', f'{FLAGS.env_name}', f'mdp_{FLAGS.mdp_num}', f'{FLAGS.outfile}', f'{FLAGS.algo_name}'), ma)

    if FLAGS.compare is not None:
        c_pi = None
        if FLAGS.compare == "opt":
            c_pi = SAOptimal(env)
        c_ma = c_pi.get_ma()
        if FLAGS.normalize:
            min_c_ma = np.min(c_ma)
            max_c_ma = np.max(c_ma)
            c_ma = (np.array(c_ma) - min_c_ma) / (max_c_ma - min_c_ma)
        ma = np.abs(ma - c_ma)

    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(20, 18))

    ax = sns.heatmap(ma, linewidth=0.5)
    ax.invert_yaxis()
    plt.title('Value')
    plt.xlabel('Q1 length')
    plt.ylabel('Q2 length')

    outfile = FLAGS.algo_name
    if "lyp" in FLAGS.algo_name:
        outfile = outfile + str(FLAGS.lyp_power)
    if FLAGS.compare is not None:
        outfile = outfile + "-" + FLAGS.compare
    if FLAGS.normalize:
        outfile = outfile + "_normalized"

    plt.savefig(f'heatmaps/{FLAGS.env_name}/mdp_{FLAGS.mdp_num}/{FLAGS.outfile}/{outfile}_heat.jpg', bbox_inches='tight')
    plt.close()

    x, y = np.meshgrid(np.arange(ma.shape[1]), np.arange(ma.shape[0]))
    fig = plt.figure(figsize=(20, 18))
    ax = fig.add_subplot(111, projection='3d')
    cmap = sns.color_palette("rocket", as_cmap=True)
    surface = ax.plot_surface(x, y, ma, cmap=cmap)
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    ax.view_init(elev=40, azim=45)  # 225 for (0,0), 45 for (30,30)

    ax.set_xlabel('Q1 length', labelpad=20)
    ax.set_ylabel('Q2 length', labelpad=20)
    ax.set_zlabel('Value', labelpad=20)

    plt.savefig(f'heatmaps/{FLAGS.env_name}/mdp_{FLAGS.mdp_num}/{FLAGS.outfile}/{outfile}_3dheat.jpg', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
