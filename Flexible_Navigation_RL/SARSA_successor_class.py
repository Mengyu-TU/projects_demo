import numpy as np
import helper_func


class TabularSuccessorAgent(object):  # not required in Python 3 to put 'object' in brackets
    def __init__(self, state_size, action_size, learning_rate, gamma, exploration_rate):
        self.state_size = state_size  # total number of states in the grid world
        self.action_size = action_size  # total number of actions
        self.M = np.stack([np.identity(state_size) for i in range(action_size)])  # default axis = 0
        self.w = np.zeros([state_size])
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate

    def Q_estimates(self, state, goal=None):
        # Generate Q values for all actions.
        if goal is None:
            goal = self.w
        else:
            goal = helper_func.onehot(goal, self.state_size)
        return np.matmul(self.M[:, state, :], goal)

    def sample_action(self, state, goal=None, epsilon=0.0):
        # Samples action using epsilon-greedy approach
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            Qs = self.Q_estimates(state, goal)
            action = np.argmax(Qs)
        return action

    def get_optimal_w(self, env):
        """

        :param env: environment with reward loc info
        :return: optimal w vector for SR
        """
        self.w = np.zeros([self.state_size])
        self.w[env.goal] = 1

    def update_w(self, current_exp):
        """
        Update reward function R(st)
        :param current_exp:
        :return:
        """
        # A simple update rule
        s_1 = current_exp[2]  # next state
        r = current_exp[3]  # reward
        error = r - self.w[s_1]
        self.w[s_1] += self.learning_rate * error
        return error

    def SARSA_update_sr(self, current_exp, next_exp):
        # SARSA TD learning rule
        s = current_exp[0]
        s_a = current_exp[1]
        s_1 = current_exp[2]
        s_a_1 = next_exp[1]
        r = current_exp[3]
        d = current_exp[4]
        I = helper_func.onehot(s, self.state_size)
        if d: # terminal state
            td_error = (I + self.gamma * helper_func.onehot(s_1, self.state_size) - self.M[s_a, s, :])
        else:
            td_error = (I + self.gamma * self.M[s_a_1, s_1, :] - self.M[s_a, s, :])
        self.M[s_a, s, :] += self.learning_rate * td_error
        return td_error




