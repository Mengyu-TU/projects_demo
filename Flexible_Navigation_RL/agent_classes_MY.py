import numpy as np
import helper_func


class Q_LearningAgent(object):
    def __init__(self, state_size, action_size, gamma, learning_rate, exploration_rate=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_size, action_size))

    def sample_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(self.action_size)
        else:
            action = self.argmax(state)
        return action

    def update_q_table(self, state, action, reward, next_state):
        max_q_value = np.max(self.q_table[next_state])
        td_error = reward + self.gamma * max_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def update_q_terminate(self, state, action, reward):
        td_error = reward - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def argmax(self, state):
        """
        Take in a list of q values and returns the index of the item with highest value
        Break tie randomly.
        :return:
        """
        top = float("-inf")
        ties = []
        for i in range(len(self.q_table[state])):
            if self.q_table[state][i] > top:
                top = self.q_table[state][i]
                ties = [i]
            elif self.q_table[state][i] == top:
                ties.append(i)

        return ties[np.random.choice(len(ties))]


class Dyna_Q(object):
    def __init__(self, state_size, action_size, gamma, learning_rate, exploration_rate=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_size, action_size))
        self.model = np.nan * np.zeros((state_size, action_size, 2))

    def sample_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(self.action_size)
        else:
            action = self.argmax(state)
        return action

    def update_q_table(self, state, action, reward, next_state):
        max_q_value = np.max(self.q_table[next_state])
        td_error = reward + self.gamma * max_q_value - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def update_q_terminate(self, state, action, reward):
        td_error = reward - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def argmax(self, state):
        """
        Take in a list of q values and returns the index of the item with highest value
        Break tie randomly.
        :return:
        """
        top = float("-inf")
        ties = []
        for i in range(len(self.q_table[state])):
            if self.q_table[state][i] > top:
                top = self.q_table[state][i]
                ties = [i]
            elif self.q_table[state][i] == top:
                ties.append(i)

        return ties[np.random.choice(len(ties))]

    def dyna_q_model_update(self, state, action, reward, next_state):
        self.model[state, action] = reward, next_state

    def dyna_q_planning(self, k):
        for _ in range(k):
            # return the indices of valid experiences
            candidates = np.array(np.where(~np.isnan(self.model[:, :, 0]))).T

            idx = np.random.choice(len(candidates))

            state, action = candidates[idx]
            reward, next_state = self.model[state, action]
            next_state = next_state.astype(int)
            # update value function
            self.update_q_table(state, action, reward, next_state)


class SARSAAgent(object):
    def __init__(self, state_size, action_size, gamma, learning_rate, exploration_rate=0.2):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((state_size, action_size))

    def sample_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.randint(self.action_size)
        else:
            action = self.argmax(state)
        return action

    def update_q_table(self, state, action, reward, next_state, next_action):
        td_error = reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def update_q_terminate(self, state, action, reward):
        td_error = reward - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def argmax(self, state):
        """
        Take in a list of q values and returns the index of the item with highest value
        Break tie randomly.
        :return:
        """
        top = float("-inf")
        ties = []
        for i in range(len(self.q_table[state])):
            if self.q_table[state][i] > top:
                top = self.q_table[state][i]
                ties = [i]
            elif self.q_table[state][i] == top:
                ties.append(i)

        return ties[np.random.choice(len(ties))]


