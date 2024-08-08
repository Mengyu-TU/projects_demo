import numpy as np

class DynamicPrograming(object):
    def __init__(self, state_size, action_size, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

    def value_iteration(self, env, theta):
        block_pos = []
        num_blocks = len(env.blocks)
        for i in range(num_blocks):
            block_pos.append(env.blocks[i][0] * env.grid_size + env.blocks[i][1])
        block_pos = set(block_pos)
        # initialize to 0 for all states
        V = np.zeros(self.state_size)
        while True:
            delta = 0
            for s in range(self.state_size):
                # if s is in blocks, value default to 0, skip to the next state
                if s in block_pos:
                    continue
                # update V[s]
                v = V[s]
                self.bellman_optimality_update(env, V, s)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        # Initialize to uniform random policy and then greedify
        pi = np.ones((self.state_size, self.action_size)) / self.action_size
        for s in range(self.state_size):
            self.q_greedify_policy(env, V, pi, s)
        return V, pi

    def bellman_optimality_update(self, env, V, s):
        best_v = float("-inf")
        # set the agent to be at state 's'
        agent_pos_grid = env.state_to_grid(s)
        # convert 1d position to 2d x,y coordinates
        x, y = np.where(agent_pos_grid != 0)
        for a in range(self.action_size): #
            v = 0
            env.reset_agent_pos([int(x), int(y)])
            # Reset done to False
            env.done = False
            reward, next_state = env.transition(a)  # P(s', r|s, a)
            # for probablistic environment:
            #for s_ in range(self.state_size):
                #v += transitions[s_][1] * transitions[s_][0] + self.gamma * V[s_]

            # for a deterministic environment
            if env.done: # reach terminal state
                v = reward
            else:
                v = reward + self.gamma*V[next_state]  #
            if best_v < v:
                best_v = v

        if s == env.goal:
            # terminal state always has 0 value
            V[s] = 0
        else:
            V[s] = best_v

    def q_greedify_policy(self, env, V, pi, s):
        best_v = float('-inf')
        best_a = 0
        # set the agent to be at state 's'
        agent_pos_grid = env.state_to_grid(s)
        x, y = np.where(agent_pos_grid != 0)

        for action in range(self.action_size):
            env.reset_agent_pos([int(x), int(y)])
            reward, next_state = env.transition(action)

            # for probablistic environment:
            #v = 0
            #for s_ in range(self.state_size):
            #    v += transitions[s_][1] * (transitions[s_][0] + + self.gamma * V[s_])

            # for a deterministic environment
            if env.done:
                v = reward
            else:
                v = reward + self.gamma * V[next_state]
            if best_v < v:
                best_v = v
                best_a = action
        pi[s] = np.zeros_like(pi[s])
        pi[s][best_a] = 1

