import numpy as np
import matplotlib.pyplot as plt
import helper_func
from grid_world import SimpleGrid
from agent_classes_MY import Dyna_Q

# Set up environment
grid_size = 7
pattern = "four_rooms_left_bridge"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode='index')  # index mode: directly return 1d agent pos
env.reset(agent_pos=[6, 6], goal_pos=[0, grid_size - 1])
plt.imshow(env.grid)  # env.grid is a grid_size * grid_size * 3 np array
# layer 1 is the agent position
# layer 2 is the target position
# layer 3 is the wall positions
plt.show()

# Set up the agent
max_steps = 1000
n_episodes = 2000
gamma = 0.95
lr = 5e-1
train_epsilon = 1.0
test_epsilon = 0.1
exploration_rate = 0.1

agent = Dyna_Q(env.state_size, env.action_size, lr, gamma, exploration_rate)
experience = []
test_experience = []
test_lengths = []
lifetime_td_errors = []

# Run learning
reward_sums = np.zeros(n_episodes)
episode_steps = np.zeros(n_episodes)

for episode in range(n_episodes):
    #if episode == n_episodes // 2:  # < 1000 trials
        #env = SimpleGrid(grid_size, block_pattern="four_rooms", obs_mode='index')
        #agent.exploration_rate = 0.5
        #goal_pos = [0, grid_size - 1]
    #else:  # > 1000 trials
        #goal_pos = [grid_size - 1, grid_size - 1]
    # initialize environment
    agent_start = [6, 6]
    goal_pos = [0, grid_size - 1]
    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    reward_sum = 0

    for t in range(max_steps):
        state = env.observation
        action = agent.sample_action(state)
        reward = env.step(action)
        reward_sum += reward

        if env.done:
            agent.update_q_terminate(state, action, reward)
            experience.append([state, action, env.observation, reward, True])
            break

        state_next = env.observation  # next state
        done = env.done  # reach the goal pos or not
        experience.append([state, action, state_next, reward, done])
        # update value function
        td_error = agent.update_q_table(state, action, reward, state_next)
        # update model
        agent.dyna_q_model_update(state, action, reward, state_next)
        # execute planner
        agent.dyna_q_planning(k=10)

    episode_steps[episode] = t + 1
    reward_sums[episode] = reward_sum


#plt.plot(reward_sums)
#plt.show()

plt.plot(episode_steps)
plt.show()

