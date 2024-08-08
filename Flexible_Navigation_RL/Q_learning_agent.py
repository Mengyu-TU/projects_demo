import numpy as np
import matplotlib.pyplot as plt
import helper_func
from grid_world import SimpleGrid
from agent_classes_MY import Q_LearningAgent

# Set up environment
grid_size = 7
pattern = "four_rooms_left_bridge"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode='index')  # index mode: directly return 1d agent pos
env.reset(agent_pos=[6, 6], goal_pos=[0, 6])
plt.imshow(env.grid)  # env.grid is a grid_size * grid_size * 3 np array
# layer 1 is the agent position
# layer 2 is the target position
# layer 3 is the wall positions
plt.show()

# Set up the agent
max_steps = 1000
n_episodes = 20
gamma = 0.95
lr = 5e-1
exploration_rate = 0.9

agent = Q_LearningAgent(env.state_size, env.action_size, gamma, lr, exploration_rate)
experience = []
test_experience = []
test_lengths = []
lifetime_td_errors = []

# Run learning
reward_sums = np.zeros(n_episodes)
episode_steps = np.zeros(n_episodes)

for episode in range(n_episodes):
    env.reset(agent_pos=[6, 6], goal_pos=[0, 6])
    if episode == 20: #n_episodes // 2:  # < 1000 trials
        agent.learning_rate /= 10
        agent.exploration_rate = 0
        #env = SimpleGrid(grid_size, block_pattern="four_rooms", obs_mode='index')
        #agent.exploration_rate = 0.5
        #goal_pos = [0, grid_size - 1]


    # initialize environment
    #if episode >= 150:
        #agent.exploration_rate = 0.1
        #agent.learning_rate = 0.5
        #env = SimpleGrid(grid_size, block_pattern="four_rooms_right_bridge", obs_mode='index')
        #env.reset(agent_pos=[6, 6], goal_pos=[0, 0])

    reward_sum = 0
    for t in range(max_steps):
        state = env.observation             # get current state
        action = agent.sample_action(state) # choose an action
        reward = env.step(action)           # taking a step
        reward_sum += reward

        if env.done:
            agent.update_q_terminate(state, action, reward)
            experience.append([state, action, env.observation, reward, True])
            break

        state_next = env.observation        # next state
        done = env.done  # reach the goal pos or not
        experience.append([state, action, state_next, reward, done])
        # after updating the q value of state, then update state to state_next
        td_error = agent.update_q_table(state, action, reward, state_next)
        state = state_next

    episode_steps[episode] = t + 1
    reward_sums[episode] = reward_sum

#plt.plot(reward_sums)
#plt.show()
fig, ax = plt.subplots()
ax.plot(episode_steps)
ax.set_xlabel('Episodes', fontsize = 16)
ax.set_ylabel('Num steps per episode',fontsize = 16)
ax.tick_params(labelsize=16)
#plt.plot(episode_steps)
plt.show()

V_q = np.max(agent.q_table, axis = 1)
V_q_2D = np.reshape(V_q, (grid_size,grid_size))

# plot the heat map
fig, ax = plt.subplots()
im = ax.imshow(V_q_2D)
# Set the plot title and axis labels
ax.set_title("State values (Q-learning)")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
cbar = ax.figure.colorbar(im, ax=ax)

# Set the ticks and tick labels for the x and y axes
ax.set_xticks(np.arange(V_q_2D.shape[1]))
ax.set_yticks(np.arange(V_q_2D.shape[0]))
ax.set_xticklabels(["C {}".format(i) for i in range(V_q_2D.shape[1])])
ax.set_yticklabels(["R {}".format(i) for i in range(V_q_2D.shape[0])])

plt.show()
