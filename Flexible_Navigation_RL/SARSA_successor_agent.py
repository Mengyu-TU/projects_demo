import numpy as np
import matplotlib.pyplot as plt
import helper_func
from grid_world import SimpleGrid
from SARSA_successor_class import TabularSuccessorAgent


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
t_max = 1000
n_episodes = 250
gamma = 0.95
lr = 5e-1

train_epsilon = 1
test_epsilon = 0
exploration_rate = 0.9

agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, exploration_rate)
experiences = []
test_experiences = []
test_lengths = []
lifetime_td_errors = []
episode_steps = np.zeros(n_episodes)

for episode in range(n_episodes):
    # training: need to update SR rep
    # agent_start = [0, 0]
    if episode == 20: #episodes // 2:
        agent.learning_rate /= 10
        agent.exploration_rate = 0.1

    if episode >= 150:
        #agent.exploration_rate = 0.1
        agent.learning_rate = 0.05
        #env = SimpleGrid(grid_size, block_pattern="four_rooms_right_bridge", obs_mode='index')
        #env.reset(agent_pos=[6, 6], goal_pos=[0, 0])

    if episode >= 150:
        #env.reset(agent_pos=[6, 6], goal_pos=[0, 0])
        env = SimpleGrid(grid_size, block_pattern="four_rooms_right_bridge", obs_mode='index')

    if episode >= 200:
        agent.learning_rate /= 10
    # initialize environment
    env.reset(agent_pos=[6, 6], goal_pos=[0, 6])
    # set the weight of the w vector in SR representation
    agent.get_optimal_w(env)
    state = env.observation
    episodic_error = []
    for j in range(t_max):
        action = agent.sample_action(state, epsilon=train_epsilon)
        reward = env.step(action)
        state_next = env.observation
        done = env.done  # reach the goal pos or not
        experiences.append([state, action, state_next, reward, done])
        state = state_next
        if (j > 1):
            td_sr = agent.SARSA_update_sr(experiences[-2], experiences[-1])
            #td_w = agent.update_w(experience[-1])
            #episodic_error.append(np.mean(np.abs(td_sr)))
        if env.done:
            td_sr = agent.SARSA_update_sr(experiences[-1], experiences[-1])
            #episodic_error.append(np.mean(np.abs(td_sr)))
            break
    #lifetime_td_errors.append(np.mean(episodic_error))
    episode_steps[episode] = j + 1

    # Test phase
    if episode >= 150:
        #env.reset(agent_pos=[6, 6], goal_pos=[0, 0])
        env = SimpleGrid(grid_size, block_pattern="four_rooms_right_bridge", obs_mode='index')

    # initialize environment
    env.reset(agent_pos=[6, 6], goal_pos=[0, 6])
    state = env.observation
    for j in range(t_max):
        action = agent.sample_action(state, epsilon=agent.exploration_rate)
        reward = env.step(action)
        state_next = env.observation
        test_experiences.append([state, action, state_next, reward])
        state = state_next
        if env.done:
            break
    test_lengths.append(j)


fig, ax = plt.subplots()
ax.plot(test_lengths)
ax.set_xlabel('Episodes', fontsize = 16)
ax.set_ylabel('Num steps per episode',fontsize = 16)
ax.tick_params(labelsize=16)
#plt.plot(episode_steps)
plt.show()

#a = np.zeros([env.state_size])
#for i in range(env.state_size):
    #Qs = agent.Q_estimates(i)
    #V = np.max(Qs)
    #a[i] = V

#V_SR_2D = np.reshape(a, (grid_size,grid_size))

# plot the heat map
#fig, ax = plt.subplots()
#im = ax.imshow(V_SR_2D)
# Set the plot title and axis labels
#ax.set_title("State values (SR-Q-learning)")
#ax.set_xlabel("X axis")
#ax.set_ylabel("Y axis")
#cbar = ax.figure.colorbar(im, ax=ax)

# Set the ticks and tick labels for the x and y axes
#ax.set_xticks(np.arange(V_SR_2D.shape[1]))
#ax.set_yticks(np.arange(V_SR_2D.shape[0]))
#ax.set_xticklabels(["C {}".format(i) for i in range(V_SR_2D.shape[1])])
#ax.set_yticklabels(["R {}".format(i) for i in range(V_SR_2D.shape[0])])

#plt.show()


occupancy_grid = np.zeros([grid_size, grid_size])
for experience in test_experiences:
    occupancy_grid += env.state_to_grid(experience[1])
occupancy_grid = np.sqrt(occupancy_grid)
occupancy_grid = helper_func.mask_grid(occupancy_grid, env.blocks)
plt.imshow(occupancy_grid)
plt.show()



