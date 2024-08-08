import numpy as np
import matplotlib.pyplot as plt
import helper_func
from grid_world import SimpleGrid
from dynamic_programing import DynamicPrograming
import matplotlib.pyplot as plt

# Set up environment
grid_size = 7
pattern = "four_rooms_right_bridge"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode='index')  # index mode: directly return 1d agent pos
env.reset(agent_pos=[4, 2], goal_pos=[0, grid_size - 1]) #grid_size - 1
plt.imshow(env.grid)  # env.grid is a grid_size * grid_size * 3 np array
# layer 1 is the agent position
# layer 2 is the target position
# layer 3 is the wall positions
plt.show()

gamma = 0.95
lr = 5e-1

# value iteration
dp_agent = DynamicPrograming(env.state_size, env.action_size, gamma)
theta = 0.1
V, pi = dp_agent.value_iteration(env, theta)
V_2D = np.reshape(V, (7,7))


# plot the heat map
fig, ax = plt.subplots()
im = ax.imshow(V_2D)
# Set the plot title and axis labels
ax.set_title("Optimal state values")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
cbar = ax.figure.colorbar(im, ax=ax)

# Set the ticks and tick labels for the x and y axes
ax.set_xticks(np.arange(V_2D.shape[1]))
ax.set_yticks(np.arange(V_2D.shape[0]))
ax.set_xticklabels(["C {}".format(i) for i in range(V_2D.shape[1])])
ax.set_yticklabels(["R {}".format(i) for i in range(V_2D.shape[0])])

plt.show()


#
import numpy as np
import matplotlib.pyplot as plt
episode_steps = np.zeros([250,1])
episode_steps[0:150] = 16
episode_steps[150] = 18
episode_steps[151:250] = 8
fig, ax = plt.subplots()
ax.plot(episode_steps)
ax.set_xlabel('Episodes', fontsize = 16)
ax.set_ylabel('Num steps per episode',fontsize = 16)
ax.tick_params(labelsize=16)
#plt.plot(episode_steps)
plt.show()