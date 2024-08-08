import numpy as np
import matplotlib.pyplot as plt
import helper_func
from grid_world import SimpleGrid
from SARSA_successor_class import TabularSuccessorAgent


# Set up environment
grid_size = 7
pattern = "four_rooms_left_bridge" #"four_rooms"
env = SimpleGrid(grid_size, block_pattern=pattern, obs_mode='index')  # index mode: directly return 1d agent pos
env.reset(agent_pos=[6, 6], goal_pos=[0, grid_size - 1])
plt.imshow(env.grid)  # env.grid is a grid_size * grid_size * 3 np array
# layer 1 is the agent position
# layer 2 is the target position
# layer 3 is the wall positions
plt.show()

# Set up the agent
train_episode_length = 1000
test_episode_length = 1000
episodes = 250
gamma = 0.95
lr = 0.15

train_epsilon = 1
test_epsilon = 0.1
exploration_rate = 0.1

agent = TabularSuccessorAgent(env.state_size, env.action_size, lr, gamma, exploration_rate)
experience = []
test_experience = []
test_lengths = []
lifetime_td_errors = []

for episode in range(episodes):
    # training: need to update SR rep
    # agent_start = [0, 0]
    if episode == 150:#episodes // 2:  # < 1000 trials
        env = SimpleGrid(grid_size, block_pattern="four_rooms_right_bridge", obs_mode='index')
        #goal_pos = [0, grid_size - 1]
    #else:  # > 1000 trials
        #goal_pos = [grid_size - 1, grid_size - 1]
    # initialize environment
    agent_start = [6, 6]
    goal_pos = [0, 0]
    #if i < episodes // 2:  # < 1000 trials
        #goal_pos = [0, grid_size - 1]
    #else:  # > 1000 trials
        #goal_pos = [grid_size - 1, grid_size - 1]
    # initialize environment
    env.reset(agent_pos=[6, 6], goal_pos=[0, 0])
    #env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    episodic_error = []
    for j in range(train_episode_length):
        action = agent.sample_action(state, epsilon=train_epsilon)
        reward = env.step(action)
        state_next = env.observation
        done = env.done  # reach the goal pos or not
        experience.append([state, action, state_next, reward, done])
        state = state_next
        if (j > 1):
            td_sr = agent.SARSA_update_sr(experience[-2], experience[-1])
            td_w = agent.update_w(experience[-1])
            episodic_error.append(np.mean(np.abs(td_sr)))
        if env.done:
            td_sr = agent.SARSA_update_sr(experience[-1], experience[-1])
            episodic_error.append(np.mean(np.abs(td_sr)))
            break
    lifetime_td_errors.append(np.mean(episodic_error))

    # Test phase
    env.reset(agent_pos=agent_start, goal_pos=goal_pos)
    state = env.observation
    for j in range(test_episode_length):
        action = agent.sample_action(state, epsilon=test_epsilon)
        reward = env.step(action)
        state_next = env.observation
        test_experience.append([state, action, state_next, reward])
        state = state_next
        if env.done:
            break
    test_lengths.append(j)

    if episode % 50 == 0:
        print('\rEpisode {}/{}, TD Error: {}, Test Lengths: {}'
              .format(episode, episodes, np.mean(lifetime_td_errors[-50:]),
                      np.mean(test_lengths[-50:])), end='')


fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(2,2,1)
ax.plot(lifetime_td_errors)
ax.set_title('TD Error')
ax = fig.add_subplot(2,2,2)
ax.plot(test_lengths)
ax.set_title("Episodes lengths")
plt.show()




