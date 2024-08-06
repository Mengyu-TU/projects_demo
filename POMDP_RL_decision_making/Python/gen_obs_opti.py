import numpy as np


def gen_obs_opti(max_t, true_coh, w_z):
    # Generate observations
    # Inputs:
    # max_t: maximum number of timesteps
    # true_coh: a sequence of signed coherences; size: num_trial x 1
    # w_z: true observation STD; size: 1x1

    # Outputs:
    # z_t: observations at each time step; size: time_step x num_trials
    # Z: cumulative sum of z_t ; size: time_step x num_trials

    num_trials = len(true_coh)

    # Format: time_step x num_trials
    mu = np.tile(true_coh, (max_t, 1))
    sigma = np.tile(w_z, (max_t, num_trials))

    # Format: time_step x num_trials
    z_t = np.random.normal(mu, sigma, (max_t, num_trials))
    # Compute the cumulative sum of z
    Z = np.cumsum(z_t, axis=0)

    return z_t, Z
