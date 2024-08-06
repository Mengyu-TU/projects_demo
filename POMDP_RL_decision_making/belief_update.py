import numpy as np


def belief_update(t, std_z, std_i, Z):
    # This function updates the belief distribution: the mean and the
    # standard deviation of it

    # Refer to Khalvati et. al. 2021 Equation 4

    mu_t = (Z * std_z ** (-2)) / (t * std_z ** (-2) + std_i ** (-2))
    sigma_t = np.sqrt(1 / (t * std_z ** (-2) + std_i ** (-2)))

    return mu_t, sigma_t
