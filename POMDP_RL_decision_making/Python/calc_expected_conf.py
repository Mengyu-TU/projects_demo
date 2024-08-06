import numpy as np
from scipy.stats import multivariate_normal


def calc_expected_conf(Z, t, std_z, std_i, mu_t, sigma_t_sqr):
    # This function calculates the expected confidence if the agent takes one more
    # observation.
    # Refer to Khalvati et. al. 2021 Python code: pomdpWithCost.py
    # Available at: https://github.com/koosha66/POMDP-Confidence

    # Same direction as the current inferred coherence
    mu1 = Z + mu_t
    mu2 = mu_t

    var2 = std_z ** 2 + sigma_t_sqr
    var1 = t * std_z ** 2 + std_z ** 2 + std_i ** (-2) * std_z ** 4 + var2

    cor = (var2 / var1) ** 0.5

    lower_bound = np.array([-100000, -100000])
    upper_bound = np.array([mu1 / var1 ** 0.5, (Z + mu2) / var2 ** 0.5])

    mu_2d = np.array([0, 0])
    cov = np.array([[1, cor], [cor, 1]])

    # CDF for a bivariate normal distribution
    p1 = multivariate_normal.cdf(upper_bound, mean=mu_2d, cov=cov) - multivariate_normal.cdf(lower_bound, mean=mu_2d,
                                                                                             cov=cov)

    # Opposite direction
    p2 = multivariate_normal.cdf(-lower_bound, mean=-mu_2d, cov=cov) - multivariate_normal.cdf(-upper_bound,
                                                                                               mean=-mu_2d, cov=cov)

    expected_conf = p1 + p2

    return expected_conf
