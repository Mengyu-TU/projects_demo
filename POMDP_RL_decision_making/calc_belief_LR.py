from scipy.stats import norm

def calc_belief_LR(mu_t, sigma_t):
    # This function calculates the beliefs for the true state being on left and
    # right respectively.

    bL = norm.cdf(0, mu_t, sigma_t)  # Left
    bR = 1 - bL                      # Right
    belief = [bL, bR]
    return belief
