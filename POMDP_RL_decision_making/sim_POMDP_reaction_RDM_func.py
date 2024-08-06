import numpy as np
from multiprocessing import Pool
from functools import partial
from gen_obs_opti import gen_obs_opti
from belief_update import belief_update
from calc_belief_LR import calc_belief_LR
from calc_expected_conf import calc_expected_conf

def sim_POMDP_reaction_RDM_func(std_i, w_z, std_z, learning_rate, reward,
                                termination_thres, termination_factor, conf_thres,
                                trial_num, max_t, iter_num, data):
    # This function simplifies the procedures to run POMDP_reaction_RDM. Just
    # specify the parameters and run

    # Trial stimulus
    stimulus = [-0.512, -0.256, -0.128, -0.064, -0.032, 0,
                0, 0.032, 0.064, 0.128, 0.256, 0.512]  # the 1st 0:left; the 2nd 0: right.

    # Simulate reaction time task
    observe = np.ones((max_t, trial_num, iter_num))

    dir_total = np.zeros((trial_num, iter_num))
    true_coh_total = np.zeros((trial_num, iter_num))
    Z_total = np.zeros((max_t, trial_num, iter_num))

    conf_total = np.zeros((trial_num, iter_num))

    action_total = np.zeros((trial_num, iter_num))
    reward_total = np.zeros((trial_num, iter_num))

    Predicted_val_total = np.zeros((trial_num, iter_num))
    RPE_total = np.zeros((trial_num, iter_num))
    belief_L_total = np.zeros((max_t, trial_num, iter_num))
    Q_total = np.zeros((2, 2, iter_num))

    # Simulation runs on parallel loops for multiple iterations
    with Pool() as pool:
        results = pool.map(partial(run_iteration, std_i=std_i, w_z=w_z, std_z=std_z,
                                   learning_rate=learning_rate, reward=reward,
                                   termination_thres=termination_thres,
                                   termination_factor=termination_factor,
                                   conf_thres=conf_thres, trial_num=trial_num,
                                   max_t=max_t, data=data),
                           range(iter_num))

    # Unpack results
    for i, result in enumerate(results):
        observe[:,:,i] = result['observe']
        dir_total[:,i] = result['dir_trial_all']
        true_coh_total[:,i] = result['true_coh']
        Z_total[:,:,i] = result['Z_all']
        conf_total[:,i] = result['conf']
        action_total[:,i] = result['action']
        reward_total[:,i] = result['reward']
        Predicted_val_total[:,i] = result['predicted_value']
        RPE_total[:,i] = result['RPE']
        Q_total[:,:,i] = result['Q']

    # Save
    simulation = {
        'learning_rate': learning_rate,
        'std_i': std_i,
        'w_z': w_z,
        'std_z': std_z,
        'termination_thres': termination_thres,
        'termination_factor': termination_factor,
        'conf_thres': conf_thres,
        'max_t': max_t,
        'reward': reward,
        'trial_num': trial_num,
        'iter_num': iter_num,
        'true_coh_total': true_coh_total,
        'Z_total': Z_total,
        'dir_total': dir_total,
        'action_total': action_total,
        'reward_total': reward_total,
        'observe': observe,
        'conf_total': conf_total,
        'RPE_total': RPE_total,
        'Predicted_val_total': Predicted_val_total,
        'Q_total': Q_total
    }

    return simulation

def run_iteration(i, std_i, w_z, std_z, learning_rate, reward,
                  termination_thres, termination_factor, conf_thres,
                  trial_num, max_t, data):
    # Initialize state-action pairs
    Q = np.zeros((2, 2))

    observe_temp = np.ones((max_t, trial_num))

    # Generate trials
    true_coh = data['scoh'][:trial_num]
    dir_trial_all = np.logical_xor(data['direction'][:trial_num], 1).astype(bool)

    _, Z_all = gen_obs_opti(max_t, true_coh, w_z)

    prev_RPE = np.pi
    conf = np.zeros(trial_num)
    action = np.zeros(trial_num)
    reward_trial = np.zeros(trial_num)
    RPE = np.zeros(trial_num)
    predicted_value = np.zeros(trial_num)

    for iTrial in range(trial_num):
        belief_ave = np.array([0, 0])

        for t in range(max_t):
            Z = Z_all[t, iTrial]

            # Posterior belief
            mu_t, sigma_t = belief_update(t+1, std_z, std_i, Z)

            # Current confidence
            belief = calc_belief_LR(mu_t, sigma_t)
            belief_ave += belief
            cur_conf = max(belief)

            # Expected confidence for making one more observation
            expected_conf = calc_expected_conf(Z, t+1, std_z, std_i, mu_t, sigma_t**2)

            if prev_RPE == np.pi:  # First trial in an iteration
                running_thres = termination_thres
            elif prev_conf < conf_thres and prev_outcome <= 0:   # Incorrect low
                running_thres = termination_thres
            elif prev_conf > conf_thres and prev_outcome > 0:    # Correct high
                running_thres = termination_thres
            elif prev_conf > conf_thres and prev_outcome <= 0:   # Incorrect high
                running_thres = termination_thres*(1-termination_factor)
            elif prev_conf < conf_thres and prev_outcome > 0:    # Correct low
                running_thres = termination_thres*(1+termination_factor)

            if expected_conf - cur_conf < running_thres or t == max_t - 1:  # stop observation
                observe_temp[t+1:, iTrial] = 0

                # Use average belief for TD
                belief_ave /= (t + 1)

                Q, action[iTrial], reward_trial[iTrial], RPE[iTrial], predicted_value[iTrial] = td_w_belief(
                    belief_ave, Q, reward, learning_rate, dir_trial_all[iTrial])
                conf[iTrial] = belief_ave[int(action[iTrial])]

                prev_outcome = reward_trial[iTrial] > 0
                prev_conf = belief_ave[int(action[iTrial])]
                prev_RPE = RPE[iTrial]
                break

    return {
        'observe': observe_temp,
        'dir_trial_all': dir_trial_all,
        'true_coh': true_coh,
        'Z_all': Z_all,
        'conf': conf,
        'action': action,
        'reward': reward_trial,
        'predicted_value': predicted_value,
        'RPE': RPE,
        'Q': Q
    }
