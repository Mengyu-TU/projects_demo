import numpy as np


def td_w_belief(belief, Q_prev, reward, learning_rate, dir_trial):
    # This function takes an action Left or Right by picking the side with
    # a higher estimated value. Then, given feedback for this trial, the
    # model updates the values for state-action pairs.

    # Outputs:
    # Q: updated state-action pairs; action: selected action for this
    # trial; 0 means left, 1 means right
    # prediction error: Reward - estimated values of the chosen side
    # predicted_value: estimated values of the side with higher value

    belief_L = belief[0]
    belief_R = belief[1]

    qll, qlr, qrl, qrr = Q_prev.flatten()

    ql = belief_L * qll + belief_R * qrl
    qr = belief_L * qlr + belief_R * qrr

    # Choose an action
    if ql > qr:
        action = 0  # 'left'
    elif ql < qr:
        action = 1  # 'right'
    else:  # ql == qr, randomly pick an action
        action = np.random.choice([0, 1])

    predicted_value = max(ql, qr)

    # Reward
    reward_trial = reward[0] if action == dir_trial else reward[1]

    # Update q-values
    if action == 1:  # right
        prediction_error = reward_trial - qr
        qlr += learning_rate * prediction_error * belief_L
        qrr += learning_rate * prediction_error * belief_R
    else:  # left
        prediction_error = reward_trial - ql
        qll += learning_rate * prediction_error * belief_L
        qrl += learning_rate * prediction_error * belief_R

    Q = np.array([[qll, qlr], [qrl, qrr]])

    return Q, action, reward_trial, prediction_error, predicted_value
