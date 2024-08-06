function [Q, action,reward_trial,prediction_error, predicted_value] = ...
    td_w_belief(belief, Q_prev, reward, learning_rate, dir_trial)
    % This function takes an action Left or Right by picking the side with 
    % a higher estimated value. Then, given feedback for this trial, the
    % model updates the values for state-action pairs.
    
    % Same algorithm as the TD update in sim_POMDP_no_accumulation_RDM.m
    
    % Outputs: 
    % Q: updated state-action pairs; action: selected action for this
    % trial; 0 means left, 1 means right
    % prediction error: Reward - estimated values of the chosen side
    % predicted_value: estimated values of the side with higher value

    
    belief_L = belief(1);
    belief_R = belief(2);

    qll = Q_prev(1,1);
    qlr = Q_prev(1,2);
    qrr = Q_prev(2,2);
    qrl = Q_prev(2,1);

    ql = belief_L * qll + belief_R * qrl;
    qr = belief_L * qlr + belief_R * qrr;

    % Choose an action
    if ql > qr
        action = 0; % 'left';
    elseif ql < qr
        action = 1; % 'right';
    else            % ql == qr, randomly pick an action
        if rand > 0.5
            action = 0;
        else
            action = 1;
        end
    end
    
    predicted_value = max(ql,qr);
    
    % Reward
    if action == dir_trial % correct
        reward_trial = reward(1);
    else
        reward_trial = reward(2);      % incorrect
    end

    % Update q-values
    if action == 1  % right
        prediction_error = reward_trial - qr;
        qlr = qlr + learning_rate * prediction_error * belief_L;
        qrr = qrr + learning_rate * prediction_error * belief_R;
    else            % left
        prediction_error = reward_trial - ql;
        qll = qll + learning_rate * prediction_error * belief_L;
        qrl = qrl + learning_rate * prediction_error * belief_R;
    
    end

    Q = [qll,qlr;qrl,qrr];

end