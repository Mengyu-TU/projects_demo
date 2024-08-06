function simulation = sim_POMDP_reaction_RDM_func(std_i, w_z, std_z, learning_rate, reward, ...
    termination_thres, termination_factor, conf_thres, trial_num, max_t, iter_num, data)
% This function simplies the procedures to run POMDP_reaction_RDM. Just
% specify the parameters and run

%% Trial stimulus
stimulus = [-0.512 -0.256 -0.128 -0.064 -0.032 0 ...
    0 0.032 0.064 0.128 0.256 0.512]; % the 1st 0:left; the 2nd 0: right.
% Real data num trial with '0' coh has 2 times more trials than other coh levels

%% Simulate reaction time task
observe = ones(max_t,trial_num,iter_num);

dir_total = zeros(trial_num,iter_num);
true_coh_total = zeros(trial_num,iter_num);
Z_total = zeros(max_t, trial_num, iter_num);

conf_total = zeros(trial_num,iter_num);

action_total = zeros(trial_num,iter_num);
reward_total = zeros(trial_num,iter_num);

Predicted_val_total = zeros(trial_num,iter_num);
RPE_total = zeros(trial_num,iter_num);
belief_L_total = zeros(max_t, trial_num, iter_num);
Q_total = zeros(2,2,iter_num);

%% Simulation runs on parallel loops for multiple iteration
parfor iIter = 1:iter_num
    % Initialize state-action pairs
    Q = [0, 0; 0, 0] % zeros(2);

    observe_temp = ones(max_t,trial_num);

    % Generate trials
    idx = unidrnd(length(stimulus),trial_num,1);   % unidrnd(max_n,sz): Random numbers from discrete uni distri
    % true_coh = (stimulus(idx))';
    true_coh = data.scoh(1:trial_num);

    % dir_trial_all = idx > 6;  % 1: right is correct; 0: left correct
    dir_trial_all =  logical(logical(data.direction(1:trial_num))-1);

    true_coh_total(:,iIter) = true_coh;
    dir_total(:,iIter) = dir_trial_all;

    [~,Z_all] = gen_obs_opti(max_t, true_coh, w_z);
    Z_total(:,:,iIter) = Z_all;

    prev_RPE = pi;
    for iTrial = 1:trial_num

        belief_ave = [0 0];

        for t = 1:max_t
            Z = Z_all(t,iTrial);

            % Posterior belief
            [mu_t, sigma_t] = belief_update(t,std_z,std_i, Z);

            % Current confidence
            belief = calc_belief_LR(mu_t, sigma_t);
            belief_ave = belief + belief_ave;
            % belief_L_total(t,iTrial,iIter) = belief(1);
            cur_conf = max(belief);

            % Expected confidence for making one more observation
            expected_conf = calc_expected_conf(Z,t,std_z,std_i,mu_t,sigma_t);

            if prev_RPE == pi  % First trial in an iteration
                running_thres = termination_thres;
            elseif prev_conf < conf_thres && prev_outcome <= 0   % Incorrect low
                running_thres = termination_thres
            elseif prev_conf > conf_thres && prev_outcome > 0    % Correct high
                running_thres = termination_thres;
            elseif prev_conf > conf_thres && prev_outcome <= 0   % Incorrect high
                running_thres = termination_thres*(1-termination_factor);
            elseif prev_conf < conf_thres && prev_outcome > 0    % Correct low
                running_thres = termination_thres*(1+termination_factor);
            end

            if expected_conf - cur_conf < running_thres || t == max_t % stop observation
                observe_temp(t+1:end,iTrial) = 0;

                % Use average belief for TD
                belief_ave = belief_ave/t;

                [Q, action,reward_trial,prediction_error, predicted_value] = td_w_belief(belief_ave, Q, reward, ...
                    learning_rate, dir_trial_all(iTrial));
                conf_total(iTrial,iIter) = belief_ave(action + 1);

                reward_total(iTrial,iIter) = reward_trial;
                action_total(iTrial,iIter) = action;
                RPE_total(iTrial,iIter) = prediction_error;
                Predicted_val_total(iTrial,iIter) = predicted_value;
                Q_total(:,:,iIter) = Q;

                prev_outcome = reward_trial > 0;
                prev_conf = belief_ave(action + 1);
                prev_RPE = prediction_error;
                break
            end

        end % end current trial

    end % end all trials
    observe(:,:,iIter) = observe_temp;
end  % end all iterations

%% Save
simulation = struct('learning_rate',{learning_rate},'std_i',...
    {std_i},'w_z',{w_z},'std_z',{std_z},'termination_thres',{termination_thres},'termination_factor',{termination_factor},...
    'conf_thres',{conf_thres},...
    'max_t',{max_t},'reward',{reward},'trial_num',{trial_num},'iter_num',{iter_num},...
    'true_coh_total',{true_coh_total},'Z_total',{Z_total},...
    'dir_total',{dir_total},'action_total',{action_total},...
    'reward_total',{reward_total},'observe',{observe},'conf_total',{conf_total},...
    'RPE_total',{RPE_total}, 'Predicted_val_total', {Predicted_val_total}, 'Q_total', {Q_total});
end