function [z_t,Z] = gen_obs_opti(max_t, true_coh, w_z)
    % Elapsed time is 0.008001 seconds.
    % Generate observations
    % Inputs:
    % max_t: maximum number of timesteps
    % true_coh: a sequence of signed coherences; size: num_trial x 1
    % w_z: true observation STD; size: 1x1
    
    % Outputs:
    % z_t: observations at each time step; size: time_step x num_trials
    % Z: cumulative sum of z_t ; size: time_step x num_trials
    
    num_trials = length(true_coh);
    
    % Format: time_step x num_trials
    mu = repmat(true_coh', [max_t, 1]);
    sigma = repmat(w_z, [max_t, num_trials]);
    
    % Format: time_step x num_trials
    z_t = normrnd(mu,sigma,[max_t,num_trials]);
    % Compute the cumulative sum of z
    Z = cumsum(z_t);

end


