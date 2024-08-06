function expected_conf = calc_expected_conf(Z,t,std_z,std_i,mu_t,sigma_t_sqr)
    % This function calculates the expected confidence if the agent takes one more
    % observation.
    % Refer to Khalvati et. al. 2021 Python code: pomdpWithCost.py
    % Available at: https://github.com/koosha66/POMDP-Confidence
    
    %% same direction as the current inferred coherence
    mu1 = Z + mu_t;
    mu2 = mu_t;
    
    var2 = std_z^2 + sigma_t_sqr;
    var1 = t*std_z^2 + std_z^2 + std_i^(-2) * std_z^4 + var2;
    
    cor = (var2/var1)^0.5;
    
    lower_bound = [-100000, -100000];
    upper_bound = [mu1/var1^0.5, (Z + mu2)/var2^0.5];
    
    mu_2d = [0,0];
    cov = [1,cor;cor 1];
    
    % CDF for a bivariate normal distribution 
    p1 = mvncdf(lower_bound,upper_bound,mu_2d,cov);
    
    %% opposite direction 
    p2 = mvncdf(lower_bound,-upper_bound,-mu_2d,cov);
    
    expected_conf = p1 + p2;

end