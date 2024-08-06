% This version and parameter produces the behavior and RT graphs in this
% folder. The difference from previous models is that the scaling of
% evidence is by constants depending previous trial confidcence and
% outcome, instead of a linear function of RPE.

% clc
% clear
load('hanzo_4_targ_RDM_beh_screened_all.mat')
%% Parameters
% Model parameters: initial belief b_0 = N(0, std_i)
std_i = 0.46;
% True std
w_z = 0.52;
% learned std:
std_z = 2.5;
 
% learning rate
learning_rate = 0.15;

% reward
reward = [1,-1]; %[2,-2]; %[100,-400]; %[20, -400]; % [20,-20];  % Correct, incorrect
 
% cost of observation
% cost = 1;
termination_thres = 0.0005;
termination_factor = 0.5;
 
% conf_thres for low and high bet
conf_thres = 0.58;
 
% Initialize task values
trial_num = 2200; % 2000; 4000 too big;
max_t = 100;
iter_num = 100;   % Either 50 or 100 will be enough
 
%% Trial stimulus
stimulus = [-0.512 -0.256 -0.128 -0.064 -0.032 0 ...
    0 0.032 0.064 0.128 0.256 0.512]; % the 1st 0:left; the 2nd 0: right.
% Real data num trial with '0' coh has 2 times more trials than other coh levels
cohs = unique(stimulus);
 
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
    Q = zeros(2);
    
    observe_temp = ones(max_t,trial_num);
    
    % Generate trials
    idx = unidrnd(length(stimulus),trial_num,1);   % unidrnd(max_n,sz): Random numbers from discrete uni distri
    true_coh = (stimulus(idx))';
    true_coh = data.scoh(1:trial_num);

    dir_trial_all = idx > 6;  % 1: right is correct; 0: left correct
    dir_trial_all =  logical(logical(data.direction(1:trial_num))-1);

    true_coh_total(:,iIter) = true_coh;
    dir_total(:,iIter) = dir_trial_all;
    
    [z_t_all,Z_all] = gen_obs_opti(max_t, true_coh, w_z);
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
            belief_L_total(t,iTrial,iIter) = belief(1);
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
 
% save('simulation_20240131_02','simulation');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% Plot learning curve
mean_reward = mean(reward_total,2);
figure;plot(mean_reward);xlabel('Episode');ylabel('Average reward')
set(gca,'FontSize',25);
xlim([-100,5000])
 
%% Psychometric curve: use the second half of episodes for each iteration because learning only converges in later episodes.
start_idx = ceil(trial_num/3);
 
plot_coh = reshape(true_coh_total(start_idx:end,:),[],1);
 
plot_reward = reshape(reward_total(start_idx:end,:) > 0,[],1);
 
plot_action = reshape(action_total(start_idx:end,:),[],1);
 
 
perc_right = zeros(length(cohs),1);
n = zeros(length(cohs),1);
 
for i = 1:length(cohs)
    n(i) = sum(plot_coh == cohs(i));
    perc_right(i) = sum(plot_action(plot_coh == cohs(i)))/n(i);
    
end
 
perc_right_SE = sqrt(perc_right.* (1 - perc_right)./n);
 
% fit logistic regression
[B, ~, stats] = glmfit(plot_coh, plot_action, 'binomial');
x = linspace(cohs(1),cohs(end),100);
y_hat = glmval(B, x,'logit');
 
 
%% psychometric curve: conditioned on previous right correct trial
idx = find(plot_action == 1 & plot_reward > 0) + 1;
idx = idx(2:end-1);
plot_coh_c = plot_coh(idx);
plot_action_c = plot_action(idx);
perc_right_c = zeros(length(cohs),1);
n_c = zeros(length(cohs),1);
 
for i = 1:length(cohs)
    n_c(i) = sum(plot_coh_c == cohs(i));
    perc_right_c(i) = sum(plot_action_c(plot_coh_c == cohs(i)))/n_c(i);
end
 
perc_right_SE_c = sqrt(perc_right_c.* (1 - perc_right_c)./n_c);
 
% fit logistic regression
[B_c, ~, stats_c] = glmfit(plot_coh_c, plot_action_c, 'binomial');
x = linspace(cohs(1),cohs(end),100);
y_hat_c = glmval(B_c, x,'logit');
 
%% Plot psychometric and conditioned psychometric curve
figure;hold on;
 
errorbar(cohs*100, perc_right, perc_right_SE, '.','MarkerSize',14);
errorbar(cohs*100, perc_right_c, perc_right_SE_c, '.','MarkerSize',14);
% plot(cohs,perc_right_c,'.r--')
% plot(cohs,perc_right,'ob');
plot(x*100,y_hat,'b','Linewidth',1.)
plot(x*100,y_hat_c,'r--','Linewidth',1.5)
legend('','','All trials','Previous R, rewarded','FontSize',18,'location','NW')
xlabel('Motion strength (Coh %)'); ylabel('Proportion rightward choices');
set(gca,'FontSize',22);
legend('boxoff')
set(gca, 'XTick',[-50:25:50])
xlim([-52,52])
 
%% Plot reaction time
observe_temp = reshape(observe(:,start_idx:end,:),size(observe,1),[],1);
 
RT = sum(observe_temp,1);
meanRT = zeros(length(cohs),1);
seRT = zeros(length(cohs),1);
for iCoh = 1:length(cohs)
    meanRT(iCoh) = mean(RT(plot_coh == cohs(iCoh)));
    seRT(iCoh) = std(RT(plot_coh == cohs(iCoh)))/sqrt(sum(plot_coh == cohs(iCoh)));
end
 
% figure;errorbar(cohs*100,meanRT,seRT,'bo--','LineWidth',2)
% xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% % set(gca, 'XTick',[-50:25:50])
% box off;
% legend('All trials','location','NW')
% 
% set(gca,'FontSize',22) %'FontWeight','bold',
% legend('boxoff')
 
% % figure;histogram(RT)
 
%% Plot reaction time conditioned on previous correct and error trials
observe_temp = reshape(observe(:,start_idx:end,:),size(observe,1),[],1);
 
% Correct
idx_corr = find(plot_reward > 0) + 1; % plot_conf < conf_thres
idx_corr = idx_corr(2:end-1);
observe_correct = observe_temp(:,idx_corr);
 
RT_correct = sum(observe_correct,1);
meanRT_correct = zeros(length(cohs),1);
seRT_correct = zeros(length(cohs),1);
for iCoh = 1:length(cohs)
    meanRT_correct(iCoh) = mean(RT_correct(plot_coh(idx_corr) == cohs(iCoh)));
    seRT_correct(iCoh) = std(RT_correct(plot_coh(idx_corr) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_corr) == cohs(iCoh)));
end
 
% figure;
% errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',2);hold on;
% errorbar(cohs*100,meanRT_correct,seRT_correct,'ro--','LineWidth',1);
% xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% % set(gca, 'XTick',[-50:25:50])
% box off;
% legend('All trials','Post-correct trials','location','NW')
%  
% set(gca,'FontSize',22) %'FontWeight','bold',
% legend('boxoff')
 
% Error
idx_incorr = find(plot_reward <= 0) + 1;
idx_incorr = idx_incorr(2:end-1);
observe_incorrect = observe_temp(:,idx_incorr);
 
RT_incorrect = sum(observe_incorrect,1);
meanRT_incorrect = zeros(length(cohs),1);
seRT_incorrect = zeros(length(cohs),1);
for iCoh = 1:length(cohs)
    meanRT_incorrect(iCoh) = mean(RT_incorrect(plot_coh(idx_incorr) == cohs(iCoh)));
    seRT_incorrect(iCoh) = std(RT_incorrect(plot_coh(idx_incorr) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_incorr) == cohs(iCoh)));
end
 
figure;
errorbar(cohs*100,meanRT,seRT,'ko-','LineWidth',1);hold on;
errorbar(cohs*100,meanRT_incorrect,seRT_incorrect,'bo--','LineWidth',2);
errorbar(cohs*100,meanRT_correct,seRT_correct,'ro--','LineWidth',2);
xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% set(gca, 'XTick',[-50:25:50])
box off;
legend('All trials','Post-error trials','Post-correct trials','location','NW')
 
set(gca,'FontSize',22) %'FontWeight','bold',
legend('boxoff')
 
%% Plot reaction time conditioned on post-low/high bet and previous correct/error trials
plot_conf = reshape(conf_total(start_idx:end,:),[],1);
 
% Correct and low bet
idx_corr_low = find(plot_reward > 0 & plot_conf < conf_thres) + 1;
idx_corr_low = idx_corr_low(2:end-1);
observe_correct_low = observe_temp(:,idx_corr_low);
 
RT_correct_low = sum(observe_correct_low,1);
meanRT_correct_low = zeros(length(cohs),1);
seRT_correct_low = zeros(length(cohs),1);
for iCoh = 1:length(cohs)
    meanRT_correct_low(iCoh) = mean(RT_correct_low(plot_coh(idx_corr_low) == cohs(iCoh)));
    seRT_correct_low(iCoh) = std(RT_correct_low(plot_coh(idx_corr_low) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_corr_low) == cohs(iCoh)));
end
 
% figure;hold on;
% errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',2)
% errorbar(cohs*100,meanRT_correct_low,seRT_correct_low,'ro--','LineWidth',1);
% xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% % set(gca, 'XTick',[-50:25:50])
% box off;
% legend('All trials','Post-correct low-conf','location','NW')
% ylim([0 45])
% set(gca,'FontSize',22) %'FontWeight','bold',
% legend('boxoff')
 
 
% Correct and high bet
idx_corr_high = find(plot_reward > 0 & plot_conf > conf_thres) + 1;
idx_corr_high = idx_corr_high(2:end-1);
observe_correct_high = observe_temp(:,idx_corr_high);
 
RT_correct_high = sum(observe_correct_high,1);
meanRT_correct_high = zeros(length(cohs),1);
seRT_correct_high = zeros(length(cohs),1);
for iCoh = 1:length(cohs)
    meanRT_correct_high(iCoh) = mean(RT_correct_high(plot_coh(idx_corr_high) == cohs(iCoh)));
    seRT_correct_high(iCoh) = std(RT_correct_high(plot_coh(idx_corr_high) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_corr_high) == cohs(iCoh)));
end
 
figure;hold on;
errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',1)
errorbar(cohs*100,meanRT_correct_high,seRT_correct_high,'ro--','LineWidth',1);
errorbar(cohs*100,meanRT_correct_low,seRT_correct_low,'ko--','LineWidth',1);
xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% set(gca, 'XTick',[-50:25:50])
box off;
legend('All trials','Post-correct high-conf','Post-correct low-conf','location','NW')
% ylim([0 45])
set(gca,'FontSize',22) %'FontWeight','bold',
legend('boxoff')
 
% Error and low bet
idx_incorr_low = find(plot_reward <= 0 & plot_conf < conf_thres) + 1;
idx_incorr_low = idx_incorr_low(2:end-1);
observe_incorrect_low = observe_temp(:,idx_incorr_low);
 
RT_incorrect_low = sum(observe_incorrect_low,1);
meanRT_incorrect_low = zeros(length(cohs),1);
seRT_incorrect_low = zeros(length(cohs),1);
 
for iCoh = 1:length(cohs)
    meanRT_incorrect_low(iCoh) = mean(RT_incorrect_low(plot_coh(idx_incorr_low) == cohs(iCoh)));
    seRT_incorrect_low(iCoh) = std(RT_incorrect_low(plot_coh(idx_incorr_low) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_incorr_low) == cohs(iCoh)));
end
 
% figure; hold on;
% errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',2)
% errorbar(cohs*100,meanRT_incorrect_low,seRT_incorrect_low,'ko--','LineWidth',1);
% errorbar(cohs*100,meanRT_incorrect_high,seRT_incorrect_high,'ro--','LineWidth',1);
% xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% % set(gca, 'XTick',[-50:25:50])
% box off;
% legend('All trials','Post-error low-conf','Post-error high-conf','location','NW')
% % ylim([0 45])
% set(gca,'FontSize',22) %'FontWeight','bold',
% legend('boxoff')
 
 
% Error and high bet
idx_incorr_high = find(plot_reward <= 0 & plot_conf > conf_thres) + 1;
idx_incorr_high = idx_incorr_high(2:end-1);
observe_incorrect_high = observe_temp(:,idx_incorr_high);
 
RT_incorrect_high = sum(observe_incorrect_high,1);
meanRT_incorrect_high = zeros(length(cohs),1);
seRT_incorrect_high = zeros(length(cohs),1);
 
for iCoh = 1:length(cohs)
    meanRT_incorrect_high(iCoh) = mean(RT_incorrect_high(plot_coh(idx_incorr_high) == cohs(iCoh)));
    seRT_incorrect_high(iCoh) = std(RT_incorrect_high(plot_coh(idx_incorr_high) == cohs(iCoh)))/sqrt(sum(plot_coh(idx_incorr_high) == cohs(iCoh)));
end
 
% figure; hold on;
% errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',2)
% errorbar(cohs*100,meanRT_incorrect_high,seRT_incorrect_high,'ko--','LineWidth',1);
% xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% % set(gca, 'XTick',[-50:25:50])
% box off;
% legend('All trials','Post-error high-conf','location','NW')
% ylim([0 45])
% set(gca,'FontSize',22) %'FontWeight','bold',
% legend('boxoff')
 
figure; hold on;
errorbar(cohs*100,meanRT,seRT,'bo-','LineWidth',1)
errorbar(cohs*100,meanRT_incorrect_low,seRT_incorrect_low,'ko--','LineWidth',1);
errorbar(cohs*100,meanRT_incorrect_high,seRT_incorrect_high,'ro--','LineWidth',1);
xlabel('Motion strength (Coh %)'); ylabel('Reaction time (Time step)');
% set(gca, 'XTick',[-50:25:50])
box off;
legend('All trials','Post-error low-conf','Post-error high-conf','location','NW')
% ylim([0 45])
set(gca,'FontSize',22) %'FontWeight','bold',
legend('boxoff')

%% Plot distribution of belief
observe2 = observe(:,start_idx:end,:);
plot_observe = reshape(observe2,max_t,[],1);
plot_belief = reshape(belief_L_total(:,start_idx:end,:),max_t,[],1);
 
for iTrial = 1:length(plot_observe(1,:))
    if isempty(strfind((plot_observe(:,iTrial))',[1 0]))  %#ok<STREMP>
        idx_conf(iTrial) = max_t;
    else
        idx_conf(iTrial) = strfind((plot_observe(:,iTrial))',[1 0]);
    end
    
    bel_ave(iTrial) = mean(plot_belief(1:idx_conf(iTrial),iTrial));
    bel(iTrial) = plot_belief(idx_conf(iTrial),iTrial);
end
 
% At last time point
figure;histogram(bel); xlabel('Belief(L) at the last timestep'); ylabel('Number of trials');
set(gca,'FontSize',22);box off;
 
% average belief
figure;histogram(bel_ave);xlabel('Belief(L) averaged within a trial'); ylabel('Number of trials');
set(gca,'FontSize',22);box off;
 
%% Plot confidence
plot_conf = reshape(conf_total(start_idx:end,:),[],1);
 
for i = 1:length(cohs)
    idx = find(plot_coh == cohs(i));
    idx_corr = find(plot_coh == cohs(i) & plot_reward > 0);
    idx_incorr = find(plot_coh == cohs(i) & plot_reward == 0);
    
    mean_conf(i) = mean(plot_conf(idx));
    mean_conf_corr(i) = mean(plot_conf(idx_corr));
    
    if length(idx_incorr) > 5
        mean_conf_incorr(i) = mean(plot_conf(idx_incorr));
    else % Not enough error trials, ignore
        mean_conf_incorr(i) = NaN;
    end
    
    std_conf(i) = std(plot_conf(idx))/sqrt(length(idx));
    std_conf_corr(i) = std(plot_conf(idx_corr))/sqrt(length(idx_corr));
    std_conf_incorr(i) = std(plot_conf(idx_incorr))/sqrt(length(idx_incorr));
end
 
figure;errorbar(cohs*100,mean_conf_corr,std_conf_corr,'r.-','MarkerSize',14);
hold on;errorbar(cohs*100,mean_conf_incorr,std_conf_incorr,'b.-','MarkerSize',14);
legend('Correct trials','Inorrect trials')
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
xlabel('Motion strength (Coh %)'); ylabel('Confidence');set(gca,'FontSize',22);
legend('boxoff'); box off;
 
 
 
figure;errorbar(cohs*100,mean_conf,std_conf,'k.-','MarkerSize',14);
xlabel('Motion strength (Coh %)'); ylabel('Confidence all trials');
legend('All trials')
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
set(gca,'FontSize',22);legend('boxoff');box off;
 
 
%% Reward prediction error
plot_RPE = reshape(RPE_total(start_idx:end,:),[],1);
 
for i = 1:length(cohs)
    idx_corr = find(plot_coh == cohs(i) & plot_reward > 0);
    idx_incorr = find(plot_coh == cohs(i) & plot_reward == 0);
    mean_RPE_corr(i) = mean(plot_RPE(idx_corr));
    
    if length(idx_incorr) > 5
        mean_RPE_incorr(i) = mean(plot_RPE(idx_incorr));
    else % Not enough error trials, ignore
        mean_RPE_incorr(i) = NaN;
    end
    
    std_RPE_corr(i) = std(plot_RPE(idx_corr))/sqrt(length(idx_corr));
    std_RPE_incorr(i) = std(plot_RPE(idx_incorr))/sqrt(length(idx_incorr));
end
 
figure;errorbar(cohs*100,mean_RPE_corr,std_RPE_corr,'r.-','MarkerSize',14);
xlabel('Motion strength (Coh %)'); ylabel('Reward prediction error');
legend('Correct trials')
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
box off;
set(gca,'FontSize',22);legend('boxoff')
 
 
figure;errorbar(cohs*100,mean_RPE_incorr,std_RPE_incorr,'k.-','MarkerSize',14);
legend('Inorrect trials')
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
xlabel('Motion strength (Coh %)'); ylabel('Reward prediction error');set(gca,'FontSize',22);
legend('boxoff');
box off;
 
%% RPE conditioned on low bet and high bet
% Correct trials, Low and High confidence
for i = 1:length(cohs)
    idx_corr_H = find(plot_coh == cohs(i) & plot_reward > 0 & plot_conf > conf_thres);
    idx_corr_L = find(plot_coh == cohs(i) & plot_reward > 0 & plot_conf < conf_thres);
    mean_RPE_corr_H(i) = mean(plot_RPE(idx_corr_H));
    mean_RPE_corr_L(i) = mean(plot_RPE(idx_corr_L));
    std_RPE_corr_H(i) = std(plot_RPE(idx_corr_H))/sqrt(length(idx_corr_H));
    std_RPE_corr_L(i) = std(plot_RPE(idx_corr_L))/sqrt(length(idx_corr_L));
end
 
figure;errorbar(cohs*100,mean_RPE_corr_H,std_RPE_corr_H,'b.-','MarkerSize',14,'LineWidth',1.5);
xlabel('Motion strength (Coh %)'); ylabel('Reward prediction error');
hold on; errorbar(cohs*100,mean_RPE_corr_L,std_RPE_corr_L,'.--','MarkerSize',14,'Color',[0, 0.4470, 0.7410],'LineWidth',1.5);
xlabel('Motion strength (Coh %)'); ylabel('Reward prediction error');
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
legend('Correct trials, high conf','Correct trials, low conf')
set(gca,'FontSize',22);
legend('boxoff')
box off;
 
% Incorrect trials, Low and High confidence
for i = 1:length(cohs)
    idx_incorr_H = find(plot_coh == cohs(i) & plot_reward == 0 & plot_conf > conf_thres);
    idx_incorr_L = find(plot_coh == cohs(i) & plot_reward == 0 & plot_conf < conf_thres);
    
    if length(idx_incorr_H) > 5
        mean_RPE_incorr_H(i) = mean(plot_RPE(idx_incorr_H));
    else % Not enough error trials, ignore
        mean_RPE_incorr_H(i) = NaN;
    end
    
    if length(idx_incorr_L) > 5
        mean_RPE_incorr_L(i) = mean(plot_RPE(idx_incorr_L));
    else % Not enough error trials, ignore
        mean_RPE_incorr_L(i) = NaN;
    end
    
    std_RPE_incorr_H(i) = std(plot_RPE(idx_incorr_H))/sqrt(length(idx_incorr_H));
    std_RPE_incorr_L(i) = std(plot_RPE(idx_incorr_L))/sqrt(length(idx_incorr_L));
end
 
figure;errorbar(cohs*100,mean_RPE_incorr_H,std_RPE_incorr_H,'k.-','MarkerSize',14,'LineWidth',1.5);
xlabel('Motion strength (Coh %)'); ylabel('Reward prediction error');
hold on; errorbar(cohs*100,mean_RPE_incorr_L,std_RPE_incorr_L,'.--','MarkerSize',14,'Color',[0.5, 0.5, 0.5],'LineWidth',1.5);
xlabel('Motion strength (Coh *100)'); ylabel('Reward prediction error');
xlim([-55, 55]);set(gca,'XTick',-50:25:50)
legend('Incorrect trials, high conf','incorrect trials, low conf')
set(gca,'FontSize',22);
legend('boxoff')
box off;
 
%% Psych curves: Conditioned on low bet and high bet, rewarded trial
rightward = 1;
% High confidence
idx_high = find(plot_action == rightward & plot_reward > 0 & plot_conf > conf_thres) + 1;
idx_high = idx_high(2:end-1);
plot_coh_c_high = plot_coh(idx_high);
plot_action_c_high = plot_action(idx_high);
perc_right_c_high = zeros(length(cohs),1);
n_c_high = zeros(length(cohs),1);
 
for i = 1:length(cohs)
    n_c_high(i) = sum(plot_coh_c_high == cohs(i));
    perc_right_c_high(i) = sum(plot_action_c_high(plot_coh_c_high == cohs(i)))/n_c_high(i);
end
 
perc_right_SE_c_high = sqrt(perc_right_c_high.* (1 - perc_right_c_high)./n_c_high);
 
% fit logistic regression
[B_c_high, ~, stats_c_high] = glmfit(plot_coh_c_high, plot_action_c_high, 'binomial');
x = linspace(cohs(1),cohs(end),100);
y_hat_c_high = glmval(B_c_high, x,'logit');
 
% Low confidence
idx_low = find(plot_action == rightward & plot_reward > 0 & plot_conf < conf_thres) + 1;
idx_low = idx_low(2:end-1);
plot_coh_c_low = plot_coh(idx_low);
plot_action_c_low = plot_action(idx_low);
perc_right_c_low = zeros(length(cohs),1);
n_c_low = zeros(length(cohs),1);
 
for i = 1:length(cohs)
    n_c_low(i) = sum(plot_coh_c_low == cohs(i));
    perc_right_c_low(i) = sum(plot_action_c_low(plot_coh_c_low == cohs(i)))/n_c_low(i);
end
 
perc_right_SE_c_low = sqrt(perc_right_c_low.* (1 - perc_right_c_low)./n_c_low);
 
% fit logistic regression
[B_c_low, ~, stats_c_low] = glmfit(plot_coh_c_low, plot_action_c_low, 'binomial');
x = linspace(cohs(1),cohs(end),100);
y_hat_c_low = glmval(B_c_low, x,'logit');
 
figure;hold on;
 
errorbar(cohs*100, perc_right, perc_right_SE, 'b.','MarkerSize',14);
errorbar(cohs*100, perc_right_c_high, perc_right_SE_c_high, 'r.','MarkerSize',14);
errorbar(cohs*100, perc_right_c_low, perc_right_SE_c_low, 'k.','MarkerSize',14);
 
plot(x*100,y_hat,'b','Linewidth',1.)
plot(x*100,y_hat_c_high,'r--','Linewidth',1.5)
plot(x*100,y_hat_c_low,'k--','Linewidth',1.5)
legend('','','','All trials','Prev R, rewarded, high bet',...
    'Prev R, rewarded, low bet','FontSize',18,'location','NW')
xlabel('Motion strength (Coh %)'); ylabel('Proportion rightward choices');
set(gca,'FontSize',22);
legend('boxoff')
set(gca,'XTick',-50:25:50)

