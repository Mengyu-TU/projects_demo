% This script plots the b0 bias term for low and high confidence across all
% coherence levels.
% Require to run sim_POMDP_no_accumulation_RDM_MY.m or
% sim_POMDP_reaction_RDM_multi_iter_MY.m and
% plot_POMDP_reaction_RDM_multi_iter.m

rightward = 1;
%% Plot conditioned on trial difficulty rightward, high confidence
abs_cohs = unique(abs(cohs));

stats_all_high = struct('coh',[],'all_trial',[]); % coherence refers to the coh in the previous trial
stats_all_high.all_trial = stats;

fn = fieldnames(stats_all_high);

for iCoh = 1 % :length(abs_cohs)
    % select trial whose previous trial is rewarded, rightward, and have a
    % specific coherence level
    idx = find(plot_action == rightward & plot_reward > 0 & plot_conf > conf_thres) + 1;
    idx = idx(2:end-1);
    plot_coh_high_c = plot_coh(idx);
    plot_action_high_c = plot_action(idx);
    perc_right_high_c = zeros(length(cohs),1);
    n_c = zeros(length(cohs),1);
    
    for i = 1:length(cohs)
        n_c(i) = sum(plot_coh_high_c == cohs(i));
        perc_right_high_c(i) = sum(plot_action_high_c(plot_coh_high_c == cohs(i)))/n_c(i);
    end
    
    perc_right_SE_high_c = sqrt(perc_right_high_c.* (1 - perc_right_high_c)./n_c);
    
    % fit logistic regression
    [B_high_c, ~, stats_high_c] = glmfit(plot_coh_high_c, plot_action_high_c, 'binomial');
    x = linspace(cohs(1),cohs(end),100);
    y_hat_high_c = glmval(B_high_c, x,'logit');
    
    stats_all_high.(fn{iCoh}) = stats_high_c;
    
%     % Plot
%     figure; hold on;
%     errorbar(cohs, perc_right, perc_right_SE, '.','MarkerSize',14);
%     errorbar(cohs, perc_right_high_c, perc_right_SE_high_c, '.','MarkerSize',14);
%     plot(x,y_hat,'b','Linewidth',1.);
%     plot(x,y_hat_high_c,'r--','Linewidth',1.5);
%     legend('','','All trials',['Prev R, rewarded, high bet,',' ', num2str(abs_cohs(iCoh))]','FontSize',16,'location','NW')
%     xlabel('Motion strength (Coh)'); ylabel('Proportion rightward choices');
%     set(gca,'FontSize',22);
%     legend('boxoff')
    
    % Save figures
%     saveas(gcf,[num2str(abs_cohs(iCoh)),'_coh_rightward_correct.fig'])
%     saveas(gcf,[num2str(abs_cohs(iCoh)),'_coh_rightward_correct.jpg'])
    
end

% save('logistic_fit_stats_high_20211229.mat','stats_all_high')

%% Plot conditioned on trial difficulty rightward, low confidence
abs_cohs = unique(abs(cohs));

stats_all_low = struct('coh',[],'all_trial',[]); % coherence refers to the coh in the previous trial
stats_all_low.all_trial = stats;

fn = fieldnames(stats_all_low);

for iCoh = 1 % :length(abs_cohs)
    % select trial whose previous trial is rewarded, rightward, and have a
    % specific coherence level
    idx = find(plot_action == rightward & plot_reward > 0 & plot_conf < conf_thres) + 1;
    idx = idx(2:end-1);
    plot_coh_low_c = plot_coh(idx);
    plot_action_low_c = plot_action(idx);
    perc_low_c = zeros(length(cohs),1);
    n_c = zeros(length(cohs),1);
    
    for i = 1:length(cohs)
        n_c(i) = sum(plot_coh_low_c == cohs(i));
        perc_low_c(i) = sum(plot_action_low_c(plot_coh_low_c == cohs(i)))/n_c(i);
    end
    
    perc_low_SE_c = sqrt(perc_low_c.* (1 - perc_low_c)./n_c);
    
    % fit logistic regression
    [B_low_c, ~, stats_low_c] = glmfit(plot_coh_low_c, plot_action_low_c, 'binomial');
    x = linspace(cohs(1),cohs(end),100);
    y_hat_low_c = glmval(B_low_c, x,'logit');
    
    stats_all_low.(fn{iCoh}) = stats_low_c;
    
%     % Plot
%     figure; hold on;
%     errorbar(cohs, perc_right, perc_right_SE, '.','MarkerSize',14);
%     errorbar(cohs, perc_low_c, perc_low_SE_c, '.','MarkerSize',14);
%     plot(x,y_hat,'b','Linewidth',1.);
%     plot(x,y_hat_low_c,'r--','Linewidth',1.5);
%     legend('','','All trials',['Prev L, rewarded, low bet',' ', num2str(abs_cohs(iCoh))]','FontSize',16,'location','NW')
%     xlabel('Motion strength (Coh)'); ylabel('Proportion rightward choices');
%     set(gca,'FontSize',22);
%     legend('boxoff')
    
    % Save figures
%     saveas(gcf,[num2str(abs_cohs(iCoh)),'_coh_lowward_correct.fig'])
%     saveas(gcf,[num2str(abs_cohs(iCoh)),'_coh_lowward_correct.jpg'])
    
end

% save('logistic_fit_stats_low_20211229.mat','stats_all_low')

%% Plot weights
if rightward
    b0 = [stats_all_high.coh.beta(1);stats_all_low.coh.beta(1)];
    b0_std = [stats_all_high.coh.se(1);stats_all_low.coh.se(1)];
    
    figure('Position',[497,385,653,374]);hold on;
    bar(1,b0(1),0.8,'Linewidth',2,'Edgecolor',[1,1,1],'FaceColor',[0, 0.4470, 0.7410]);
    bar(2,b0(2),0.8,'Linewidth',2,'Edgecolor',[1,1,1],'FaceColor',[0.3010 0.7450 0.9330]);
    set(gca,'xtick',[1:2],'xticklabel', {'High','Low'});
    hold on;
    er = errorbar(1:2,b0,b0_std,'LineWidth',1.5);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    xlabel('Previous bet')
    ylabel('Logistic coefficient, b_0')
    set(gca,'FontSize',25,'YTick',[0:0.3:0.90])
    ylim([-0.00, 0.90])
    box off;
else
    b0 = [stats_all_high.coh.beta(1);stats_all_low.coh.beta(1)];
    b0_std = [stats_all_high.coh.se(1);stats_all_low.coh.se(1)];
    
    figure('Position',[497,385,653,374]);hold on;
    bar(1,b0(1),0.8,'Linewidth',2,'Edgecolor',[1,1,1],'FaceColor',[0.8500, 0.3250, 0.0980]);
    bar(2,b0(2),0.8,'Linewidth',2,'Edgecolor',[1,1,1],'FaceColor',[0.9290, 0.6940, 0.1250]);
    set(gca,'xtick',[1:2],'xticklabel', {'High','Low'});
    hold on;
    er = errorbar(1:2,b0,b0_std,'LineWidth',1.5);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    xlabel('Previous bet')
    ylabel('Logistic coefficient, b_0')
    set(gca,'FontSize',25,'YTick',[-0.90:0.3:0.])
    ylim([-1.0, 0.00])
    box off;

end
