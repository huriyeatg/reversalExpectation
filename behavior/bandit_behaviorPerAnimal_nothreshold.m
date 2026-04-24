function bandit_behaviorPerAnimal_nothreshold(dataIndex,save_path)
% % bandit_behaviorPerAnimal %
%PURPOSE:   Analyze bandit behavior averaged across animals
%AUTHORS:   H Atilgan and AC Kwan 191204
%
%INPUT ARGUMENTS
%   dataIndex:    a database index table for the sessions to analyze
%   save_path:    path for saving the plots
%
%OUTPUT ARGUMENTS
%

%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each animal
animalList = unique(dataIndex.Animal);
%animalList = animalList([1 2 5 6 7 8 10])

disp('-----------------------------------------------------------');
disp(['--- Analyzing - summary of ', int2str(numel(animalList)) ' animals']);
disp('-----------------------------------------------------------');

Lcriteria = 200; % no criteria
for j = 1:numel(animalList)
    
    %which session belong to this one animal
    currAnimalSessions = ismember(dataIndex.Animal,animalList(j));
    
    %concatenate the sessions for this one animal
    [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));
    stats = value_getTrialStats(trials, nRules);
    stats = value_getTrialStatsMore(stats);
   % stats.blockTrialRandomAdded = stats.blockLength;
    %% plot choice behavior - around switches left to right
    trials_back=10;  % set number of previous trials
    
    sw_output{j}=choice_switch(stats,trials_back);
    
    %% plot choice behavior - around switch high-probability side to low-probability side
    sw_hrside_output{j}=choice_switch_hrside(stats,trials_back);
    
    %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
    L1_ranges=[10 Lcriteria;10 Lcriteria;10 Lcriteria;10 Lcriteria]; %consider only subset of blocks within the range, for trials to criterion
    L2_ranges=[0 4;5 9;10 14;15 30];% [ 15 19;20 24;25 29;30 45];%   %consider only subset of blocks within the range, for random added number of trials
    sw_random_output{j}=choice_switch_random(stats,trials_back,L1_ranges,L2_ranges);
    
    sw_hrside_random_output{j}=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);
    
    %% plot tendency to predict upcoming reversal
    %store the x and y variables to be made into a scatter plot
    ind = find(stats.blockTrialtoCrit<=Lcriteria);% & stats.followingBlockChangePointTrialIndex<20);
    sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
    sw_LRVsChoice{j}.range{1}=[0 30];
    sw_LRVsChoice{j}.range{2}=[0.6 1];
    sw_LRVsChoice{j}.label{1}={'L_{Random}'};
    sw_LRVsChoice{j}.label{2}={'Fraction of trials';'selecting initial better option'};
    
    %Slope / MidPoint
    L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
    L1_ranges= ones(size(L2_ranges,1),2).*[10 Lcriteria];    % BehCriteria
    trials_forward = 50;
    sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,trials_forward,L1_ranges,L2_ranges);
    
    sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
    sw_LRVsMidpoint{j}.range{1}=[0 30];
    sw_LRVsMidpoint{j}.range{2}=[0 10];
    sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
    sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'};{'fraction of trials = 0.5'}];
    
    sw_LRVsSlope{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(2,:)'];
    sw_LRVsSlope{j}.range{1}=[0 30];
    sw_LRVsSlope{j}.range{2}=[-0.08 0];
    sw_LRVsSlope{j}.label{1}={'L_{Random}'};
    sw_LRVsSlope{j}.label{2}={'Slope'};
    
    sw_LRVsIntercept{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(3,:)'];
    sw_LRVsIntercept{j}.range{1}=[0 30];
    sw_LRVsIntercept{j}.range{2}=[0.5 1];
    sw_LRVsIntercept{j}.label{1}={'L_{Random}'};
    sw_LRVsIntercept{j}.label{2}={'Intercept'};
    
    %% More stats on block patterns
    sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
    sw_LRVsRR{j}.range{1}=[0 30];
    sw_LRVsRR{j}.range{2}=[40 60];
    sw_LRVsRR{j}.label{1}={'L_{Random}'};
    sw_LRVsRR{j}.label{2}={'Reward rates (%)'};
    
    sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
    sw_LRVsHR{j}.range{1}=[0 30];
    sw_LRVsHR{j}.range{2}=[50 80];
    sw_LRVsHR{j}.label{1}={'L_{Random}'};
    sw_LRVsHR{j}.label{2}={'Hit rates (%)'};
    
    sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
    sw_LRVsWinStay{j}.range{1}=[0 30];
    sw_LRVsWinStay{j}.range{2}=[0.85 1];
    sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
    sw_LRVsWinStay{j}.label{2}={'P(win|stay)'};
    
    sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
    sw_LRVsLooseSwitch{j}.range{1}=[0 30];
    sw_LRVsLooseSwitch{j}.range{2}=[0.1 0.3];
    sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
    sw_LRVsLooseSwitch{j}.label{2}={'P(lose|switch)'};
    
    %% plot lick rates
    trialType={{'left','reward'},{'left','noreward'},{'right','reward'},{'right','noreward'}};
    edges=[-0.5:0.02:3];   % edges to plot the lick rate histogram
    lick_trType{j}=get_lickrate_byTrialType(trialData,trials,trialType,edges);
    
    %% plot response times
    valLabel='Response time (s)';
    trialType={'go','left','right'};
    edges=[0:0.01:1];
    respTime_trType{j}=get_val_byTrialType(trialData.rt,trials,trialType,edges,valLabel);
    
    %% plot ITI
    valLabel='Inter-trial interval (s)';
    trialType={'go','reward','noreward'};
    edges=[0:0.1:30];
    iti_trType{j}=get_val_byTrialType(trialData.iti,trials,trialType,edges,valLabel);
    
    %% plot logistic regression analysis
    num_regressor = 10;
    [lreg{j}, ~, ~, ~]=logreg_RCUC(stats,num_regressor);
    
    %% plot logistic regression analysis - rewarded/unrewarded left/right choices
    % num_regressor = 10;
    % [lreg_LR{j}, ~, ~, ~]=logreg_RCUC_LR(stats,num_regressor);
    
end

tlabel = ['Summary of ' int2str(numel(animalList)) ' animals'];

plot_switch(sw_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches_lateral'));
saveas(gcf, fullfile(save_path,'switches_lateral'), 'fig');

plot_switch_hrside(sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside'));
saveas(gcf, fullfile(save_path,'switches_hrside'), 'fig');

plot_switch_hrside_witheachanimals(sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside_witheachanimals'));
saveas(gcf, fullfile(save_path,'switches_hrside_witheachanimals'), 'fig');

plot_switch_random(sw_random_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches_lateral_random'));
saveas(gcf, fullfile(save_path,'switches_lateral_random'), 'fig');

plot_switch_hrside_random(sw_hrside_random_output,tlabel);
%legend off
print(gcf,'-dpng',fullfile(save_path,'switches_hrside_random'));
saveas(gcf, fullfile(save_path,'switches_hrside_random'), 'svg');

plot_switch_random_distn(sw_random_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_random_distn'));
saveas(gcf, fullfile(save_path,'switches_random_distn'), 'svg');

plot_multibinxaveragey([ {sw_LRVsChoice},{sw_LRVsMidpoint},...
    {sw_LRVsSlope},{sw_LRVsIntercept}]);
print(gcf,'-dpng',fullfile(save_path,'LR_vs_switchPatterns'));
saveas(gcf, fullfile(save_path,'LR_vs_switchPatterns'), 'svg');

plot_multibinxaveragey([ {sw_LRVsHR};{sw_LRVsRR}; ...
    {sw_LRVsWinStay};{sw_LRVsLooseSwitch}]);
print(gcf,'-dpng',fullfile(save_path,'LR_vs_blockPatterns'));
saveas(gcf, fullfile(save_path,'LR_vs_blockPatterns'), 'fig');
% 
% plot_lickrate_byTrialType(lick_trType);
% print(gcf,'-dpng',fullfile(save_path,'lickrates_byTrialType'));
% saveas(gcf, fullfile(save_path,'lickrates_byTrialType'), 'fig');
% 
% plot_val_byTrialType(respTime_trType);
% print(gcf,'-dpng',fullfile(save_path,'rt_byTrialType'));
% saveas(gcf, fullfile(save_path,'rt_byTrialType'), 'fig');
% 
% plot_val_byTrialType(iti_trType);
% print(gcf,'-dpng',fullfile(save_path,'iti_byTrialType'));
% saveas(gcf, fullfile(save_path,'iti_byTrialType'), 'fig');
% 
% plot_logreg(lreg,tlabel);
% print(gcf,'-dpng',fullfile(save_path,'logreg'));
% saveas(gcf, fullfile(save_path,'logreg'), 'fig');

% plot_logreg(lreg_LR,tlabel);
% print(gcf,'-dpng',fullfile(save_path,'logreg_lateral'));
% saveas(gcf, fullfile(save_path,'logreg_lateral'), 'fig')

%close all
disp (' Figures save in :')
disp(['   ',save_path])
