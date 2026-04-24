function bandit_session_PerAnimal(dataIndex,save_path)
% % bandit_session_PerAnimal %
%PURPOSE:   Preparing to analyze a single session of mouse behavior
%AUTHORS:   H Atilgan and AC Kwan 191203
%
%INPUT ARGUMENTS
%   BehPath:        path for the location of the analysis folder containing
%                   the behavioral .mat file
%   LogFileName:    name of the logfile
%
%OUTPUT ARGUMENTS
%

%% load the behavioral data
if ~exist(save_path,'dir')
    mkdir(save_path);
end

disp('-----------------------------------------------------------');
disp('--- Analyzing one animal behavioral sessions ');
disp('-----------------------------------------------------------');
model_type ='Experimental';
% Get trial information
%concatenate the sessions for this one animal
[trialData, trials, nRules] = merge_sessions(dataIndex);

stats = value_getTrialStats(trials, nRules);
stats = value_getTrialStatsMore(stats);

%% HRSide random
trials_back = 10;

L1_ranges=[10 20;10 20;10 20;10 20]; %consider only subset of blocks within the range, for trials to criterion
L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials
sw_hrside_random_output=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);

%% plot tendency to predict upcoming reversal
j=1;
%store the x and y variables to be made into a scatter plot
ind = find(stats.blockTrialtoCrit<20);% & stats.followingBlockChangePointTrialIndex<20);
sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
sw_LRVsChoice{j}.range{1}=[0 30];
sw_LRVsChoice{j}.range{2}=[0.6 1];
sw_LRVsChoice{j}.label{1}={'L_{Random}'};
sw_LRVsChoice{j}.label{2}={'Fraction of trials';'selecting initial better option'};

%Slope / MidPoint
L2_ranges=[1:2:30 ; 3:2:32]';        % Random Block length steps
L1_ranges= ones(size(L2_ranges,1),2).*[10 20];    % BehCriteria
sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,L1_ranges,L2_ranges);


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


%%
sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
sw_LRVsHR{j}.range{1}=[0 30];
sw_LRVsHR{j}.range{2}=[0.7 0.9];
sw_LRVsHR{j}.label{1}={'L_{Random}'};
sw_LRVsHR{j}.label{2}={'Hit rates'};

sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
sw_LRVsRR{j}.range{1}=[0 30];
sw_LRVsRR{j}.range{2}=[0.4 0.7];
sw_LRVsRR{j}.label{1}={'L_{Random}'};
sw_LRVsRR{j}.label{2}={'Reward rates'};

sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
sw_LRVsWinStay{j}.range{1}=[0 30];
sw_LRVsWinStay{j}.range{2}=[0.9 1];
sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
sw_LRVsWinStay{j}.label{2}={'P(win|stay)'};

sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
sw_LRVsLooseSwitch{j}.range{1}=[0 30];
sw_LRVsLooseSwitch{j}.range{2}=[0.1 0.3];
sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
sw_LRVsLooseSwitch{j}.label{2}={'P(loose|switch)'};


%% plot figures
tlabel=[model_type];
lb='00';
plot_switch_hrside_random(sw_hrside_random_output,tlabel);
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_hrside_random']));
saveas(gcf, fullfile(save_path,[model_type,'_switches_hrside_random']), 'fig');

plot_multibinxaveragey([ {sw_LRVsChoice},{sw_LRVsMidpoint}])
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_LRVsChoice']));
saveas(gcf, fullfile(save_path,[model_type,'_switches_LRVsChoice']), 'fig');

plot_multibinxaveragey([ {sw_LRVsHR};{sw_LRVsRR}; ...
    {sw_LRVsWinStay};{sw_LRVsLooseSwitch}]);
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_hitRatesVsLrandom']));
saveas(gcf, fullfile(save_path,[model_type,'_hitRatesVsLrandom']), 'fig');


end