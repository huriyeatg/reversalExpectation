function bandit_PerSession_longitudinal(dataIndex,save_path, sessionNumber)
% % bandit_PerSession_longitudinal %
%PURPOSE:   Analyze bandit behavior averaged across sessions, comparing
%           first couple of sessions  versus last couple sessions
%AUTHORS:   H Atilgan, AC Kwan and S Koc 211031
%
%INPUT ARGUMENTS
%   dataIndex:    dataIndex
%   save_path:    path for saving the plots
%   sessionNumber: the number of session to compare
%OUTPUT ARGUMENTS
% figures saved in the save_path

%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each session

disp('-----------------------------------------------------------');
disp(['--- Analyzing the first', num2str(sessionNumber),' phase 3 sessions to the last'...
    num2str(sessionNumber) ' phase 3 sessions']);
disp('-----------------------------------------------------------');

%% Organize data

for kk = 1:2 % THe points you will compare
    % COMPARE first 2, 3, 5 sessions to last 2, 3, 5 sessions
    % COMPARE first vs last session
    % COMPARE session 1,3,5,7,9,11 - can you see gradiual increase?
    % TRY PER ANIMAL to see if there is anything
    if kk==1     %Calculate the first set of sessions
        currList = find(dataIndex.Phase ==3 & dataIndex.sessionIndex<=sessionNumber);
    elseif kk==2 %Calculate the last set of sessions
        currList = find(dataIndex.Phase ==3 & dataIndex.reverseSessionIndex<=sessionNumber);
    end
    
    for j = 1:numel(currList)
     
        load(fullfile(dataIndex.BehPath{currList(j)},[dataIndex.LogFileName{currList(j)}(1:end-4),'_beh.mat']));
        
        trials = value_getTrialMasks(trialData);
        stats = value_getTrialStats(trials, sessionData.nRules);
        stats = value_getTrialStatsMore(stats);
        
        %% plot basic behavioral performance
        dat(kk).beh_output{j}=beh_performance(stats);
        
        %% plot choice behavior - around switches left to right
        trials_back=10;  % set number of previous trials
        
        dat(kk).sw_output{j}=choice_switch(stats,trials_back);
        
        %% plot choice behavior - around switch high-probability side to low-probability side
        dat(kk).sw_hrside_output{j}=choice_switch_hrside(stats,trials_back);
        
        %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
        L1_ranges=[10 200;10 200;10 200;10 200]; %consider only subset of blocks within the range, for trials to criterion
        L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials
        dat(kk).sw_random_output{j}=choice_switch_random(stats,trials_back,L1_ranges,L2_ranges);
        
        dat(kk).sw_hrside_random_output{j}=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);
        
        %% hit rates for selected blocks
        ind1 = stats.blockTrialtoCrit<300 & stats.blockTrans==1; % general criteria
        ind2 = stats.blockTrialtoCrit<300 & stats.blockTrans==2;
        dat(kk).hitRates{j}.dat=[nanmean(stats.hitrates(ind1)) nanmean(stats.hitrates(ind2))];
        
        %% More stats on block patterns
        L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
        L1_ranges= ones(size(L2_ranges,1),2).*[10 200];    % BehCriteria
        trials_forward = 50;
        sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,trials_forward,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpoint{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpoint{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'};{'fraction of trials = 0.5'}];
        
        ind = stats.blockTrialtoCrit<=200;
        dat(kk).sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoice{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoice{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoice{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoice{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
        dat(kk).sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRR{j}.range{2}=[35 55];
        dat(kk).sw_LRVsRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRR{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHR{j}.range{2}=[50 70];
        dat(kk).sw_LRVsHR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHR{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStay{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStay{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStay{j}.label{2}={'P(stay|win)'};
        
        dat(kk).sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitch{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitch{j}.range{2}=[0.1 0.4];
        dat(kk).sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitch{j}.label{2}={'P(lose|switch)'};
        
        %% Trials to reach midpoints - bilateral
        L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
        L1_ranges= ones(size(L2_ranges,1),2).*[10 200];    % BehCriteria
        trials_forward = 50;
        sw_stats_output{j} = choice_switch_stats_random_lesion(stats,trials_back,trials_forward,L1_ranges,L2_ranges);
        
        %% More stats on block patterns - Left
        ind = stats.blockTrialtoCrit<=200 & stats.blockTrans==1; %general criteria
        dat(kk).sw_LRVsChoiceL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceL{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceL{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoiceL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceL{j}.label{2}={'Fraction of trials'};%'selecting initial  better option'};
        
        dat(kk).sw_LRVsMidpointL{j}.dat=[L2_ranges(:,1) squeeze(sw_stats_output{j}.statl(1,1,:))];
        dat(kk).sw_LRVsMidpointL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointL{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointL{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
        dat(kk).sw_LRVsRRL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRRL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRRL{j}.range{2}=[35 55];
        dat(kk).sw_LRVsRRL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRRL{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHRL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHRL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHRL{j}.range{2}=[50 70];
        dat(kk).sw_LRVsHRL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHRL{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStayL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStayL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStayL{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsWinStayL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStayL{j}.label{2}={'P(win|stay)'};
        
        dat(kk).sw_LRVsLooseSwitchL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitchL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitchL{j}.range{2}=[0.1 0.4];
        dat(kk).sw_LRVsLooseSwitchL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitchL{j}.label{2}={'P(lose|switch)'};
        
        
        %% More stats on block patterns - Right
        ind = stats.blockTrialtoCrit<=200 & stats.blockTrans==2; % general criteria
        
        dat(kk).sw_LRVsChoiceR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceR{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceR{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoiceR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceR{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
        dat(kk).sw_LRVsMidpointR{j}.dat=[L2_ranges(:,1) squeeze(sw_stats_output{j}.statr(1,2,:))];
        dat(kk).sw_LRVsMidpointR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointR{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointR{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
        dat(kk).sw_LRVsRRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRRR{j}.range{2}=[35 55];
        dat(kk).sw_LRVsRRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRRR{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHRR{j}.range{2}=[50 70];
        dat(kk).sw_LRVsHRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHRR{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStayR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStayR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStayR{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsWinStayR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStayR{j}.label{2}={'P(stay|win)'};
        
        dat(kk).sw_LRVsLooseSwitchR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitchR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitchR{j}.range{2}=[0.1 0.4];
        dat(kk).sw_LRVsLooseSwitchR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitchR{j}.label{2}={'P(lose|switch)'};
        
        %Slope / MidPoint - linear fit needs more data, cannot be
        %calculated by session - check perAnimal code
        
        %% plot lick rates
        trialType={{'left','reward'},{'left','noreward'},{'right','reward'},{'right','noreward'}};
        edges=[-0.5:0.02:3];   % edges to plot the lick rate histogram
        dat(kk).lick_trType{j}=get_lickrate_byTrialType(trialData,trials,trialType,edges);
        
        %% plot response times
        valLabel='Response time (s)';
        trialType={'go','left','right'};
        edges=[0:0.01:1];
        dat(kk).respTime_trType{j}=get_val_byTrialType(trialData.rt,trials,trialType,edges,valLabel);
        
        %% plot ITI
        valLabel='Inter-trial interval (s)';
        trialType={'go','reward','noreward'};
        edges=[0:0.1:30];
        dat(kk).iti_trType{j}=get_val_byTrialType(trialData.iti,trials,trialType,edges,valLabel);
        
    end
end

%% Visualise data
tlabel = ['Longitudinal analysis of ', num2str(sessionNumber),' sessions'];

plot_behperf_longitudinal(dat(1).beh_output,dat(2).beh_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'behperf'));
saveas(gcf, fullfile(save_path,'behperf'), 'fig');

% UGLY CODE plot_behperf_lesion_hitRates(dat(1).hitRates,dat(2).hitRates,tlabel);

plot_switch_longitudinal(dat(1).sw_output,dat(2).sw_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches'));
saveas(gcf, fullfile(save_path,'switches'), 'fig');

plot_switch_random_longitudinal(dat(1).sw_random_output,dat(2).sw_random_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches_random'));
saveas(gcf, fullfile(save_path,'switches_random'), 'fig');

plot_switch_hrside_longitudinal(dat(1).sw_hrside_output,dat(2).sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside'));
saveas(gcf, fullfile(save_path,'switches_hrside'), 'fig');

plot_switch_hrside_random_longitudinal(dat(1).sw_hrside_random_output,dat(2).sw_hrside_random_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside_random'));
saveas(gcf, fullfile(save_path,'switches_hrside_random'), 'fig');

L2_ranges = [0 4;5 9;10 14;15 30];

plot_binxaveragey_longitudinal({dat(1).sw_LRVsMidpoint,dat(2).sw_LRVsMidpoint},tlabel, L2_ranges,save_path);

plot_binxaveragey_longitudinal({dat(1).sw_LRVsChoice,dat(2).sw_LRVsChoice},tlabel, L2_ranges,save_path);

plot_binxaveragey_longitudinal({dat(1).sw_LRVsRR,dat(2).sw_LRVsRR},tlabel, L2_ranges,save_path);

plot_binxaveragey_longitudinal({dat(1).sw_LRVsHR,dat(2).sw_LRVsHR},tlabel, L2_ranges,save_path);

plot_binxaveragey_longitudinal({dat(1).sw_LRVsWinStay,dat(2).sw_LRVsWinStay},tlabel, L2_ranges,save_path);

plot_binxaveragey_longitudinal({dat(1).sw_LRVsLooseSwitch,dat(2).sw_LRVsLooseSwitch},tlabel, L2_ranges,save_path);


plot_lickrate_byTrialType_longitudinal(dat(1).lick_trType,dat(2).lick_trType);
print(gcf,'-dpng',fullfile(save_path,'lickrates_byTrialType'));
saveas(gcf, fullfile(save_path,'lickrates_byTrialType'), 'fig');

plot_val_byTrialType_longitudinal(dat(1).respTime_trType,dat(2).respTime_trType);
print(gcf,'-dpng',fullfile(save_path,'rt_byTrialType'));
saveas(gcf, fullfile(save_path,'rt_byTrialType'), 'fig');

plot_val_byTrialType_longitudinal(dat(1).iti_trType,dat(2).iti_trType);
print(gcf,'-dpng',fullfile(save_path,'iti_byTrialType'));
saveas(gcf, fullfile(save_path,'iti_byTrialType'), 'fig');


close all hidden
disp ( 'Figures saved.')