function bandit_M2stimulationPerSession_vs2(dataIndex,save_path)
% % bandit_M2stimulationPerSession %
%PURPOSE:   Analyze bandit behavior averaged across sessions, comparing
%           stimulated vs non-stimulated
%AUTHORS:   H Atilgan 210321
%
%INPUT ARGUMENTS
%   dataIndex :   vector corresponding to dataIndex
%   save_path:    path for saving the plots
%
%OUTPUT ARGUMENTS
%


%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% %%%%% SWITCH PLOTS
 %concatenate the sessions for this one animal
[trialData, trials, nRules] = merge_sessions(dataIndex);
stats = value_getTrialStats(trials, nRules);
stats = value_getTrialStatsMore(stats);
%stimulation
stats.stRegion = trialData.stimulationRegion-1000;
stats.st=nan(numel(trialData.stimulation),1);
stats.st = stats.stRegion;
stats.st (stats.stRegion==2) =1;
    
plot_photoStimulationSwitchPlots (stats, save_path)

nCrit = 200;
kk = 1;
trials_back = 10;


%% %%%%% Look for ipsi vs 
clear dat
for j = 1: size(dataIndex,1)
    load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_beh.mat']));
    trials = value_getTrialMasks(trialData);
    stats = value_getTrialStats(trials, sessionData.nRules);
    stats = value_getTrialStatsMore(stats);
    %% Add stimulation info
    % stimulated side
    stats.stRegion = trialData.stimulationRegion-1000;
    
    %stimulation: yes=1; no=1;
    stats.st=nan(numel(trialData.stimulation),1);
    stats.st = stats.stRegion;
    stats.st (stats.stRegion==2) =1;
    
    % which block stimulated
    x0=stats.rule';
    stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    stats.blockStRegion = stats.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    %% now divide the blocks for stimulated vs non-stimulated
    nTrials = numel(stats.c);
    nBlocks = size(stats.blockRule,1);
    
    % define contra vs ipsi blocks in stimulated blocks - contra means 1, ipsi means 0
    bStimInfo =  nan(nBlocks,1);
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 1)  = 0 ;% contra: left stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 2)  = 1 ;% ipsi: left stimulation & left side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 2)  = 1 ;% ipsi: right stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 1)  = 0 ;% contra: right stimulation & left side being initial better option
    stats.blockStimulationSide = bStimInfo(1:end-1); % exclude last block
    
     % define contra vs ipsi blocks in control blocks - contra means 1, ipsi means 0
     % two blocks before stimulated blocks 
     bControlInfo = [ bStimInfo(3:end);nan; nan];
    %bControlInfo = [ nan;nan; bStimInfo(1:end-2)];
    stats.blockControlSide = bControlInfo(1:end-1); % exclude last block
   
         
         % Stimulated: contra vs ipsi
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==1; %ipsi 
        dat(kk).sw_LRVsChoiceIpsi{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
        dat(kk).sw_LRVsChoiceIpsi{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceIpsi{j}.range{2}=[0.5 1];
        dat(kk).sw_LRVsChoiceIpsi{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceIpsi{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
        
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==0; %contra
        dat(kk).sw_LRVsChoiceContra{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
        dat(kk).sw_LRVsChoiceContra{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceContra{j}.range{2}=[0.5 1];
        dat(kk).sw_LRVsChoiceContra{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceContra{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
      
        % Control: contra vs ipsi
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==0; %general criteria
        dat(kk).sw_LRVsChoiceIpsiControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.range{2}=[0.5 1];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceIpsiControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
        
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==1; %general criteria
        dat(kk).sw_LRVsChoiceContraControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
        dat(kk).sw_LRVsChoiceContraControl{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceContraControl{j}.range{2}=[0.5 1];
        dat(kk).sw_LRVsChoiceContraControl{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceContraControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
        
        %% trials to reach midpoint
        L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
        L1_ranges= ones(size(L2_ranges,1),2).*[10 nCrit];    % BehCriteria
        trials_forward =50;
        sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,trials_forward,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpoint{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpoint{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
       % Stimulated: Trials to reach midpoints - bilateral
        L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
        L1_ranges= ones(size(L2_ranges,1),2).*[10 20];    % BehCriteria
        trials_forward = 50;
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==0; %general criteria
        sw_stats_output{j} = choice_switch_stats_random_M2stimulation(ind, stats,trials_back,trials_forward,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpointIpsi{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpointIpsi{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointIpsi{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointIpsi{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointIpsi{j}.label{2}= [{'Trials to reach midpoint bilateral'}];%;{'fraction of trials = 0.5'}];

        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==1; %general criteria
        sw_stats_output{j} = choice_switch_stats_random_M2stimulation(ind, stats,trials_back,trials_forward,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpointContra{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpointContra{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointContra{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointContra{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointContra{j}.label{2}= [{'Trials to reach midpoint bilateral'}];%;{'fraction of trials = 0.5'}];

        
          % Control: Trials to reach midpoints - bilateral
         ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==0; %general criteria
         sw_stats_output{j} = choice_switch_stats_random_M2stimulation(ind, stats,trials_back,trials_forward,L1_ranges,L2_ranges);
         
        dat(kk).sw_LRVsMidpointIpsiControl{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpointIpsiControl{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointIpsiControl{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointIpsiControl{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointIpsiControl{j}.label{2}= [{'Trials to reach midpoint bilateral'}];%;{'fraction of trials = 0.5'}];

        ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==1; %general criteria
        sw_stats_output{j} = choice_switch_stats_random_M2stimulation(ind, stats,trials_back,trials_forward,L1_ranges,L2_ranges);

        dat(kk).sw_LRVsMidpointContraControl{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpointContraControl{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointContraControl{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointContraControl{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointContraControl{j}.label{2}= [{'Trials to reach midpoint bilateral'}];%;{'fraction of trials = 0.5'}];
end

%% Visualise data

L2_ranges = [0 4;5 9;10 14; 15 30];%[0:2:29; 1:2:30]' %
tlabelBilateral   = [{'Ipsi blocks'}; {'Contra block'}]';
plot_binxaveragey_bilateral_M2stimulation([{dat(1).sw_LRVsChoiceIpsiControl,dat(1).sw_LRVsChoiceIpsi};...
    {dat(1).sw_LRVsChoiceIpsiControl,dat(1).sw_LRVsChoiceContra}],tlabelBilateral, L2_ranges,save_path);

%  plot_binxaveragey_bilateral_M2stimulation([{dat(1).sw_LRVsMidpointIpsiControl,dat(1).sw_LRVsMidpointIpsi};...
%      {dat(1).sw_LRVsMidpointContraControl,dat(1).sw_LRVsMidpointContra}],tlabelBilateral, L2_ranges,save_path);


% 
% 
% close all hidden
% disp ( 'Figures saved.')

