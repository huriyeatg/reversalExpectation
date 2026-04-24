function bandit_M2stimulationPerSession(dataIndex,save_path)
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

nCrit = 20;
%%
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
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 1)  = 0 ;% left stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 2)  = 1 ;% left stimulation & left side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 2)  = 0 ;% right stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 1)  = 1 ;% right stimulation & left side being initial better option
    stats.blockStimulationSide = bStimInfo(1:end-1); % exclude last block
    
     % define contra vs ipsi blocks in control blocks - contra means 1, ipsi means 0
     % two blocks before stimulated blocks 
    bControlInfo = [ bStimInfo(3:end);nan; nan];
    stats.blockControlSide = bControlInfo(1:end-1); % exclude last block
    
    statsOriginal = stats; % stats will be used below function
      
    for kk = 1:2 % control trials vs stimulated trials
        if kk==1  % for control trials
            trialInd = statsOriginal.st==0;
            blockInd = statsOriginal.blockSt==0;
        else      % for stimulated trials
            trialInd = statsOriginal.st==1;
            blockInd = statsOriginal.blockSt==1;
        end
        clear stats trialDataSelected
        fields=fieldnames(statsOriginal);
        for jj = 1:numel(fields)
            if size(statsOriginal.(fields{jj}),1)==nTrials
                stats.(fields{jj}) = statsOriginal.(fields{jj})(trialInd);
            elseif size(statsOriginal.(fields{jj}),1)==nBlocks
                stats.(fields{jj}) = statsOriginal.(fields{jj})(blockInd);
            elseif size(statsOriginal.(fields{jj}),1)==nBlocks-1
                stats.(fields{jj}) = statsOriginal.(fields{jj})(blockInd(1:nBlocks-1));
            end
        end
        
        stats.rule_labels = statsOriginal.rule_labels;
        stats.ruletransList = statsOriginal.ruletransList;
        
        %% plot basic behavioral performance
        dat(kk).beh_output{j}=beh_performance(stats);
        
        %% plot choice behavior - around switches left to right
        trials_back=10;  % set number of previous trials
        
        dat(kk).sw_output{j}=choice_switch(stats,trials_back);
        
        %% plot choice behavior - around switch high-probability side to low-probability side
        dat(kk).sw_hrside_output{j}=choice_switch_hrside(stats,trials_back);
        
        %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
        L1_ranges=[10 nCrit;10 nCrit;10 nCrit;10 nCrit]; %consider only subset of blocks within the range, for trials to criterion
        L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials
        dat(kk).sw_random_output{j}=choice_switch_random(stats,trials_back,L1_ranges,L2_ranges);
        
        dat(kk).sw_hrside_random_output{j}=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);
        
        %% initial better option plots
        ind = stats.blockTrialtoCrit<=nCrit;
        dat(kk).sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
        dat(kk).sw_LRVsChoice{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoice{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoice{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoice{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
         % Stimulated: contra vs ipsi
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==0; %general criteria
        dat(kk).sw_LRVsChoiceIpsi{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceIpsi{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceIpsi{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoiceIpsi{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceIpsi{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
        
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==1; %general criteria
        dat(kk).sw_LRVsChoiceContra{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceContra{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceContra{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoiceContra{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceContra{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
      
        % Control: contra vs ipsi
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==0; %general criteria
        dat(kk).sw_LRVsChoiceIpsiControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.range{2}=[0.6 1];
        dat(kk).sw_LRVsChoiceIpsiControl{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceIpsiControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
        
        ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==1; %general criteria
        dat(kk).sw_LRVsChoiceContraControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceContraControl{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceContraControl{j}.range{2}=[0.6 1];
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

        %% More stats on block patterns: RR, HR, Pwin, Ploose
        ind = stats.blockTrialtoCrit<=nCrit;
        dat(kk).sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRR{j}.range{2}=[45 65];
        dat(kk).sw_LRVsRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRR{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHR{j}.range{2}=[50 80];
        dat(kk).sw_LRVsHR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHR{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStay{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStay{j}.range{2}=[0.9 1];
        dat(kk).sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStay{j}.label{2}={'P(stay|win)'};
        
        dat(kk).sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitch{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitch{j}.range{2}=[0.1 0.3];
        dat(kk).sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitch{j}.label{2}={'P(lose|switch)'};
        
        
    end
end

%% Visualise data
tlabel = ['Stimulation'];

plot_behperf_M2stimulation(dat(1).beh_output,dat(2).beh_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'behperf'));
saveas(gcf, fullfile(save_path,'behperf'), 'fig');

plot_switch_hrside_M2stimulation(dat(1).sw_hrside_output,dat(2).sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside'));
saveas(gcf, fullfile(save_path,'switches_hrside'), 'fig');

plot_switch_hrside_random_M2stimulation(dat(1).sw_hrside_random_output,dat(2).sw_hrside_random_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'switches_hrside_random'));
saveas(gcf, fullfile(save_path,'switches_hrside_random'), 'fig');

L2_ranges = [0 4;5 9;10 14;15 30];
plot_binxaveragey_M2stimulation({dat(1).sw_LRVsChoice,dat(2).sw_LRVsChoice},tlabel, L2_ranges,save_path);

plot_binxaveragey_M2stimulation({dat(1).sw_LRVsMidpoint,dat(2).sw_LRVsMidpoint},tlabel, L2_ranges,save_path);

plot_binxaveragey_M2stimulation({dat(1).sw_LRVsRR,dat(2).sw_LRVsRR},tlabel, L2_ranges,save_path);

plot_binxaveragey_M2stimulation({dat(1).sw_LRVsHR,dat(2).sw_LRVsHR},tlabel, L2_ranges,save_path);

plot_binxaveragey_M2stimulation({dat(1).sw_LRVsWinStay,dat(2).sw_LRVsWinStay},tlabel, L2_ranges,save_path);

plot_binxaveragey_M2stimulation({dat(1).sw_LRVsLooseSwitch,dat(2).sw_LRVsLooseSwitch},tlabel, L2_ranges,save_path);

L2_ranges = [0 4;5 9;10 14;15 30];
tlabelBilateral   = [{'Ipsi'}; {'Contra'}];
plot_binxaveragey_bilateral_M2stimulation([{dat(1).sw_LRVsChoiceIpsiControl,dat(2).sw_LRVsChoiceIpsi};...
    {dat(1).sw_LRVsChoiceContraControl,dat(2).sw_LRVsChoiceContra}],tlabelBilateral, L2_ranges,save_path);

 plot_binxaveragey_bilateral_M2stimulation([{dat(1).sw_LRVsMidpointIpsiControl,dat(2).sw_LRVsMidpointIpsi};...
     {dat(1).sw_LRVsMidpointContraControl,dat(2).sw_LRVsMidpointContra}],tlabelBilateral, L2_ranges,save_path);

% %% Codes needs to be modified for these functions -
% plot_lickrate_byTrialType_M2stimulation(dat(1).lick_trType,dat(2).lick_trType);
% print(gcf,'-dpng',fullfile(save_path,'lickrates_byTrialType'));
% saveas(gcf, fullfile(save_path,'lickrates_byTrialType'), 'fig');
%
% plot_val_byTrialType_M2stimulation(dat(1).respTime_trType,dat(2).respTime_trType);
% print(gcf,'-dpng',fullfile(save_path,'rt_byTrialType'));
% saveas(gcf, fullfile(save_path,'rt_byTrialType'), 'fig');
%
% plot_val_byTrialType_M2stimulation(dat(1).iti_trType,dat(2).iti_trType);
% print(gcf,'-dpng',fullfile(save_path,'iti_byTrialType'));
% saveas(gcf, fullfile(save_path,'iti_byTrialType'), 'fig');


close all hidden
disp ( 'Figures saved.')

