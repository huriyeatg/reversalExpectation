function bandit_M2stimulationPerBlock(dataIndex,save_path)
% % bandit_M2stimulationPerBlock %
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
    stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);               %reward probabilities / rule associated with each block
    stats.blockStRegion = stats.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);   %reward probabilities / rule associated with each block
    %% now divide the blocks for stimulated vs non-stimulated
    nTrials = numel(stats.c);
    nBlocks = numel(stats.blockRule);
    statsOriginal = stats; % stats will be used below function
    
    for kk = 1:3 % 3 different block switch type
        if kk==1  % control--> control
            tempNextBlock  = [statsOriginal.blockSt(2:end);nan];
            blockInd = statsOriginal.blockSt==0 & tempNextBlock==0;
        elseif  kk==2   % stimulated to control block
            tempNextBlock  = [statsOriginal.blockSt(2:end);nan];
            blockInd = statsOriginal.blockSt==1 & tempNextBlock==0;
        elseif  kk==3   % control to stimulated block
            tempNextBlock  = [statsOriginal.blockSt(2:end);nan];
            blockInd = statsOriginal.blockSt==0 & tempNextBlock==1;
        end
        clear stats trialDataSelected
        stats = statsOriginal;
        stats.blockSelected = blockInd;
        
        %% plot choice behavior - around switches initially better to worse
        trials_back=10;  % set number of previous trials
           dat(kk).sw_output{j}=choice_switchSelected(stats,trials_back);
        dat(kk).sw_hrside_output{j}=choice_switch_hrsideSelected(stats,trials_back);
        
        %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
        L1_ranges=[10 20;10 20;10 20;10 20]; %consider only subset of blocks within the range, for trials to criterion
        L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials      
        dat(kk).sw_hrside_random_output{j}=choice_switch_hrside_randomSelected(stats,trials_back,L1_ranges,L2_ranges);
        
        %% initial better option plots
        ind = stats.blockTrialtoCrit<=20 & stats.blockSelected(1:end-1)==1;
        dat(kk).sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoice{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoice{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsChoice{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoice{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
        %% More stats on block patterns: RR, HR, Pwin, Ploose
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
        dat(kk).sw_LRVsWinStay{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStay{j}.label{2}={'P(stay|win)'};
        
        dat(kk).sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitch{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitch{j}.range{2}=[0 0.3];
        dat(kk).sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitch{j}.label{2}={'P(lose|switch)'};
        
        
    end
end

%%

tlabel='Stimulated to control blocks';
%plot_switch_lesion(dat(1).sw_output,dat(2).sw_output,tlabel,stats.rule_labels);

plot_switch_hrside_M2stimulation(dat(1).sw_hrside_output,dat(2).sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,[tlabel,'_switches_hrside']));
saveas(gcf, fullfile(save_path,[tlabel,'_switches_hrside']), 'fig');

plot_switch_hrside_random_M2stimulation(dat(1).sw_hrside_random_output,dat(2).sw_hrside_random_output,tlabel);
legend off
print(gcf,'-dpng',fullfile(save_path,[tlabel,'switches_random']));
saveas(gcf, fullfile(save_path,[tlabel,'switches_random']), 'fig');

tlabel='Control to stimulated blocks';
plot_switch_hrside_M2stimulation(dat(1).sw_hrside_output,dat(3).sw_hrside_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,[tlabel,'_switches_hrside']));
saveas(gcf, fullfile(save_path,[tlabel,'_switches_hrside']), 'fig');

plot_switch_hrside_random_M2stimulation(dat(1).sw_hrside_random_output,dat(3).sw_hrside_random_output,tlabel);
legend off
print(gcf,'-dpng',fullfile(save_path,[tlabel,'switches_random']));
saveas(gcf, fullfile(save_path,[tlabel,'switches_random']), 'fig');


L2_ranges = [0 4;5 9;10 14;15 30];
tlabel = 'Stimulation';
plot_binxaveragey_bilateral_M2stimulation({dat(1).sw_LRVsChoice,dat(2).sw_LRVsChoice, dat(3).sw_LRVsChoice},tlabel, L2_ranges,save_path);

plot_binxaveragey_bilateral_M2stimulation({dat(1).sw_LRVsWinStay,dat(2).sw_LRVsWinStay,dat(3).sw_LRVsWinStay},tlabel, L2_ranges,save_path);

plot_binxaveragey_bilateral_M2stimulation({dat(1).sw_LRVsLooseSwitch,dat(2).sw_LRVsLooseSwitch,dat(3).sw_LRVsLooseSwitch},tlabel, L2_ranges,save_path);





