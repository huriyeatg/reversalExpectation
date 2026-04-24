function bandit_M2stimulationPerSession_vs3(dataIndex,save_path)
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
 %concatenate the sessions for all animals
[trialData, trials, nRules] = merge_sessions(dataIndex);
stats = value_getTrialStats(trials, nRules);
stats = value_getTrialStatsMore(stats);
%stimulation
stats.stRegion = trialData.stimulationRegion-1000;
stats.st = nan(numel(trialData.stimulation),1);
stats.st = stats.stRegion; %stRegion=1 means right, stRegion 2 means left; c = -1 means left; c = 1 means right
stats.st (stats.stRegion==2) = 1;

% which block stimulated
x0=stats.rule';
stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
stats.blockStRegion = stats.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
stats.blockHrSide = stats.hr_side([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block

% Params
nCrit = 200;
trials_back=10;  % set number of previous trials
L1_ranges=[10 nCrit;10 nCrit;10 nCrit;10 nCrit]; %consider only subset of blocks within the range, for trials to criterion
L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials

%% IPSI BLOCKS
% TO REMINDER: stats.stRegion(k)==1 means Right stimulated
%if stats.st(k)==1 && stats.stRegion(k)==2 && stats.c(k)==-1 % ipsilateral side Left stimulation & left choice

indIpsi = [find(stats.blockStRegion == 1 & stats.blockHrSide == 1); find(stats.blockStRegion == 2 & stats.blockHrSide ==-1)];% get index for blocks for control:
stats.selectedBlocks = nan(numel(stats.blockStRegion),1);
stats.selectedBlocks(indIpsi) =1;
sw_hrside_random_output_ipsi{1} = choice_switch_hrside_random_opto(stats,trials_back,L1_ranges,L2_ranges);

% control blocks
stats.selectedBlocks = [stats.selectedBlocks(3:end);nan;nan];% get index for blocks for control:
sw_hrside_random_output_control{1} = choice_switch_hrside_random_opto(stats,trials_back,L1_ranges,L2_ranges);

% plot the figure
tlabel = ' Pre';
plot_switch_hrside_random_M2stimulation(sw_hrside_random_output_control,sw_hrside_random_output_ipsi,tlabel);
legend off
print(gcf,'-dsvg',fullfile(save_path,'switches_hrside_random_ipsi'));
saveas(gcf, fullfile(save_path,'switches_hrside_random_ipsi'), 'fig');

%% CONTA BLOCKS
indContra =[find(stats.blockStRegion == 1 & stats.blockHrSide == -1); find(stats.blockStRegion == 2 & stats.blockHrSide ==1)];% get index for blocks for control:
stats.selectedBlocks = nan(numel(stats.stRegion),1);
stats.selectedBlocks(indContra) = 1;
sw_hrside_random_output_contra{1} = choice_switch_hrside_random_opto(stats,trials_back,L1_ranges,L2_ranges);

% control blocks
stats.selectedBlocks = [stats.selectedBlocks(3:end);nan;nan];% get index for blocks for control:
sw_hrside_random_output_control{1} = choice_switch_hrside_random_opto(stats,trials_back,L1_ranges,L2_ranges);

% plot the figure
tlabel = ' Pre';
plot_switch_hrside_random_M2stimulation(sw_hrside_random_output_control,sw_hrside_random_output_contra,tlabel);
legend off
print(gcf,'-dsvg',fullfile(save_path,'switches_hrside_random_contra'));
saveas(gcf, fullfile(save_path,'switches_hrside_random_contra'), 'fig');
