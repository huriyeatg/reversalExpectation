function plot_photoStimulationSwitchPlots (stats, save_path)

nCrit =200;
j =1;

x0=stats.rule';
stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
%% now divide the blocks for stimulated vs non-stimulated
nTrials = numel(stats.c);
nBlocks = size(stats.blockRule,1);

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
        
    %% plot choice behavior - around switches left to right
    trials_back=10;  % set number of previous trials
    
    dat(kk).sw_output{j}=choice_switch(stats,trials_back);
    
    %% plot choice behavior - around switch high-probability side to low-probability side
    dat(kk).sw_hrside_output{j}=choice_switch_hrside(stats,trials_back);
    
      %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
        L1_ranges=[10 nCrit;10 nCrit;10 nCrit;10 nCrit]; %consider only subset of blocks within the range, for trials to criterion
        L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials

        dat(kk).sw_hrside_random_output{j}=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);


end

%% Visualise data
tlabel = ['Stimulation']

plot_switch_hrside_M2stimulation(dat(1).sw_hrside_output,dat(2).sw_hrside_output,tlabel);
print(gcf,'-dsvg',fullfile(save_path,'switches_hrside_noerrorbar'));
saveas(gcf, fullfile(save_path,'switches_hrside'), 'fig');

plot_switch_hrside_random_M2stimulation(dat(1).sw_hrside_random_output,dat(2).sw_hrside_random_output,tlabel);
legend off
print(gcf,'-dsvg',fullfile(save_path,'switches_hrside_random'));
saveas(gcf, fullfile(save_path,'switches_hrside_random_bilateral'), 'fig');




