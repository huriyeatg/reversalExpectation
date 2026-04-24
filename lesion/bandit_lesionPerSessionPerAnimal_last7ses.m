function bandit_lesionPerSessionPerAnimal_last7ses(dataIndex,save_path)
% % bandit_lesionPerSessionPerAnimal_last7ses %
%PURPOSE:   Analyze bandit behavior for each animal, comparing
%           pre- versus post-lesion
%AUTHORS:   H Atilgan and AC Kwan 191208
%
%INPUT ARGUMENTS
%   lesionList:   vector corresponding to dataIndex, with NaN if pre-lesion, 1 if post-lesion
%   save_path:    path for saving the plots
%
%OUTPUT ARGUMENTS
%

%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each session

preLesionList = find(isnan(dataIndex.Lesioned));
postLesionList = find(~isnan(dataIndex.Lesioned));

disp('-----------------------------------------------------------');
disp(['--- Analyzing - ', int2str(numel(preLesionList)) ' pre- and '...
    int2str(numel(postLesionList)) ' post-lesion sessions']);
disp('-----------------------------------------------------------');

%%
uniLesion = false;   %was this an analysis involved unilateral lesion?
animalList = unique(dataIndex.Animal);
for kk = 1:2
    if kk==1     %pre-lesion sessions
        currList = isnan(dataIndex.Lesioned);
    elseif kk==2 %post-lesion sessions
        currList = ~isnan(dataIndex.Lesioned);
    end
    
    for j = 1:numel(animalList)
        
        %which session belong to this one animal
        currAnimalSessions = find(ismember(dataIndex.Animal,animalList(j))& currList==1);
        
if k==1
        %concatenate the sessions for this one animal
        [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));
else
        %concatenate the sessions for this one animal
        [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions(end-10:end),:));
end

        trialData.presCodeSet = trialData.presCodeSet(1);
        % if unilateral lesion data:
        % leave right lesion data alone and flip choice vector for left lesion data,
        % so effectively left == side contralateral to an unilateral lesion
        ind =currAnimalSessions(1);
        if dataIndex.LesionSide(ind(1))==1  %if lesioned side is LEFT
            trialData = fliptrialData(trialData);
            uniLesion = true;  %set flag that we have flipped a subset of data
        end
        
        trials = value_getTrialMasks(trialData);
        stats = value_getTrialStats(trials, nRules);
        stats = value_getTrialStatsMore(stats);
        %% plot basic behavioral performance
        dat(kk).beh_output{j}=beh_performance(stats);
        
        %% plot choice behavior - around switches left to right
        trials_back=10;  % set number of previous trials
        
        dat(kk).sw_output{j}=choice_switch(stats,trials_back);
        
        %% plot choice behavior - around switch high-probability side to low-probability side
        dat(kk).sw_hrside_output{j}=choice_switch_hrside(stats,trials_back);

        %% plot choice behavior - around switches left to right, as a function of the statistics of the block preceding the switch
        L1_ranges=[10 20;10 20;10 20;10 20]; %consider only subset of blocks within the range, for trials to criterion
        L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials
        dat(kk).sw_random_output{j}=choice_switch_random(stats,trials_back,L1_ranges,L2_ranges);
        
        dat(kk).sw_hrside_random_output{j}=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);
        
        %% initial better option plots
        ind = stats.blockTrialtoCrit<20;
        dat(kk).sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoice{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoice{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsChoice{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoice{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
        ind = stats.blockTrialtoCrit<20 & stats.blockTrans==1; % no criteria
        dat(kk).sw_LRVsChoiceL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceL{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceL{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsChoiceL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceL{j}.label{2}={'Fraction of trials'};%'selecting initial  better option'};
        
        ind = stats.blockTrialtoCrit<20 & stats.blockTrans==2;
        dat(kk).sw_LRVsChoiceR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
        dat(kk).sw_LRVsChoiceR{j}.range{1}=[1 30];
        dat(kk).sw_LRVsChoiceR{j}.range{2}=[0.7 1];
        dat(kk).sw_LRVsChoiceR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChoiceR{j}.label{2}={'Fraction of trials'};%;'selecting initial better option'};
        
        
        %% More stats on block patterns
        ind = stats.blockTrialtoCrit<20;
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
        dat(kk).sw_LRVsWinStay{j}.label{2}={'P(win|stay)'};
        
        dat(kk).sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitch{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitch{j}.range{2}=[0.1 0.3];
        dat(kk).sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitch{j}.label{2}={'P(lose|switch)'};
        
        %% More stats on block patterns - Left
        ind = stats.blockTrialtoCrit<20 & stats.blockTrans==1; % no criteria
        dat(kk).sw_LRVsRRL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRRL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRRL{j}.range{2}=[45 65];
        dat(kk).sw_LRVsRRL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRRL{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHRL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHRL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHRL{j}.range{2}=[70 90];
        dat(kk).sw_LRVsHRL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHRL{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStayL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStayL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStayL{j}.range{2}=[0.9 1];
        dat(kk).sw_LRVsWinStayL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStayL{j}.label{2}={'P(win|stay)'};
        
        dat(kk).sw_LRVsLooseSwitchL{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitchL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitchL{j}.range{2}=[0.1 0.3];
        dat(kk).sw_LRVsLooseSwitchL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitchL{j}.label{2}={'P(lose|switch)'};
        
        
        %% More stats on block patterns - Right
        ind = stats.blockTrialtoCrit<20 & stats.blockTrans==2; % no criteria
        dat(kk).sw_LRVsRRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
        dat(kk).sw_LRVsRRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsRRR{j}.range{2}=[45 65];
        dat(kk).sw_LRVsRRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsRRR{j}.label{2}={'Reward rates (%)'};
        
        dat(kk).sw_LRVsHRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
        dat(kk).sw_LRVsHRR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsHRR{j}.range{2}=[70 90];
        dat(kk).sw_LRVsHRR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsHRR{j}.label{2}={'Hit rates (%)'};
        
        dat(kk).sw_LRVsWinStayR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
        dat(kk).sw_LRVsWinStayR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsWinStayR{j}.range{2}=[0.9 1];
        dat(kk).sw_LRVsWinStayR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsWinStayR{j}.label{2}={'P(win|stay)'};
        
        dat(kk).sw_LRVsLooseSwitchR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
        dat(kk).sw_LRVsLooseSwitchR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsLooseSwitchR{j}.range{2}=[0.1 0.3];
        dat(kk).sw_LRVsLooseSwitchR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsLooseSwitchR{j}.label{2}={'P(lose|switch)'};
          
        %% Slope / MidPoint
        L1_ranges=[10 20;10 20;10 20;10 20];   % BehCriteria
        L2_ranges=[0 4;5 9;10 14;15 30];       % Random Block length steps
        
        sw_stats_output{j} = choice_switch_stats_random_lesion(stats, trials_back,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpointL{j}.dat=[L2_ranges(:,1) squeeze(sw_stats_output{j}.statl(1,1,:))];
        dat(kk).sw_LRVsMidpointL{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointL{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointL{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointL{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
        dat(kk).sw_LRVsMidpointR{j}.dat=[L2_ranges(:,1) squeeze(sw_stats_output{j}.statr(1,2,:))];
        dat(kk).sw_LRVsMidpointR{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpointR{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpointR{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpointR{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
        L2_ranges=[1:2:30 ; 3:2:32]';        % Random Block length steps
        L1_ranges= ones(size(L2_ranges,1),2).*[10 20];    % BehCriteria
        sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,L1_ranges,L2_ranges);
        
        dat(kk).sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
        dat(kk).sw_LRVsMidpoint{j}.range{1}=[0 30];
        dat(kk).sw_LRVsMidpoint{j}.range{2}=[0 10];
        dat(kk).sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'}];%;{'fraction of trials = 0.5'}];
        
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


tlabel = [int2str(numel(preLesionList)) ' pre- and ' int2str(numel(postLesionList)) ' post-lesion sessions'];

plot_behperf_lesion(dat(1).beh_output,dat(2).beh_output,tlabel);
print(gcf,'-dpng',fullfile(save_path,'behperf'));
saveas(gcf, fullfile(save_path,'behperf'), 'fig');

plot_switch_lesion(dat(1).sw_output,dat(2).sw_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches'));
saveas(gcf, fullfile(save_path,'switches'), 'fig');

plot_switch_random_lesion(dat(1).sw_random_output,dat(2).sw_random_output,tlabel,stats.rule_labels);
print(gcf,'-dpng',fullfile(save_path,'switches_random'));
saveas(gcf, fullfile(save_path,'switches_random'), 'fig');

if uniLesion == true   %these plots make more sense for unilateral lesion
    L2_ranges = [0 4;5 9;10 14;15 30];
    tlabel    = [{'Intact'}; {'Lesion'}];
    plot_binxaveragey_lesion([{dat(1).sw_LRVsChoiceL,dat(2).sw_LRVsChoiceL};...
        {dat(1).sw_LRVsChoiceR,dat(2).sw_LRVsChoiceR}],tlabel, L2_ranges,save_path);

    plot_binxaveragey_lesion([{dat(1).sw_LRVsRRL,dat(2).sw_LRVsRRL};...
        {dat(1).sw_LRVsRRR,dat(2).sw_LRVsRRR}],tlabel, L2_ranges,save_path);
   
    plot_binxaveragey_lesion([{dat(1).sw_LRVsHRL,dat(2).sw_LRVsHRL};...
        {dat(1).sw_LRVsHRR,dat(2).sw_LRVsHRR}],tlabel, L2_ranges,save_path);

    plot_binxaveragey_lesion([{dat(1).sw_LRVsWinStayL,dat(2).sw_LRVsWinStayL};...
        {dat(1).sw_LRVsWinStayR,dat(2).sw_LRVsWinStayR}],tlabel, L2_ranges,save_path);
 
    plot_binxaveragey_lesion([{dat(1).sw_LRVsLooseSwitchL,dat(2).sw_LRVsLooseSwitchL};...
        {dat(1).sw_LRVsLooseSwitchR,dat(2).sw_LRVsLooseSwitchR}],tlabel, L2_ranges,save_path);

    plot_binxaveragey_lesion([{dat(1).sw_LRVsMidpointL,dat(2).sw_LRVsMidpointL};...
        {dat(1).sw_LRVsMidpointR,dat(2).sw_LRVsMidpointR}],tlabel, L2_ranges,save_path);

end

if uniLesion == false  %this plot only makes sense for bilateral/saline, not appropriate for unilateral lesion
    plot_switch_hrside_lesion(dat(1).sw_hrside_output,dat(2).sw_hrside_output,tlabel);
    print(gcf,'-dpng',fullfile(save_path,'switches_hrside'));
    saveas(gcf, fullfile(save_path,'switches_hrside'), 'fig');
    
    plot_switch_hrside_random_lesion(dat(1).sw_hrside_random_output,dat(2).sw_hrside_random_output,tlabel);
    print(gcf,'-dpng',fullfile(save_path,'switches_hrside_random'));
    saveas(gcf, fullfile(save_path,'switches_hrside_random'), 'fig');
    
    L2_ranges = [0 4;5 9;10 14;15 30];
    plot_binxaveragey_lesion({dat(1).sw_LRVsChoice,dat(2).sw_LRVsChoice},tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion({dat(1).sw_LRVsRR,dat(2).sw_LRVsRR},tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion({dat(1).sw_LRVsHR,dat(2).sw_LRVsHR},tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion({dat(1).sw_LRVsWinStay,dat(2).sw_LRVsWinStay},tlabel, L2_ranges,save_path);

    plot_binxaveragey_lesion({dat(1).sw_LRVsLooseSwitch,dat(2).sw_LRVsLooseSwitch},tlabel, L2_ranges,save_path);

    plot_binxaveragey_lesion({dat(1).sw_LRVsMidpoint,dat(2).sw_LRVsMidpoint},tlabel, L2_ranges,save_path);

    
end

plot_lickrate_byTrialType_lesion(dat(1).lick_trType,dat(2).lick_trType);
print(gcf,'-dpng',fullfile(save_path,'lickrates_byTrialType'));
saveas(gcf, fullfile(save_path,'lickrates_byTrialType'), 'fig');

plot_val_byTrialType_lesion(dat(1).respTime_trType,dat(2).respTime_trType);
print(gcf,'-dpng',fullfile(save_path,'rt_byTrialType'));
saveas(gcf, fullfile(save_path,'rt_byTrialType'), 'fig');

plot_val_byTrialType_lesion(dat(1).iti_trType,dat(2).iti_trType);
print(gcf,'-dpng',fullfile(save_path,'iti_byTrialType'));
saveas(gcf, fullfile(save_path,'iti_byTrialType'), 'fig');


close all hidden
disp ( 'Figures saved.')


