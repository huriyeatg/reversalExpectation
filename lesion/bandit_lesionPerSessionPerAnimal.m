function bandit_lesionPerSessionPerAnimal(dataIndex,save_path)
% % bandit_lesionPerSessionPerAnimal %
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
        currAnimalSessions = ismember(dataIndex.Animal,animalList(j))& currList==1;
        
        %concatenate the sessions for this one animal
        [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));
        trialData.presCodeSet = trialData.presCodeSet(1);
        % if unilateral lesion data:
        % leave right lesion data alone and flip choice vector for left lesion data,
        % so effectively left == side contralateral to an unilateral lesion
        ind =find(currAnimalSessions==1);
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
            
    end
end


if uniLesion == false  %this plot only makes sense for bilateral/saline, not appropriate for unilateral lesion
   L2_ranges = [0 4;5 9;10 14;15 30];
   tlabel=[{'Both side'}];
   plot_binxaveragey_lesion({dat(1).sw_LRVsMidpoint,dat(2).sw_LRVsMidpoint},tlabel, L2_ranges,save_path);
    
else
    L2_ranges = [0 4;5 9;10 14;15 30];
    tlabel    = [{'Intact'}; {'Lesion'}];
    plot_binxaveragey_lesion([{dat(1).sw_LRVsMidpointL,dat(2).sw_LRVsMidpointL};...
        {dat(1).sw_LRVsMidpointR,dat(2).sw_LRVsMidpointR}],tlabel, L2_ranges,save_path);
end


close all hidden;

