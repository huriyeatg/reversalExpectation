function bandit_neuromodulatorPerAnimal(dataIndex,save_path)
% % bandit_neuromodulatorPerAnimal %
%PURPOSE:   Analyze bandit behavior for each animal
%AUTHORS:   H Atilgan and AC Kwan 200210
%
%INPUT ARGUMENTS
%   dataIndex:    a table of the data files
%   save_path:    path for saving the plots
%
%OUTPUT ARGUMENTS
%

%% go through each session

animalList = unique(dataIndex.Animal);

for j = 1:numel(animalList)
    
    %which session belong to this one animal
    currAnimalSessions = ismember(dataIndex.Animal,animalList(j));
    savebehfigpath = fullfile(save_path,animalList{j});
    if ~exist(savebehfigpath,'dir')
        mkdir(savebehfigpath);
    end
    
    % order based on the date
    curr_dataIndex = dataIndex(currAnimalSessions,:);
    curr_dataIndex = sortrows(curr_dataIndex,'DateNumber','ascend');
    %concatenate the sessions for this one animal
    [trialData, trials, nRules] = merge_sessions_neuromodulator(curr_dataIndex);
    trialData.presCodeSet = 31;
    stats = value_getTrialStats(trials, nRules);
    stats = value_getTrialStatsMore(stats);
    
    tlabel = animalList{j};
    %% plot session
    plot_session_neuromodulator(stats,trials, tlabel,savebehfigpath)    
    
    %analyze_session_expt(trialData,trials,tlabel,savebehfigpath);
    
end
