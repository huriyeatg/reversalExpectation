function [trialDataCombined, trialsCombined, nRules] = merge_sessions_neuromodulator(dataIndex)
% % merge_sessions %
%PURPOSE:   Merge different sessions into one long session, with the gaps
%           between sessions filled with NaNs
%AUTHORS:   H Atilgan and AC Kwan 201002
%
%INPUT ARGUMENTS
%   dataIndex:    a database index table for the sessions to analyze
%
%OUTPUT ARGUMENTS
%   trialDataCombined:  the concatenated 'trialData' structure
%   trialsCombined:     the concatenated 'trials' structure
%   nRules:             number of sets of reward probabilities

n_nan = 20;   %insert this many NaN in each gap
fs = 20;
tWindow = fs*6; % for -2 to 4 sec - total 6 sec.
trialDataCombined = struct;
trialDataCombined.presCodeSet = 31;
trialsCombined = struct; 
for i=1:size(dataIndex,1)
    
    load(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']));
    trials = value_getTrialMasks(trialData);
    
    load(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_dff.mat']));
    trials.dffN_zscore = zscore(data.dffN(1:end-1,1:tWindow));
    trials.dffN = data.dffN(1:end-1,1:tWindow);
    trials.dff  = data.dff(1:end-1,1:tWindow);
    
    % check trial numbers across beh and dff files
    nTrials = min(numel(trials.go), size(trials.dffN,1));
    
    if i==1
        nRules = sessionData.nRules;
        fields=fieldnames(trialData);
        for j = 2:numel(fields)
            trialDataCombined.(fields{j}) = trialData.(fields{j})(1:nTrials,:);
        end
        
        fields=fieldnames(trials);
        for j = 1:numel(fields)
            trialsCombined.(fields{j}) = trials.(fields{j})(1:nTrials,:);
        end
        
    else
        fields=fieldnames(trialData);
        for j = 2:numel(fields)
            if iscell(trialDataCombined.(fields{j}))    %licktimes are stored in cells
                trialDataCombined.(fields{j}) = [trialDataCombined.(fields{j}); cell(n_nan,1); trialData.(fields{j})(1:nTrials)];
            else
                trialDataCombined.(fields{j}) = [trialDataCombined.(fields{j}); nan(n_nan,1); trialData.(fields{j})(1:nTrials)];
            end
        end
        
        fields=fieldnames(trials);
        for j = 1:numel(fields)-3
            trialsCombined.(fields{j}) = [trialsCombined.(fields{j}); nan(n_nan,1); trials.(fields{j})(1:nTrials)];
        end
        for j =numel(fields)-2: numel(fields)
        trialsCombined.(fields{j})  = [trialsCombined.(fields{j}); nan(n_nan,tWindow); trials.(fields{j})(1:nTrials,:)];
        end
        
        if nRules ~= sessionData.nRules
            error('Error in merge_sessions: the nRules for the sessions do not match');
        end
    end
    
    
end

end


