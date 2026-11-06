function [ stats ] = value_getTrialStatsModel( stats_input, model_type, model_path,animalID,ps )
% % value_getTrialStatsMore %
%PURPOSE:   Process 'stats to add more fields to data structure: 'stats'
%           containing dummy codes for task variables
%AUTHORS:   H Atilgan and AC Kwan 200428
%
%INPUT ARGUMENTS
%   stats_iput: Structure containing previously calculated stats value.
%               (e.g., trial.c = -1 for left, 1 for right)
%  model_type:  String of model name
%  model_path:  Path for saved-model fitting values.
%  animalID:    Animal ID is necessary to fit the data.
%  ps:        - struct with few fields;
%     ps.params: model fitting params ( either animal ID or params is
%               required)
%     ps.simulation: if stats_input is simulated data, and does not need
%     simultion again, needs to be 'true'
%
%OUTPUT VARIABLES
%   stats:      Structure containing fields, each dummy-coded
%               tabulating the animal's choice, but more!
%               (e.g., trial.c = -1 for left, 1 for right)

if nargin==5
    fit_params = ps.params;
else
    ps =[];
end
stats = stats_input;
%% get fitted model parameters and calculate latent variables
if ~isfield(ps, 'simulation') % load best fitted values
    load(fullfile(model_path,[model_type '.mat']));
    if ~isempty (animalID)
        idx_sim = find(ismember(animal,animalID)==1);
        player_fit.label=['algo_' model_type];
        player_fit.params=fitpar{idx_sim};
    elseif ~isempty (fit_params)
        player_fit =['algo_' model_type];
        player_fit.params= fit_params;
    end
    stats_sim = predictAgent(player_fit,stats);
    
    f_sim = fieldnames(stats_sim);
    f     = fieldnames(stats);
    for kk = 1:length(f_sim)
        if ~ismember(f_sim(kk), f)
            stats.(f_sim{kk}) = stats_sim.(f_sim{kk});
        end
    end
    
elseif isfield(ps, 'simulation')  % simulated data
    % no need to simulate, latent variables already exist in stats.
    player_fit.label=['algo_' model_type];
    player_fit.params=fit_params; 
end

%% changePoint analysis: look for a robust change in animal choice
% behavior in the session based on hazard rate
if strcmp(model_type, 'belief_CK_bilateral') % for bilateral model fitting, there are two hazard rate.
    hazardrate = mean( [player_fit.params(1) player_fit.params(2)]);
elseif strcmp(model_type, 'belief_CK') 
    hazardrate = player_fit.params(1);
else
    hazardrate = 1/11; % true value
end
stats.hazardrate = hazardrate;
% calculate change-point-probability based on dataLL and fitted hazard rate:
Q = (.5.*hazardrate)./( exp(stats.dataLL).*(1-hazardrate));
stats.CPP = Q./(1+Q);
% % Get CPP peaks
% if hazardrate == 0
%     warning ( 'Hazard rate is equal to zero!')
%     [ ~, index ] = findpeaks(stats.CPP,'MinPeakDistance',1);
% else
%     [ ~, index ] = findpeaks(stats.CPP,'MinPeakDistance',ceil(1/hazardrate));
% end
 stats.CPP_peaks = nan(size(stats.r)); % create nans
% stats.CPP_peaks(index) = 1; % detected change points
% 
%% go through each block, identify transition, trial to performance,
%  and random number added
trials_back =10;

stats.blockChangePointDetection = nan(numel(stats.blockLength)-1,1);
stats.blockChangePointMeanProbability = nan(numel(stats.blockLength)-1,1);
stats.meanLearningRate = nan(numel(stats.blockLength)-1,1);
stats.meanTrialEntropy = nan(numel(stats.blockLength)-1,1);
stats.trials2changepoint = nan(numel(stats.blockLength)-1,1);
stats.switchLearningRate = cell(numel(stats.blockLength)-1,1);
stats.switchCPP          = cell(numel(stats.blockLength)-1,1);
stats.switchRU           = cell(numel(stats.blockLength)-1,1);
stats.meanLearningRate_last5trials = nan(numel(stats.blockLength)-1,1);

for i=1:numel(stats.blockLength)-1
    if ~isnan(stats.blockTrans(i))
        %sum change point porbability & number of detected change points
        %within each block
        index_block = (sum(stats.blockLength(1:i-1))+1):(sum(stats.blockLength(1:i)));
        stats.blockChangePointMeanProbability(i) = nanmedian(stats.CPP(index_block));
%        stats.blockChangePointDetection(i) = nansum(stats.CPP_peaks(index_block));
        stats.meanLearningRate(i) = nanmean(stats.lr(index_block));
        stats.meanTrialEntropy(i) = nanmean(stats.trialEntropy(index_block));
        
        trialStart = (sum(stats.blockLength(1:i))+1);
        if (trialStart-trials_back)>1 &&  (trialStart+trials_back)<numel(stats.lr)
            stats.switchLearningRate{i} = stats.lr(trialStart-trials_back:trialStart+trials_back);
            stats.switchCPP{i} = stats.CPP(trialStart-trials_back:trialStart+trials_back);
            stats.switchRU{i} = stats.trialEntropy(trialStart-trials_back:trialStart+trials_back);
            stats.meanLearningRate_last5trials(i) = nanmean(stats.lr(trialStart-5:trialStart));
        else
            stats.switchLearningRate{i} = nan(1+2*trials_back,1);
            stats.switchCPP{i} = nan(1+2*trials_back,1);
            stats.switchRU{i} = nan(1+2*trials_back,1);
            stats.meanLearningRate_last5trials(i) = nan;
        end
        
        % trials to reach change points - next block
        index_block = (sum(stats.blockLength(1:i))+1):(sum(stats.blockLength(1:(i+1)))); % next block
%        idx = find(stats.CPP_peaks(index_block)==1,1, 'first');
        %idx = find(stats.CPP(index_block)>prctile(stats.CPP(index_block),75),1, 'first');
%         if ~isempty(idx)
%             stats.trials2changepoint(i) = idx; % For next block
%         end
    end
    
end
%stats.blockChangePointDetection(stats.blockChangePointDetection==0)=nan;

end