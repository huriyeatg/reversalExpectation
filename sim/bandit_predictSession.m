function savebehfigpath = bandit_predictSession(dataIndex,exampleLogName,exampleAnimal,model_path,model_type)
% % bandit_predictSession %
%PURPOSE:   Take parameters extracted from model fitting to predict
%           behavior using experimental data
%AUTHORS:   AC Kwan 191212
%
%INPUT ARGUMENTS
%   dataIndex:      a database index table for the sessions to analyze
%   exampleLogName: name of the logfile to compare animal vs model
%   exampleAnimal:  name of the logfile to compare animal vs model
%   model_path:     path containing the .mat file for the models
%   model_type:     the name of the model, e.g. 'Q_RPE' or 'WSLS',
%                   corresponding to the .m file names
%
%OUTPUT ARGUMENTS
%   savebehfigpath: the path related to analysis figure for that session

%% load the animal data
idx = find(ismember(dataIndex.LogFileName,exampleLogName)==1);
load(fullfile(dataIndex.BehPath{idx},[dataIndex.LogFileName{idx}(1:end-4),'_beh.mat']));
   
trials = value_getTrialMasks(trialData);
stats = value_getTrialStats(trials, sessionData.nRules);
stats = value_getTrialStatsMore(stats);
stats = value_getTrialStatsModel( stats, model_type, model_path,exampleAnimal);

%% plot the animal vs simulated behaviors

n_plot = 100*floor(numel(stats.c)/100); %plot up to the nearest 100 trials
if n_plot > 1000   %for computer simulations, #trials can be too high to plot effectively
    n_plot = 1000;
end

% Create a subfolder to save the images for this session
% folder named using year/month/day of file
yr=num2str(sessionData.dateTime{1}(9:10));
mo=num2str(sessionData.dateTime{1}(1:2));
day=num2str(sessionData.dateTime{1}(4:5));
savebehfigpath = fullfile(dataIndex.BehPath{idx},[yr mo day]);

if ~exist(savebehfigpath,'dir')
    mkdir(savebehfigpath);
end

tlabel= ['Animal: ' exampleLogName];
tlabel2=[model_type];
plot_session_sim(stats,n_plot);

end

