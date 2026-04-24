function bandit_session(BehPath,LogFileName, FigPath)
% % bandit_session %
%PURPOSE:   Preparing to analyze a single session of mouse behavior
%AUTHORS:   H Atilgan and AC Kwan 191203
%
%INPUT ARGUMENTS
%   BehPath:        path for the location of the analysis folder containing
%                   the behavioral .mat file
%   LogFileName:    name of the logfile
%   FigPath:        path to save figures
%
%OUTPUT ARGUMENTS
%

%% load the behavioral data

disp('-----------------------------------------------------------');
disp('--- Analyzing a single behavioral session ');
disp('-----------------------------------------------------------');
disp(['Loading ' LogFileName]);

load(fullfile(BehPath,[LogFileName(1:end-4),'_beh.mat']));

% Get trial information
trials = value_getTrialMasks(trialData);
stats = value_getTrialStats(trials,sessionData.nRules);
stats = value_getTrialStatsMore(stats);

% What to put as title for some of the figures generated
tlabel=strcat('Subject=',sessionData.subject{1},', Time=',sessionData.dateTime(1),'-',sessionData.dateTime(2));

% Create a subfolder to save the images for this session
% folder named using year/month/day of file
yr=num2str(sessionData.dateTime{1}(9:10));
mo=num2str(sessionData.dateTime{1}(1:2));
day=num2str(sessionData.dateTime{1}(4:5));
savebehfigpath = fullfile(FigPath,['sample-session-', yr mo day]);

analyze_session(stats,tlabel,savebehfigpath);

analyze_session_expt(trialData,trials,tlabel,savebehfigpath);

end