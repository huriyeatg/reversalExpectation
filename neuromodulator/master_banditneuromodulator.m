% Master file for analyzing the behavioral outcome for two-armed bandit task
% while recording with 1photon miniscope from prelimbic cortex
clc
clearvars;
close all;
setup_figprop;
setup_compprop; % add your computer settings here

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Preprocessing: parse each log file and tabulate the data set

disp('-----------------------------------------------------------');
disp('--- HERE WE GO ');
disp('-----------------------------------------------------------');

% Look for data files and create a database index
logfilepath = fullfile(root_path,'data','data-behavior');
nmfilepath = fullfile(root_path,'data','data-miniscope');
analysispath = fullfile(root_path,'analysis');
figurepath = fullfile(root_path,'figs');
dataIndex = makeDataIndex(logfilepath, analysispath,'bandit_neuromodulator');% this folder only
% Parse and analyze each logfile, save as .mat files, as needed
dataIndex = createBehMatFiles(dataIndex);

% Add information about neural data
dataIndex = addIndexNeuromodulator(dataIndex,nmfilepath); 

% Parse the signal for trials
dataIndex = creatDffMatFiles_miniscope(dataIndex); 
nansum(dataIndex.dffCreated)

%% Per session analysis - For NE signal
save_path = fullfile(figurepath,'neuromodulator-per-session-ne');
disp('--- Bandit task, NE signal');
criteriaSubset = (ismember(dataIndex.Phase,31)==1)& dataIndex.experiment==1 &~isnan(dataIndex.dffCreated) ; % sessions with 
bandit_neuromodulatorPerSession(dataIndex(criteriaSubset,:),save_path);

%% plot whole animal session - concatenate and session order is important
save_path = fullfile(figurepath,'neuromodulator-per-animal');

disp('--- Whole behaviour with all neuromodulator signal');
criteriaSubset = (ismember(dataIndex.Phase,31)==1)&~isnan(dataIndex.dffCreated) ; % sessions with 

bandit_neuromodulatorPerAnimal(dataIndex(criteriaSubset,:),save_path);



%% 

