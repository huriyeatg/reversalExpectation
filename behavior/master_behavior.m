% Master file for analyzing the behavioral outcome for two-armed bandit task
% A Kwan & H Atilgan 060421
%
% To run this code:
% 1) Add all the subfolders to the Path in MATLAB
% 2) Add your computer & path to setup_compprop.m file - Change the
% variable 'root_path' accordingly
%
% List of unresolved issues:
% 1) Why do the response times have two peaks?
% 2) Some switches seem to occur following less than 10 choices on the high-probability side?
% 3) Why is there an excess of L1=10 in the switches_random_distn plot?
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
analysispath = fullfile(root_path,'analysis');
figurepath = fullfile(root_path,'figs');
dataIndex = makeDataIndex(logfilepath, analysispath,'bandit_R71_lesion');% this folder only

% Parse and analyze each logfile, save as .mat files, as needed
dataIndex = createBehMatFiles(dataIndex);

% Add information about lesion
% Since the first set of behavior data collected under lesion project, 
% there are many animals with lesion data - to exclude this data, we need
% lesion information. 
dataIndex = addIndexLesion(dataIndex); 

% Determine if each session fulfill performance criteria
[dataIndex, ~] = determineBehCriteria(dataIndex); 

% Create table for manuscript: per Animal, total session, total switch etc 
InfoTable = makeAnimalInfoTable(dataIndex);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - Example session

exampleLogName = '19107_phase3_R71NoCue_1904051629.log';
idx = find(ismember(dataIndex.LogFileName,exampleLogName)==1);

bandit_session(dataIndex.BehPath{idx},dataIndex.LogFileName{idx},figurepath);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - Reversal, all control/pre-lesion animals

normalSubset = isnan(dataIndex.Lesioned);  %control or pre-lesion

save_path = fullfile(figurepath,'naive-summary-per-animal');
bandit_behaviorPerAnimal(dataIndex(normalSubset,:),save_path);

save_path = fullfile(figurepath,'naive-summary-per-session');
bandit_behaviorPerSession(dataIndex(normalSubset,:),save_path); % few analysis specific that might have a different pattern if averaged across sessions (not include all analysis)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Model fitting and comparison for naive data

% For session fitting it neeeds to be all sessions - including lesion
% session - that's why it was runned here.
simPath.model_mat        = fullfile(root_path,'sim','mat_models');
bandit_fittingPerSession(dataIndex,simPath.model_mat); % for all sessions 

normalSubset = isnan(dataIndex.Lesioned); 
bandit_modelfitting(dataIndex(normalSubset,:), root_path); % for non-lesion data



