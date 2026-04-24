% Master file for analyzing effects of Cg1/M2 photostimulation on two-armed bandit task
% H Atilgan and AC Kwan,23/03/21
%
% To run this code:
% 1) Add all the subfolders to the Path in MATLAB
% 2) Change the variable 'root_path' below
%
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
dataIndex = makeDataIndex(logfilepath, analysispath,'bandit_R71_M2stimulation');% this folder only

% Parse and analyze each logfile, save as .mat files, as needed
dataIndex = createBehMatFiles(dataIndex);

% Determine if each session fulfill performance criteria
[dataIndex, ~] = determineBehCriteria(dataIndex); 

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - pre-choice stimulation
save_path = fullfile(figurepath,'M2stimulation-pre');
preSubset = (ismember(dataIndex.Phase,21)==1);    %post-stimulation

disp('--- M2stimulation-pre:');
%bandit_M2stimulationPerSession_vs3(dataIndex(preSubset,:),save_path);
bandit_M2stimulationPerSession_vs2(dataIndex(preSubset,:),save_path);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - post-choice stimulation
save_path = fullfile(figurepath,'M2stimulation-post');
postSubset = (ismember(dataIndex.Phase,22)==1);    %post-stimulation

disp('--- M2stimulation-post:');
%bandit_M2stimulationPerSession_vs3(dataIndex(postSubset,:),save_path);
bandit_M2stimulationPerSession_vs2(dataIndex(postSubset,:),save_path);
%% Large ANOVA
test_choiceinM2stimulation(dataIndex, figurepath)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Model fitting and comparison for lesion data
bandit_modelfitting_M2stimulation(dataIndex, root_path)


%%
close all hidden
