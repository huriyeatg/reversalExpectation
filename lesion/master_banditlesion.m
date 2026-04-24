% Master file for analyzing effects of Cg1/M2 lesion on two-armed bandit task
% H Atilgan and AC Kwan, 04/30/20
%
% To run this code:
% 1) Add all the subfolders to the Path in MATLAB
% 2) Change the variable 'root_path' below
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
% Future - Add information about pupil, imaging, etc.
dataIndex = addIndexLesion(dataIndex); 

% Determine if each session fulfill performance criteria
[dataIndex, ~] = determineBehCriteria(dataIndex); 

% Create table for manuscript: per Animal, total session, total switch etc 
InfoTable = makeAnimalInfoTable_lesion(dataIndex);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - Reversal, unilateral lesion
 
save_path = fullfile(figurepath,'lesion-uni-per-session');
uniSubset = (ismember(dataIndex.LesionSide,[1 2])==1);  %unilateral lesion

disp('--- Unilateral lesions:');
bandit_lesionPerSession(dataIndex(uniSubset ,:),save_path);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - Reversal, bilateral lesion

save_path = fullfile(figurepath,'lesion-bi-per-session');
biSubset = (ismember(dataIndex.LesionSide,[3])==1);    %bilateral lesion

disp('--- Bilateral lesions:');
bandit_lesionPerSession(dataIndex(biSubset ,:),save_path);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Behavior - Reversal, saline control

save_path = fullfile(figurepath,'lesion-saline-per-session');
salineSubset = (ismember(dataIndex.LesionSide,[4])==1); %saline control

disp('--- Saline injection:');
bandit_lesionPerSession(dataIndex(salineSubset,:),save_path);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Model fitting and comparison for lesion data
bandit_modelfitting_lesion(dataIndex, root_path);

%%% End of the lesion analysis


