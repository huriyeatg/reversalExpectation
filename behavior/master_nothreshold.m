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
figurepath = fullfile(root_path,'figs','no-threshold');
dataIndex = makeDataIndex(logfilepath, analysispath,'bandit_R71_noHitCriteria');% this folder only

% Parse and analyze each logfile, save as .mat files, as needed
dataIndex = createBehMatFiles(dataIndex);

% Determine if each session fulfill performance criteria
[dataIndex, ~] = determineBehCriteria(dataIndex); 

% Create table for manuscript: per Animal, total session, total switch etc 
InfoTable = makeAnimalInfoTable(dataIndex);

%% 
save_path = fullfile(figurepath,'noThreshold-summary-per-animal');
bandit_behaviorPerAnimal_nothreshold(dataIndex,save_path);

save_path = fullfile(figurepath,'noThreshold-summary-per-session');
bandit_behaviorPerSession_nothreshold(dataIndex,save_path);

