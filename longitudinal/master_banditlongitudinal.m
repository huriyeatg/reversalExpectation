% master_banditlongitudinal
% Master file for analyzing the learning variables across sessions in
% bandit task

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
logfilepath = fullfile(root_path,'data','data-behavior','bandit_R71_longitudinal');
analysispath = fullfile(root_path,'analysis');
figurepath = fullfile(root_path,'figs');
dataIndex = makeDataIndex(logfilepath, analysispath);% this folder only

% Parse and analyze each logfile, save as .mat files, as needed
dataIndex = createBehMatFiles_longitudinal(dataIndex); 
% CHECK LATER! There are lots of session not being calculated correctly. 

% Add information about longitudinial
dataIndex = addIndexLongitudinial(dataIndex); %% determine criteria

%% COMPARE FIRST 3 PHASE 3 sessions to LAST 3 PHASE 3 sessions
sessionNumber = 5;
save_path = figurepath;
bandit_PerSession_longitudinal(dataIndex,save_path, sessionNumber)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Behavioural effect %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% behavioural session analysis

% Points to explore;
% how reward rate, hit rate,reaction times changes across session?



%% How switch criteria effects animals?
%Transient to no-criteria sessions and back to with criteria session. 
save_path = fullfile(figurepath,'Transition2noCriteriaTask-per-animal-101620');
bandit_Transition2noCriteriaTask(dataIndex,save_path)




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL FITTING     %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Model fitting
reversalSubset = (ismember(dataIndex.Phase,[3 8])==1);  %reversal
controlSubset = isnan(dataIndex.Lesioned);  %control or pre-lesion data
%bandit_fittingPerAnimal(dataIndex(controlSubset & reversalSubset,:),model_path);
bandit_fittingPerSession(dataIndex(controlSubset & reversalSubset,:),model_path);

%% Simulate 
n_stim = 1e4*35; 
model_type = 'hybridFQ_RPE_CK';% model based learning
bandit_stimulation (model_type,fullfile(figurepath,model_type),model_path, n_stim,[], ls)


%% Compare actual choices and outcomes from session with simulated data 
exampleLogName = '19107_phase3_R71NoCue_1904051629.log';
exampleAnimal = '19107';
model_type = 'FQ_RPE_CK';

bandit_predictSession(dataIndex,...
                      exampleLogName,exampleAnimal,model_path,model_type);

print(gcf,'-dsvg',fullfile(figurepath,['session-sample' model_type]));
saveas(gcf, fullfile(figurepath,['session-sample' model_type]), 'fig');

%% Look for latent variable changes
% For example: beta ratio
plot_betaratio(model_type, model_path)

% Points to explore;
% how beta ratio changes over sessions?
% how hazard rate or choice kernel learning rate changes over sessions?
% how L random effect on switches changes over sessions?
% Can you see any change on best model fittings over time? 




