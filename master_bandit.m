% Master file for analyzing lick port bandit task in different projects.
% see Read Me for details & ReadMe - DataCollectionForBanditTask.txt for
% details of data. 
% H Atilgan 031120
%
% To run this code:
% 1) Add all the subfolders to the Path in MATLAB
% 2) Add your computer & path to setup_compprop.m file
%

clearvars;
close all;
setup_figprop;
setup_compprop; % add your computer settings here

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Behaviour analysis of bandit task
% Analyse lickport two-bandit task across blocks and fits to reinforcement
% models.
%%%% Code Folder: \behavior 
%%%% Data Folder: \data-behavior\bandit_R71_lesion
master_behavior

master_nothreshold
plot_summaryfittingvalues

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Simulation of different RL/Bayesian models
master_simulation

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Effects of Cg1/M2 lesion on bandit task
% Analyse lickport two-bandit task across blocks and fits to reinforcement
% models.
%%%% Code Folder: \lesion 
%%%% Data Folder: \data\data-behavior\bandit_R71_lesion
master_banditlesion

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Effects of Cg1/M2 photo stimulation on bandit task
%%%% Code Folder:  \M2stimulation
%%%% Data Folder:  \data\data-behavior\bandit-R71_M2stimulation
master_banditM2stimulation

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Neural dynamic changes of NE & ACh in prelimbic 
%%%%%%%%%%% cortex - Inscopix recording
master_banditneuromodulator

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Behavioural changes over time in bandit task
%%%% Code Folder: \longitudinal 
%%%% Data Folder: \data\longitudinal
master_banditlongitudinal

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Project: Wholecortex mapping - Opto steering rig
master_banditwholecortex

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Effect of reward rate  in two armed bandit task
%%%% Code Folder: \rewardRate 
%%%% Data Folder: \data\data-behavior\rewardRate
master_banditrewardrate % Phase6

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Effect of reward magnitude in two armed bandit task
%%%% Code Folder: \longitudinal 
%%%% Data Folder: \data\longitudinal
master_banditrewardamount % Heather's data



