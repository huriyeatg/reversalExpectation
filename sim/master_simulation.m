% Master file for simulating different model options for two-bandit task
% This model simulation uses BADS optimizer and requires;
%           Optimization Toolbox
%           Global Optimization Toolbox
%           Installed BADS 
%               to install: (bandit-master\simulation\bads-master\install)
%               to test: bads('test')    
%   
% H Atilgan, H Wang and AC Kwan, 03/11/20

%% Make sure the figure folder created
save_sim_path = fullfile(root_path, 'figs','simulation1906');
if ~exist(save_sim_path,'dir')
    mkdir(save_sim_path);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Stimulation: Model based Belief Model

% Task parameters
n_stim=10000;    % number of trials to simulate
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean 

% Stimulate model
player_sim.model_type='FQ_RPE_CK';
player_sim.params= [0.3 5];% two free parameter - hazard rate, beta

stats_sim=simBandit(player_sim,taskparams,n_stim);

% plot session related figures 
tlabel=[player_sim.model_type ', n=' num2str(n_stim) ' trials'];
save_path = fullfile(save_sim_path,['simulation ' player_sim.model_type]);
analyze_session(stats_sim,tlabel,save_path);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Stimulation: Model based Belief Model

% Task parameters
n_stim=500;    % number of trials to simulate
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean 

% Stimulate model
player_sim.model_type='belief_CK';

% using median value from fit with animal data if created
model_path = fullfile(root_path,'sim','mat_models');
%load(fullfile(model_path, [player_sim.model_type '.mat']));
%fitparMat = cell2mat(fitpar');
player_sim.params = [ 0.01 3];% fitparMat(4,:);%nanmedian(fitparMat,1);

% Stimulate accordingly: 
stats_sim=simBandit(player_sim,taskparams,n_stim);

% check fitting
model.name = 'belief_CK';      % text label to refer to the model
model.fun = 'funbelief_CK';    % the corresponding .m code for the model
model.initpar=[0.3 5 0.2 2];   % initial [hazard rate beta alpha CK beta CK]
model.lb=[0 0 0 0];            % lower bound of parameters
model.ub=[1 10 1 100];         % upper bound of parameters

[fitpar, ~, bic, ~]=fit_fun(stats_sim,model.fun,model.initpar,model.lb,model.ub);
  
% Plots for stimulation
tlabel=[player_sim.model_type ', n=' num2str(n_stim) ' trials'];
save_path = fullfile(save_sim_path,['simulation ' player_sim.model_type]);
analyze_session(stats_sim,tlabel,save_path);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Model comparisons

%%%%% BIC comparisons - this requires model fittings first!
% see  'bandit_lesion_modelfitting' function for model fittings.
bandit_compareModels(model_path, save_sim_path);   % model comparison from control data

%%%%% Behavioural outcome
animalid = '1810';
compareModels = [{'experimental'};{'belief_CK'};{'FQ_RPE_CK'}];
simPath.root             = root_path;
simPath.fig              = fullfile(root_path,'figs','simulation1909');
simPath.model_mat        = fullfile(root_path,'sim','mat_models');
simPath.stats_mat        = fullfile(root_path,'sim','mat_stats1909');

hFig = bandit_compareSimulation2experimentalData([],compareModels,simPath,animalid);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MODEL TESTING SECTION %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% OPTIMIZER : BADS  
% Optimizer is used to fit the parameters in models. 
% We have decided to use BADS (by Wei Ma Lab) for optimizer - this part is
% to compare BADS with other optimizers - Current code compares BADS, 
% fminsearch and fmincon - there are other optimizers within the code,
% currently opt-out.

% Task parameters
n_stim=5000;    % number of trials to simulate
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean 

% Stimulate model
player_sim.model_type='FQ_RPE'; 
% alternatives: 'Q_RPE'; 'DQ_RPE'; 'hybridFQ_RPE'; 'hybridFQ_RPE_CK'
player_sim.params=[0.3 5];
stats_sim = simBandit(player_sim,taskparams,n_stim);
model.fun = @funFQ_RPE;

% Test optimizers 
optimviz_bandit(model.fun,player_sim.params,0,100,stats_sim)
savefigurename = fullfile(save_sim_path,['optimviz-',player_sim.model_type]);
print(gcf,'-dsvg',savefigurename);
saveas(gcf, [savefigurename, '.fig']);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Parameter recovery

%%%%%  Single parameter recovery
% Task parameters
n_stim=50000;    % number of trials to simulate
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean 

% Stimulate model
player_sim.model_type='FQ_RPE';
player_sim.params=[0.3 5]; % alpha, beta ----> We aim to find these values after fitting!
stats_sim=simBandit(player_sim,taskparams,n_stim);

% To recover the initially stimulated parameter
save_path = fullfile(save_sim_path, ['recovery', player_sim.model_type]);
bandit_fitOneSession(stats_sim,'FQ_RPE',save_path); % saves mat file.

%%%%% Multiple parameter recovery
% To recover systematically a range of parameters - this will take some time
model_type = 'FQ_RPE_CK';
nCount = 1000; % number of simulation
bandit_parameterRecovery( model_type,nCount)
print(gcf,'-dpng',fullfile(save_sim_path, ['figs_simulation-recovery', player_sim.model_type]));
saveas(gcf, fullfile(save_sim_path, ['figs_simulation-recovery', player_sim.model_type]), 'fig');


%% Compare Qvalue differences and action selection for an animal
% Be careful: This only works without choice kernel models - Choice kernel
% effects action selection as well as Q value - so it is natural this code
% fails for model with choice kernels.
exampleAnimal = '18107';
model_type = 'FQ_RPE_CK';

animalSubset = (ismember(dataIndex.Animal,exampleAnimal)==1);
plot_qvaluehistogram(model_type,dataIndex(animalSubset & controlSubset & reversalSubset,:),model_path);
print(gcf,'-dpng',fullfile(save_sim_path,[exampleAnimal, '_qhistogram']));
saveas(gcf, fullfile(save_sim_path,[exampleAnimal, '_qhistogram']), 'fig');




