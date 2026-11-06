function bandit_modelfitting(dataIndex, root_path)
% % bandit_modelfitting %
%PURPOSE:   Analysis of naive data by model fittings.
%AUTHORS:   H Atilgan 04/30/2020
%
%INPUT ARGUMENTS
%   dataIndex:  the stats variable that contains choice/outcome information
%   root_path:  text that describe the name of the model
%
%OUTPUT ARGUMENTS
%

%% create paths and folders
simPath.root             = root_path;
simPath.fig              = fullfile(root_path,'figs','simulation');
simPath.model_mat        = fullfile(root_path,'sim','mat_models');
simPath.stats_mat        = fullfile(root_path,'sim','mat_stats');
fnames = fieldnames(simPath);
for k=1:numel(fnames)
    temp = simPath.(fnames{k});  
    if ~exist(temp,'dir') % create folders
        mkdir(temp);
    end
end

%%  Model fitting
bandit_fittingPerAnimal(dataIndex,simPath.model_mat);
%bandit_fittingPerSession(dataIndex,simPath.model_mat);

%% Model BIC comparison figure
bandit_compareModels(simPath.model_mat,simPath.fig); 

%% Additional: Compare actual choices and outcomes from session with 
% simulated data 
exampleLogName = '19107_phase3_R71NoCue_1904051629.log';
exampleAnimal = '19107';
model_type = 'belief_CK';

bandit_predictSession(dataIndex,...
                      exampleLogName,exampleAnimal,...
                      simPath.model_mat,model_type);

print(gcf,'-dsvg',fullfile(simPath.fig,['session-sample' model_type]));
saveas(gcf, fullfile(simPath.fig,['session-sample' model_type]), 'fig');
close all
%% Latent variables analysis (learning rates, CPP, RU)
%model_type = 'belief_CK';
%plot_latentvariables(dataIndex,model_type,simPath);

% %% explore lateralization effect - needs to be updated!
% model_type = 'belief_CK';
% bandit_simulateLateralization(model_type,simPath.fig)

%% Compare behavioural outcomes across different models + data

animalid = '1810';
compareModels = [{'experimental'};{'belief_CK'};{'FQ_RPE_CK'}];
hFig = bandit_compareSimulation2experimentalData(dataIndex,compareModels,simPath,animalid);

%%  Additional: Model simulations 
% % simulates and plots general switch figures for specified parameters or
% % specified animal id fittings or mean fittings - see function for more
% % details.
% 
% n_stim = 1e3*30; % 350K trials gives similar estimation to our experimental
%                  % data that takes ~2.5 hours to generate.
% model_type = 'belief_CK';% model based learning
% bandit_stimulation (model_type,simPath, n_stim,[],'19107');
% % simulates for mean values if animal id or customize params are not given.
% 
% n_stim = 1e3*10; % 350K trials gives similar estimation to our experimental
%                  % data that takes ~2.5 hours to generate.
% model_type = 'FQ_RPE_CK';% model based learning
% bandit_stimulation (model_type,simPath, n_stim,[],'19107');
% 
% %%% >>> see function: "master_simulation" for more details of model
% %%% comparisons.

end

