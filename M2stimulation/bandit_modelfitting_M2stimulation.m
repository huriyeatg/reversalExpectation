function bandit_modelfitting_M2stimulation(dataIndex, root_path)
% % bandit_modelfitting_M2stimulation%
%PURPOSE:   Analysis of lesion data by model fittings.
%AUTHORS:   H Atilgan 04/30/2020
%
%INPUT ARGUMENTS
%   dataIndex:  the stats variable that contains choice/outcome information
%   root_path:  text that describe the name of the model
%

%% create paths and folders
simPath.root             = root_path;
simPath.fig              = fullfile(root_path,'figs','M2stimulation-sim');
simPath.model_mat_M2stimulationPre = fullfile(root_path,'sim','mat_models_M2stimulationPre');
simPath.model_mat_M2stimulationPost = fullfile(root_path,'sim','mat_models_M2stimulationPost');
simPath.stats_mat        = fullfile(root_path,'sim','mat_stats');
simPath.sim_path         =  fullfile(root_path,'sim'); % path for both mat folder. 
fnames = fieldnames(simPath);
for k=1:numel(fnames)
    temp = simPath.(fnames{k});
    if ~exist(temp,'dir') % create folders
        mkdir(temp);
    end
end

%%  Model fitting - per Animal
preSubset = (ismember(dataIndex.Phase,21)==1); %pre-stimulation data
bandit_fittingPerAnimal_M2stimulation(dataIndex(preSubset,:),simPath.model_mat_M2stimulationPre);

postSubset = (ismember(dataIndex.Phase,22)==1); %post-stimulation data
bandit_fittingPerAnimal_M2stimulation(dataIndex(postSubset,:),simPath.model_mat_M2stimulationPost);

 %%  Model fitting - per Session
%Using one session to estimate 10 fittings - not very reliable. 
preSubset = (ismember(dataIndex.Phase,21)==1); %pre-stimulation data
bandit_fittingPerSession_M2stimulation(dataIndex(preSubset,:),simPath.model_mat_M2stimulationPre);

postSubset = (ismember(dataIndex.Phase,22)==1); %post-stimulation data
bandit_fittingPerSession_M2stimulation(dataIndex(postSubset,:),simPath.model_mat_M2stimulationPost);

%% Compare parameters by Animal 
compareParameters_M2stimulation_perAnimal(simPath);

%% Compare parameters by Session
preSubset = (ismember(dataIndex.Phase,21)==1); %pre-stimulation data
save_path = fullfile(simPath.fig ,'pre');
bandit_M2stimulationPerSession(dataIndex(preSubset,:),save_path)

preSubset = (ismember(dataIndex.Phase,22)==1); %pre-stimulation data
save_path = fullfile(simPath.fig ,'post');
bandit_M2stimulationPerSession(dataIndex(preSubset,:),save_path)

% % % %% DELETE: Compare parameters by Animal (n=6 mice) - This is not used in the MS - so little data points
% % % preSubset = (ismember(dataIndex.Phase,21)==1); % same animals were used for pre-stimulation & post-stimulation
% % % animalList = unique(dataIndex.Animal(preSubset)); 
% % % compareParameters_M2stimulation(simPath ,animalList);

end

