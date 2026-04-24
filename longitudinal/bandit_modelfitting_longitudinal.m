function bandit_modelfitting_longitudinal(dataIndex, root_path)
% % bandit_modelfitting_longitudinal(dataIndex, root_path)
%PURPOSE:   Analysis of  longitudinal data by model fittings.
%AUTHORS:   H Atilgan 07/12/2020
%
%INPUT ARGUMENTS
%   dataIndex:  the stats variable that contains choice/outcome information
%   root_path:  root_path of the repository
%
%OUTPUT ARGUMENTS
%   figures saved in fis folder

%% create paths and folders
simPath.root             = root_path;
simPath.fig              = fullfile(root_path,'figs','longitudinal-sim');
simPath.sim_path         = fullfile(root_path,'longitudinal','sim'); % path for both mat folder. 
simPath.model_mat        = fullfile(root_path,'longitudinal','sim','models');
simPath.stats_mat        = fullfile(root_path,'longitudinal','sim','mat_stats');

fnames = fieldnames(simPath);
for k=1:numel(fnames)
    temp = simPath.(fnames{k});
    if ~exist(temp,'dir') % create folders
        mkdir(temp);
    end
end

%%  Model fitting
sessionNumber = 5; 
earlyPhaseSubset = (dataIndex.sessioninPhase3_CHANGETHIS < sessionNumber); 
latePhaseSubset = (dataIndex.sessioninPhase3_CHANGETHIS < sessionNumber); 

save_path = fullfile(simPath.model_mat, 'earlyPhase');
bandit_fittingPerAnimal(dataIndex(earlyPhaseSubset,:),save_path);

save_path = fullfile(simPath.model_mat, 'latePhase');
bandit_fittingPerAnimal(dataIndex(latePhaseSubset,:),save_path);

%% Compare models
model_path = fullfile(simPath.model_mat, 'earlyPhase');
save_sim_path = fullfile (simPath.fig, 'earlyPhase')
bandit_compareModels(model_path, save_sim_path);   % model comparison from control data

model_path = fullfile(simPath.model_mat, 'latePhase');
save_sim_path = fullfile (simPath.fig, 'latePhase')
bandit_compareModels(model_path, save_sim_path);   % model comparison from control data

