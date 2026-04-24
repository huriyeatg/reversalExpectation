function bandit_fittingPerAnimal_longitudinal(dataIndex,save_path)
% % bandit_fittingPerAnimal_longitudinal %
%PURPOSE:   Fit various learning algorithms to experimental data, on a
%           per-animal basis by merging sessions from same animal
%AUTHORS:   H Atilgan and AC Kwan 191212
% modified for longitudinal analysis by HA 07/12/2021
%
%INPUT ARGUMENTS
%   dataIndex:    a database index table for the sessions to analyze
%   save_path:    path for saving the analysis
%
%OUTPUT ARGUMENTS
%

model{1}.name = 'WSLS';   % text label to refer to the model
model{1}.fun = 'funWSLS'; % the corresponding .m code for the model
model{1}.initpar=0.8;     % initial parameter: [prob_WSLS]
model{1}.lb=0;            % upper bound of parameters
model{1}.ub=1;            % lower bound of parameters

model{2}.name = 'Q_RPE';        % text label to refer to the model
model{2}.fun = 'funQ_RPE';      % the corresponding .m code for the model
model{2}.initpar=[0.3 5];       % initial [alpha beta]
model{2}.lb=[0 0];              % upper bound of parameters
model{2}.ub=[1 100];            % lower bound of parameters

model{3}.name = 'DFQ_RPE';       % text label to refer to the model
model{3}.fun = 'funDFQ_RPE';     % the corresponding .m code for the model
model{3}.initpar=[0.3 0.3 5];    % initial [alpha lambda beta]
model{3}.lb=[0 0 0 ];            % upper bound of parameters
model{3}.ub=[1 1 100];           % lower bound of parameters

model{4}.name = 'FQ_RPE';      % text label to refer to the model
model{4}.fun = 'funFQ_RPE';    % the corresponding .m code for the model
model{4}.initpar=[0.3 5];      % initial [alpha beta]
model{4}.lb=[0 0];             % upper bound of parameters
model{4}.ub=[1 100];           % lower bound of parameters

model{5}.name = 'FQ_RPE_CK';      % text label to refer to the model
model{5}.fun = 'funFQ_RPE_CK';    % the corresponding .m code for the model
model{5}.initpar=[0.3 5 0.2 3];   % initial [alpha beta alpha CK beta CK]]
model{5}.lb=[0 0 0 0];            % upper bound of parameters
model{5}.ub=[1 100 1 100];        % lower bound of parameters

model{6}.name = 'DFQ_RPE_CK';      % text label to refer to the model
model{6}.fun = 'funDFQ_RPE_CK';    % the corresponding .m code for the model
model{6}.initpar=[0.3 0.3 5 0.2 3];   % initial [alpha lambda beta alphaCK betaCK]]
model{6}.lb=[0 0 0 0 0];            % upper bound of parameters
model{6}.ub=[1 1 100 1 100];        % lower bound of parameters

%%%%%%% FQ More options 
model{7}.name = 'belief';          % text label to refer to the model
model{7}.fun =  'funbelief';       % the corresponding .m code for the model
model{7}.initpar=[0.1 3];          % initial [hazardrate beta]
model{7}.lb=[0 0];                 % lower bound of parameters
model{7}.ub=[1 100];               % upper bound of parameters

model{8}.name = 'belief_CK';      % text label to refer to the model
model{8}.fun =  'funbelief_CK';   % the corresponding .m code for the model
model{8}.initpar=[0.3 5 0.2 3];   % initial [hazard rate beta alpha CK beta CK]
model{8}.lb=[0 0 0 0];            % lower bound of parameters
model{8}.ub=[1 100 1 100];         % upper bound of parameters

model{9}.name = 'belief_CK_bilateral';      % text label to refer to the model
model{9}.fun = 'funbelief_CK_bilateral';    % the corresponding .m code for the model
model{9}.initpar=[0.3 0.3 0.2 0.2 5 3];     % initial % initial [H Left, H Right, alpha CK Left, alpha CK right, beta, beta CK]
model{9}.lb=[0 0 0 0 0 0];                  % lower bound of parameters
model{9}.ub=[1 1 1 1 100 100];              % upper bound of parameters


%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each animal
animalList = unique(dataIndex.Animal);

disp('-----------------------------------------------------------');
disp(['--- Fitting models - summary of ', int2str(numel(animalList)) ' animals']);
disp('-----------------------------------------------------------');

for k = 1:numel(model)
    
    disp(['Considering model "' model{k}.name '"']);
    
    % Does the analysis file exist?
    fn = dir(fullfile(save_path, [model{k}.fun(4:end) '.mat']));
    if size(fn,1)>0
        answer = questdlg(['There is already .mat file for model fitting for ' model{k}.fun '. Run the analysis again? (may take time)'], ...
            'Run model fitting?', ...
            'Yes','No','Yes');
        if strcmp(answer,'Yes')
            runFit = true;
        else
            runFit = false;
        end
    else
        runFit = true;
    end
    
    if (runFit)
        for j = 1:numel(animalList)
            
            disp(['Processing animal # ' animalList{j} '...']);
            disp(['   ' num2str(sum(strcmp(dataIndex.Animal,animalList(j)))) ' sessions associated with this animal']);
            
            %which session belong to this one animal
            currAnimalSessions = ismember(dataIndex.Animal,animalList(j));
            
            %concatenate the sessions for this one animal
            [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));

            stats = value_getTrialStats(trials, nRules);
            stats = value_getTrialStatsMore(stats);
            
            if isfield(model{k},'lb')
                [fitpar{j},negloglike{j}, bic{j}, nlike{j}]=fit_fun(stats,model{k}.fun,model{k}.initpar,model{k}.lb,model{k}.ub);
            else
                [fitpar{j},negloglike{j}, bic{j}, nlike{j}]=fit_fun(stats,model{k}.fun,model{k}.initpar);
            end
            fitpar{j}
            animal{j} = animalList{j};
        end
        
         %save behavioral .mat file
         save(fullfile(save_path, [model{k}.fun(4:end) '.mat']),...
             'animal','fitpar','bic','nlike','negloglike');
         
        fitparMat = cell2mat(fitpar');
        disp('Median fitted parameters:');
        nanmedian(fitparMat,1)
    end
    
end

end
% 
