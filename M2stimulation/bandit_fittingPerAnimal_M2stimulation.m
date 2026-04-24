function bandit_fittingPerAnimal_M2stimulation(dataIndex,save_path)
% % bandit_fittingPerAnimal_M2stimulation %
%PURPOSE:   Fit various learning algorithms to experimental data, on a
%           per-animal basis by merging sessions from same animal
%AUTHORS:   H Atilgan and AC Kwan 191212
%
%INPUT ARGUMENTS
%   dataIndex:    a database index table for the sessions to analyze
%   save_path:    path for saving the analysis
%
%OUTPUT ARGUMENTS
%

model{1}.name = 'funbelief_CK_stimulation_bilateral';  % text label to refer to the model
model{1}.fun = 'funbelief_CK_stimulation_bilateral';   % the corresponding .m code for the model
model{1}.initpar=[0.1 3 0.2 3 0.1 0.2 0.1 0.2 3 3]; % inits:  [H-CNT,beta-CNT,alpha-CNT,betaCK-CNT, H-contra,alpha-contra, H-ipsi, alpha-ipsi, beta-ST, betaCK-ST]
model{1}.lb= [ 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{1}.ub=[1 100 1 100 1 1 1 1 100 100];              % lower bound of parameters

model{2}.name = 'funbelief_CK_stimulation_bilateral_10params';   % text label to refer to the model
model{2}.fun = 'funbelief_CK_stimulation_bilateral_10params';    % the corresponding .m code for the model
model{2}.initpar=[0.1 0.1 0.1 0.2 0.2 0.2 3 5 3 5]; % inits:  C
model{2}.lb= [ 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{2}.ub=[1 1 1 1 1 1 100 100 100 100];              % lower bound of parameters

model{3}.name = 'funbelief_CK_stimulation_8params';   % text label to refer to the model
model{3}.fun = 'funbelief_CK_stimulation_8params';    % the corresponding .m code for the model
model{3}.initpar=[0.1 0.1 0.2 0.2 3 3 5 5];     % inits:  C
model{3}.lb= [ 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{3}.ub=[1 1 1 1 100 100 100 100];              % lower bound of parameters

model{4}.name = 'funbelief_CK_stimulation_delta_8params';   % text label to refer to the model
model{4}.fun = 'funbelief_CK_stimulation_delta_8params';    % the corresponding .m code for the model
model{4}.initpar=[0.1 0.1 0.2 0.2 3 3 5 5];     % inits:  C
model{4}.lb= [ 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{4}.ub=[1 1 1 1 100 100 100 100];          % lower bound of parameters

model{5}.name = 'funbelief_CK_stimulation_bilateral_delta_10params';   % text label to refer to the model
model{5}.fun = 'funbelief_CK_stimulation_bilateral_delta_10params';    % the corresponding .m code for the model
model{5}.initpar=[0.1 0.1 0.1 0.1 0.1 0.1 3 3 3 3]; % inits:  C
model{5}.lb= [ 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{5}.ub=[1 1 1 1 1 1 100 100 100 100];              % lower bound of parameters

%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each animal
animalList = unique(dataIndex.Animal);

disp('-----------------------------------------------------------');
disp(['--- Fitting models - summary of ', int2str(numel(animalList)) ' animals']);
disp('-----------------------------------------------------------');

for k = 5
    
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
            % stimulated side
            stats.stRegion = trialData.stimulationRegion-1000;
            
            %stimulation: yes=1; no=1;
            stats.st=nan(numel(trialData.stimulation),1);
            stats.st = stats.stRegion;
            stats.st (stats.stRegion==2) =1;
            
            
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
