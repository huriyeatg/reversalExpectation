function bandit_fittingPerSession_M2stimulation(dataIndex,save_path)
% % bandit_fittingPerSession_M2stimulation %
%PURPOSE:   Fit various learning algorithms to experimental data, on a
%           per-session basis
%AUTHORS:   A Kwan & H Atilgan 231121
%
%INPUT ARGUMENTS
%   dataIndex:    a database index table for the sessions to analyze
%   save_path:    path for saving the analysis
%
%OUTPUT ARGUMENTS
%

% model{1}.name = 'funbelief_CK_stimulation_bilateral_8params';   % text label to refer to the model
% model{1}.fun = 'funbelief_CK_stimulation_bilateral_8params';    % the corresponding .m code for the model
% model{1}.initpar=[0.1 0.1 0.1 0.2 0.2 0.2 3 5]; % inits:  C
% model{1}.lb= [ 0 0 0 0 0 0 0 0];                % upper bound of parameters
% model{1}.ub=[1 1 1 1 1 1 10 10];              % lower bound of parameters

model{1}.name = 'funbelief_CK_stimulation_8params';   % text label to refer to the model
model{1}.fun = 'funbelief_CK_stimulation_8params';    % the corresponding .m code for the model
model{1}.initpar=[0.1 0.1 0.2 0.2 3 3 5 5];     % inits:  C
model{1}.lb= [ 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{1}.ub=[1 1 1 1 100 100 100 100];              % lower bound of parameters

model{2}.name = 'funbelief_CK_stimulation_bilateral_10params';   % text label to refer to the model
model{2}.fun = 'funbelief_CK_stimulation_bilateral_10params';    % the corresponding .m code for the model
model{2}.initpar=[0.1 0.1 0.1 0.2 0.2 0.2 3 5 3 5]; % inits:  C
model{2}.lb= [ 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{2}.ub=[1 1 1 1 1 1 100 100 100 100];              % lower bound of parameters

model{3}.name = 'funbelief_CK_stimulation_bilateral_12params';   % text label to refer to the model
model{3}.fun = 'funbelief_CK_stimulation_bilateral_12params';    % the corresponding .m code for the model
model{3}.initpar=[0.1 0.1 0.1 0.2 0.2 0.2 3 5 3 5 3 5]; % inits:  C
model{3}.lb= [ 0 0 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{3}.ub=[1 1 1 1 1 1 10 10 10 10 10 10];              % lower bound of parameters

model{4}.name = 'funbelief_CK_stimulation_delta_8params';   % text label to refer to the model
model{4}.fun = 'funbelief_CK_stimulation_delta_8params';    % the corresponding .m code for the model
model{4}.initpar=[0.1 0.1 0.2 0.2 3 3 5 5];     % inits:  C
model{4}.lb= [ 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{4}.ub=[1 1 1 1 100 100 100 100];          % lower bound of parameters

model{5}.name = 'funbelief_CK_stimulation_bilateral_delta_10params';   % text label to refer to the model
model{5}.fun = 'funbelief_CK_stimulation_bilateral_delta_10params';    % the corresponding .m code for the model
model{5}.initpar=[0.1 0.1 0.1 0.2 0.2 0.2 3 5 3 5]; % inits:  C
model{5}.lb= [ 0 0 0 0 0 0 0 0 0 0];                % upper bound of parameters
model{5}.ub=[1 1 1 1 1 1 100 100 100 100];              % lower bound of parameters


%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each session

disp('-----------------------------------------------------------');
disp(['--- Fitting models - summary of ', int2str(size(dataIndex,1)) ' sessions']);
disp('-----------------------------------------------------------');

for k=5
    
    disp(['Considering model "' model{k}.fun '"']);
    
    % Does the analysis file exist?
    fn = dir(fullfile(save_path, [model{k}.fun(4:end) '_perSession.mat']));
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
        for j = 1:size(dataIndex,1)
            
            disp(['Processing session # ' int2str(j) '...']);
            
            load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_beh.mat']));
            
            trials = value_getTrialMasks(trialData);
            stats = value_getTrialStats(trials, sessionData.nRules);
            stats = value_getTrialStatsMore(stats);
            
            % stimulated side
            stats.stRegion = trialData.stimulationRegion-1000;
            
            %stimulation: yes=1; no=1;
            stats.st=nan(numel(trialData.stimulation),1);
            stats.st = stats.stRegion;
            stats.st (stats.stRegion==2) = 1;
            
            
            [fitpar{j}, ~, bic{j}, ~]=fit_fun(stats,model{k}.fun,model{k}.initpar,model{k}.lb,model{k}.ub);
            
            session{j} = dataIndex.LogFileName{j};
            
            fitpar{j};
        end
        
        %save behavioral .mat file
        save(fullfile(save_path, [model{k}.fun(4:end), '_upperBound100_perSession.mat']),...
            'session','fitpar','bic', 'model');
    end
end

end

