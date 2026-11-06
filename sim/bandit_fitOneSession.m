function bandit_fitOneSession(stats,modelName,save_path)
% % bandit_fittingPerSession %
%PURPOSE:   Fit one session with one learning algorithm
%AUTHORS:   AC Kwan 200210
%
%INPUT ARGUMENTS
%   stats:      the stats variable that contains choice/outcome information
%   modelName:  text that describe the name of the model
%   save_path:  path for saving the analysis
%
%OUTPUT ARGUMENTS
%

if strcmp(modelName,'WSLS')
    model.name = 'Win-stay lose-switch';   % text label to refer to the model
    model.fun = 'funWSLS'; % the corresponding .m code for the model
    model.initpar=0.8;     % initial parameter: [prob_WSLS]
    model.lb=0;            % upper bound of parameters
    model.ub=1;            % lower bound of parameters
    
elseif strcmp(modelName,'Q_RPE')
    model.name = 'Q_RPE';            % text label to refer to the model
    model.fun = 'funQ_RPE';      % the corresponding .m code for the model
    model.initpar=[0.2 5];       % initial [alpha beta]
    model.lb=[0 0];              % upper bound of parameters
    model.ub=[1 inf];            % lower bound of parameters
    
elseif strcmp(modelName,'DQ_RPE')
    model.name = 'DQ_RPE';           % text label to refer to the model
    model.fun = 'funDQ_RPE';     % the corresponding .m code for the model
    model.initpar=[0.5 0.2 5];   % initial [alpha_reward alpha_noreward beta]
    model.lb=[0 0 0];            % upper bound of parameters
    model.ub=[1 inf 1];          % lower bound of parameters
    
elseif strcmp(modelName,'FQ_RPE')
    model.name = 'FQ_RPE';      % text label to refer to the model
    model.fun = 'funFQ_RPE';    % the corresponding .m code for the model
    model.initpar=[0.5 5];      % initial [alpha beta]
    model.lb=[0 0];             % upper bound of parameters
    model.ub=[1 100];           % lower bound of parameters
    
elseif strcmp(modelName,'FQ_RPE_CK')
    model.name = 'FQ_RPE_CK';      % text label to refer to the model
    model.fun = 'funFQ_RPE_CK';    % the corresponding .m code for the model
    model.initpar=[0.5 5 0.5 5];   % initial [hazardrate, beta, alpha CK, beta CK]
    model.lb=[0 0 0 0];            % lower bound of parameters
    model.ub=[1 100 1 100];
    
elseif strcmp(modelName,'hybridFQ_RPE')
    model.name = 'hybridFQ_RPE';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE';    % the corresponding .m code for the model
    model.initpar=[0.5 5];     % initial [hazardrate beta]
    model.lb=[0 0];            % lower bound of parameters
    model.ub=[1 100];          % upper bound of parameters
    
elseif strcmp(modelName,'hybridFQ_RPE_CK')
    model.name = 'hybridFQ_RPE_CK';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE_CK';    % the corresponding .m code for the model
    model.initpar=[0.5 5 0.5 5];     % initial [hazardrate, beta, alpha CK, beta CK]
    model.lb=[0 0 0 0 ];             % lower bound of parameters
    model.ub=[1 100 1 100];          % upper bound of parameters
    
elseif strcmp(modelName,'hybridFQ_RPE_CK_bilateral')
    model.name = 'hybridFQ_RPE_CK_bilateral';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE_CK_bilateral';    % the corresponding .m code for the model
    model.initpar=[0.5 0.5 0.5 0.5 5 5];     % initial [H Left, H Right, alpha CK Left, alpha CK right, beta, beta CK]
    model.lb=[0 0 0 0 0 0 ];                 % lower bound of parameters
    model.ub=[1 1 1 1 100 100];              % upper bound of parameters
    
end

%%
if ~exist(save_path,'dir')
    mkdir(save_path);
end

%% go through each session

disp(['Considering model "' model.fun '"']);

% Does the analysis file exist?
fn = dir(fullfile(save_path, [model.fun(4:end) '.mat']));
if size(fn,1)>0
    answer = questdlg(['There is already .mat file for model fitting for ' model.fun '. Run the analysis again? (may take time)'], ...
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
    
    [fitpar, ~, bic, ~]=fit_fun(stats,model.fun,model.initpar,model.lb,model.ub)
    
    %save behavioral .mat file
    save(fullfile(save_path, [model.fun(4:end) '.mat']),...
        'fitpar','bic');
    
end

end

