function bandit_parameterRecovery( modelName,nCount, save_path)
% %bandit_parameterRecovery %
%PURPOSE:   Based on the model, creates nCount params and fit it and check
%if it can recover the parameter.
%AUTHORS:   H Atilgan & AC Kwan 05012020
%
%INPUT ARGUMENTS
%   modelName:  text that describe the name of the model
%
%OUTPUT ARGUMENTS
%

%% Task parameters
n_stim=20000;    % number of trials to simulate
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean

%% Initialize model
if strcmp(modelName,'WSLS')
    model.name = 'Win-stay lose-switch';   % text label to refer to the model
    model.fun = 'funWSLS'; % the corresponding .m code for the model
    model.initpar=0.8;     % initial parameter: [prob_WSLS]
    model.lb=0;            % upper bound of parameters
    model.ub=1;            % lower bound of parameters
    model.params ={ 'prob'};
    model.paramsfnc= { 'rand'};
    
    
elseif strcmp(modelName,'Q_RPE')
    model.name = 'Q_RPE';            % text label to refer to the model
    model.fun = 'funQ_RPE';      % the corresponding .m code for the model
    model.initpar=[0.2 5];       % initial [alpha beta]
    model.lb=[0 0];              % upper bound of parameters
    model.ub=[1 inf];            % lower bound of parameters
    model.params ={ 'alpha' 'beta'};
    model.paramsfnc= { 'rand' 'exprnd(10)'};
    
elseif strcmp(modelName,'DQ_RPE')
    model.name = 'DQ_RPE';           % text label to refer to the model
    model.fun = 'funDQ_RPE';     % the corresponding .m code for the model
    model.initpar=[0.5 0.2 5];   % initial [alpha_reward alpha_noreward beta]
    model.lb=[0 0 0];            % upper bound of parameters
    model.ub=[1 inf 1];          % lower bound of parameters
    model.params ={ 'alphaR' 'alphaNR' 'beta'};
    model.paramsfnc= { 'rand' 'rand' 'exprnd(10)'};
    
    
elseif strcmp(modelName,'FQ_RPE')
    model.name = 'FQ_RPE';      % text label to refer to the model
    model.fun = 'funFQ_RPE';    % the corresponding .m code for the model
    model.initpar=[0.5 5];      % initial [alpha beta]
    model.lb=[0 0];             % upper bound of parameters
    model.ub=[1 100];           % lower bound of parameters
    model.params ={ 'alpha' 'beta'};
    model.paramsfnc= { 'rand' 'exprnd(10)'};
    
elseif strcmp(modelName,'FQ_RPE_CK')
    model.name = 'FQ_RPE_CK';      % text label to refer to the model
    model.fun = 'funFQ_RPE_CK';    % the corresponding .m code for the model
    model.initpar=[0.5 5 0.5 5];   % initial [alpha, beta, alpha CK, beta CK]
    model.lb=[0 0 0 0];            % lower bound of parameters
    model.ub=[1 100 1 100];
    model.params ={ 'alpha' 'beta' 'alphaCK' 'betaCK'};
    model.paramsfnc= { 'rand' 'exprnd(10)' 'rand' 'exprnd(10)'};
    
elseif strcmp(modelName,'hybridFQ_RPE')
    model.name = 'hybridFQ_RPE';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE';    % the corresponding .m code for the model
    model.initpar=[0.5 5];     % initial [hazardrate beta]
    model.lb=[0 0];            % lower bound of parameters
    model.ub=[1 100];          % upper bound of parameters
    model.params ={ 'hazrate' 'beta' };
    model.paramsfnc= { 'rand' 'exprnd(10)'}; 
    
elseif strcmp(modelName,'hybridFQ_RPE_CK')
    model.name = 'hybridFQ_RPE_CK';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE_CK';    % the corresponding .m code for the model
    model.initpar=[0.2 5 0.2 5];     % initial [hazardrate, beta, alpha CK, beta CK]
    model.lb=[0 0 0 0 ];             % lower bound of parameters
    model.ub=[0.5 100 0.5 100];          % upper bound of parameters
    model.params ={ 'hazrate' 'beta' 'alphaCK' 'betaCK'};
    model.paramsfnc= { 'rand' 'exprnd(10)' 'rand' 'exprnd(10)'};
    
elseif strcmp(modelName,'hybridFQ_RPE_CK_bilateral')
    model.name = 'hybridFQ_RPE_CK_bilateral';      % text label to refer to the model
    model.fun = 'funhybridFQ_RPE_CK_bilateral';    % the corresponding .m code for the model
    model.initpar=[0.5 0.5 0.5 0.5 5 5];     % initial [H Left, H Right, alpha CK Left, alpha CK right, beta, beta CK]
    model.lb=[0 0 0 0 0 0 ];                 % lower bound of parameters
    model.ub=[0.5 0.5 0.5 0.5 100 100];              % upper bound of parameters
    model.params ={ 'hazrateL' 'hazrateR' 'hazrateCKL' 'hazrateCKR' 'beta' 'betaCK'};
    model.paramsfnc= { 'rand' 'rand' 'rand' 'rand' 'exprnd(10)' 'exprnd(10)'};
    
end

%% Add progress bar
f = waitbar(0,'1','Name','Simulating 1000 random fittings',...
    'CreateCancelBtn','setappdata(gcbf,''canceling'',1)');

setappdata(f,'canceling',0);

%% Stimulate
sim = nan(numel(model.paramsfnc), nCount);
fit = nan(numel(model.paramsfnc), nCount);

player_sim.model_type = modelName;

for count = 1:nCount
    % Check for clicked Cancel button
    if getappdata(f,'canceling')
        break
    end
    % Update waitbar and message
    waitbar(count/nCount,f,sprintf('%3.3f %%',(count*100/nCount) ));
    % Create random params for model fittings
    params =[]; 
    for k = 1:numel(model.paramsfnc)
        params=[params,eval(model.paramsfnc{k})];
    end
    %Stimulate model
    player_sim.params = params;
    stats_sim=simBandit(player_sim,taskparams,n_stim);
    %Fit stimulated results
    [fitpar, ~, ~, ~]=fit_fun(stats_sim,model.fun,model.initpar,model.lb,model.ub);
    
    fit (:,count) = player_sim.params;
    sim (:,count) = fitpar;
end
delete(f)
%% Figure
figure; 

for k=1:numel(model.paramsfnc)
    subplot(1,numel(model.paramsfnc),k); hold on
    plot( sim(k,:),fit(k,:),'r+','MarkerSize',20);
    ylabel('fit');xlabel('stimulated')
    title(model.params{k})
    if ismember(model.paramsfnc(k), 'exprnd(10)')
        set(gca,'xscale','log','yscale','log')
    end 
     xl = get(gca, 'xlim');
    plot(xl, xl, 'k--')
end

simID = num2str(round(rand(1)*100));
print(gcf,'-dpng',fullfile(save_path, ['figs_simulation-recovery', player_sim.model_type, '_' simID ]));
saveas(gcf, fullfile(save_path, ['figs_simulation-recovery', player_sim.model_type, '_' simID]), 'fig');

model.nStim = n_stim;
save (fullfile(save_path,['parameterRecoveryValues','_', simID,'.mat']),'model', 'sim', 'fit')

end