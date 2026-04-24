function bandit_neuromodulatorPerSession(dataIndex,savebehfigpath)
% % bandit_neuromodulatorPerSession %
%PURPOSE:   Analyze bandit behavior for each animal
%AUTHORS:   H Atilgan and AC Kwan 200210
%
%INPUT ARGUMENTS
%   dataIndex:    a table of the data files
%   save_path:    path for saving the plots
%
%OUTPUT ARGUMENTS
%

%% go through each session
nSession = size(dataIndex,1);

fs = 20;
tWindow = fs*6; % for -2 to 4 sec - total 6 sec.

if ~exist(savebehfigpath,'dir')
    mkdir(savebehfigpath);
end
for j = 1:nSession
    
    % load session data
    load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_beh.mat']));
    trials = value_getTrialMasks(trialData);
    
    load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_dff.mat']));
    trials.dffN_zscore = zscore(data.dffN(1:end-1,1:tWindow));
    trials.dffN = data.dffN(1:end-1,1:tWindow);
    trials.dff  = data.dff(1:end-1,1:tWindow);
    
    % check trial numbers across beh and dff files
    if numel(trials.go)~= size(trials.dff,1)
        nTrials = min(numel(trials.go), size(trials.dffN,1));
        clear trialsNew
        fields=fieldnames(trials);
        for j = 1:numel(fields)
            trialsNew.(fields{j}) = trials.(fields{j})(1:nTrials,:);
        end
        trials = trialsNew;
    end
    
    nRules = sessionData.nRules;
    stats = value_getTrialStats(trials, nRules);
    stats = value_getTrialStatsMore(stats);
    
    tlabel = dataIndex.LogFileName{j}(1:end-4);
    %% Calculate trial-averaged dF/F
    figure; hold on
    cond=[[{'High reward'},    {{'left','reward','L70R10'}},{{'right','reward','L10R70'}}];...
        [{'High no reward'},{{'left','noreward','L70R10'}},{{'right','noreward','L10R70'}}];...
        [{'Low reward'},    {{'left','reward','L10R70'}},{{'right','reward','L70R10'}}];...
        [{'Low no reward'}, {{'left','noreward','L10R70'}},{{'right','noreward','L70R10'}}]];
    for k = 1:size(cond,1)
        fieldname = cond(k,2);
        trialMask1 = getMask(trials,fieldname{:});
        fieldname = cond(k,3);
        trialMask2 = getMask(trials,fieldname{:});
        trialMask = (trialMask1 + trialMask2)>0;
        
        temp_psth.signal = trials.dff(trialMask,:);
        temp_psth.t = -1.95:1/fs:4;
        temp_psth.psth_label = cond{k,1};
        
        xlabel_st = 'Time from stimulus (s)';
        subplot(1,4,k)
        plot_snake(temp_psth,[0 6.5],xlabel_st);
    end
    print(gcf,'-dsvg',fullfile(savebehfigpath,[tlabel,'_neuralSignal']));
    saveas(gcf, fullfile(savebehfigpath,[tlabel,'_neuralSignal']), 'fig');

end

    
    %% CHOICE AND OUTCOME: Multiple linear regression  - choice and reward and their interaction
    
    params=[];
    
    %first predictor is choice; dummy-code: left=-1, right=1, miss=NaN
    params.choiceEvent=NaN(size(trials.left));
    params.choiceEvent(trials.left) = -1;
    params.choiceEvent(trials.right) = 1;
    %second predictor is outcome; dummy-code: reward=0, omit/error=-1, double-reward=1, miss=NaN
    params.outcomeEvent=NaN(size(trials.hit));
    params.outcomeEvent(trials.hit) = 0;  %rationale: for well-learned subjects, this is par on course
    params.outcomeEvent(trials.err) = NaN;  %well-learned animal, source of error unclear
    params.outcomeEvent(trials.omitreward) = -1;
    params.outcomeEvent(trials.doublereward) = 1;
    
    params.trigTime = trialData.cueTimes;
    params.xtitle = 'Time from stimulus (s)';
    params.window = [-2:0.5:6.5];
    params.nback = 2;       %how many trials back to regress against
    params.interaction = true; %consider interaction terms (our data do not have enough trials)
    params.pvalThresh = 0.01;   %p-value for coefficient be considered significant
    
    %only perform analysis on trials with a response (when animal is engaged)
    fieldname={'left','right'};
    trialMask = getAnyMask(trials,fieldname);
    for j=1:numel(cells.dFF)
        reg_cr{j}=linear_regr( cells.dFF{j}, cells.t, [params.choiceEvent params.outcomeEvent], params.trigTime, trialMask, params );
    end
    
    tlabel={'C(n)','C(n-1)','C(n-2)','R(n)','R(n-1)','R(n-2)','C(n)xR(n)','C(n-1)xR(n-1)','C(n-2)xR(n-2)'};
    plot_regr(reg_cr,params.pvalThresh,tlabel,params.xtitle);
    print(gcf,'-dpng','MLR-choiceoutcome');    %png format
    saveas(gcf, 'MLR-choiceoutcome', 'fig');
    
    save(fullfile(savematpath,'dff_and_beh.mat'),...
        'reg_cr','-append');
    
    
    
end
