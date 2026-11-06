function hFig = bandit_compareSimulation2experimentalData(dataIndex,compareModels,simPath,animalid)
% %%bandit_compareSimulation2experimentalData(compareModels,simPath)%%


%% Create stats if not created already.
for k=1:numel(compareModels)
    if strcmp(compareModels{k}, 'experimental')
        save_name = fullfile(simPath.stats_mat,[compareModels{k},'_stats.mat']);
        if ~exist(save_name,'file')
            fprintf(['Creating stats for %s.\n'],compareModels{k});
            %concatenate the sessions for this one animal
            [~, trials, nRules] = merge_sessions(dataIndex);
            stats = value_getTrialStats(trials, nRules);
            stats = value_getTrialStatsMore(stats);
            save(save_name, 'stats');
        else
            fprintf(['Previosly created values will be used. Path: %s \n'],simPath.stats_mat);
        end
    else
        save_name = fullfile(simPath.stats_mat,[compareModels{k},'_',animalid,'_stats.mat']);
        if ~exist(save_name,'file')
            fprintf(['Simulation file for ''%s'' is not found in stats_mat folder.'...
                '\nSimulation will take ~ 2 hours.\n'],compareModels{k});
            n_stim = 1e3*350;
            bandit_stimulation (compareModels{k},simPath,n_stim,[],animalid,1)
            % bandit_stimulation(model_type,simPath,...
            %                    n_stim,custom_params,animalid,statsonly) %%
        else
            fprintf(['Previosly created values will be used. Path: %s \n'],simPath.stats_mat);
        end
    end
end% model for loop ends

%% get stats - calculated vaulue saved in mat_stats folder
for k= 1:numel(compareModels)
    if strcmp(compareModels{k}, 'experimental')
    load (fullfile(simPath.stats_mat,[compareModels{k},'_stats.mat']))
    else
    load (fullfile(simPath.stats_mat,[compareModels{k},'_',animalid,'_stats.mat']))
    end
    stats = value_getTrialStatsMore(stats);
    dat(k).stats = stats;
    dat(k).title = compareModels{k};
    
    %% plot tendency to predict upcoming reversal
    j=1;
    trials_back =10;
    %store the x and y variables to be made into a scatter plot
    ind = find(stats.blockTrialtoCrit<=20);% & stats.followingBlockChangePointTrialIndex<20);
    dat(k).sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
    dat(k).sw_LRVsChoice{j}.range{1}=[0 30];
    dat(k).sw_LRVsChoice{j}.range{2}=[0.6 1];
    dat(k).sw_LRVsChoice{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsChoice{j}.label{2}={'P(better option)';'pre-switch'};
    
    %Slope / MidPoint
    L2_ranges=[1:1:30 ; 1:1:30]' ;       % Random Block length steps
    L1_ranges= ones(size(L2_ranges,1),2).*[10 20];    % BehCriteria  
    trials_forward =50;

    sw_stats_output = choice_switch_stats_random(stats,trials_back,trials_forward,L1_ranges,L2_ranges);
    
    dat(k).sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output.stath(1,:)'];
    dat(k).sw_LRVsMidpoint{j}.range{1}=[0 30];
    dat(k).sw_LRVsMidpoint{j}.range{2}=[3 10];
    dat(k).sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'};{'fraction of trials = 0.5'}];
    %% more behavioural outcome across Lrandom
    dat(k).sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
    dat(k).sw_LRVsHR{j}.range{1}=[0 30];
    dat(k).sw_LRVsHR{j}.range{2}=[70 90];
    dat(k).sw_LRVsHR{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsHR{j}.label{2}={'Hit rates (%)'};
    
    dat(k).sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
    dat(k).sw_LRVsRR{j}.range{1}=[0 30];
    dat(k).sw_LRVsRR{j}.range{2}=[45 65];
    dat(k).sw_LRVsRR{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsRR{j}.label{2}={'Reward rates (%)'};
    
    dat(k).sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
    dat(k).sw_LRVsWinStay{j}.range{1}=[0 30];
    dat(k).sw_LRVsWinStay{j}.range{2}=[0.7 1];
    dat(k).sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsWinStay{j}.label{2}={'P (stay | win)'};
    
    dat(k).sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
    dat(k).sw_LRVsLooseSwitch{j}.range{1}=[0 30];
    dat(k).sw_LRVsLooseSwitch{j}.range{2}=[0 0.3];
    dat(k).sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
    dat(k).sw_LRVsLooseSwitch{j}.label{2}={'P (switch | lose)'};
end

%% plot figures
hFig = plot_binxaveragey_simulation([{dat(1).sw_LRVsChoice,dat(2).sw_LRVsChoice,dat(3).sw_LRVsChoice};...
    {dat(1).sw_LRVsMidpoint,dat(2).sw_LRVsMidpoint,dat(3).sw_LRVsMidpoint};...
    {dat(1).sw_LRVsHR,dat(2).sw_LRVsHR,dat(3).sw_LRVsHR};...
    {dat(1).sw_LRVsRR,dat(2).sw_LRVsRR,dat(3).sw_LRVsRR};...
    {dat(1).sw_LRVsWinStay,dat(2).sw_LRVsWinStay,dat(3).sw_LRVsWinStay};...
    {dat(1).sw_LRVsLooseSwitch,dat(2).sw_LRVsLooseSwitch,dat(3).sw_LRVsLooseSwitch}]);
print(gcf,'-dpng',fullfile(simPath.fig,['LR_vs_simulation_all_',animalid]));
saveas(gcf, fullfile(simPath.fig,['LR_vs_simulation_all_',animalid]), 'fig');

