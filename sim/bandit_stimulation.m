function bandit_stimulation (model_type,simPath,n_stim,custom_params,animalid,statsonly)
% bandit_stimulation(model_type,simPath,...
%                    n_stim,custom_params,animalid,statsonly) %%

%%
save_path  = fullfile(simPath.fig,model_type);
model_path = simPath.model_mat;
stat_path  = simPath.stats_mat;
if ~exist(save_path,'dir')
    mkdir(save_path);
end

if nargin==3
    n_stim = 1e5;
    disp (['Simulation will be based on the mean values for ',model_type, 'for ',num2str(n_stim),' trials.']);
    animalid  = [];
    statsonly = 0;
elseif nargin==5 || nargin==4
    disp (['Simulation will be based on the mean values for ',model_type, 'for ',num2str(n_stim),' trials.']);
    statsonly = 0;
end
%% Bandit task parameters
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean

switch model_type   
    case 'belief_CK'
        player_sim.model_type='belief_CK';
        load(fullfile(model_path, [player_sim.model_type '.mat']));
        fitparMat = cell2mat(fitpar');
        player_sim.median = nanmedian(fitparMat,1);
        if ~isempty(animalid)
            nlabel = animalid;
            ind = strcmp(animalid, animal);
            player_sim.params = fitparMat(ind,:)
        else
            nlabel  = 'median';
            player_sim.params =  player_sim.median
        end
        
    case 'FQ_RPE_CK'
        player_sim.model_type='FQ_RPE_CK';
        load(fullfile(model_path, [player_sim.model_type '.mat']));
        fitparMat = cell2mat(fitpar');
        player_sim.median = nanmedian(fitparMat,1);
        if ~isempty(animalid)
            nlabel = animalid;
            ind = ismember(animalid, animal);
            player_sim.params = fitparMat(ind,:)
        else
            nlabel  = 'median';
            player_sim.params =  player_sim.median
        end
        
    case 'customize'
        nlabel  = 'customize'; 
        player_sim.model_type=custom_params.model_type;
        player_sim.params = custom_params.params
        
    otherwise
        error ('Model parameters have not defined yet.')
end
stats = simBandit(player_sim,taskparams,n_stim);
stats = value_getTrialStatsMore(stats);
player_sim.simulation = 'true';
%stats = value_getTrialStatsModel( stats, model_type, model_path,animalid,player_sim);

if statsonly
    save_name = fullfile(stat_path,[model_type,'_',nlabel,'_stats.mat']);
    save(save_name, 'stats', 'player_sim');
else
    %% HRSide random
    trials_back = 10;
    
    L1_ranges=[10 20;10 20;10 20;10 20]; %consider only subset of blocks within the range, for trials to criterion
    L2_ranges=[0 4;5 9;10 14;15 30];      %consider only subset of blocks within the range, for random added number of trials
    sw_hrside_random_output=choice_switch_hrside_random(stats,trials_back,L1_ranges,L2_ranges);
    
    %% plot tendency to predict upcoming reversal
    j=1;
    %store the x and y variables to be made into a scatter plot
    ind = find(stats.blockTrialtoCrit<20);% & stats.followingBlockChangePointTrialIndex<20);
    sw_LRVsChoice{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceAtSwitch(ind)];
    sw_LRVsChoice{j}.range{1}=[0 30];
    sw_LRVsChoice{j}.range{2}=[0.6 1];
    sw_LRVsChoice{j}.label{1}={'L_{Random}'};
    sw_LRVsChoice{j}.label{2}={'Fraction of trials';'selecting initial better option'};
    
    %Slope / MidPoint
    L2_ranges=[1:2:30 ; 3:2:32]';        % Random Block length steps
    L1_ranges= ones(size(L2_ranges,1),2).*[10 20];    % BehCriteria
    sw_stats_output{j} = choice_switch_stats_random(stats,trials_back,L1_ranges,L2_ranges);
      
    sw_LRVsMidpoint{j}.dat=[L2_ranges(:,1) sw_stats_output{j}.stath(1,:)'];
    sw_LRVsMidpoint{j}.range{1}=[0 30];
    sw_LRVsMidpoint{j}.range{2}=[0 10];
    sw_LRVsMidpoint{j}.label{1}={'L_{Random}'};
    sw_LRVsMidpoint{j}.label{2}= [{'Trials to reach midpoint'};{'fraction of trials = 0.5'}];
    
    %% plot L random mean block learning rate
    
%     ind = find(stats.blockTrialtoCrit<21 & stats.blockTrialtoCrit>9);% & stats.followingBlockChangePointTrialIndex<20);
%     sw_LRVslearningrate{j}.dat=[stats.blockTrialRandomAdded(ind) stats.meanLearningRate(ind)];
%     sw_LRVslearningrate{j}.range{1}=[0 30];
%     sw_LRVslearningrate{j}.range{2}=[0.15 0.3];
%     sw_LRVslearningrate{j}.label{1}={'L_{Random}'};
%     sw_LRVslearningrate{j}.label{2}={'Average learning rate'};
%     
%     %block change point - Probability
%     sw_LRVsChangePointProbability{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointMeanProbability(ind)];
%     sw_LRVsChangePointProbability{j}.range{1}=[0 30];
%     sw_LRVsChangePointProbability{j}.range{2}=[0.02 0.12];
%     sw_LRVsChangePointProbability{j}.label{1}={'L_{Random}'};
%     sw_LRVsChangePointProbability{j}.label{2}={'Average change-point probability'};
%     
%     % trials to reach next block changepoint.
%     sw_LRVstrials2changepoint{j}.dat=[stats.blockTrialRandomAdded(ind,1) stats.trials2changepoint(ind,1)];
%     sw_LRVstrials2changepoint{j}.range{1}=[0 30];
%     sw_LRVstrials2changepoint{j}.range{2}=[0 20];
%     sw_LRVstrials2changepoint{j}.label{1}={'L_{Random}'};
%     sw_LRVstrials2changepoint{j}.label{2}=[{'Trials to the first change-point'} ;{'in the next block'}];
    
%     %% switch learning rate
%     trials_back =10;
%     n=[-trials_back:1:trials_back]';
%     L1_ranges=[10 20;10 20;10 20;10 20];
%     L2_ranges=[10 14;15 20;21 40;41 60];
%     numRange = size(L1_ranges,1);
%     numtrans=zeros(1,numRange);
%     tempLR=zeros(1+2*trials_back,numRange);
%     tempCPP=zeros(1+2*trials_back,numRange);
%     for i=1:numel(stats.blockTrialtoCrit)
%         range_idx = (stats.blockTrialtoCrit(i) >= L1_ranges(:,1)) & (stats.blockTrialtoCrit(i) <= L1_ranges(:,2)) &...
%             (stats.blockLength(i) >= L2_ranges(:,1)) & (stats.blockLength(i) <= L2_ranges(:,2));
%         if sum(range_idx) == 1  && ~isempty(stats.switchLearningRate{i})
%             numtrans(range_idx)=numtrans(range_idx)+1;
%             tempLR(:,range_idx)=tempLR(:,range_idx)+stats.switchLearningRate{i};
%         end
%     end
%     
%     sw_lr_output.n=n;
%     sw_lr_output.prob=tempLR./numtrans;
%     sw_lr_output.numRange=numRange;
%     sw_lr_output.L1_ranges=L1_ranges;
%     sw_lr_output.L2_ranges=L2_ranges;
%     
%     sw_cpp_output.n=n;
%     sw_cpp_output.prob=tempCPP./numtrans;
%     sw_cpp_output.numRange=numRange;
%     sw_cpp_output.L1_ranges=L1_ranges;
%     sw_cpp_output.L2_ranges=L2_ranges;
    
    %% more behavioural outcome across Lrandom
    sw_LRVsHR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.hitrates(ind)];
    sw_LRVsHR{j}.range{1}=[0 30];
    sw_LRVsHR{j}.range{2}=[0.5 1];
    sw_LRVsHR{j}.label{1}={'L_{Random}'};
    sw_LRVsHR{j}.label{2}={'Hit rates'};
    
    sw_LRVsRR{j}.dat=[stats.blockTrialRandomAdded(ind) stats.rewardrates(ind)];
    sw_LRVsRR{j}.range{1}=[0 30];
    sw_LRVsRR{j}.range{2}=[0.4 0.9];
    sw_LRVsRR{j}.label{1}={'L_{Random}'};
    sw_LRVsRR{j}.label{2}={'Reward rates'};
    
    sw_LRVsWinStay{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pWinStay(ind)];
    sw_LRVsWinStay{j}.range{1}=[0 30];
    sw_LRVsWinStay{j}.range{2}=[0.7 1];
    sw_LRVsWinStay{j}.label{1}={'L_{Random}'};
    sw_LRVsWinStay{j}.label{2}={'P(win|stay)'};
    
    sw_LRVsLooseSwitch{j}.dat=[stats.blockTrialRandomAdded(ind) stats.pLooseSwitch(ind)];
    sw_LRVsLooseSwitch{j}.range{1}=[0 30];
    sw_LRVsLooseSwitch{j}.range{2}=[0 0.4];
    sw_LRVsLooseSwitch{j}.label{1}={'L_{Random}'};
    sw_LRVsLooseSwitch{j}.label{2}={'P(loose|switch)'};
    
    
    %% plot figures
    tlabel=[player_sim.model_type ', n=' num2str(n_stim) ' trials',...
        ' \alpha=',num2str(player_sim.params(1)),...
        ' \beta=',num2str(player_sim.params(2))];
    
    plot_switch_hrside_random(sw_hrside_random_output,tlabel);
    print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_hrside_random_',num2str(n_stim),'_',nlabel]));
    saveas(gcf, fullfile(save_path,[model_type,'_switches_hrside_random',num2str(n_stim),'_',nlabel]), 'fig');
    
    plot_multibinxaveragey([ {sw_LRVsChoice},{sw_LRVsMidpoint}])
    print(gcf,'-dsvg',fullfile(save_path,[model_type,'_LRVsChoice_',num2str(n_stim),'_',nlabel]));
    saveas(gcf, fullfile(save_path,[model_type,'_switches_LRVsChoice',num2str(n_stim),'_',nlabel]), 'fig');
%     
%     plot_multibinxaveragey([ {sw_LRVslearningrate},{sw_LRVstrials2changepoint},...
%        {sw_LRVsChangePointProbability}]);
%     print(gcf,'-dsvg',fullfile(save_path,[model_type,'_parametersVsLrandom_',num2str(n_stim),'_',nlabel]));
%     saveas(gcf, fullfile(save_path,[model_type,'_parametersVsLrandom_',num2str(n_stim),'_',nlabel]), 'fig');
    
    plot_multibinxaveragey([ {sw_LRVsHR};{sw_LRVsRR}; ...
        {sw_LRVsWinStay};{sw_LRVsLooseSwitch}]);
    print(gcf,'-dsvg',fullfile(save_path,[model_type,'_hitRatesVsLrandom_',num2str(n_stim),'_',nlabel]));
    saveas(gcf, fullfile(save_path,[model_type,'_hitRatesVsLrandom_',num2str(n_stim),'_',nlabel]), 'fig');
    
    % params.ylim = [ 0.2 0.4];
    % params.ylabel ={'Learning rate'};
    % plot_switch_lr_random(sw_lr_output,params);
    % print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_learningrate_',num2str(n_stim),'_',lb]));
    % saveas(gcf, fullfile(save_path,[model_type,'_switchesLR',num2str(n_stim),'_',lb]), 'fig');
    
    % params.ylim = [ 0 0.5];
    % params.ylabel ={'CPP'};
    % plot_switch_lr_random(sw_cpp_output,params);
    % print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_CPP_',num2str(n_stim),'_',lb]));
    % saveas(gcf, fullfile(save_path,[model_type,'_switchesCPP',num2str(n_stim),'_',lb]), 'fig');
    
    save_name = fullfile(save_path,[model_type,'_',nlabel,'_stats.mat']);
    save(save_name, 'stats', 'player_sim');
    close all
end
end
