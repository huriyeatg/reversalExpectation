function plot_latentvariables(dataIndex,model_type,simPath)
% %plot_latentvariables(dataIndex,model_type,simPath) %%


save_path  = fullfile(simPath.fig,['latent variables_',model_type]);
if ~exist(save_path,'dir') % create folders
    mkdir(save_path);
end
    
model_path = simPath.model_mat;
animalList = unique(dataIndex.Animal);
k = 1;
sw_LRVslearningrateVsCCP{1}.dat =[];
sw_LRVslearningrateVsRU{1}.dat =[];
sw_hazardrate = nan(1, numel(animalList));
meanLearningRateForEachAnimal = nan(1, numel(animalList));
animalList = animalList([1:26, 28:31]);
for kk = 1: numel(animalList)
    %% load the animal data
    %which session belong to this one animal
    currAnimalSessions = ismember(dataIndex.Animal,animalList(kk));
    
    %concatenate the sessions for this one animal
    [~, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));
    
    stats = value_getTrialStats(trials, nRules);
    stats = value_getTrialStatsMore(stats);
    stats = value_getTrialStatsModel( stats, model_type, model_path,animalList(kk));
    
    if ceil(1/stats.hazardrate)<20
    %% hazardrate
    sw_hazardrate(k) = stats.hazardrate;
    meanLearningRateForEachAnimal(k) = nanmean(stats.meanLearningRate); 
    %% plot L random mean block learning rate
    
    ind = find(stats.blockTrialtoCrit<21 & stats.blockTrialtoCrit>9);% & stats.followingBlockChangePointTrialIndex<20);
    sw_LRVslearningrate{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanLearningRate(ind)];
    sw_LRVslearningrate{k}.range{1}=[0 30];
    sw_LRVslearningrate{k}.range{2}=[0.4 0.5];
    sw_LRVslearningrate{k}.label{1}={'L_{Random}'};
    sw_LRVslearningrate{k}.label{2}={'Average learning rate'};

    % relative uncertainy
    sw_LRVstrialEntropy{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanTrialEntropy(ind)];
    sw_LRVstrialEntropy{k}.range{1}=[0 30];
    sw_LRVstrialEntropy{k}.range{2}=[0.25 0.35];
    sw_LRVstrialEntropy{k}.label{1}={'L_{Random}'};
    sw_LRVstrialEntropy{k}.label{2}={'Average relative uncertainty'};
    
%     %number of detected change points
%     sw_LRVsChangePointDetection{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointDetection(ind)];
%     sw_LRVsChangePointDetection{k}.range{1}=[0 30];
%     sw_LRVsChangePointDetection{k}.range{2}=[0 4];
%     sw_LRVsChangePointDetection{k}.label{1}={'L_{Random}'};
%     sw_LRVsChangePointDetection{k}.label{2}={'Number of change-points'};
    
    %block change point - Probability
    sw_LRVsChangePointProbability{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointMeanProbability(ind)];
    sw_LRVsChangePointProbability{k}.range{1}=[0 30];
    sw_LRVsChangePointProbability{k}.range{2}=[0.15 0.3 ];
    sw_LRVsChangePointProbability{k}.label{1}={'L_{Random}'};
    sw_LRVsChangePointProbability{k}.label{2}={'Average change-point probability'};
    
    % learning rate vs CPP
    sw_LRVslearningrateVsCCP{1}.dat=[sw_LRVslearningrateVsCCP{1}.dat;stats.blockChangePointMeanProbability(ind) stats.meanLearningRate(ind)];
    sw_LRVslearningrateVsCCP{1}.range{1}=[0 1];
    sw_LRVslearningrateVsCCP{1}.range{2}=[0 1];
    sw_LRVslearningrateVsCCP{1}.label{1}={'Average change-point probability'};
    sw_LRVslearningrateVsCCP{1}.label{2}={'Average learning rate'};
    
    % learning rate vs RU
    sw_LRVslearningrateVsRU{1}.dat=[sw_LRVslearningrateVsRU{1}.dat;stats.meanTrialEntropy(ind) stats.meanLearningRate(ind)];
    sw_LRVslearningrateVsRU{1}.range{1}=[0 1];
    sw_LRVslearningrateVsRU{1}.range{2}=[0 1];
    sw_LRVslearningrateVsRU{1}.label{1}={'Average relative uncertainty'};
    sw_LRVslearningrateVsRU{1}.label{2}={'Average learning rate'};
    %% latent variables around switch
    trials_back =10;
    n=[-trials_back:1:trials_back]';
    L1_ranges=[10 20;10 20;10 20;10 20]; 
    L2_ranges=[0 4;5 9;10 14;15 30];        % Random Block length steps
    % L2_ranges=[10 14;15 20;21 40;41 60];  % Block length -all steps
    numRange = size(L1_ranges,1);
    numtrans=zeros(1,numRange);
    tempLR=zeros(1+2*trials_back,numRange); 
    tempCPP=zeros(1+2*trials_back,numRange); 
    tempRU =zeros(1+2*trials_back,numRange); 
    for i=1:numel(stats.blockTrialtoCrit)
         range_idx = (stats.blockTrialtoCrit(i) >= L1_ranges(:,1)) & (stats.blockTrialtoCrit(i) <= L1_ranges(:,2)) &...
            (stats.blockTrialRandomAdded(i) >= L2_ranges(:,1)) & (stats.blockTrialRandomAdded(i) <= L2_ranges(:,2));
      if sum(range_idx) == 1  && ~isempty(stats.switchLearningRate{i})
          numtrans(range_idx)=numtrans(range_idx)+1;
          tempLR(:,range_idx)=tempLR(:,range_idx)+stats.switchLearningRate{i};
          tempCPP(:,range_idx)=tempCPP(:,range_idx)+stats.switchCPP{i};
          tempRU(:,range_idx) =tempRU(:,range_idx)+stats.switchRU{i};
      end
    end

    sw_lr_output{k}.n=n;
    sw_lr_output{k}.prob=tempLR./numtrans;
    sw_lr_output{k}.numRange=numRange;
    sw_lr_output{k}.L1_ranges=L1_ranges;
    sw_lr_output{k}.L2_ranges=L2_ranges;

    sw_cpp_output{k}.n=n;
    sw_cpp_output{k}.prob=tempCPP./numtrans;
    sw_cpp_output{k}.numRange=numRange;
    sw_cpp_output{k}.L1_ranges=L1_ranges;
    sw_cpp_output{k}.L2_ranges=L2_ranges;

    sw_ru_output{k}.n=n;
    sw_ru_output{k}.prob=tempRU./numtrans;
    sw_ru_output{k}.numRange=numRange;
    sw_ru_output{k}.L1_ranges=L1_ranges;
    sw_ru_output{k}.L2_ranges=L2_ranges;
    k= k+1;
    end
end

%% learning rate analysis
figure % compare with Change point probabilities
subplot(2,2,1) 
scatter(sw_LRVslearningrateVsCCP{1}.dat(:,1), sw_LRVslearningrateVsCCP{1}.dat(:,2),30,'.k');
axis('square')
xlabel( sw_LRVslearningrateVsCCP{1}.label{1});
ylabel( sw_LRVslearningrateVsCCP{1}.label{2});
print(gcf,'-dpng',fullfile(save_path,'LRvsCPP'));
saveas(gcf, fullfile(save_path,'LRvsCPP'), 'fig');


figure % compare with trial entropy
subplot(2,2,1) 
scatter(sw_LRVslearningrateVsRU{1}.dat(:,1), sw_LRVslearningrateVsRU{1}.dat(:,2),30,'.k');
axis('square')
xlabel( sw_LRVslearningrateVsRU{1}.label{1});
ylabel( sw_LRVslearningrateVsRU{1}.label{2});
print(gcf,'-dpng',fullfile(save_path,'LRvstrialentropy'));
saveas(gcf, fullfile(save_path,'LRvstrialentropy'), 'fig');

% learning rate for switch
params.ylim = [ 0 1];
params.ylabel ={'Learning rate'};
plot_switch_latent(sw_lr_output,params); 
print(gcf,'-dpng',fullfile(save_path,'switches_learningrate'));
saveas(gcf, fullfile(save_path,'switches_learningrate'), 'fig');

% learning rates at switch for different block lengths (L random)
params.ylim = [ 0 1];
params.ylabel ={'Learning rate'};
plot_switch_latent_random(sw_lr_output,params); 
print(gcf,'-dpng',fullfile(save_path,'switches_learningrate-random'));
saveas(gcf, fullfile(save_path,'switches_learningrate-random'), 'fig');

%% hazard rate analysis
figure; subplot(2,2,1) 
edges = 0.05:0.05:0.9;
histogram(sw_hazardrate,edges, 'FaceColor','k'); box off; hold on
plot(ones(1,11)*nanmean(sw_hazardrate),0:10,'r')
plot(ones(1,11)*0.0909090,0:10,'g')
ylabel('Number of animals'); xlabel('Hazard rate')
print(gcf,'-dpng',fullfile(save_path,'hazardrate-histogram'));
saveas(gcf, fullfile(save_path,'hazardrate-histogram'), 'fig');

% mean learning rate vs hazard rate
figure 
subplot(2,2,1) 
scatter(meanLearningRateForEachAnimal, sw_hazardrate,100,'vk','MarkerFaceColor','k');
xlabel( 'Mean learning rate');
ylabel( 'Fitted hazard rate');
print(gcf,'-dpng',fullfile(save_path,'LRvsHazardRate'));
saveas(gcf, fullfile(save_path,'LRvsHazardRate'), 'fig');

%% Change-point probability analysis
% CPP for switch
params.ylim = [ 0.25 0.5];
params.ylabel ={'CPP '};
plot_switch_latent(sw_cpp_output,params); 
axis('square')
print(gcf,'-dpng',fullfile(save_path,'switches_CPP'));
saveas(gcf, fullfile(save_path,'switches_CPP'), 'fig');

% Cpp for different block lengths 
params.ylim = [ 0.25 0.5];
params.ylabel ={'CPP'};
plot_switch_latent_random(sw_cpp_output,params);%15
print(gcf,'-dpng',fullfile(save_path,'switches_random_CPP'));
saveas(gcf, fullfile(save_path,'switches_random_CPP'), 'fig');

%% Relative uncertainty analysis
% RU for switch
params.ylim = [ 0.2 0.5];
params.ylabel ={'RU'};
plot_switch_latent(sw_ru_output,params); 
axis('square')
print(gcf,'-dpng',fullfile(save_path,'switches_RU'));
saveas(gcf, fullfile(save_path,'switches_RU'), 'fig');

% RU for different block lengths 
params.ylim = [ 0.2 0.5];
params.ylabel ={'RU'};
plot_switch_latent_random(sw_ru_output,params);
print(gcf,'-dpng',fullfile(save_path,'switches_random_RU'));
saveas(gcf, fullfile(save_path,'switches_random_RU'), 'fig');

%% Comparisons across block length ( Lrandom)
plot_multibinxaveragey([ {sw_LRVslearningrate},{sw_LRVstrialEntropy},...
    {sw_LRVsChangePointProbability}]);
print(gcf,'-dpng',fullfile(save_path,'parametersVsLrandom'));
saveas(gcf, fullfile(save_path,'parametersVsLrandom'), 'fig');

%% compare beta for value vs 
plot_betaratio(model_type, model_path)

print(gcf,'-dpng',fullfile(save_path,'beta-ratio'));
saveas(gcf, fullfile(save_path,'beta-ratio'), 'fig');


end

