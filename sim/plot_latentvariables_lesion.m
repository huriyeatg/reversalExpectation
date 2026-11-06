function plot_latentvariables_lesion(dataIndex,model_type,simPath)
% %plot_latentvariables_lesion %

save_path = simPath.fig;
if ~exist(save_path,'dir') % create folders
    mkdir(save_path);
end

dat = struct;
animalList = unique(dataIndex.Animal);
uniLesion = false;
for kk =1:2 % pre/post
    
    for k = 1: numel(animalList)
        %% load the animal data
        if kk == 1
            %pre lesion sessions
            currAnimalSessions = ismember(dataIndex.Animal,animalList(k))& isnan(dataIndex.Lesioned);
            curr_path = simPath.model_mat;
        else
            %post lesion sessions
            currAnimalSessions = ismember(dataIndex.Animal,animalList(k))& ~isnan(dataIndex.Lesioned);
            curr_path = simPath.model_mat_lesion;
        end
        %concatenate the sessions for this one animal
        [trialData, trials, nRules] = merge_sessions(dataIndex(currAnimalSessions,:));
        trialData.presCodeSet = trialData.presCodeSet(1);
        % if unilateral lesion data:
        % leave right lesion data alone and flip choice vector for left lesion data,
        % so effectively left == side contralateral to an unilateral lesion
        ind =find(currAnimalSessions==1);
        if dataIndex.LesionSide(ind(1))==1  %if lesioned side is LEFT
            trialData = fliptrialData(trialData);
            uniLesion = true;  %set flag that we have flipped a subset of data
        end
        
        trials = value_getTrialMasks(trialData);
        stats = value_getTrialStats(trials, nRules);
        stats = value_getTrialStatsMore(stats);
        stats = value_getTrialStatsModel( stats, model_type, curr_path, animalList(k));
        
        %% Learning rate
        ind = stats.blockTrialtoCrit<21 ; % Left/criteria fitting block
        dat(kk).sw_LRVslearningrate{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanLearningRate(ind)];
        dat(kk).sw_LRVslearningrate{k}.range{1}=[0 30];
        dat(kk).sw_LRVslearningrate{k}.range{2}=[0.2 0.28];
        dat(kk).sw_LRVslearningrate{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVslearningrate{k}.label{2}={'Average learning rate'};

        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==1; % Left/criteria fitting block
        dat(kk).sw_LRVslearningrateL{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanLearningRate(ind)];
        dat(kk).sw_LRVslearningrateL{k}.range{1}=[0 30];
        dat(kk).sw_LRVslearningrateL{k}.range{2}=[0.45 0.5];
        dat(kk).sw_LRVslearningrateL{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVslearningrateL{k}.label{2}={'Average learning rate'};
        
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==2; % Right/ criteria fitting blocks
        dat(kk).sw_LRVslearningrateR{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanLearningRate(ind)];
        dat(kk).sw_LRVslearningrateR{k}.range{1}=[0 30];
        dat(kk).sw_LRVslearningrateR{k}.range{2}=[0.2 0.28];
        dat(kk).sw_LRVslearningrateR{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVslearningrateR{k}.label{2}={'Average learning rate'};
        
        % relative uncertainy
        ind = stats.blockTrialtoCrit<21; % Left/criteria fitting block
        dat(kk).sw_LRVstrialEntropy{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanTrialEntropy(ind)];
        dat(kk).sw_LRVstrialEntropy{k}.range{1}=[0 30];
        dat(kk).sw_LRVstrialEntropy{k}.range{2}=[4.2 4.3];
        dat(kk).sw_LRVstrialEntropy{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVstrialEntropy{k}.label{2}={'Average relative uncertainty'};

        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==1; % Left/criteria fitting block
        dat(kk).sw_LRVstrialEntropyL{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanTrialEntropy(ind)];
        dat(kk).sw_LRVstrialEntropyL{k}.range{1}=[0 30];
        dat(kk).sw_LRVstrialEntropyL{k}.range{2}=[4.2 4.3];
        dat(kk).sw_LRVstrialEntropyL{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVstrialEntropyL{k}.label{2}={'Average relative uncertainty'};
        
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==2; % Right/ criteria fitting blocks
        dat(kk).sw_LRVstrialEntropyR{k}.dat=[stats.blockTrialRandomAdded(ind) stats.meanTrialEntropy(ind)];
        dat(kk).sw_LRVstrialEntropyR{k}.range{1}=[0 30];
        dat(kk).sw_LRVstrialEntropyR{k}.range{2}=[4.2 4.3];
        dat(kk).sw_LRVstrialEntropyR{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVstrialEntropyR{k}.label{2}={'Average relative uncertainty'};
        
        %block change point - Probability
        ind = stats.blockTrialtoCrit<21;
        dat(kk).sw_LRVsChangePointProbability{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointMeanProbability(ind)];
        dat(kk).sw_LRVsChangePointProbability{k}.range{1}=[0 30];
        dat(kk).sw_LRVsChangePointProbability{k}.range{2}=[0.09 0.13];
        dat(kk).sw_LRVsChangePointProbability{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChangePointProbability{k}.label{2}={'Average change-point probability'};

        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==1;
        dat(kk).sw_LRVsChangePointProbabilityL{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointMeanProbability(ind)];
        dat(kk).sw_LRVsChangePointProbabilityL{k}.range{1}=[0 30];
        dat(kk).sw_LRVsChangePointProbabilityL{k}.range{2}=[0.09 0.13];
        dat(kk).sw_LRVsChangePointProbabilityL{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChangePointProbabilityL{k}.label{2}={'Average change-point probability'};
        
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==2;
        dat(kk).sw_LRVsChangePointProbabilityR{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointMeanProbability(ind)];
        dat(kk).sw_LRVsChangePointProbabilityR{k}.range{1}=[0 30];
        dat(kk).sw_LRVsChangePointProbabilityR{k}.range{2}=[0.09 0.13];
        dat(kk).sw_LRVsChangePointProbabilityR{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChangePointProbabilityR{k}.label{2}={'Average change-point probability'};
        
        %number of detected change points
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==1;
        dat(kk).sw_LRVsChangePointDetectionL{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointDetection(ind)];
        dat(kk).sw_LRVsChangePointDetectionL{k}.range{1}=[0 30];
        dat(kk).sw_LRVsChangePointDetectionL{k}.range{2}=[0 4];
        dat(kk).sw_LRVsChangePointDetectionL{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChangePointDetectionL{k}.label{2}={'Number of change-points'};
        
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==2;
        dat(kk).sw_LRVsChangePointDetectionR{k}.dat=[stats.blockTrialRandomAdded(ind) stats.blockChangePointDetection(ind)];
        dat(kk).sw_LRVsChangePointDetectionR{k}.range{1}=[0 30];
        dat(kk).sw_LRVsChangePointDetectionR{k}.range{2}=[0 4];
        dat(kk).sw_LRVsChangePointDetectionR{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVsChangePointDetectionR{k}.label{2}={'Number of change-points'};
        
        % trials to reach next block changepoint.
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==1;
        dat(kk).sw_LRVstrials2changepointL{k}.dat=[stats.blockTrialRandomAdded(ind,1) stats.trials2changepoint(ind,1)];
        dat(kk).sw_LRVstrials2changepointL{k}.range{1}=[0 30];
        dat(kk).sw_LRVstrials2changepointL{k}.range{2}=[0 20];
        dat(kk).sw_LRVstrials2changepointL{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVstrials2changepointL{k}.label{2}=[{'Trials to the first change-point'} ;{'in the next block'}];
        
        % trials to reach next block changepoint.
        ind = stats.blockTrialtoCrit<21 & stats.blockTrans==2;
        dat(kk).sw_LRVstrials2changepointR{k}.dat=[stats.blockTrialRandomAdded(ind,1) stats.trials2changepoint(ind,1)];
        dat(kk).sw_LRVstrials2changepointR{k}.range{1}=[0 30];
        dat(kk).sw_LRVstrials2changepointR{k}.range{2}=[0 20];
        dat(kk).sw_LRVstrials2changepointR{k}.label{1}={'L_{Random}'};
        dat(kk).sw_LRVstrials2changepointR{k}.label{2}=[{'Trials to the first change-point'} ;{'in the next block'}];
        
    end
end

%% plot figures
if uniLesion==true
    tlabel    = [{'Intact'}; {'Lesion'}];
    L2_ranges = [0 4;5 9;10 14;15 30];
    plot_binxaveragey_lesion([ {dat(1).sw_LRVslearningrateL,dat(2).sw_LRVslearningrateL};...
        {dat(1).sw_LRVslearningrateR,dat(2).sw_LRVslearningrateR}],tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion([ {dat(1).sw_LRVsChangePointDetectionL,dat(2).sw_LRVsChangePointDetectionL};...
        {dat(1).sw_LRVsChangePointDetectionR,dat(2).sw_LRVsChangePointDetectionR}],tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion([ {dat(1).sw_LRVstrialEntropyL,dat(2).sw_LRVstrialEntropyL};...
        {dat(1).sw_LRVstrialEntropyR,dat(2).sw_LRVstrialEntropyR}],tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion([ {dat(1).sw_LRVstrials2changepointL,dat(2).sw_LRVstrials2changepointL};...
        {dat(1).sw_LRVstrials2changepointR,dat(2).sw_LRVstrials2changepointR}],tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion([ {dat(1).sw_LRVsChangePointProbabilityL,dat(2).sw_LRVsChangePointProbabilityL};...
        {dat(1).sw_LRVsChangePointProbabilityR,dat(2).sw_LRVsChangePointProbabilityR}],tlabel, L2_ranges,save_path);
    
else
    tlabel    = [{'Intact'}; {'Lesion'}];
    L2_ranges = [0 4;5 9;10 14;15 30];
    plot_binxaveragey_lesion({dat(1).sw_LRVslearningrate,dat(2).sw_LRVslearningrate},tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion( {dat(1).sw_LRVsChangePointProbability,dat(2).sw_LRVsChangePointProbability},tlabel, L2_ranges,save_path);
    
    plot_binxaveragey_lesion({dat(1).sw_LRVstrialEntropy,dat(2).sw_LRVstrialEntropy},tlabel, L2_ranges,save_path);
     
end
end

