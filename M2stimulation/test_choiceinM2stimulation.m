function test_choiceinM2stimulation(dataIndexAll, save_path)

nCrit = 200;
bin = [0 4;5 9;10 14;15 30]; %L2_ranges
nbin=size(bin,1);
stat_data =[];
stat_group =[];
kk=1;

%% Get values for preChoice
clear dat
subset = (ismember(dataIndexAll.Phase,21)==1);    %post-stimulation
dataIndex = dataIndexAll(subset,:);

for j = 1: size(dataIndex,1)
    load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_beh.mat']));
    trials = value_getTrialMasks(trialData);
    stats = value_getTrialStats(trials, sessionData.nRules);
    stats = value_getTrialStatsMore(stats);
    %% Add stimulation info
    % stimulated side
    stats.stRegion = trialData.stimulationRegion-1000;
    
    %stimulation: yes=1; no=1;
    stats.st=nan(numel(trialData.stimulation),1);
    stats.st = stats.stRegion;
    stats.st (stats.stRegion==2) =1;
    
    % which block stimulated
    x0=stats.rule';
    stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    stats.blockStRegion = stats.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    %% now divide the blocks for stimulated vs non-stimulated
    nBlocks = size(stats.blockRule,1);
    
    % define contra vs ipsi blocks in stimulated blocks - contra means 1, ipsi means 0
    bStimInfo =  nan(nBlocks,1);
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 1)  = 0 ;% left stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 2)  = 1 ;% left stimulation & left side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 2)  = 0 ;% right stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 1)  = 1 ;% right stimulation & left side being initial better option
    stats.blockStimulationSide = bStimInfo(1:end-1); % exclude last block
    
    % define contra vs ipsi blocks in control blocks - contra means 1, ipsi means 0
    % two blocks before stimulated blocks
    bControlInfo = [ bStimInfo(3:end);nan; nan]; % 2 blocks before stimulated blocks
    %bControlInfo = [ nan;nan; bStimInfo(1:end-2)]; % 2 blocks after
    %stimulated blocks
    stats.blockControlSide = bControlInfo(1:end-1); % exclude last block
    %% initial better option plots
    % Stimulated: contra vs ipsi
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==0; %general criteria
    dat(kk).sw_LRVsChoiceIpsi{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceIpsi{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceIpsi{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceIpsi{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceIpsi{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==1; %general criteria
    dat(kk).sw_LRVsChoiceContra{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceContra{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceContra{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceContra{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceContra{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    % Control: contra vs ipsi
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==0; %general criteria
    dat(kk).sw_LRVsChoiceIpsiControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceIpsiControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==1; %general criteria
    dat(kk).sw_LRVsChoiceContraControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceContraControl{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceContraControl{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceContraControl{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceContraControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
end

%control
val=[];
for j=1:numel(dat(1).sw_LRVsChoiceIpsiControl)
    val=[val; dat(1).sw_LRVsChoiceIpsiControl{j}.dat];
end
val2=[];
for j=1:numel(dat(1).sw_LRVsChoiceContraControl)
    val2=[val2; dat(1).sw_LRVsChoiceContraControl{j}.dat];
end

y=[]; y2=[];
for j = 1:nbin
    y  = [val(val(:,1)>=bin(j,1) & val(:,1)<=bin(j,2),2)];
    y2 = [val2(val2(:,1)>=bin(j,1) & val2(:,1)<=bin(j,2),2)];
    stat_data = [ stat_data; y;y2];
    stat_group = [stat_group; ones(size(y)).*[j,1,1,1]; ...
        ones(size(y2)).*[j,2,1,1]]; % blockLength,ipsi/contra side,stimulated(2) /control (1), pre (1)/post(2) choice
end

%stimulated
val=[];
for j=1:numel(dat(1).sw_LRVsChoiceIpsi)
    val=[val; dat(1).sw_LRVsChoiceIpsi{j}.dat];
end
val2=[];
for j=1:numel(dat(1).sw_LRVsChoiceContra)
    val2=[val2; dat(1).sw_LRVsChoiceContra{j}.dat];
end

y=[]; y2=[];
for j = 1:nbin
    y  = [val(val(:,1)>=bin(j,1) & val(:,1)<=bin(j,2),2)];
    y2 = [val2(val2(:,1)>=bin(j,1) & val2(:,1)<=bin(j,2),2)];
    stat_data = [ stat_data; y;y2];
    stat_group = [stat_group; ones(size(y)).*[j,1,2,1]; ...
        ones(size(y2)).*[j,2,2,1]]; % blockLength,ipsi/contra side,stimulated(2) /control (1) /control (1), pre (1)/post(2) choice
end

%% Get values for postChoice
clear dat
subset = (ismember(dataIndexAll.Phase,22)==1);    %post-stimulation
dataIndex = dataIndexAll(subset,:);

for j = 1: size(dataIndex,1)
    load(fullfile(dataIndex.BehPath{j},[dataIndex.LogFileName{j}(1:end-4),'_beh.mat']));
    trials = value_getTrialMasks(trialData);
    stats = value_getTrialStats(trials, sessionData.nRules);
    stats = value_getTrialStatsMore(stats);
    %% Add stimulation info
    % stimulated side
    stats.stRegion = trialData.stimulationRegion-1000;
    
    %stimulation: yes=1; no=1;
    stats.st=nan(numel(trialData.stimulation),1);
    stats.st = stats.stRegion;
    stats.st (stats.stRegion==2) =1;
    
    % which block stimulated
    x0=stats.rule';
    stats.blockSt = stats.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    stats.blockStRegion = stats.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);        %reward probabilities / rule associated with each block
    %% now divide the blocks for stimulated vs non-stimulated
    nBlocks = size(stats.blockRule,1);
    
    % define contra vs ipsi blocks in stimulated blocks - contra means 1, ipsi means 0
    bStimInfo =  nan(nBlocks,1);
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 1)  = 0 ;% left stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 1 &  stats.blockRule == 2)  = 1 ;% left stimulation & left side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 2)  = 0 ;% right stimulation & right side being initial better option
    bStimInfo(stats.blockStRegion== 2 &  stats.blockRule == 1)  = 1 ;% right stimulation & left side being initial better option
    stats.blockStimulationSide = bStimInfo(1:end-1); % exclude last block
    
    % define contra vs ipsi blocks in control blocks - contra means 1, ipsi means 0
    % two blocks before stimulated blocks
    %bControlInfo = [ bStimInfo(3:end);nan; nan];
     bControlInfo = [ nan;nan; bStimInfo(1:end-2)];
    stats.blockControlSide = bControlInfo(1:end-1); % exclude last block
    
    
    %% initial better option plots
    % Stimulated: contra vs ipsi
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==0; %general criteria
    dat(kk).sw_LRVsChoiceIpsi{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceIpsi{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceIpsi{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceIpsi{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceIpsi{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockStimulationSide==1; %general criteria
    dat(kk).sw_LRVsChoiceContra{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceContra{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceContra{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceContra{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceContra{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    % Control: contra vs ipsi
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==0; %general criteria
    dat(kk).sw_LRVsChoiceIpsiControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceIpsiControl{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceIpsiControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
    
    ind = stats.blockTrialtoCrit<=nCrit & stats.blockControlSide==1; %general criteria
    dat(kk).sw_LRVsChoiceContraControl{j}.dat=[stats.blockTrialRandomAdded(ind) stats.blockPreSwitchBetterChoiceRate(ind)];
    dat(kk).sw_LRVsChoiceContraControl{j}.range{1}=[1 30];
    dat(kk).sw_LRVsChoiceContraControl{j}.range{2}=[0.6 1];
    dat(kk).sw_LRVsChoiceContraControl{j}.label{1}={'L_{Random}'};
    dat(kk).sw_LRVsChoiceContraControl{j}.label{2}={'Fraction of trials bilateral'};%'selecting initial  better option'};
end

%control
val=[];
for j=1:numel(dat(1).sw_LRVsChoiceIpsiControl)
    val=[val; dat(1).sw_LRVsChoiceIpsiControl{j}.dat];
end
val2=[];
for j=1:numel(dat(1).sw_LRVsChoiceContraControl)
    val2=[val2; dat(1).sw_LRVsChoiceContraControl{j}.dat];
end

y=[]; y2=[];
for j = 1:nbin
    y  = [val(val(:,1)>=bin(j,1) & val(:,1)<=bin(j,2),2)];
    y2 = [val2(val2(:,1)>=bin(j,1) & val2(:,1)<=bin(j,2),2)];
    stat_data = [ stat_data; y;y2];
    stat_group = [stat_group; ones(size(y)).*[j,1,1,2]; ...
        ones(size(y2)).*[j,2,1,2]]; % blockLength,ipsi/contra side,stimulated(2) /control (1), pre (1)/post(2) choice
end

%stimulated
val=[];
for j=1:numel(dat(1).sw_LRVsChoiceIpsi)
    val=[val; dat(1).sw_LRVsChoiceIpsi{j}.dat];
end
val2=[];
for j=1:numel(dat(1).sw_LRVsChoiceContra)
    val2=[val2; dat(1).sw_LRVsChoiceContra{j}.dat];
end

y=[]; y2=[];
for j = 1:nbin
    y  = [val(val(:,1)>=bin(j,1) & val(:,1)<=bin(j,2),2)];
    y2 = [val2(val2(:,1)>=bin(j,1) & val2(:,1)<=bin(j,2),2)];
    stat_data = [ stat_data; y;y2];
    stat_group = [stat_group; ones(size(y)).*[j,1,2,2]; ...
        ones(size(y2)).*[j,2,2,2]]; % blockLength,ipsi/contra side,stimulated(2) /control (1) /control (1), pre (1)/post(2) choice
end

%% stats - one way ANOVA
[p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2),stat_group(:,3),stat_group(:,4)},...
    'varnames',{'Block length', 'Side', 'Stimulation','Choice'},'model','full');
hTable = figure(1);
print(hTable,'-dpng',fullfile(save_path,'LargeANOVAAfterblocks'));

