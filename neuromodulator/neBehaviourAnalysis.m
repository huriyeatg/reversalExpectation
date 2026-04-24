setup_figprop;
%% Get the data
dataIndex = makeBlockIndexPR_NE;
% Create beh file for logfiles
createBehFile %

%% Create Error Bar for HPS/LPS rewarded vs unrewarded
NEpath = 'A:\HuriyeAtilgan\NESensorData\Data\exported\';
fs = round(20.9);
pre = 1; % 1 sec
dffAll =[]; trialDataAll.rule=[]; trialDataAll.outcome=[];
dffAllBase=[];
for i = 1: size(dataIndex,1)
    if dataIndex.MeanRuleSwitch(i)<51
         %% Trial Type
        fname = dataIndex.LogFileName{i};
        [ logData ] = parseLogfileMixStructure (dataIndex.LogFilePath{i},fname);
        % Get the information you need for each trial
        [ sessionData, trialData ] = value_getSessionData( logData,31 );
        trialDataAll.rule    = [trialDataAll.rule;trialData.rule];
        trialDataAll.outcome = [trialDataAll.outcome;trialData.outcome];
        
        %% Get NE data
        % Load NE data
        fname = fullfile(NEpath,dataIndex.NEDataPath{i});
        data = csvread(fname,2,0); % First two row is header - excluded.
        
        fname = fullfile(NEpath,dataIndex.NEDataTimeEventPath{i});
        temp =readtable(fname); % First two row is header - excluded.
        ind = find(strcmp(temp.ChannelName,'IO1')); % Make a list of all datasources.
        datatime = [temp.Time_s_(ind),temp.Value(ind)];
        
        % Now, get each trials dff
        x0 = datatime(:,2)';
        l =datatime(find(x0(1:end-1) ~= x0(2:end))',1);
        l = l(2:2:end);
        % Exclude last switch
        
        dff     = nan(size(trialData.outcome,1),30);
        dffN    = nan(size(trialData.outcome,1),30);
        dffbase = nan(size(trialData.outcome,1),round(pre*fs));
        
        for k = 1:(numel(l)-1)
            [~,stIndex]  = min(abs(data(:,1)-l(k)));
            [~,endIndex] = min(abs(data(:,1)-l(k+1)));
            ind = (stIndex-round(pre*fs)):endIndex-1;
            dff(k,1:numel(ind))  = data(ind,2);
            dffN(k,1:numel(ind)) = data(ind,2)-mean(data(ind(1):ind(round(pre*fs)),2));
            %data(ind,2)- mean(data(ind,2));
            
            dffbase (k,1:round(pre*fs)) = data(ind(1):ind(round(pre*fs)),2);%-mean(data(ind(1):ind(round(pre*fs)),2));
            clear ind
        end
        dffAll     = [dffAll;dffN(:,1:200)];
        dffAllBase = [dffAllBase; dffbase];
        % %     %% All Trials Combined
        % %     tW = -1:1/round(fs): 13*round(fs);
        % %     tW = tW(1: size(dff,2));
        % %     figure;
        % %     subplot(2,1,1)
        % %     plot( data(:,1),data(:,2))
        % %
        % %     subplot(2,2,3)
        % %     plot(tW,(dff*100)','LineWidth',1); hold on
        % %     plot(tW,mean(dff)*100,'k')
        % %     xlim([-1 3]); ylabel('dff (Raw)')
        % %     box off
        % %
        % %     subplot(2,2,4)
        % %     plot(tW,(dffN*100)','LineWidth',1); hold on
        % %     plot(tW, mean(dffN)*100,'k')
        % %     xlim([-1 3]); ylabel('dff (Normalized to pre-cue)')
        % %     box off
        
       
        %% HPS vs LPS Reward/Noreward
        indHP_Reward = [find(trialData.rule ==41 & trialData.outcome==5);...
            find(trialData.rule ==42 & trialData.outcome==6) ];
        
        indHP_NoReward = [find(trialData.rule ==41 & trialData.outcome==75);...
            find(trialData.rule ==42 & trialData.outcome==76) ];
        
        indLP_Reward = [find(trialData.rule ==42 & trialData.outcome==5);...
            find(trialData.rule ==41 & trialData.outcome==6) ];
        
        indLP_NoReward = [find(trialData.rule ==42 & trialData.outcome==75);...
            find(trialData.rule ==41 & trialData.outcome==76) ];
        
        c = [numel(indHP_Reward),numel(indLP_Reward),numel(indHP_NoReward),numel(indLP_NoReward)]
        try
            %% HPS/LPS reward/NoReward
            tEnd = 4;
            tW   = -pre:(1/fs):(tEnd-pre);
            figure
            mn     = nanmean(dffN(indHP_Reward,1:tEnd*fs+1)*100,1);
            errlow = nanstd(dffN(indHP_Reward,1:tEnd*fs+1)*100)./sqrt(numel(indHP_Reward));
            h(1) = errorbar(tW,mn,errlow, 'r'); hold on
            
            mn     = nanmean(dffN(indLP_Reward,1:tEnd*fs+1)*100,1);
            errlow = nanstd(dffN(indLP_Reward,1:tEnd*fs+1)*100,1)./sqrt(numel(indLP_Reward));
            h(2) = errorbar(tW,mn,errlow, 'k'); hold on
            
            mn     = nanmean(dffN(indHP_NoReward,1:tEnd*fs+1)*100,1);
            errlow = nanstd(dffN(indHP_NoReward,1:tEnd*fs+1)*100)./sqrt(numel(indHP_NoReward));
            h(3) = errorbar(tW,mn,errlow, 'g'); hold on
            
            mn     = nanmean(dffN(indLP_NoReward,1:tEnd*fs+1)*100,1);
            errlow = nanstd(dffN(indLP_NoReward,1:tEnd*fs+1)*100)./sqrt(numel(indLP_NoReward));
            h(4) = errorbar(tW,mn,errlow, 'b'); hold on
            
            legend(h,[{'HPS Reward'};{'LPS Reward'};{'HPS NoReward'};{'LPS NoReward'}])  ; box off
            ylabel ( 'Normalized NE signal dff(%)')
            xlabel( 'Seconds')
            title(num2str(i));
            
            % Explorative/ Static State
            dd = nanmean(dffbase,2);
            figure;plot(dd); hold on
            plot(sessionData.ruleSwitches, max(dd),'g*')
            ind = indHP_Reward;
            plot(ind,dd(ind) ,'r*')
            
            ind = indLP_Reward;
            plot(ind,dd(ind) ,'k*')
            
            ind = indHP_NoReward;
            plot(ind,dd(ind) ,'g*')
            
            ind = indLP_NoReward;
            plot(ind,dd(ind) ,'b*')
            
        catch
            disp('Check index values, it will give error for automatic figure generation');
            % pause;
        end
    end
end

%% Population Analysis
trialData = trialDataAll;
dffN      = dffAll;

indHP_Reward = [find(trialData.rule ==41 & trialData.outcome==5);...
    find(trialData.rule ==42 & trialData.outcome==6) ];

indHP_NoReward = [find(trialData.rule ==41 & trialData.outcome==75);...
    find(trialData.rule ==42 & trialData.outcome==76) ];

indLP_Reward = [find(trialData.rule ==42 & trialData.outcome==5);...
    find(trialData.rule ==41 & trialData.outcome==6) ];

indLP_NoReward = [find(trialData.rule ==42 & trialData.outcome==75);...
    find(trialData.rule ==41 & trialData.outcome==76) ];

c = [numel(indHP_Reward),numel(indLP_Reward),numel(indHP_NoReward),numel(indLP_NoReward)]


figure
mn     = nanmean(dffN(indHP_Reward,1:tEnd*fs+1)*100,1);
errlow = nanstd(dffN(indHP_Reward,1:tEnd*fs+1)*100)./sqrt(numel(indHP_Reward));
h(1) = errorbar(tW,mn,errlow, 'r'); hold on

mn     = nanmean(dffN(indLP_Reward,1:tEnd*fs+1)*100,1);
errlow = nanstd(dffN(indLP_Reward,1:tEnd*fs+1)*100,1)./sqrt(numel(indLP_Reward));
h(2) = errorbar(tW,mn,errlow, 'k'); hold on

mn     = nanmean(dffN(indHP_NoReward,1:tEnd*fs+1)*100,1);
errlow = nanstd(dffN(indHP_NoReward,1:tEnd*fs+1)*100)./sqrt(numel(indHP_NoReward));
h(3) = errorbar(tW,mn,errlow, 'g'); hold on

mn     = nanmean(dffN(indLP_NoReward,1:tEnd*fs+1)*100,1);
errlow = nanstd(dffN(indLP_NoReward,1:tEnd*fs+1)*100)./sqrt(numel(indLP_NoReward));
h(4) = errorbar(tW,mn,errlow, 'b'); hold on

legend(h,[{'HPS Reward'};{'LPS Reward'};{'HPS NoReward'};{'LPS NoReward'}])  ; box off
ylabel ( 'Normalized NE signal dff(%)')
title ('Population')

figure
pr = 50;
plot(prctile(dffN(indHP_Reward,1:tEnd*fs+1),pr)); hold on
plot(prctile(dffN(indHP_NoReward,1:tEnd*fs+1),pr))
plot(prctile(dffN(indLP_NoReward,1:tEnd*fs+1),pr))
plot(prctile(dffN(indLP_Reward,1:tEnd*fs+1),pr))

%% %Explorative/ Static State
trialData = trialDataAll;
dffbase     = dffAllBase;
figure
dd = nanmean(dffbase,2);
dd = [nanmean(dd(indHP_Reward)),nanmean(dd(indHP_NoReward)),...
    nanmean(dd(indLP_Reward)),nanmean(dd(indLP_NoReward))];
bar(dd )
xticklabels([{'HP Reward'};{'HP NoReward'};{'LP Reward'};{'LP No Reward'}])
title ('PreCue Baseline NE Activity')







