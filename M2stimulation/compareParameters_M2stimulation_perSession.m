function compareParameters_M2stimulation_perSession(simPath)
%% compareParameters_M2stimulation_perSession %%

if ~exist(simPath.fig,'dir') % create folders
    mkdir(simPath.fig);
end

%% load parameters for pre and post stimulation fittings
dat(1).input = load(fullfile(simPath.model_mat_M2stimulationPre,'belief_CK_stimulation_bilateral_10paramsNEW.mat'));
dat(1).label = 'PreStimulation';
dat(1).color = [ 0 0 0];
dat(2).input = load(fullfile(simPath.model_mat_M2stimulationPost,'belief_CK_stimulation_bilateral_10params.mat'));
dat(2).label = 'PostStimulation';
dat(2).color = [0 0 1];
clear temp temp2
for k = 1:size(dat(1).input.fitpar,2)
    temp (:,k) = dat(1).input.fitpar{k}'; % for pre
end

for k = 1:size(dat(2).input.fitpar,2)
    temp2 (:,k) = dat(2).input.fitpar{k}'; % for post
end

% clean fittings beta is not in the range
ind =[];
for k =1:size (temp,2)
    if sum([ temp(1:10,k)==0; temp(1:6,k)==1; temp(7:10,k)>=10])==0 % exclude any outliers beta value
        ind = [ind, k];
    end
end
temp = temp(:,ind);

% clean fittings not in the range
ind2 =[];
for k =1:size (temp2,2)
    if sum([  temp2(1:10,k)==0; temp2(1:6,k)==1; temp2(7:10,k)>=10])==0 % exclude any outliers beta value
        ind2 = [ind2, k];
    end
end
temp2 = temp2(:,ind2);

% calculate beta Sum & beta Ratio
pre_betaSum    = [ temp(7,:)'+temp(8,:)', temp(9,:)'+ temp(10,:)'];
post_betaSum  =  [ temp2(7,:)'+temp2(8,:)', temp2(9,:)'+ temp2(10,:)'];
pre_betaRatio   = [temp(7,:)'./pre_betaSum(:,1) ,  temp(9,:)'./pre_betaSum(:,2)] ;
post_betaRatio = [temp2(7,:)'./post_betaSum(:,1) , temp2(9,:)'./post_betaSum(:,2)] ;

%% ViolinPlot versionn
% violinplot does not work nicely - too many data points within a specific
% range - bar plot is nicer.
% violinplot(temp(:,1:2),[ones(n_pre,1),ones(n_pre,1)*2],'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);hold on
x = [1 2 3];
titleSt = [{'Hazard rate'};{'Learning rate CK'};{'betaSum'};{'betaRatio'}];

for k=1:4
    figure
    if k==1 % for hazard rate
        tempInd = [temp(1,:);temp(2,:);temp(3,:)]';%control, contra, ipsi
        tempInd2 = [temp2(1,:);temp2(2,:);temp2(3,:)]';
        tempY = [ones(size(tempInd,2),1),ones(size(tempInd,2),1)*2,ones(size(tempInd,2),1)*3];
    elseif k==2 % for CK
        tempInd = [temp(4,:);temp(5,:);temp(6,:)]';%control, contra, ipsi
        tempInd2 = [temp2(4,:);temp2(5,:);temp2(6,:)]';
        tempY = [ones(size(tempInd,2),1),ones(size(tempInd,2),1)*2,ones(size(tempInd,2),1)*3];
    elseif k==3 % beta
        tempInd = [pre_betaSum(:,1), pre_betaSum(:,2)];%control, contra, ipsi
        tempInd2 = [post_betaSum(:,1), post_betaSum(:,2)];
        tempY = [ones(size(tempInd,2),1),ones(size(tempInd,2),1)*2];
    elseif k==4
        tempInd = [pre_betaRatio(:,1), pre_betaRatio(:,2)];%control, contra, ipsi
        tempInd2 = [post_betaRatio(:,1), post_betaRatio(:,2)];
        tempY = [ones(size(tempInd,2),1),ones(size(tempInd,2),1)*2];
    end
    
    % plot pre stimulation
    %%%%% PRE
    subplot(2,3,1); hold on
    hFig = violinplot(tempInd,tempY,...
        'ViolinColor',[0 0 0],'ViolinAlpha',0.7,'EdgeColor',[0 0 0],...
        'BoxColor',[0.5 0.5 0.5],'ShowData',true);hold on
    hFig(2).ViolinPlot.FaceColor = [0 0 1];
    hFig(2).ViolinPlot.EdgeColor = [0 0 1];
    if k<3
        hFig(3).ViolinPlot.FaceColor = [0 0 1];
        hFig(3).ViolinPlot.EdgeColor = [0 0 1];
        [p2 h] = signrank(tempInd(:,1),tempInd(:,3));
        [p3 h] = signrank(tempInd(:,2), tempInd(:,3));
    else
        p2 = 0;
        p3 = 0;
    end
    set(gca,'xtick',x);
    set(gca,'xticklabel',{'control','contra','ipsi'})
    
    % stat
    [p1 h] = signrank(tempInd(:,1),tempInd(:,2));
    
    subplot(2,2,3)
    plot(1,[1,2,3,4])
    legend([{ ['PRE ' , titleSt{k} ]}; {['contra: ' , num2str(p1)]};...% p/3 for correction
        {['stimulated: ' , num2str(p2)]};...
        {['contraVSipsi: ' , num2str(p3)]}]); box off;axis off
    
    %%%%%%% POST
    
    subplot(2,3,2)
    hFig = violinplot(tempInd2,tempY,...
        'ViolinColor',[0 0 0],'ViolinAlpha',0.7,'EdgeColor',[0 0 0],...
        'BoxColor',[0.5 0.5 0.5],'ShowData',true);hold on
    hFig(2).ViolinPlot.FaceColor = [0 0 1];
    hFig(2).ViolinPlot.EdgeColor = [0 0 1];
    if k<3
        hFig(3).ViolinPlot.FaceColor =  [0 0 1];
        hFig(3).ViolinPlot.EdgeColor = [0 0 1];
        [p2 h] = ranksum(tempInd2(:,1),tempInd2(:,3));
        [p3 h] = ranksum(tempInd2(:,2), tempInd2(:,3));
    else
        p2 = 0;
        p3 = 0;
    end
    set(gca,'xtick',x);
    set(gca,'xticklabel',{'control','contra','ipsi'})
    
    % stat
    [p1 h] = signrank(tempInd2(:,1),tempInd2(:,2));
    
    
    subplot(2,2,4)
    plot(1,[1,2,3,4])
    legend([{ ['POST ' , titleSt{k} ]}; {['contra: ' , num2str(p1)]};...% p/3 for correction
        {['stimulated: ' , num2str(p2)]};...
        {['contraVSipsi: ' , num2str(p3)]}]); box off;axis off
    
    print(gcf,'-dpng',fullfile(simPath.fig ,['violin_perSession_',titleSt{k}]));
    print(gcf,'-dsvg',fullfile(simPath.fig ,['violin_perSession_',titleSt{k}]));
    saveas(gcf, fullfile(simPath.fig ,['violin_perSession_',titleSt{k}]), 'fig');
    pause
    
    %%% ANOVA
    if k<3
        % stats - one way ANOVA
        figure
        stat_data = [tempInd(:);tempInd2(:)];
        grouptemp = ones(size(tempInd)).*[1 2 3];
        grouptemp2 = ones(size(tempInd2)).*[1 2 3];
        stat_group = [grouptemp(:),ones(size(tempInd(:)));grouptemp2(:),ones(size(tempInd2(:)))*2];
        [p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2)},...
            'varnames',{'Stimulation', 'Choice'},'model','full');
    else
        figure
        stat_data = [tempInd(:);tempInd2(:)];
        grouptemp = ones(size(tempInd)).*[1 2 ];
        grouptemp2 = ones(size(tempInd2)).*[1 2 ];
        stat_group = [grouptemp(:),ones(size(tempInd(:)));grouptemp2(:),ones(size(tempInd2(:)))*2];
        
        [p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2)},...
            'random',2,'varnames',{'Stimulation', 'Choice'},'model','full') ;
        
    end
    
    pause
    close all
    
end

%% BarPlot versionn
% violinplot does not work nicely - too many data points within a specific
% range - bar plot is nicer.
% violinplot(temp(:,1:2),[ones(n_pre,1),ones(n_pre,1)*2],'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);hold on

titleSt = [{'Hazard rate'};{'Learning rate CK'} ;{'beta'};{' beta CK'};{'betaSum'};{'betaRatio'}];

for k=[1 2 3 5 6]
    figure
    if k==1
        tempInd = [temp(1,:);temp(2,:);temp(3,:)]';%control, contra, ipsi
        tempInd2 = [temp2(1,:);temp2(2,:);temp2(3,:)]';
        x = [1 2 3];
    elseif k==2
        tempInd = [temp(4,:);temp(5,:);temp(6,:)]';%control, contra, ipsi
        tempInd2 = [temp2(4,:);temp2(5,:);temp2(6,:)]';
        x = [1 2 3];
    elseif k==5
        tempInd = [pre_betaSum(:,1), pre_betaSum(:,2)];%control, contra, ipsi
        tempInd2 = [post_betaSum(:,1), post_betaSum(:,2)];
        x = [1 2];
    elseif k==6
        tempInd = [pre_betaRatio(:,1), pre_betaRatio(:,2)];%control, contra, ipsi
        tempInd2 = [post_betaRatio(:,1), post_betaRatio(:,2)];
        x = [1 2 ];
    end
    %%%%% PRE
    subplot(2,3,1); hold on
    dd = nanmean(tempInd) ;
    dd_sem = nanstd(tempInd)/sqrt(size(tempInd,1));
    hFig = bar(x, dd,'FaceColor','flat');hold on
    hFig.CData(1,:) = [ 0 0 0];
    errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
    plot(tempInd','o','color',[0.5 0.5 0.5],'MarkerSize',8)
    box off
    set(gca,'xtick',x);
    set(gca,'xticklabel',{'control','stimulated'});
    
    % stat
    [p1 h] = signrank(tempInd(:,1),tempInd(:,2));
    p2 = 0; p3 = 0;
    hFig.CData(2,:) = [ 0 0 1];
    if k<3
        set(gca,'xticklabel',{'control','contra','ipsi'})
        
        hFig.CData(3,:) = [ 0 0 1];
        [p2 h] = signrank(tempInd(:,1),tempInd(:,3));
        [p3 h] = signrank(tempInd(:,2), tempInd(:,3));
    end
    subplot(2,2,3)
    plot(1,[1,2,3,4])
    legend([{ ['PRE ' , titleSt{k} ]}; {['contra: ' , num2str(p1)]};...% p/3 for correction
        {['stimulated: ' , num2str(p2)]};...
        {['contraVSipsi: ' , num2str(p3)]}]); box off;axis off
    
    %%%%% POST
    subplot(2,3,2); hold on
    dd = nanmean(tempInd2) ;
    dd_sem = nanstd(tempInd2)/sqrt(size(tempInd2,1));
    hFig = bar(x, dd,'FaceColor','flat');hold on
    hFig.CData(1,:) = [ 0 0 0];
    
    errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
    plot(tempInd2','o','color',[0.5 0.5 0.5],'MarkerSize',8)
    box off
    set(gca,'xtick',x);
    set(gca,'xticklabel',{'control','stimulated'})
    
    % stat
    [p1 h] = signrank(tempInd2(:,1),tempInd2(:,2));
    p2 = 0; p3 = 0;
    hFig.CData(2,:) = [ 0 0 1];
    if k<3
        set(gca,'xticklabel',{'control','contra','ipsi'})
        
        hFig.CData(3,:) = [ 0 0 1];
        [p2 h] = signrank(tempInd2(:,1),tempInd2(:,3));
        [p3 h] = signrank(tempInd2(:,2), tempInd2(:,3));
        
    end
    subplot(2,2,4)
    plot(1,[1,2,3,4])
    legend([{ ['POST ' , titleSt{k} ]}; {['contra: ' , num2str(p1)]};...% p/3 for correction
        {['stimulated: ' , num2str(p2)]};...
        {['contraVSipsi: ' , num2str(p3)]}]); box off;axis off
    
    
    print(gcf,'-dpng',fullfile(simPath.fig ,['barPlot_withDots_perSession_',titleSt{k}]));
    print(gcf,'-dsvg',fullfile(simPath.fig ,['barPlot_withDots_perSession_',titleSt{k}]));
    saveas(gcf, fullfile(simPath.fig ,['barPlot_withDots_perSession_',titleSt{k}]), 'fig');
    
end

end