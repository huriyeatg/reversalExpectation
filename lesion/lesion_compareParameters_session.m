function lesion_compareParameters_session(model_type,simPath,dataIndex,condition)
% lesion_compareParameters_session%
% Current labelling and parameter setup is written only for :hybridFQ_RPE_CK_bilateral
% may not work for other models.

sim_path = simPath.sim_path;
if ~strcmp(model_type, 'belief_CK_bilateral') & ~strcmp(model_type,'belief_CK_varyingH_delta_bilateral')
    error(['This function works only one model: ''belief_CK_bilateral''']) ;
end

if ~exist(simPath.fig,'dir') % create folders
    mkdir(simPath.fig);
end
%% load parameters for pre and post fittings
dat(1).input = load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'_perSession_flipped.mat']));
dat(1).label = 'Pre-lesion';
dat(1).color = [0 0 0];
dat(2).input = [];% lesion data calculated together with non-lesion data -
dat(2).label = 'Post-lesion';
dat(2).color = [0 150/256 0];

% % get the values, and exclude the 
fitpar = cell2mat(dat(1).input.fitpar'); % [# of sessions x 6 params]
% [H Left, H Right, alpha CK Left, alpha CK right, beta, beta CK]

switch condition
    case 'unilateral' %% UNILATERAL
        temp = fitpar(isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[1 2 ])==1,:);
        % if betas were at boundaries, then exclude them. 
        temp = temp( temp(:,5)~=0 & temp(:,6)~=0 & temp(:,5)~=10 & temp(:,6)~=10,:);
        temp2 = fitpar(~isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[1 2 ])==1,:);
        temp2 = temp2( temp2(:,5)~=0 & temp2(:,6)~=0 & temp2(:,5)~=10 & temp2(:,6)~=10,:);
        save_st ='uni_';
        
    case 'bilateral' %% BILATERAL
        temp = fitpar(isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[3 ])==1,:);
        temp = temp( temp(:,5)~=0 & temp(:,6)~=0 & temp(:,5)~=10 & temp(:,6)~=10,:);
        temp2 = fitpar(~isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[3 ])==1,:);
        temp2 = temp2( temp2(:,5)~=0 & temp2(:,6)~=0 & temp2(:,5)~=10 & temp2(:,6)~=10,:);
        save_st ='bi_';
        
    case 'saline' %% SALINE
        temp = fitpar(isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[4 ])==1,:);
        temp = temp( temp(:,5)~=0 & temp(:,6)~=0 & temp(:,5)~=10 & temp(:,6)~=10,:);
        temp2 = fitpar(~isnan(dataIndex.Lesioned) & ismember(dataIndex.LesionSide,[4])==1,:);
        temp2 = temp2( temp2(:,5)~=0 & temp2(:,6)~=0 & temp2(:,5)~=10 & temp2(:,6)~=10,:);
        save_st ='saline_';
end

%% Bar - hazardrate
% violinplot does not work nicely - too many data points within a specific
% range - bar plot is nicer. 
% violinplot(temp(:,1:2),[ones(n_pre,1),ones(n_pre,1)*2],'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);hold on

dd = [nanmean(temp(:,1:2)) nanmean(temp2(:,1:2))];
dd_sem = [nanstd(temp(:,1:2))/sqrt(size(temp,1)) nanstd(temp2(:,1:2))/sqrt(size(temp2,1))];
x = [2 1 4.5 3.5];
figure
bar(x, dd,'FaceColor',[0.5 0.5 0.5],'LineWidth',3);hold on
errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
box off
x = [1 2 3.5 4.5];
set(gca,'xtick',x);
set(gca,'xticklabel',{'preL','preR','postL','postR'})
ylabel('Hazard rate')
ylim([0.25 0.6])
%stats
disp ("Contra vs lesion: ")
[p1,h,stats]  =  signrank(temp(:,1), temp(:,2))
[p2,h,stats]  =  signrank(temp2(:,1), temp2(:,2))

disp("pre vs post")
[p1,h,stats]  =  ranksum(temp(:,1), temp2(:,1))
[p1,h,stats]  =  ranksum(temp(:,2), temp2(:,2))

title(['Pairwise ttest, pre: ',num2str(p1),' ,post: ', num2str(p2)])
print(gcf,'-dsvg',fullfile(simPath.fig ,[save_st,'hrate_barplot']));
saveas(gcf, fullfile(simPath.fig ,[save_st,'hrate_barplot']), 'fig');

%% Bar - learningrate
dd = [nanmean(temp(:,3:4)) nanmean(temp2(:,3:4))];
dd_sem = [nanstd(temp(:,3:4))/sqrt(size(temp,1)) nanstd(temp2(:,3:4))/sqrt(size(temp2,1))];
x = [2 1 4.5 3.5];
figure
bar(x, dd,'FaceColor',[0.5 0.5 0.5],'LineWidth',3);hold on
errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
box off
x = [1 2 3.5 4.5];
set(gca,'xtick',x);
set(gca,'xticklabel',{'preL','preR','postL','postR'})
ylabel('Learning rate')
ylim([0.35 0.65]);
% stat
[p1,h,stats]  =  signrank(temp(:,3), temp(:,4));
[p2,h,stats]  =  signrank(temp2(:,3), temp2(:,4));

disp("pre vs post")
[p1,h,stats]  =  ranksum(temp(:,3), temp2(:,3))
[p1,h,stats]  =  ranksum(temp(:,4), temp2(:,4))

title(['Pairwise ttest, pre: ',num2str(p1),' ,post: ', num2str(p2)])
print(gcf,'-dsvg',fullfile(simPath.fig ,[save_st,'learningrate_barplot']));
saveas(gcf, fullfile(simPath.fig ,[save_st,'learningrate_barplot']), 'fig');

%% Beta ratio comparison
figure;
subplot(2,2,1); hold on
edges = 0:0.02:1;
y = 0:20;

x= temp(:,5)./ (temp(:,5)+ temp(:,6));
histogram( x,edges,'FaceColor','k')

x2 = temp2(:,5)./ (temp2(:,5)+ temp2(:,6));
histogram( x2,edges,'FaceColor',dat(2).color)

plot(median(x2)*ones(numel(y),1),y,'color',dat(2).color)
plot(median(x)*ones(numel(y),1),y,'k')
[p, ~, ~]=ranksum(x,x2);
box off
title(['beta H /betaH+betaCK p=',num2str(p)])
ylabel('Number of sessions')
xlabel('Inverse temp. ratio')

print(gcf,'-dsvg',fullfile(simPath.fig ,[save_st,'betaRatio']));
saveas(gcf, fullfile(simPath.fig ,[save_st,'betaRatio']), 'fig');
%% Beta sum - comparison
figure;
subplot(2,2,1); hold on
edges = 1:1:20;
y = 0:50;

x= (temp(:,5)+ temp(:,6));
histogram( x,edges,'FaceColor','k')

x2 = (temp2(:,5)+ temp2(:,6));
histogram( x2,edges,'FaceColor',dat(2).color)

plot(median(x2)*ones(numel(y),1),y,'color',dat(2).color)
plot(median(x)*ones(numel(y),1),y,'k')
[p, ~, ~]=ranksum(x,x2);
box off
title(['betaH+betaCK p=',num2str(p)]);
ylabel('Number of sessions');
xlabel('Beta + Beta CK');

print(gcf,'-dsvg',fullfile(simPath.fig ,[save_st,'betaSum']));
saveas(gcf, fullfile(simPath.fig ,[save_st,'betaSum']), 'fig');

%% bar beta Sum
dd = [nanmean(nansum(temp(:,5:6),2)), nanmean(nansum(temp2(:,5:6),2))];
dd_sem = [nanstd(nansum(temp(:,5:6),2))/sqrt(size(temp,1)) nanstd(nansum(temp2(:,5:6),2))/sqrt(size(temp2,1))];
x = [1 2];
figure
subplot(1,2,1)
bar(x, dd,'FaceColor',[0.5 0.5 0.5],'LineWidth',3);hold on
errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
box off
set(gca,'xtick',x);
set(gca,'xticklabel',{'pre','post'})
ylabel('Beta Sum')
ylim([3 7]);

disp("pre vs post")
[p1,h,stats]  =  ranksum(nansum(temp(:,5:6),2), nansum(temp2(:,5:6),2))


betaRatioPre =temp(:,5)./ (temp(:,5)+ temp(:,6));
betaRatioPost =temp2(:,5)./ (temp2(:,5)+ temp2(:,6));
dd = [nanmean(betaRatioPre) nanmean(betaRatioPost)];
dd_sem = [nanstd(betaRatioPre)/sqrt(numel(betaRatioPre)) nanstd(betaRatioPost)/sqrt(numel(betaRatioPost))];
subplot(1,2,2)
bar(x, dd,'FaceColor',[0.5 0.5 0.5],'LineWidth',3);hold on
errorbar(x, dd,dd_sem,dd_sem,'k.','LineWidth',3)
box off
set(gca,'xtick',x);
set(gca,'xticklabel',{'pre','post'})
ylabel('Beta Ratio')
ylim([0 0.5]);

[p1,h,stats]  =  ranksum(betaRatioPre, betaRatioPost)

print(gcf,'-dsvg',fullfile(simPath.fig ,[save_st,'beta_barplot']));
saveas(gcf, fullfile(simPath.fig ,[save_st,'beta_barplot']), 'fig');

end