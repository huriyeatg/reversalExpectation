function lesion_compareParameters_unilateral_fornobilateralmodel(model_type,simPath,animalList)
% lesion_compareParameters_unilateral%
% Current labelling and parameter setup is written only for :hybridFQ_RPE_CK_bilateral
% may not work for other models.

sim_path = simPath.sim_path;
if ~strcmp(model_type, 'belief_CK')
    error('This function works only one model: ''belief_CK''') ;
else
    titleSt = [{'Hazard rate'};{'\beta'};{'\alpha_C_K'}; {'\beta_C_K'}];
end

if ~exist(simPath.fig,'dir') % create folders
    mkdir(simPath.fig);
end
%% load parameters for pre and post fittings
dat(1).input = load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'.mat']));
dat(1).label = 'Pre-lesion';
dat(1).color = [ 0 0 0];
dat(2).input = load(fullfile(fullfile(sim_path,'mat_models_lesion'),[model_type,'.mat']));
dat(2).label = 'Post-lesion';
dat(2).color = [0 150/256 0];

nAnimal= numel(animalList);
nParam = numel(dat(2).input.fitpar{1});
temp = nan(nParam,nAnimal);
temp2 = nan(nParam,nAnimal);

for k = 1:nAnimal
    animal = animalList(k);
    temp (:,k) = dat(1).input.fitpar{ismember(dat(1).input.animal,animal)}; % for pre
    temp2 (:,k) = dat(2).input.fitpar{ismember(dat(2).input.animal,animal)}; % for post
end

pre_betaSum = temp(2,:)'+temp(4,:)';
post_betaSum = temp2(2,:)'+temp2(4,:)';
betaRatio_pre = temp(2,:)'./pre_betaSum;
betaRatio_post=  temp2(2,:)'./post_betaSum;

%% Plot
figure
orthogonalLine_min = [0 1 0.4 2 ];
orthogonalLine_max = [0.75 6 0.8 4];
temp_ind =[ 1 nan; 3 nan; 3 nan; 4 nan];

for k=1:numel(titleSt)
    subplot(2,2,k); hold on
    ln = [orthogonalLine_min(k) orthogonalLine_max(k)]; plot(ln, ln , 'k--','Linewidth',1);
    % plot right
    h(2) = scatter( temp(k,:), temp2(k,:),150,'+','MarkerEdgeColor','k','Linewidth',2);
    title (titleSt{k})
    axis([ln ln])
    
    axis([ln ln])
    xlabel('pre');ylabel('post')
    axis('square')
    clear h
end

print(gcf,'-dpng',fullfile(simPath.fig ,['paramsComparison_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['paramsComparison_',model_type]), 'fig');

%% Beta Sum plot
figure
subplot(2,2,1); hold on
ln = [3 6.5]; plot(ln, ln , 'k--','Linewidth',1);
scatter( pre_betaSum, post_betaSum,150,'+','MarkerEdgeColor','k','Linewidth',2);
axis([ln ln])
xlabel('pre');ylabel('post')
axis('square')
[p,~] = signrank(pre_betaSum,post_betaSum);
title(['Beta sum, p = ',num2str(p)])

subplot(2,2,2); hold on
ln = [0.08 0.6]; plot(ln, ln , 'k--','Linewidth',1);
scatter( betaRatio_pre, betaRatio_post,150,'+','MarkerEdgeColor','k','Linewidth',2);
axis([ln ln])
xlabel('pre');ylabel('post')
axis('square')
[p,~] = signrank(betaRatio_pre,betaRatio_post);
title(['Beta ratio, p = ',num2str(p)])

print(gcf,'-dpng',fullfile(simPath.fig ,['betaSumRatio_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['betaSumRatio__',model_type]), 'fig');

%% Violin pairwise comparison for all plots
figure% violin plot - default params for figure
v_color  = [0 0 0]; % will change according to lesion side
v_alpha  = 0.8;
v_edge   = [0.5 0.5 0.5];
v_box    = [0.5 0.5 0.5];
v_median = [0.5 0.5 0.5];

for i=1:6
    temp_x = [ones(nAnimal,1),ones(nAnimal,1)*2]; % same for all plots
    if i==1 % Hazard rate - intact side
        temp_y = [temp(1,:)',temp2(1,:)'];
        [h,p,stats]  = ttest2(temp_y(:,1),temp_y(:,2));
        y_axis = [0 0.55];
        v_color = dat(1).color;
        title_st =['HR', num2str(p)];
        
    elseif i==2
        temp_y = [temp(2,:)',temp2(2,:)'];
        [h,p,stats]  =  ttest2(temp_y(:,1),temp_y(:,2));
        y_axis   = [1 4];
        v_color = dat(1).color;
        title_st =['beta ', num2str(p)];
        
    elseif i==3 % Learning rate - intact side
        temp_y = [temp(3,:)',temp2(3,:)'];
        [h,p,stats]  = ttest2(temp_y(:,1),temp_y(:,2));
        y_axis = [0.3 0.75];
        v_color = dat(1).color;
        title_st =['LR CK ', num2str(p)];
        
    elseif i==4
        temp_y = [temp(4,:)',temp2(4,:)'];
        [h,p,stats]  = ttest2(temp_y(:,1),temp_y(:,2));
        y_axis = [2 4];
        v_color = dat(1).color;
        title_st =['beta CK ', num2str(p)];
        
    elseif i==5 % BetaSum
        temp_y = [pre_betaSum,post_betaSum];
        [h,p,stats]  = ttest2(temp_y(:,1),temp_y(:,2));
        y_axis = [3 8];
        v_color = dat(1).color;
        title_st =['BetaSum ', num2str(p)];
        
    elseif i==6
        temp_y = [betaRatio_pre,betaRatio_post];
        [p,h,stats]  = ttest2(temp_y(:,1),temp_y(:,2));
        y_axis = [0 0.75];
        v_color = dat(1).color;
        title_st =['BetaRatio', num2str(p)];
        
    end
    subplot(2,4,i)
    violinplot(temp_y,temp_x,'ViolinColor',v_color,'ViolinAlpha',v_alpha,...
        'EdgeColor',v_edge,'BoxColor',v_box,'MedianColor',v_median,'ShowData',false);hold on
    ylim(y_axis);
    set(gca,'xtick',1:2);
    set(gca,'xticklabel',{'pre','post'});
    title(title_st)
end

print(gcf,'-dpng',fullfile(simPath.fig ,['paramsComparisonSUB_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['paramsComparisonSUB_',model_type]), 'fig');

%% stats

close all hidden
% stat figures
titleSt = [{'HazardRate'};{'alphaCK'};{'beta'};{'betaCK'}]; % for stat file name
subjID = (1:nAnimal)';
for k=1:numel(titleSt)
    if k<3 % look for side effect as well
        clear stat_data stat_group
        stat_data = [temp(temp_ind(k,1),:),temp(temp_ind(k,2),:),...
            temp2(temp_ind(k,1),:),temp2(temp_ind(k,2),:)]'; % left-pre,right-pre, left-post, right-post
        stat_group = [[ones(nAnimal,1).*[1 1],subjID];[ones(nAnimal,1).*[2 1],subjID];...
            [ones(nAnimal,1).*[1 2],subjID];[ones(nAnimal,1).*[2 2],subjID]];
        
        [p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2),stat_group(:,3)},...
            'random',3,'varnames',{'Side', 'Session','SubjID'},'model','full') ;
    else
        clear stat_data stat_group
        stat_data = [temp(temp_ind(k,1),:),temp2(temp_ind(k,1),:)]; % pre,post
        stat_group = [ones(nAnimal,1).*[1 1];ones(nAnimal,1).*[1 2]];
        
        [p,ANOVATAB,STATS] = anovan(stat_data', {stat_group(:,2)},...
            'varnames',{'Session'},'model','full') ;
        
    end
    hFig = figure(1);
    print(hFig,'-dpng',fullfile(simPath.fig ,['paramsComparison_stats_',titleSt{k}]));
    close (hFig);
    
end


%% lesion-induced changes
H_left= (temp2(1,:)'-temp(1,:)')./(temp2(1,:)'+temp(1,:)');
H_right= (temp2(2,:)'-temp(2,:)')./(temp2(2,:)'+temp(2,:)');

CK_left= (temp2(3,:)'-temp(3,:)')./(temp2(3,:)'+temp(3,:)');
CK_right= (temp2(4,:)'-temp(4,:)')./(temp2(4,:)'+temp(4,:)');

betaRatio = (betaRatio_post-betaRatio_pre)./(betaRatio_pre+betaRatio_post);
betaSum = (post_betaSum-pre_betaSum)./(pre_betaSum+post_betaSum);

figure
for i = 1:6
    if i==1
        temp_x = [CK_right;CK_left];
        temp_y = [H_right;H_left];
        xlabel_st = [{'Lesion-induced change'};{'in CK'}];
        ylabel_st = [{'Lesion-induced change'};{'in H'}];
        axis_lb    = [ -0.05    0.2   -0.8    0.6];
        
    elseif i==2
        temp_x = [betaRatio;betaRatio];
        temp_y = [H_right;H_left];
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in H'}];
        axis_lb =  [ -0.1200    0.2500   -0.8    0.6];
        
    elseif i==3
        temp_x = [betaSum;betaSum];
        temp_y = [H_right;H_left];
        xlabel_st = [{'Lesion-induced change'};{'in betaSum'}];
        ylabel_st = [{'Lesion-induced change'};{'in H'}];
        axis_lb =  [-0.1500    0.1500   -0.8   0.6];
        
    elseif i==4
        temp_x = [betaRatio;betaRatio];
        temp_y = [CK_right;CK_left];
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in CK'}];
        axis_lb =  [ -0.1200    0.2500   -0.05    0.2];
        
    elseif i==5
        temp_x = [betaSum;betaSum];
        temp_y = [CK_right;CK_left];
        xlabel_st = [{'Lesion-induced change'};{'in betaSum'}];
        ylabel_st = [{'Lesion-induced change'};{'in CK'}];
        axis_lb =  [-0.1500    0.1500    -0.05    0.2];
        
    elseif i==6
        temp_x = [betaSum];
        temp_y = [betaRatio];
        xlabel_st = [{'Lesion-induced change'};{'in betaSum'}];
        ylabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        axis_lb =  [-0.1500    0.1500   -0.1200    0.2500];
        
    end
    subplot(2,3,i)
    if numel(temp_y) > 9 % if there are two side data
        scatter(temp_x(1:numel(temp_x)/2),temp_y(1:numel(temp_x)/2),150,'+k','Linewidth',2); hold on
        scatter(temp_x((numel(temp_x)/2)+1:end),temp_y((numel(temp_x)/2)+1:end),150,'s','MarkerEdgeColor',[1 0.4 0.2],'Linewidth',2)
    else
        scatter(temp_x(1:numel(temp_x)),temp_y(1:numel(temp_x)),150,'+k','Linewidth',2); hold on
    end
    [p, S] = polyfit(temp_x,temp_y,1); % x = x data, y = y data, 1 = order of the polynomial i.e a straight line
    [y_ext,~] = polyconf(p,temp_x,S);
    plot(temp_x,y_ext,'color','k')
    
    [r,p]=corr(temp_x,temp_y);
    xlabel(xlabel_st)
    ylabel(ylabel_st)
    axis(axis_lb)
    title(['p = ', num2str(p), ', r = ', num2str(r)])
    axis('square')
end

%print(gcf,'-dpng',fullfile(simPath.fig ,'lesion-induced_HvsLearningCK'));
%saveas(gcf, fullfile(simPath.fig ,'lesion-induced_HvsLearningCK'), 'fig');


%% lesion-induced changes - differences bewtween two cortices 
H_left= (temp2(1,:)'-temp(1,:)')./(temp2(1,:)'+temp(1,:)');
H_right= (temp2(2,:)'-temp(2,:)')./(temp2(2,:)'+temp(2,:)');

CK_left= (temp2(3,:)'-temp(3,:)')./(temp2(3,:)'+temp(3,:)');
CK_right= (temp2(4,:)'-temp(4,:)')./(temp2(4,:)'+temp(4,:)');

betaRatio = (betaRatio_post-betaRatio_pre)./(betaRatio_pre+betaRatio_post);
betaSum = (post_betaSum-pre_betaSum)./(pre_betaSum+post_betaSum);

figure
for i = 1:4
    if i==1
        temp_x = betaRatio;
        temp_y = (H_right-H_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in  H'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
        
    elseif i==2
        temp_x = betaRatio;
        temp_y = (CK_right-CK_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in  CK'}];
        axis_lb =  [-0.1200    0.2500   -0.300    0.3000];
     
   elseif i==3
        temp_x = betaSum;
        temp_y = (H_right-H_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaSum'}];
        ylabel_st = [{'Lesion-induced change'};{'in  H'}];
        axis_lb =  [-0.1500    0.1500   -1.0500    1.0000];
        
    elseif i==4
        temp_x = betaSum;
        temp_y = (CK_right-CK_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaSum'}];
        ylabel_st = [{'Lesion-induced change'};{'in  CK'}];
        axis_lb =  [-0.1500    0.1500   -0.300    0.3000];    
    end
    subplot(2,3,i)
    if numel(temp_x) > 9 % if there are two side data
        scatter(temp_x(1:numel(temp_x)/2),temp_y(1:numel(temp_x)/2),150,'+k','Linewidth',2); hold on
        scatter(temp_x((numel(temp_x)/2)+1:end),temp_y((numel(temp_x)/2)+1:end),150,'s','MarkerEdgeColor',[1 0.4 0.2],'Linewidth',2)
    else
        scatter(temp_x(1:numel(temp_x)),temp_y(1:numel(temp_x)),150,'+k','Linewidth',2); hold on
    end
    [p, S] = polyfit(temp_x,temp_y,1); % x = x data, y = y data, 1 = order of the polynomial i.e a straight line
    [y_ext,~] = polyconf(p,temp_x,S);
    plot(temp_x,y_ext,'color','k')
    
    [r,p]=corr(temp_x,temp_y);
    xlabel(xlabel_st)
    ylabel(ylabel_st)
    axis(axis_lb)
    title(['p = ', num2str(p), ', r = ', num2str(r)])
    axis('square')
end

print(gcf,'-dpng',fullfile(simPath.fig ,'lesion-induced_HvsLearningCK'));
saveas(gcf, fullfile(simPath.fig ,'lesion-induced_HvsLearningCK'), 'fig');

%% only differences after lesiondifferences bewtween two cortices 
H_left= temp2(1,:)';
H_right= temp2(2,:)';

CK_left= temp2(3,:)';
CK_right= temp2(4,:)';

betaRatio = betaRatio_post;
betaSum   = post_betaSum;

figure
for i = 1:3
    if i==1
        temp_x = (CK_right-CK_left);
        temp_y = (H_right-H_left);
        xlabel_st = [{'Lesion-induced change'};{'in CK'}];
        ylabel_st = [{'Lesion-induced change'};{'in  H'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
        
    elseif i==2
        temp_x = betaRatio;
        temp_y = (CK_right-CK_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in  CK'}];
        axis_lb =  [-0.1200    0.2500   -0.300    0.3000];
     
   elseif i==3
      temp_x = betaRatio;
        temp_y = (H_right-H_left);
        xlabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        ylabel_st = [{'Lesion-induced change'};{'in  H'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
           
    end
    subplot(2,3,i)
    if numel(temp_x) > 9 % if there are two side data
        scatter(temp_x(1:numel(temp_x)/2),temp_y(1:numel(temp_x)/2),150,'+k','Linewidth',2); hold on
        scatter(temp_x((numel(temp_x)/2)+1:end),temp_y((numel(temp_x)/2)+1:end),150,'s','MarkerEdgeColor',[1 0.4 0.2],'Linewidth',2)
    else
        scatter(temp_x(1:numel(temp_x)),temp_y(1:numel(temp_x)),150,'+k','Linewidth',2); hold on
    end
    [p, S] = polyfit(temp_x,temp_y,1); % x = x data, y = y data, 1 = order of the polynomial i.e a straight line
    [y_ext,~] = polyconf(p,temp_x,S);
    plot(temp_x,y_ext,'color','k')
    
    [r,p]=corr(temp_x,temp_y);
    xlabel(xlabel_st)
    ylabel(ylabel_st)
   % axis(axis_lb)
    title(['p = ', num2str(p), ', r = ', num2str(r)])
    axis('square')
end

print(gcf,'-dpng',fullfile(simPath.fig ,'after-lesion_HvsLearningCK'));
saveas(gcf, fullfile(simPath.fig ,'after-lesion_HvsLearningCK'), 'fig');

%% Final set of comparisons 
% (H_post_contra - H_post_lesion) - (H_pre_contra - H_pre_lesion)  versus  (betaRatio_post - betaRatio_pre)
% (H_post_contra - H_pre_contra)  versus  (betaRatio_post - betaRatio_pre)
% (H_post_contra - H_post_lesion) - (H_pre_contra - H_pre_lesion)  versus  (alphaCK_contra_post - alphaCK_contra_pre)
% (H_post_contra - H_pre_contra)  versus   (alphaCK_contra_post - alphaCK_contra_pre)
 
H_contra_pre= temp(1,:)';
H_lesion_pre= temp(2,:)';

H_contra_post= temp2(1,:)';
H_lesion_post= temp2(2,:)';

alphaCK_contra_pre= temp(3,:)';
alphaCK_lesion_pre= temp(4,:)';

alphaCK_contra_post= temp2(3,:)';
alphaCK_lesion_post= temp2(4,:)';

betaRatio_pre = betaRatio_pre;
betaSum_pre   = pre_betaSum;
betaRatio_post = betaRatio_post;
betaSum_post   = post_betaSum;

figure
for i = 1:4 
    if i==1 %(H_post_contra - H_post_lesion) - (H_pre_contra - H_pre_lesion)  versus  (betaRatio_post - betaRatio_pre)
        temp_x = (H_contra_post - H_lesion_post)-(H_contra_pre - H_lesion_pre);
        temp_y = (betaRatio_post - betaRatio_pre);
        xlabel_st = [{'Lesion-induced change'};{'in (H_post_contra - H_post_lesion) - (H_pre_contra - H_pre_lesion)'}];
        ylabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
        
    elseif i==2 %(H_post_contra - H_pre_contra)  versus  (betaRatio_post - betaRatio_pre)
        temp_x = H_post_contra - H_pre_contra;
        temp_y = (betaRatio_post - betaRatio_pre);
        xlabel_st = [{'Lesion-induced change'};{'in H contra'}];
        ylabel_st = [{'Lesion-induced change'};{'in betaRatio'}];
        axis_lb =  [-0.1200    0.2500   -0.300    0.3000];
     
   elseif i==3 %(H_contra_post - H_lesion_post)  versus  (alphaCK_contra_post - alphaCK_contra_pre)
      temp_x   = (H_contra_post - H_lesion_post)-(H_contra_pre - H_lesion_pre);
        temp_y = (alphaCK_contra_post - alphaCK_contra_pre);
        xlabel_st = [{'Lesion-induced change'};{'in (H_contra_post - H_lesion_post)-(H_contra_pre - H_lesion_pre)'}];
        ylabel_st = [{'Lesion-induced change'};{'in alphaCK_contra'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
        
   elseif i==4 %(H_contra_post - H_contra_pre)  versus   (alphaCK_contra_post - alphaCK_contra_pre)
        temp_x = (H_contra_post - H_lesion_post)-(H_contra_pre - H_lesion_pre);
        temp_y = (alphaCK_contra_post - alphaCK_contra_pre);
        xlabel_st = [{'Lesion-induced change'};{'in (H_contra_post - H_lesion_post)-(H_contra_pre - H_lesion_pre)'}];
        ylabel_st = [{'Lesion-induced change'};{'in (alphaCK_contra_post - alphaCK_contra_pre)'}];
        axis_lb =  [-0.1200    0.2500   -0.300    0.3000];
     
   elseif i==5 %(betaRatio_post - betaRatio_pre)  versus  (alphaCK_contra_post - alphaCK_contra_pre)
      temp_x =(betaRatio_post - betaRatio_pre);
        temp_y = (alphaCK_contra_post - alphaCK_contra_pre);
        xlabel_st = [{'Lesion-induced change'};{'in (betaRatio_post - betaRatio_pre)'}];
        ylabel_st = [{'Lesion-induced change'};{'in  (alphaCK_contra_post - alphaCK_contra_pre)'}];
        axis_lb =  [ -0.1200    0.2500   -1.0500    1.0000];
           
    end
    subplot(2,3,i)
    if numel(temp_x) > 9 % if there are two side data
        scatter(temp_x(1:numel(temp_x)/2),temp_y(1:numel(temp_x)/2),150,'+k','Linewidth',2); hold on
        scatter(temp_x((numel(temp_x)/2)+1:end),temp_y((numel(temp_x)/2)+1:end),150,'s','MarkerEdgeColor',[1 0.4 0.2],'Linewidth',2)
    else
        scatter(temp_x(1:numel(temp_x)),temp_y(1:numel(temp_x)),150,'+k','Linewidth',2); hold on
    end
    [p, S] = polyfit(temp_x,temp_y,1); % x = x data, y = y data, 1 = order of the polynomial i.e a straight line
    [y_ext,~] = polyconf(p,temp_x,S);
    plot(temp_x,y_ext,'color','k')
    
    [r,p]=corr(temp_x,temp_y);
    xlabel(xlabel_st,'FontSize', 9)
    ylabel(ylabel_st,'FontSize', 9)
   % axis(axis_lb)
    title(['p = ', num2str(p), ', r = ', num2str(r)])
    axis('square')
end

print(gcf,'-dpng',fullfile(simPath.fig ,'after-lesion_allParamsComparison'));
saveas(gcf, fullfile(simPath.fig ,'after-lesion_allParamsComparison'), 'fig');
end