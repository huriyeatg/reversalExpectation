function lesion_compareParameters_bilateral(model_type,simPath,animalList)
% lesion_compareParameters_bilateral%
% Current labelling and parameter setup is written only for :hybridFQ_RPE_CK_bilateral
% may not work for other models.

sim_path = simPath.sim_path;
if ~strcmp(model_type, 'belief_CK_bilateral')
    error(['This function works only one model: ''belief_CK_bilateral''']) ;
else
    titleSt = [{'Hazard rate'};{'\alpha_C_K'}; {'\beta'};{'\beta_C_K'}];
end

%% load parameters for pre and post fittings
dat(1).input = load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'_flipped.mat']));
dat(1).label = 'Pre-lesion';
dat(2).input = load(fullfile(fullfile(sim_path,'mat_models_lesion'),[model_type,'_flipped.mat']));
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

figure
orthogonalLine_max = [0.6 0.9 3 4 ];
temp_ind =[ 1 2; 3 4; 5 nan; 6 nan];

for k=1:numel(titleSt)
    subplot(2,2,k); hold on
    ln = [0 orthogonalLine_max(k)]; plot(ln, ln , 'k--','Linewidth',1);
    % plot right
    h(2) = scatter( temp(temp_ind(k,1),:), temp2(temp_ind(k,1),:),100,'+','MarkerEdgeColor','k','Linewidth',2);
    if k<3
        h(1) = scatter( temp(temp_ind(k,2),:),temp2(temp_ind(k,2),:),80,'s',...
            'MarkerEdgeColor',[0 150/256 0],'Linewidth',2);
        
         [~, hobj, ~, ~] =legend(h, [{'Lesion side'}; {'Intact side'}],'location','southeast');
         M = findobj(hobj,'type','patch');
         set(M,'MarkerSize',10,'Linewidth',2);
         legend boxoff
    end
    title (titleSt{k})
    
    axis([ln ln])
    xlabel('pre');ylabel('post')
    axis('square')
    clear h
end
 
print(gcf,'-dpng',fullfile(simPath.fig ,['paramsComparison_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['paramsComparison_',model_type]), 'fig');

%%

%%%%%% Learning rate choice kernel
figure% violin plot
subplot(2,4,1)
x = [ones(nAnimal,1),ones(nAnimal,1)*2];
y = [temp(1,:)',temp2(1,:)'];

violinplot(y,x,'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);hold on
ylim([0 0.6]);
set(gca,'xtick',[1:2]);
set(gca,'xticklabel',{'pre','post'})
title('Hazard rate intact side')

subplot(2,4,2)
x = [ones(nAnimal,1)*1,ones(nAnimal,1)*2];
y = [temp(2,:)',temp2(2,:)'];
violinplot(y,x,'ViolinColor',dat(2).color,'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);

ylim([0 0.6]);
set(gca,'xtick',[1:2]);
set(gca,'xticklabel',{'pre','post'});
title('Hazard rate lesion side')


%%%%%% Learning rate choice kernel
subplot(2,4,3)
x = [ones(nAnimal,1),ones(nAnimal,1)*2];
y = [temp(3,:)',temp2(3,:)'];

violinplot(y,x,'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);hold on
ylim([0.3 0.7]);
set(gca,'xtick',[1:2]);
set(gca,'xticklabel',{'pre','post'})
title('Learning rate CK  intact side')

subplot(2,4,4)
x = [ones(nAnimal,1)*1,ones(nAnimal,1)*2];
y = [temp(4,:)',temp2(4,:)'];
violinplot(y,x,'ViolinColor',dat(2).color,'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[0.5 0.5 0.5],'ShowData',false);

ylim([0.3 0.7]);
set(gca,'xtick',[1:2]);
set(gca,'xticklabel',{'pre','post'});
title('Learning rate CK  lesion side')


print(gcf,'-dpng',fullfile(simPath.fig ,['paramsComparisonSUB_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['paramsComparisonSUB_',model_type]), 'fig');

%% stats

close all hidden
% stat figures
titleSt = [{'Hazard rate'};{'alphaCK'};{'beta'};{'betaCK'}]; % for stat file name
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


end