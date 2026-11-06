function plot_betaratio_lesion(model_type, animalList, sim_path)
%  plot_betaratio%

%%
if ~strcmp(model_type, 'belief_CK_bilateral')
    error( 'Only works with belief_CK_bilateral model fittings.')
end

%% load lesion data
load(fullfile(sim_path ,'mat_models_lesion',model_type));
fitpar =cell2mat(fitpar');
val_lesion = nan(numel(animalList),2); % beta & beta CK
for k = 1: numel(animalList)
    ind = strcmp(animal, animalList{k});
    val_lesion (k,:) = fitpar(ind,[5 6]);
end

%% load pre-lesion data
load(fullfile(sim_path ,'mat_models',model_type));
fitpar =cell2mat(fitpar');
val = nan(numel(animalList),2); % beta & beta CK
for k = 1: numel(animalList)
    ind = strcmp(animal, animalList{k});
    val (k,:) = fitpar(ind,[5 6]);
end

%%
figure
lesionColor = [0 150/256 0];
subplot(2,2,1); hold on
scatter(val(:,2),val(:,1), 100, 'dk','filled')
scatter(val_lesion(:,2),val_lesion(:,1), 100, lesionColor,'filled')

%plot([0 4],[0 4],'k--','LineWidth',3)

xlabel( '\beta_{CK}')
ylabel( '\beta')
axis('square')
 axis([2 4 0 2.5])
legend('pre - test','post - test'); legend box off

%% % stats
% 
% nAnimal =numel(animalList);
% subjID = (1:nAnimal)';
% 
% stat_data = [val_lesion(:,1);val(:,1); val_lesion(:,2);val(:,2)];
% stat_group = [[ones(nAnimal,1).*[2 1],subjID];[ones(nAnimal,1).*[1 1],subjID];...
%             [ones(nAnimal,1).*[1 2],subjID];[ones(nAnimal,1).*[2 2],subjID]];
% 
% % stats - one way ANOVA
% [p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2)},...
%     'varnames',{'Session', 'betaType'},'model','full') ;
% 
