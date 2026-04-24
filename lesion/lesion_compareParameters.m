function lesion_compareParameters(model_type,simPath,animalList)
% lesion_compareParameters%
% Current labelling and parameter setup is written only for :hybridFQ_RPE_CK_bilateral
% may not work for other models.

sim_path = simPath.sim_path;
if ~strcmp(model_type, 'belief_CK')
    error(['This function works only one model: ''belief_CK''']) ;
else
    titleSt = [{'Hazard rate'};{'\alpha_C_K'}; {'\beta'};{'\beta_C_K'}];
end

if ~exist(simPath.fig ,'dir') % create folders
    mkdir(simPath.fig );
end

%% load parameters for pre and post fittings
dat(1).input = load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'.mat']));
dat(1).label = 'Pre-lesion';
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

figure

for k=1:numel(titleSt)
    subplot(2,4,k); hold on
    % plot right
    h(2) = bar(1:2, nanmean([temp(k,:); temp2(k,:)],2),'FaceColor','white','EdgeColor','k','Linewidth',2) 
    plot([ones(1,size(temp(k,:),2));ones(1,size(temp(k,:),2))*2], [temp(k,:); temp2(k,:)],'color',[0.5 0.5 0.5])
    ylabel (titleSt{k})   
    set(gca,'xtick',[1:2]);
    set(gca,'xticklabel',{'pre','post'})
end

print(gcf,'-dpng',fullfile(simPath.fig ,['paramsComparison_',model_type]));
saveas(gcf, fullfile(simPath.fig ,['paramsComparison_',model_type]), 'fig');

end