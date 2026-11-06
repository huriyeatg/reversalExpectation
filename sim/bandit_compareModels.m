function bandit_compareModels(model_path,save_path)
% % bandit_compareModels(model_path,save_path) %
%PURPOSE:   Compare BIC values for all the models
%AUTHORS:   H Atilgan and AC Kwan 191212
%
%INPUT ARGUMENTS
%   model_path:    path where all the model fittings are saved
%                 (e.g: for control data: 'simPath.model_mat' folder)
%   save_path:     path to save figures
%OUTPUT ARGUMENTS
%

%% find all the models and BIC values
AllModelfiles = dir(fullfile(model_path,'*.mat'));

nFile = numel(AllModelfiles);

for k = 1:nFile
    
   load(fullfile(AllModelfiles(k).folder,AllModelfiles(k).name));
    
    models(k).name = AllModelfiles(k).name(1:end-4);
    models(k).bic = cell2mat(bic');
    models(k).nlike = cell2mat(nlike');
    models(k).negloglike = cell2mat(negloglike');
    
end

%% Models labels and reference model
% Reference model:
idx = ismember({models.name},'belief_CK');
if any(idx) == 0
    idx = 1;
end

modelnames = [{'WSLS'}; {'Q_RPE'};{'FQ_RPE'};...
    {'DFQ_RPE'};...%
    {'belief'};...
    {'FQ_RPE_CK'};...
    {'DFQ_RPE_CK'};...
    {'belief_CK'};...
    %{'belief_CK_bilateral'};...
    %{'hybridFQ_RPE_CK_bilateral'};...
    ];

modelLabels =[ ...
     {'WSLS'};...
     {'Q\_RPE'};{'FQ\_RPE'};...
     {'DFQ\_RPE'};...
     {'Belief model without choice kernel'};...
     {'FQ\_RPE choice kernel'};...
     {'DFQ\_RPE choice kernel'};...
     {'Belief model with choice kernel'};...
    %{'Belief + CK bilateral'};...
    %{'Belief2 + CK bilateral'};...
    %{'(\beta, \beta_C_K \alpha_C_K_L,\alpha_C_K_R, \alpha_L, \alpha_R) FQ\_RPE CK Bilateral'};...
    %{'(\beta, \beta_C_K \alpha_C_K_L,\alpha_C_K_R, H_L, H_R) Hybrid FQ\_RPE CK Bilateral'};...
    ];

%% Check how many best fitting animals
nModel = [1:8];
temp = struct2cell(models);
ax = squeeze(temp(1,1,nModel));
tx = squeeze(cell2mat(temp(2,1,nModel)) );% Exclude bilateral & flipped version & session - which gives 7 version of models.
[i d ] =min(tx(:,nModel)'); 
[counts, bins] = hist(d,nModel);
figure('position',[30 30 500 500])
barh(bins, counts,'k')
box off
yticklabels (ax)
xlabel('Number of best fitting animals');% ()
print(gcf,'-dsvg',fullfile(save_path,'modelcomparison-numOfAnimal'));
saveas(gcf, fullfile(save_path,'modelcomparison-numOfAnimal'), 'fig');

%% BIC Plot
figure('position',[30 30 1000 500])
hold on;

x=[];
y=[];
temp = struct2cell(models);
mnames = temp(1,:);
for k =1:numel(modelnames)
    ind = find(ismember(mnames, [modelnames{k}])); % find specified model
    
    x = [x; k*ones(size(models(ind).bic))];
    y = [y; log10(models(ind).bic)-log10(models(idx).bic)];  %calculate everything relative to reference model (default Q_RPE)
end

plot([0 0], [0 nFile+1],'k--','LineWidth',2);

if nFile > 50 %violin plot if a lot of data points
    violinplot(y,x,'ViolinColor',[0 0 0],'ViolinAlpha',0.8,'EdgeColor',[0.5 0.5 0.5],'BoxColor',[0.5 0.5 0.5],'MedianColor',[1 0 0],'ShowData',false);
else          %beeswarm plot works well for fewer number of points
    % beeswarm(x,y,'rand','none',3); hold on;
    
    % box plot
    h = boxplot(y,x,'Colors','k','Notch','on','Symbol',' ', 'orientation','horizontal');
    set(h,{'linew'},{2})
end

xlabel(['log BIC relative to Full model'],'interpreter','none');% ()
%set(gca,'TickLabelInterpreter','tex');
set(gca,'ytick',[1:numel(modelnames)]);
set(gca, 'yticklabel',modelLabels)
box off

print(gcf,'-dsvg',fullfile(save_path,'modelcomparison-bic'));
saveas(gcf, fullfile(save_path,'modelcomparison-bic'), 'fig');

% stat 6 -8
disp (' Comparison belief vs FQ -model')
BICvaluesForFQ = log10(models(6).bic);
BICvaluesForBelief = log10(models(8).bic);

[H,P,CI,STATS] = ttest2(BICvaluesForFQ,BICvaluesForBelief)

% stat 6 -8
disp (' Comparison belief vs DFQ - model')
BICvaluesForDF = log10(models(7).bic);
BICvaluesForBelief = log10(models(8).bic);

 [H,P,CI,STATS] = ttest2(BICvaluesForDF,BICvaluesForBelief)


