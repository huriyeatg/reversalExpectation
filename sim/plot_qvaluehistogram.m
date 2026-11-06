function plot_qvaluehistogram(model_type,dataIndex,model_path)
 % plot_qvaluedifferences %%


 
%% concatenate the sessions for this one animal
exampleAnimal = dataIndex.Animal(1);
[trialData, trials, nRules] = merge_sessions(dataIndex);

stats = value_getTrialStats(trials, nRules);
stats = value_getTrialStatsMore(stats);

%% use the animal data and fitted model parameters to simulate agent updating

load(fullfile(model_path,[model_type '.mat']));
if exist('session','var')   %if the model fitting was done per session, then find the parameters for that specific session
    idx_sim = find(ismember(session,exampleLogName)==1);
else  %else the model fitting was done per animal
    idx_sim = find(ismember(animal,exampleAnimal)==1);
end

player_fit.label=['algo_' model_type];
player_fit.params=fitpar{idx_sim};
beta =   player_fit.params(2); 
stats_sim=predictAgent(player_fit,stats);


%% Plot Q value & action seletion

qdiff =stats_sim.qr-stats_sim.ql;
%plot histogram of action value differences
maxVal = ceil(nanmax(qdiff));
minVal = floor(nanmin(qdiff));
edges  = [minVal:0.05:maxVal];
edges_center = edges(1:end-1)+mean(diff(edges))/2; %center of the bins, for plotting

n=histc(qdiff,edges);
n=n(1:end-1);

figure;
subplot(2,2,1)
hold on;
yyaxis left;
bar(edges_center,n,'k');
xlabel('Q_R - Q_L');
ylabel('Occurrence');
ax = gca;
ax.YAxis(1).Color = 'k';
ylim([0 1.1*max(n)]);
xlim([edges(1) edges(end)]);

%plot probability of choosing left as a function of action value diff
yyaxis right;

probR=nan(numel(edges_center),1);
for j=1:numel(edges_center)
    numL=sum(stats.c(qdiff>=edges(j) & qdiff<edges(j+1))==-1);
    numR=sum(stats.c(qdiff>=edges(j) & qdiff<edges(j+1))==1);
    probR(j)=numR/(numL+numR);
end

plot(edges_center,probR,'.','MarkerSize',35);
%if softmax rule was used, what is the expected curve?
exp_probR=(1)./(1+exp(-beta*(edges_center))); 
plot(edges_center,exp_probR,'--','LineWidth',3);
ylabel('P_{Right}');

ylim([-0.02 1.02]);


