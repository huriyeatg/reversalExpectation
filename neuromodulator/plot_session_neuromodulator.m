function plot_session_neuromodulator( stats,trials, tlabel, savebehfigpath)
% %%plot_session_neuromodulator( stats)%%
%%
colors=cbrewer('seq','OrRd',256);
% plot choice behavior - whole sessions
n_plot = 100*ceil(numel(stats.c)/100); %plot up to the nearest 100 trials
% analyze_session(stats,tlabel,savebehfigpath);
figure;

subplot(3,1,1); hold on; % Reward probabilities
plot(stats.rewardprob(:,1),'r','LineWidth',2);
hold on;
plot(stats.rewardprob(:,2),'b','LineWidth',2);
ylabel('Reward probability (%)');
legend('Left','Right');
legend box off;
xlim([0 n_plot]);
ylim([0 1]);
set(gca,'xticklabel',[]);
set(gca,'ytick',[0 0.1 0.7 1]);
set(gca,'yticklabel',{'','10','70', ''});
title(tlabel,'interpreter','none');

subplot(3,1,2); hold on; % Choices and outcomes
bar(-1*(stats.c(:,1)==-1 & stats.r(:,1)==1),1,'FaceColor','k','EdgeColor','none');
bar(-0.8*(stats.c(:,1)==-1 & stats.r(:,1)==1),1,'FaceColor','w','EdgeColor','none'); %use the white color to create space
bar(-0.7*(stats.c(:,1)==-1),1,'FaceColor','r','EdgeColor','none');
bar(1*(stats.c(:,1)==1 & stats.r(:,1)==1),1,'FaceColor','k','EdgeColor','none');
bar(0.8*(stats.c(:,1)==1 & stats.r(:,1)==1),1,'FaceColor','w','EdgeColor','none');
bar(0.7*(stats.c(:,1)==1),1,'FaceColor','b','EdgeColor','none');
ylabel({'Choice'},'interpreter','none');
xlim([0 n_plot]);
ylim([-1 1]);
xlabel('Trial');
set(gca,'ytick',[-1 -0.7 0.7 1]);
set(gca,'yticklabel',{'Reward','Left','Right','Reward'});
title(['Overall reward rate = ' num2str(sum(stats.r(:,1)==1)/(sum(stats.r(:,1)==1)+sum(stats.r(:,1)==0)))]);

subplot(3,1,3); hold on; % Choices and outcomes
tWindow = -1.95:1/20:4;
nTrials = size(trials.dffN',2);
imagesc(1:nTrials, tWindow,trials.dff','CDataMapping','scaled');
colormap(colors);

xlim([0 n_plot]);
ylim([-2 4]);
xlabel('Trial');
ylabel('Time (sec)')

print(gcf,'-dsvg',fullfile(savebehfigpath,'session'));
saveas(gcf, fullfile(savebehfigpath,'session'), 'fig');

figure

temp_psth.signal = trials.dff;
temp_psth.t = -1.95:1/20:4;
temp_psth.psth_label = ' ';

xlabel_st = 'Time from stimulus (s)';
plot_snake(temp_psth,[0 6.5],xlabel_st);

print(gcf,'-dsvg',fullfile(savebehfigpath,'neuralSignal'));
saveas(gcf, fullfile(savebehfigpath,'neuralSignal'), 'fig');

end

