function plot_beliefEffectinLatentVariables(stats,n_plot)
% % plot_beliefEffectinLatentVariables %
%PURPOSE:   Plot performance of a single session, comparing mice with
%           simulation
%AUTHORS:    HA 091520
%
%INPUT ARGUMENTS
%   stats:  stats of the task
%   n_plot: plot up to this number of trials
%   tlabel: Text string that will be put on top of the figure

%%
figure;

subplot(6,1,1); hold on; % Reward probabilities
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
%title(tlabel2,'interpreter','none');

subplot(6,1,2); hold on; % Choices and outcomes
bar(-1*(stats.c(:,1)==-1 & stats.r(:,1)==1),1,'FaceColor','k','EdgeColor','none');
bar(-0.8*(stats.c(:,1)==-1 & stats.r(:,1)==1),1,'FaceColor','w','EdgeColor','none'); %use the white color to create space
bar(-0.7*(stats.c(:,1)==-1),1,'FaceColor','r','EdgeColor','none');
bar(1*(stats.c(:,1)==1 & stats.r(:,1)==1),1,'FaceColor','k','EdgeColor','none');
bar(0.8*(stats.c(:,1)==1 & stats.r(:,1)==1),1,'FaceColor','w','EdgeColor','none');
bar(0.7*(stats.c(:,1)==1),1,'FaceColor','b','EdgeColor','none');
ylabel({'Choice'},'interpreter','none');
xlim([0 n_plot]);
ylim([-1 1]);
set(gca,'xticklabel',[]);
set(gca,'ytick',[-1 -0.7 0.7 1]);
set(gca,'yticklabel',{'Reward','Left','Right','Reward'});

subplot(6,1,3); hold on; % Probability of choosing right
plot(movmean((stats.c==1),10),'b-','Linewidth',2); %experiment, moving average of 10 trials
plot(1-stats.pl,'m-','LineWidth',2); %model
ylabel({'P(Right)'},'interpreter','none');
xlim([0 n_plot]);
ylim([0 1]);
set(gca,'xticklabel',[]);
legend('Animal','Model');
legend box off;

subplot(6,1,4); hold on; % Latent variables
if isfield(stats,'ql')
    plot(stats.ql,'k-','LineWidth',2);
end
if isfield(stats,'qr')
    plot(stats.qr,'g-','LineWidth',2);
end
ylabel({'Latent variables'},'interpreter','none');
xlim([0 n_plot]);
set(gca,'xticklabel',[]);
legend({'Q_L','Q_R'});
legend box off;

subplot(6,1,5) % change-point-probability
temp = cell2mat(stats.p(1:n_plot)');
if size(temp,1)>2
    imagesc(temp);
else
    plot(temp');
    box off
end

xlim([0 n_plot]);
ylabel('ps')
xlabel('Trial')


end

