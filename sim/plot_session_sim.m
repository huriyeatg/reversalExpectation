function plot_session_sim(stats,n_plot)
% % plot_session_task %
%PURPOSE:   Plot performance of a single session, comparing mice with
%           simulation
%AUTHORS:   AC Kwan 191212- edited by HA 042020
%
%INPUT ARGUMENTS
%   stats:  stats of the task
%   n_plot: plot up to this number of trials
%   tlabel: Text string that will be put on top of the figure

%%
figure;

subplot(4,1,1); hold on; % Reward probabilities
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

subplot(4,1,2); hold on; % Probability of choosing right
plot(movmean((stats.c==1),10),'b-','Linewidth',2); %experiment, moving average of 10 trials
plot(1-stats.pl,'m-','LineWidth',2); %model
ylabel({'P(Right)'},'interpreter','none');
xlim([0 n_plot]);
ylim([0 1]);
set(gca,'xticklabel',[]);
legend('Animal','Model');
legend box off;

subplot(4,1,3); hold on; % belief distribution
if isfield(stats,'p')
    plot(stats.p(:,1),'b-','LineWidth',2);
    plot(stats.p(:,2),'r-','LineWidth',2);
end
ylabel({'Belief'},'interpreter','none');
xlim([0 n_plot]);
set(gca,'xticklabel',[]);
legend({'pL10','pL70'});
legend box off;

subplot(4,1,4); hold on  %plot choice kernel
plot(stats.ckl,'r'); 
plot(stats.ckr,'b'); 
ylabel('Choice kernel')

ylim([0 1]); 
xlim([0 n_plot]);
box off
xlabel('Trial');


end

