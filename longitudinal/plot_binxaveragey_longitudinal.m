function plot_binxaveragey_longitudinal(input,tlabel,bin,save_path)
% % plot_binxaveragey_longitudinal %
%PURPOSE:   Make scatter plot
%AUTHORS:   H Atilgan & AC Kwan (created:191211, modified: 200416)
%
%INPUT ARGUMENTS
%   input:      Paired data
%   tlabel:     Figure title (a string)
%   bin:        Averaged across specifid bin. If not defined, will be in 3
%               steps.
%%

nData = size(input,1);
figure;

stat_data =[];stat_group=[];

for k=1:nData
    temp = input(k,:);
    label=temp{1,1}{1}.label;
    xrange=temp{1,1}{1}.range{1};
    yrange=temp{1,1}{1}.range{2};
    
    if nargin==2 % if bin not defined, use 3 step bin.
        bin = xrange(1):3:xrange(2)-1;    
    end
    nbin=size(bin,1);
    
    val=[];
    for j=1:numel(temp{1,1})
        val=[val; temp{1,1}{j}.dat];
    end
    val2=[];
    for j=1:numel(temp{1,2})
        val2=[val2; temp{1,2}{j}.dat];
    end
    
    %% plot
    subplot(2,3,k);
    hold on;
    
    yMean  = nan(nbin,1);
    y2Mean = nan(nbin,1);
    str = string;
    ymin=1; ymax=0;
    for j = 1:nbin
        y = val(val(:,1)>=bin(j,1) & val(:,1)<=bin(j,2),2);
        y2 = val2(val2(:,1)>=bin(j,1) & val2(:,1)<=bin(j,2),2);
        
        plot(j+0.95,nanmean(y),'k.','MarkerSize',30);
        plot(j+1.05,nanmean(y2),'.','MarkerSize',30,'color',[0 150/256 0]);
        plot([j+0.95 j+0.95],nanmean(y)+nanstd(y)/sqrt(numel(y))*[-1 1],'k-','LineWidth',3);
        plot([j+1.05 j+1.05],nanmean(y2)+nanstd(y2)/sqrt(numel(y2))*[-1 1],'-','LineWidth',3,'color',[0 150/256 0]);
        str (j,:) = [num2str(bin(j,1)), '-', num2str(bin(j,2))];
        ymax = max([ ymax nanmean(y) nanmean(y2)]);
        ymin = min([ ymin nanmean(y) nanmean(y2)]);
        yMean(j,1) = nanmean(y);
        y2Mean(j,1) = nanmean(y2);
        
        stat_data = [ stat_data; y;y2];
        stat_group = [stat_group; ones(size(y)).*[j,k,1]; ones(size(y2)).*[j,k,2]]; % blockLength,left/right side,LesionSes
    end
    plot([1:nbin]+0.9,yMean,'-k')
    plot([1:nbin]+1.1,y2Mean,'-','color',[0 150/256 0])
    xlim([1.5 nbin+1.5]);
    xticks([2:nbin+1])
    xticklabels(str)
    ylim([yrange(1) yrange(2)]);
    xlabel(label{1});
    ylabel(label{2});
    legend('naive', 'trained');
    legend boxoff;
    title(tlabel(k,:),'interpreter','none');
      
end

if strcmp(label{2}{1},'P(stay|win)')
    print(gcf,'-dpng',fullfile(save_path,'Pstay'));
    saveas(gcf, fullfile(save_path,'Pstay'), 'fig');
    
elseif  strcmp(label{2}{1},'P(lose|switch)')
    print(gcf,'-dpng',fullfile(save_path,'Plose'));
    saveas(gcf, fullfile(save_path,'Plose'), 'fig');
else  
    print(gcf,'-dpng',fullfile(save_path,label{2}{1}));
    saveas(gcf, fullfile(save_path,label{2}{1}), 'fig');
end
%  close all hidden

% % stats - one way ANOVA
% [p,ANOVATAB,STATS] = anovan(stat_data, {stat_group(:,1),stat_group(:,2),stat_group(:,3)},...
%     'varnames',{'Block length', 'Side', 'Session'},'model','full') 
% 
% hTable = figure(1);
% if strcmp(label{2}{1},'P(stay|win)')
%     print(hTable,'-dpng',fullfile(save_path,'Pstay_statsOutput'));
% elseif  strcmp(label{2}{1},'P(lose|switch)')
%     print(hTable,'-dpng',fullfile(save_path,'Plose_statsOutput'));
% else
%     print(hTable,'-dpng',fullfile(save_path,[label{2}{1},'_statsOutput']));
% end


end

