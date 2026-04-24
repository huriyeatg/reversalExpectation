function plot_multibinxaveragey (input)
% % plot_multibinxaveragey %
%PURPOSE:   Bin x and average y
%AUTHORS:   H Atilgan & AC Kwan 03142020
%
%INPUT ARGUMENTS
%   input:      Paired data in cells

%%
nInput = numel(input);

figure;
for k=1:nInput
    subplot(2,2,k)%round(nInput/2),k); 
    hold on
    temp = input{k};
    
    label=temp{1}.label;
    xrange=temp{1}.range{1};
    yrange=temp{1}.range{2};
    
    val=[];
    for j=1:numel(temp)
        val=[val; temp{j}.dat];
    end
    
    %% plot
    stat_data =[];stat_group=[];
    for j=xrange(1):2:xrange(2)-1
        y = val(val(:,1)==j| val(:,1)==j+1,2);%| val(:,1)==j+2,2);
       % y=val(val(:,1)==j,2);
        plot(j,nanmean(y),'k.','MarkerSize',20);
        plot([j j],nanmean(y)+nanstd(y)/sqrt(numel(y))*[-1 1],'k-','LineWidth',3);
        stat_data = [ stat_data; y];
        stat_group = [stat_group; ones(size(y)).*j];
    end
    box off
    xlim([xrange(1)-1 xrange(2)+1]);
    ylim([yrange(1) yrange(2)]);
    xlabel(label{1});
  %  ylabel(label{2});  
    % stats - one way ANOVA
  %  [p,ANOVATAB,STATS] = anova1(stat_data, stat_group, 'off')
 %   title( ['F=',num2str(ANOVATAB{2,5}) ' df=' num2str(STATS.df) ' p=', num2str(p)]);
end