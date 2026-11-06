function data_forStat =plot_binxaveragey_simulation(input)
% % plot_binxaveragey_simulation %
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
if size(input(1,:),2)~=3
    error( 'Function only compares three models')
end

figure('position',[30 30 1200 700])
for k=1:nData
    temp = input(k,:);
    label=temp{1,1}{1}.label;
    xrange=temp{1,1}{1}.range{1};
    yrange=temp{1,1}{1}.range{2};
    
    val=[];
    for j=1:numel(temp{1,1})
        val=[val; temp{1,1}{j}.dat];
    end
    val2=[];
    for j=1:numel(temp{1,2})
        val2=[val2; temp{1,2}{j}.dat];
    end
    val3=[];
    for j=1:numel(temp{1,3})
        val3=[val3; temp{1,3}{j}.dat];
    end
    
    %% plot
    
    cc =[ 52 52 52; 256 0 256; 8 133 161]/256;
    subplot(2,ceil(nData/2),k);
    nStep = numel(xrange(1):3:xrange(2)-1);
    mean_y = nan(nStep,1);
    mean_y2 = nan(nStep,1);
    mean_y3 = nan(nStep,1);
    j_ind =1;
    hold on;
    for j=xrange(1):2:xrange(2)-1  %x in step of 3, to reduce noise becsuse of fewer sessions
        y = val(val(:,1)==j| val(:,1)==j+1,2);%| val(:,1)==j+2,2);
        y2= val2(val2(:,1)==j |val2(:,1)==j+1 ,2);%| val2(:,1)==j+2,2);
        y3= val3(val3(:,1)==j |val3(:,1)==j+1,2);% | val3(:,1)==j+2,2);
        
        plot(j+0.8,nanmean(y),'k.','MarkerSize',30);
        plot(j+1,nanmean(y2),'.','MarkerSize',30,'color',cc(2,:));
        plot(j+1.2,nanmean(y3),'.','MarkerSize',30,'color',cc(3,:));
        
        plot([j+0.8 j+0.8],nanmean(y)+nanstd(y)/sqrt(numel(y))*[-1 1],'k-','LineWidth',3);
        plot([j+1 j+1],nanmean(y2)+nanstd(y2)/sqrt(numel(y2))*[-1 1],'-','LineWidth',3,'color',cc(2,:));
        plot([j+1.2 j+1.2],nanmean(y3)+nanstd(y3)/sqrt(numel(y3))*[-1 1],'-','LineWidth',3,'color',cc(3,:));
        mean_y(j_ind)=nanmean(y);
        mean_y2(j_ind)=nanmean(y2);
        mean_y3(j_ind)=nanmean(y3);
        j_ind = j_ind+1;
    end
    xlim([xrange(1)-1 xrange(2)+1]);
    ylim([yrange(1) yrange(2)]);
    xlabel(label{1});
    ylabel(label{2});
    data_forStat(k).val= mean_y;
    data_forStat(k).val2= mean_y2;
    data_forStat(k).val3= mean_y3;
    data_forStat(k).entryType = label{2};
    
end

legend('Experimental','Belief model','FQ-RPE model','location','northwest');
legend box off;
end


