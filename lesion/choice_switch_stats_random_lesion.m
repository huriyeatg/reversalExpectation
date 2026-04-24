function output = choice_switch_stats_random_lesion(stats,trials_back,trials_forw,L1_ranges,L2_ranges)
% % choice_switch_stats_random side specific %

%%

if size(L1_ranges,1) ~= size(L2_ranges,1)
    error('Error in choice_switch_random: The L1_range and L2_range should have the same number of rows');
else
    numRange = size(L1_ranges,1);
end

%how many different types of rule transitions?
numruletrans=size(stats.ruletransList,1);


chosel=zeros(1+(trials_back+trials_forw),numruletrans,numRange); %prob of choosing left
choser=zeros(1+(trials_back+trials_forw),numruletrans,numRange); %prob of choosing right
choseneither=zeros(1+(trials_back+trials_forw),numruletrans,numRange); %prob of not choosing (miss)

n=[-trials_back:1:trials_forw]';
numtrans=zeros(numruletrans,numRange);    %number of that type of transition

for i=1:numel(stats.blockLength)-1
    %which kind of transition is it?
    idx=stats.blockTrans(i);
    
    if ~isnan(idx)   %NaN could arise if we are using merge_session
        
        %within which range does this block fall into?
        range_idx = (stats.blockTrialtoCrit(i) >= L1_ranges(:,1)) & (stats.blockTrialtoCrit(i) <= L1_ranges(:,2)) &...
            (stats.blockTrialRandomAdded(i) >= L2_ranges(:,1)) & (stats.blockTrialRandomAdded(i) <= L2_ranges(:,2));
        
        if sum(range_idx) == 1   %if this transition falls into one of the subset of switches to be considered
            %what were the choices around that transition
            %note: output.n=0 is the first trial with the switched probabilities
            trial1=(sum(stats.blockLength(1:i))+1)-trials_back;  %trials before switch
            trial2=(sum(stats.blockLength(1:i))+1)+trials_forw;  %trials after switch
            
            if trial2<=numel(stats.c)   %if trial after switch does not exceed end of session
                numtrans(idx,range_idx)=numtrans(idx,range_idx)+1;
                chosel(:,idx,range_idx)=chosel(:,idx,range_idx)+(stats.c(trial1:trial2)==-1);
                choser(:,idx,range_idx)=choser(:,idx,range_idx)+(stats.c(trial1:trial2)==1);
                choseneither(:,idx,range_idx)=choseneither(:,idx,range_idx)+isnan(stats.c(trial1:trial2));
            end
        end
    end
end


statl=nan(3,2,numRange); %left - midpoint, slope, intercept
statr=nan(3,2,numRange); %right
statn=nan(3,2,numRange); %prob of missing
for j=1:numruletrans
    for k=1:numRange
        if numtrans(j,k)>0
            probl=chosel(:,j,k)/numtrans(j,k);
            probr=choser(:,j,k)/numtrans(j,k);
            probn=choseneither(:,j,k)/numtrans(j,k);
        % get trials to midpoint - 0.5
        val = find((probl(n>0)<=0.5),1);
        if ~isempty(val)
            statl(1,j,k) = val; % After switch, first trial below 0.5
        else
            statl(1,j,k) = NaN;
        end
        
        % get trials to midpoint - 0.5
        val = find((probr(n>0)<=0.5),1);
        if ~isempty(val)
            statr(1,j,k) = val; % After switch, first trial below 0.5
        else
            statr(1,j,k) = NaN;
        end
        
        % get trials to midpoint - 0.5
        val = find((probn(n>0)<=0.5),1);
        if ~isempty(val)
            statn(1,j,k) = val; % After switch, first trial below 0.5
        else
            statn(1,j,k) = NaN;
        end
             
        % slope & intercept:  not computed for lesion data: 
        % Check choice_switch_stats_random.m file if needed
        end
    end
end



% get trials to reach 0.5- midpoint

output.n=n;
output.statl=statl;
output.statr=statr;
output.statn=statn;
end


