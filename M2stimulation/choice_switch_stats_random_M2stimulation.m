function output = choice_switch_stats_random_M2stimulation(selectedBlocks, stats, trials_back,trials_forw,L1_ranges,L2_ranges)
% % choice_switch_stats_random block specific %
%%

if size(L1_ranges,1) ~= size(L2_ranges,1)
    error('Error in choice_switch_random: The L1_range and L2_range should have the same number of rows');
else
    numRange = size(L1_ranges,1);
end

%selectedBlocks = [ selectedBlocks(2:end); nan]; % take the following block
choseh=zeros(1+(trials_back+trials_forw),numRange); %prob of choosing high rew side
chosel=zeros(1+(trials_back+trials_forw),numRange); %prob of choosing low rew side
choseneither=zeros(1+(trials_back+trials_forw),numRange); %prob of choosing neither (miss)

n=[-trials_back:1:trials_forw]';
numtrans=zeros(1,numRange);

for i=1:numel(stats.blockLength)-1
    
    %which kind of transition is it?
    idx=stats.blockTrans(i);
    
    if ~isnan(idx) && selectedBlocks(i)==1   %NaN could arise if we are using merge_session
        
        %within which range does this block fall into?
        range_idx = (stats.blockTrialtoCrit(i) >= L1_ranges(:,1)) & (stats.blockTrialtoCrit(i) <= L1_ranges(:,2)) &...
            (stats.blockTrialRandomAdded(i) >= L2_ranges(:,1)) & (stats.blockTrialRandomAdded(i) <= L2_ranges(:,2));
        
        if sum(range_idx) == 1   %if this transition falls into one of the subset of switches to be considered
            %what were the choices around that transition
            %note: output.n=0 is the first trial with the switched probabilities
            trial1=(sum(stats.blockLength(1:i))+1)-trials_back;
            trial2=(sum(stats.blockLength(1:i))+1)+trials_forw;
            if trial2<=numel(stats.c)
                numtrans(range_idx)=numtrans(range_idx)+1;
                choseh(:,range_idx)=choseh(:,range_idx)+(stats.c(trial1:trial2) == stats.hr_side(trial1));
                chosel(:,range_idx)=chosel(:,range_idx)+(stats.c(trial1:trial2) == -1*stats.hr_side(trial1));
                choseneither(:,range_idx)=choseneither(:,range_idx)+isnan(stats.c(trial1:trial2));
            end
        end
    end
end

stath=nan(3,numRange); %prob of choosing initial best option
statl=nan(3,numRange); %prob of choosing initial worse option
statn=nan(3,numRange); %prob of missing
for j=1:numRange
    probh = choseh(:,j)/numtrans(j);
    probl = chosel(:,j)/numtrans(j);
    probn = choseneither(:,j)/numtrans(j);
    % get trials to midpoint - 0.5
    val = find((probh(n>0)<=0.5),1);
    if ~isempty(val)
        stath(1,j) = val; % After switch, first trial below 0.5
    else
        stath(1,j) = NaN;
    end
    
    val = find((probl(n>0)>=0.5),1);
    if ~isempty(val)
        statl(1,j) = val; % After switch, first trial above 0.5
    else
        statl(1,j) = NaN;
    end
    
    val = find((probn(n>0)>=0.5),1);
    if ~isempty(val)
        statn(1,j) = val; % After switch, first trial above 0.5
    else
        statn(1,j) = NaN;
    end
      
    % get slope
    stath(2:3,j) = polyfit(n(n>0 & n<=10),probh(n>0 & n<=10),1);
    statl(2:3,j) = polyfit(n(n>0 & n<=10),probl(n>0 & n<=10),1);
    statn(2:3,j) = polyfit(n(n>0 & n<=10),probn(n>0 & n<=10),1); 
end

% get trials to reach 0.5- midpoint

output.n=n;
output.stath=stath;
output.statl=statl;
output.statn=statn;
end


