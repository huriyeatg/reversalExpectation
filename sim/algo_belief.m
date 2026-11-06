function stats=algo_belief(stats,params)
% % algo_belief %
%PURPOSE:   Simulate player based on hazard rate/belief
%AUTHORS:   H Atilgan & AC Kwan 09152021
%
%INPUT ARGUMENTS
%   stats:  stats of the game thus far
%   params: parameters that define the player's strategy
%       params(1) = hazard rate
%       params(2) = inverse temperature
%
%OUTPUT ARGUMENTS
%   stats:  updated with player's probability to choose left for next step
%updated 23/09/2021
%%
hazardrate = params(1); % hazard rate
beta = params(2); % inverse temperature

%ps=(0:0.01:1)';
ps=[0.1 0.7]';  %possible states of reward probability distribution
cpPrior=ones(size(ps))./length(ps);  %prior is the states are equally likely

if stats.currTrial == 1  %if this is the first trial
    % BELIEF updates
    stats.p(stats.currTrial,:) = cpPrior';
    stats.lr(stats.currTrial) = 0.5;
    stats.mu_left(stats.currTrial)=sum(ps.*cpPrior);
    stats.mu_right(stats.currTrial)=sum(ps.*(1-cpPrior));
    
    %relative uncertainty (trial entropy) and CPP
    p = cpPrior;

else
    %% Update
    choice = stats.c(stats.currTrial-1);
    outcome = stats.r(stats.currTrial-1);
    p = stats.p(stats.currTrial-1,:)';
    
    if ~isnan(choice) && ~isnan(outcome)
                
        %% Hazard rate
        if choice==1
            pR = 1-p;
            pR = (pR.*(1-hazardrate)) +(cpPrior.*hazardrate);
            pR = pR./nansum(pR);
            p = 1-pR;
        else
            p = p.*(1-hazardrate)+cpPrior.*hazardrate;
        end
        p = p./nansum(p); % normalize;

        %% Multiply prior distribution by likelihood of producing new outcome   
        if choice == -1
            if outcome == 1     %left and rewarded
                p=p.*ps;
            elseif outcome == 0 %left and unrewarded
                p=p.*(1-ps);
            end
        end
        if choice == 1
            pR = 1-p;
            if outcome == 1     %right and rewarded
                pR=pR.*ps;
            elseif outcome == 0 %right and unrewarded
                pR=pR.*(1-ps);
            end
            p = 1-pR;
            clear pR;
        end
        p = p./nansum(p);
        stats.p(stats.currTrial,:) = p;

        %% Compute learning rate
        stats.mu_left(stats.currTrial)=sum(ps.*p);      % expected reward rate for left
        stats.mu_right(stats.currTrial)=sum(ps.*(1-p)); % expected reward rate for right
    else
        stats.mu_left(stats.currTrial) = stats.mu_left(stats.currTrial-1);
        stats.mu_right(stats.currTrial) = stats.mu_right(stats.currTrial-1);
        stats.p(stats.currTrial,:) = stats.p(stats.currTrial-1,:);
    end
end
  
    %% Softmax based on expected reward rates
    Q = [stats.mu_right(stats.currTrial) stats.mu_left(stats.currTrial)];
    V = beta*Q;
    pChoice = exp(V)/sum(exp(V));
    stats.pr(stats.currTrial) = pChoice(1);
    stats.pl(stats.currTrial) = pChoice(2);

end
