function stats=algo_belief_CK(stats,params)
% % algo_hybridFQ_RPE_CK %
%PURPOSE:   Simulate player based on hazard rate/belief
%AUTHORS:   H Atilgan & AC Kwan 
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
alpha_c = params(3);
beta_c = params(4);

ps=[0.1 0.7]';  %possible states of reward probability distribution
cpPrior=ones(size(ps))./length(ps);  %prior is the states are equally likely

if stats.currTrial == 1  %if this is the first trial
    % BELIEF updates
    stats.p(stats.currTrial,:)  = cpPrior';
    stats.lr(stats.currTrial) = 0.5;
    stats.mu_left(stats.currTrial)=sum(ps.*cpPrior);
    stats.mu_right(stats.currTrial)=sum(ps.*(1-cpPrior));
    
    % choice kernels updates
    stats.ckl(stats.currTrial) = 0;
    stats.ckr(stats.currTrial) = 0;
    
    %relative uncertainty (trial entropy) and CPP
    p = cpPrior;
    gSel=p>0;
    stats.trialEntropy(stats.currTrial)=sum(p(gSel).*(-log(p(gSel))));
    stats.dataLL(stats.currTrial) = NaN;
else
    %% update choice kernel
    
    if stats.c(stats.currTrial-1)==1   % else, chose right
        CK = [stats.ckr(stats.currTrial-1) stats.ckl(stats.currTrial-1)];
        CK = (1-alpha_c) *CK;   
        stats.ckr(stats.currTrial) = CK(1)+ alpha_c * 1;
        stats.ckl(stats.currTrial) = CK(2);
    elseif stats.c(stats.currTrial-1)==-1
        CK = [stats.ckr(stats.currTrial-1) stats.ckl(stats.currTrial-1)];
        CK = (1-alpha_c) *CK;
        stats.ckr(stats.currTrial) = CK(1);
        stats.ckl(stats.currTrial) = CK(2) + alpha_c * 1; % left
    else
        stats.ckl(stats.currTrial) = stats.ckl(stats.currTrial-1);
        stats.ckr(stats.currTrial) = stats.ckr(stats.currTrial-1);
    end
    
    %% Update value
    choice = stats.c(stats.currTrial-1);
    outcome = stats.r(stats.currTrial-1);
    p = stats.p(stats.currTrial-1,:)';
    
    if ~isnan(choice) && ~isnan(outcome)
        
        %% calculate trial entropy (estimation for relative uncertainty)
        gSel=p>0;
        stats.trialEntropy(stats.currTrial)=sum(p(gSel).*(-log(p(gSel))));
        
        %% calculate loglikelihood
        if outcome && choice==-1
            condProb=p.*ps;
        elseif ~outcome && choice==-1
            condProb=p.*flip(ps);
        elseif outcome && choice==1
            condProb=p.*flip(ps);
        elseif ~outcome && choice==1
            condProb=p.*ps;
        end
        stats.dataLL(stats.currTrial)  = log(sum(condProb)); % will be used for CPP
        
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
        if choice == -1
            UP = stats.mu_left(stats.currTrial)- stats.mu_left(stats.currTrial-1);
            PE = outcome - stats.mu_left(stats.currTrial);  %predicted outcome for left versus actual outcome for left
        elseif choice == 1
            UP = stats.mu_right(stats.currTrial)- stats.mu_right(stats.currTrial-1);
            PE = outcome - stats.mu_right(stats.currTrial); %predicted outcome for right versus actual outcome for right
        end
        stats.lr(stats.currTrial) = UP./PE; % learningrate
    else
        stats.lr(stats.currTrial) = stats.lr(stats.currTrial-1);
        stats.mu_left(stats.currTrial) = stats.mu_left(stats.currTrial-1);
        stats.mu_right(stats.currTrial) = stats.mu_right(stats.currTrial-1);
        stats.p(stats.currTrial,:) = stats.p(stats.currTrial-1,:);
    end
end

%% Softmax based on expected reward rates
Q = [stats.mu_right(stats.currTrial) stats.mu_left(stats.currTrial)];
CK = [stats.ckr(stats.currTrial) stats.ckl(stats.currTrial)];
V = beta*Q + beta_c*CK;
pChoice = exp(V)/sum(exp(V));
stats.pr(stats.currTrial) = pChoice(1);
stats.pl(stats.currTrial) = pChoice(2);

end
