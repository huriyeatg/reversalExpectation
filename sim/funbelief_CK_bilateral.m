function varargout=funbelief_CK_bilateral(xpar,stats)
% % funhybridFQ_RPE %
%PURPOSE:   Function for maximum likelihood estimation, called by fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       hazardrate, beta
%   stats:      info about animal/agent's performance
%
%OUTPUT ARGUMENTS
%   negloglike:      the negative log-likelihood to be minimized
% updated: 23/17/2020
%%
hazardrateL =xpar(1);
hazardrateR =xpar(2);
alpha_cL  = xpar(3);
alpha_cR  = xpar(4);
beta    = xpar(5);
beta_c  = xpar(6);

ps=[0.1 0.7]';  %possible states of reward probability distribution
cpPrior=ones(size(ps))./length(ps);  %prior is the states are equally likely
p=cpPrior; % for first trial.- will be updated.

nt=size(stats.c,1);
ChoiceProb = nan(nt,1);
CK =[0 0]; %for first trial, will be updated.
stats.mu_right = nan(nt,1);
stats.mu_left = nan(nt,1);


for k=1:nt
    %% softmax - action selection
    Q = [stats.mu_right(k) stats.mu_left(k)];
    V = (beta *Q) + (beta_c*CK) ;
    pChoice = exp(V)/sum(exp(V));
    pright = pChoice(1);
    pleft  = pChoice(2);
    
    if pright==0
        pright=realmin;   % Smallest positive normalized floating point number, because otherwise log(zero) is -Inf
    end
    if pleft==0
        pleft=realmin;
    end
    
    % calculate choice probability for actual choice
    if stats.c(k)==1
        ChoiceProb(k) = pright;
    elseif stats.c(k)==-1
        ChoiceProb(k) = pleft;
    else
        ChoiceProb(k) = nan;
    end
    
    %% calculate learning rate
    choice = stats.c(k);
    outcome = stats.r(k);
    
    if ~isnan(choice) && ~isnan(outcome)
        
        %% Compute p(mu_{t+1}|mu{t},H)
        if choice==1
            pR = 1-p;
            pR = (pR.*(1-hazardrateR)) +(cpPrior.*hazardrateR);
            pR = pR./nansum(pR);
            p = 1-pR;
        else
            p = (p.*(1-hazardrateL))+ (cpPrior.*hazardrateL);
        end
        p = p./nansum(p); % normalize;
        
        %% Multiply prior distribution by likelihood of producing new outcome
               
        if choice == -1
            if outcome == 1     %left and rewarded
                p=p.*ps;   
            elseif outcome == 0 %left and unrewarded
                p=p.*(1-ps);  
            end
            p = p./nansum(p);
        end
        if choice == 1
            pR = 1-p;
            if outcome == 1     %right and rewarded
                pR=pR.*ps;    
            elseif outcome == 0 %right and unrewarded
                pR=pR.*(1-ps);  
            end
            pR = pR./nansum(pR);
            
            p = 1-pR;
            clear pR;
        end
        
        %% Compute learning rate
        stats.mu_left(k+1)=sum(ps.*p);      % expected reward rate for left
        stats.mu_right(k+1)=sum(ps.*(1-p)); % expected reward rate for right
    else
        if k>1
        stats.mu_left(k) = stats.mu_left(k-1);
        stats.mu_right(k) = stats.mu_right(k-1);
        end
    end
    
      %% choice kernel update
    if stats.c(k)==1   % if chosen right
        CK(1) = (1-alpha_cR)*CK(1)+ (alpha_cR *1);
        CK(2) = (1-alpha_cR)*CK(2);
    elseif stats.c(k)==-1
        CK(1) = (1-alpha_cL)*CK(1);
        CK(2) = (1-alpha_cL)*CK(2)+ (alpha_cL *1); % left
    else % for miss - no action, no update
        CK(1) = CK(1);%(1-alpha_cR)*CK(1);
        CK(2) = CK(2);%(1-alpha_cL)*CK(2);
    end
  
end

negloglike = - sum(log(ChoiceProb(~isnan(ChoiceProb))));
varargout{1} = negloglike;
end


%
%
% % Note that the output of this function is the negative of the
% % log-likelihood not the true log likelihood. This is because Matlab's
% % optimization algorithms are setup to minimize functions not maximize
% % them. Obviously minimizing the negative log-likelihood is the same as
% %maximizing the log-likelihood. (Robert Wilson'notes, 2019 Summer School)
