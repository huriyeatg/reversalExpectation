function stats=algo_FQ_RPE_CK(stats,params)
% % algo_FQ_RPE_CK %
%PURPOSE:   Simulate player based on q-learning with reward prediction
%           errors, with differential learning rates with choice tendency
%AUTHORS:   H Atilgan & AC Kwan 04302020
%
%INPUT ARGUMENTS
%   stats:  stats of the game thus far
%   params: parameters that define the player's strategy
%       params(1) = a - learning rate for choice yielding reward
%       params(2) = beta - inverse temperature
%       params(3) = a CK- learning rate for choice tendency
%       params(4) = beta CK - inverse temperature for choice tendency
%
%OUTPUT ARGUMENTS
%   stats:  updated with player's probability to choose left for next step
%%
a = params(1);
beta = params(2);
alpha_c = params(3);
beta_c = params(4);

if stats.currTrial == 1  %if this is the first trial
    stats.ql(stats.currTrial) = 0.5;
    stats.qr(stats.currTrial) = 0.5;
    stats.rpe(stats.currTrial) = NaN;
    stats.pl(stats.currTrial) = 0.5;
    stats.pr(stats.currTrial) = 0.5;
    % choice kernels updates
    stats.ckl(stats.currTrial) = 0;
    stats.ckr(stats.currTrial) = 0;
    
else
        %% update choice kernel
    CK = [stats.ckr(stats.currTrial-1) stats.ckl(stats.currTrial-1)];
    CK = (1-alpha_c) *CK;
    if stats.c(stats.currTrial-1)==1   % else, chose right
        stats.ckr(stats.currTrial) = CK(1)+ alpha_c * 1;
        stats.ckl(stats.currTrial) = CK(2);
    elseif stats.c(stats.currTrial-1)==-1
        stats.ckr(stats.currTrial) = CK(1);
        stats.ckl(stats.currTrial) = CK(2) + alpha_c * 1; % left
    else
        stats.ckl(stats.currTrial) = stats.ckl(stats.currTrial-1);
        stats.ckr(stats.currTrial) = stats.ckr(stats.currTrial-1);
    end

    %% update action values   
    if stats.c(stats.currTrial-1)==-1   % if chose left on last trial
        stats.rpe(stats.currTrial-1)=stats.r(stats.currTrial-1)-stats.ql(stats.currTrial-1);
        stats.ql(stats.currTrial)=stats.ql(stats.currTrial-1)+a*stats.rpe(stats.currTrial-1);
        stats.qr(stats.currTrial)=(1-a)*stats.qr(stats.currTrial-1);
    elseif stats.c(stats.currTrial-1)==1   % else, chose right
        stats.rpe(stats.currTrial-1)=stats.r(stats.currTrial-1)-stats.qr(stats.currTrial-1);
        stats.ql(stats.currTrial)=(1-a)*stats.ql(stats.currTrial-1);
        stats.qr(stats.currTrial)=stats.qr(stats.currTrial-1)+a*stats.rpe(stats.currTrial-1);

    else  %miss trials
        stats.rpe(stats.currTrial-1) = 0;
        stats.ql(stats.currTrial)=stats.ql(stats.currTrial-1);
        stats.qr(stats.currTrial)=stats.qr(stats.currTrial-1);
    end
    
    %% softmax with choice kernel
    Q = [stats.qr(stats.currTrial) stats.ql(stats.currTrial)];
    CK = [stats.ckr(stats.currTrial) stats.ckl(stats.currTrial)];
    V = beta*Q + beta_c*CK;
    p = exp(V)/sum(exp(V));
    stats.pr(stats.currTrial) = p(1);
    stats.pl(stats.currTrial) = p(2);
    
end

end
