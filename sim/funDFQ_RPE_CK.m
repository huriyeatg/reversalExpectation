function varargout=funDFQ_RPE_CK(xpar,stats)
% % funDQ_RPE %
%PURPOSE:   Function for maximum likelihood estimation, called by fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       alpha,lambda, beta, alphaCK, betaCK
%   stats:      info about animal/agent's performance
%
%OUTPUT ARGUMENTS
%   negloglike:      the negative log-likelihood to be minimized
% updated: 04/30/2020
%%
if nargin == 0 % Used for optimizer testing
    varargout{1} = [0,0];                   % LB
    varargout{2} = [1,10];                  % UB
    varargout{3} = [0,3000];                % Zlim
    varargout{4} = [-30 75];                % view
    varargout{5} = [0.5 0.5 3];             % xmin
    varargout{6} = 'DFQ_RPE function';       % name
    varargout{7} = [0.5 0.5 3];             % default x0
    return;
end

alpha=xpar(1);
lambda=xpar(2);
beta=xpar(3);
alpha_c = xpar(4);
beta_c = xpar(5);


nt=size(stats.c,1);
ChoiceProb = nan(nt,1);
CK =[0 0];  % right, left choice kernel, will be updated in each trial.

v_right=0.5;
v_left=0.5;

for k=1:nt
    %% softmax with choice kernel
    Q = [v_right v_left];
    V = (beta*Q) + (beta_c*CK);
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
    %% choice kernel update
    if stats.c(k)==1    % if chosen right
        CK = (1-alpha_c) * CK;
        CK(1) = CK(1)+ (alpha_c *1);
    elseif stats.c(k)==-1
        CK = (1-alpha_c) * CK;
        CK(2) = CK(2)+ (alpha_c *1); % left
    else % for miss - no action, no update
        CK = CK;
    end
    
    
    %% update value for the performed action
    if stats.c(k)==1       %chose right
        v_right= v_right+alpha*(stats.r(k)-v_right);
        v_left =(1-lambda)*v_left;
    elseif stats.c(k)==-1  %chose left
        v_left= v_left+alpha*(stats.r(k)-v_left);
        v_right =(1-lambda)*v_right;
    end
end

negloglike = - sum(log(ChoiceProb(~isnan(ChoiceProb))));
varargout{1} = negloglike;
end
