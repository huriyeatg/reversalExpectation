function varargout=funWSLS(xpar,stats)
% % funWSLS %
%PURPOSE:   Function for maximum likelihood estimation, called by
%           fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       p (probability of following win-stay, lose-switch
%   dat:        data
%               dat(:,1) = choice vector
%               dat(:,2) = reward vector
%
%OUTPUT ARGUMENTS
%   negloglike:      the negative log-likelihood to be minimized
% updated: 04/30/2020
%%
if nargin == 0 % Used for optimizer testing
    varargout{1} = [0];                  % LB
    varargout{2} = [1];                  % UB
    varargout{3} = [0,3000];             % Zlim
    varargout{4} = [-30 75];             % view
    varargout{5} = [0.5];                % xmin
    varargout{6} = 'WSLS function';      % name
    varargout{7} = [0.5];                % default x0
    return;
end

p=xpar(1);

nt=size(stats.c,1);
ChoiceProb = nan(nt,1);

for k=1:nt
  
    %probability of choosing right
    if k==1  %first trial
        pright=0.5;
    elseif stats.r(k-1)>0 && stats.c(k-1)==1   %last trial = win + right
        pright=p;
    elseif stats.r(k-1)>0 && stats.c(k-1)==-1  %last trial = win + left
        pright=1-p;
    elseif stats.r(k-1)==0 && stats.c(k-1)==1  %last trial = lose + right
        pright=1-p;
    elseif stats.r(k-1)==0 && stats.c(k-1)==-1 %last trial = lose + left
        pright=p;
    else                                       %last trial is a miss
        pright=0.5;
    end
    pleft=1-pright;
        
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
    
end
negloglike = - sum(log(ChoiceProb(~isnan(ChoiceProb))));
varargout{1} = negloglike;
end
