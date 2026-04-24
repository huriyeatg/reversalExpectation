function CPP = changepointprobability (input, hazardrate)
% changepointprobability %%
% This function was adapted from MR Nassar code (getOptimalLRs.m)to get
% optimal inferences over the probability of a binary variable (input) that
% evolves according to a change-point process with a prior over
% changepoints (hazardrate).
%
% Ref paper: Wilson, R. C., Nassar, M. R., & Gold, J. I. (2013). A
% Delta-rule approximation to Bayesian inference in change-point problems.
% PLoS Computational Biology, 9(7), e1003150

%AUTHORS:   H Atilgan 200417
%
%INPUT ARGUMENTS
%    input       : binary outcomes
%    hazardrate  : hazard rate

%OUTPUT ARGUMENTS
%   CPP: change-point probability
%%

% Check data vector - excludes nans (inlude them at the end):
data = input(~isnan(input));

% Create an grid for possible probability:
ps=(0:.01:1)';
cpPrior=ones(size(ps))./length(ps);
p=cpPrior;

% Calculate log likelihood based on actions:
dataLL=nan(size(data));  % Log likelihood
for i = 1:length(data)-1
    if ~isnan(data(i))
        if data(i)
            condProb=p.*ps;
        else
            condProb=p.*(1-ps);
        end
        dataLL(i+1)=log(sum(condProb)); % changep point probability for next trial
        % change-point probability - measure of suprise of switch
        p=p.*(1-hazardrate)+cpPrior.*(hazardrate);
        p=p./nansum(p); % normalize;
        
        % account for likelihood of data:
        if data(i)
            p=p.*ps;
        else
            p=p.*(1-ps);
        end
        p=p./nansum(p);
    end
end

% Calculate change-point probability based on data likelihood and
% true hazard rate:
Q = (.5.*hazardrate)./( exp(dataLL).*(1-hazardrate));
temp_CPP = Q./(1+Q);

CPP = nan(size(input)); % put nans back.
CPP(~isnan(input)) = temp_CPP;



