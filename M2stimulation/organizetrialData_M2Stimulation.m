function newtrialData = organizetrialData_M2Stimulation(trialData)
% % organizetrialData_M2Stimulation for M2Stimulation data %
%
%INPUT ARGUMENTS
%   trialData:     the trialData structure
%
%OUTPUT ARGUMENTS
%   newtrialData:  trialData, with the choice direction flipped
%

%% Add stimulation info
% stimulated side
trialData.stRegion = trialData.stimulationRegion-1000;

%stimulation: yes=1; no=1;
trialData.st=nan(numel(trialData.stimulation),1);
trialData.st = trialData.stRegion;
trialData.st (trialData.stRegion==2) =1;

    
%% copy directly except for a few entries that require flipping
newtrialData = trialData;
% newtrialData = rmfield(newtrialData,'outcome');
% newtrialData = rmfield(newtrialData,'rule');
% newtrialData = rmfield(newtrialData,'response');
% newtrialData = rmfield(newtrialData,'leftlickTimes');
% newtrialData = rmfield(newtrialData,'rightlickTimes');

% get left stimulation, and swap 'left stimulation choices' as if it was
% right stimulated choices
leftStimulatedTrials = trialData.stRegion==1; 

%% for lick times, simply swap 
newtrialData.leftlickTimes(leftStimulatedTrials) = trialData.rightlickTimes(leftStimulatedTrials);
newtrialData.rightlickTimes(leftStimulatedTrials) = trialData.leftlickTimes(leftStimulatedTrials);

%% for events, need to be more careful depending on the event type

[ STIM, RESP, OUTCOME, RULE, EVENT ] = value_getPresentationCodes(trialData.presCodeSet);

% flip left and right responses
temp_resp = trialData.response;
temp_resp(trialData.response == RESP.LEFT) = RESP.RIGHT;
temp_resp(trialData.response == RESP.RIGHT) = RESP.LEFT;
newtrialData.response = temp_resp;

% flip left and right outcomes
temp_outcome = trialData.outcome;
temp_outcome(trialData.outcome == OUTCOME.REWARDLEFT) = OUTCOME.REWARDRIGHT;
temp_outcome(trialData.outcome == OUTCOME.REWARDRIGHT) = OUTCOME.REWARDLEFT;
temp_outcome(trialData.outcome == OUTCOME.NOREWARDLEFT) = OUTCOME.NOREWARDRIGHT;
temp_outcome(trialData.outcome == OUTCOME.NOREWARDRIGHT) = OUTCOME.NOREWARDLEFT;
newtrialData.outcome = temp_outcome;

% flip left and right reward probabilities (rules)
if trialData.presCodeSet == 3 || trialData.presCodeSet == 8 
    temp_rule = trialData.rule;
    temp_rule(trialData.rule == RULE.L70R10) = RULE.L10R70;
    temp_rule(trialData.rule == RULE.L10R70) = RULE.L70R10;
    newtrialData.rule = temp_rule;
else
    error('Error in fliptrialData: Currently the code does not support task beyond reversal');
end

% which block stimulated
x0=stats.rule';
trialData.blockSt = trialData.st([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);               %reward probabilities / rule associated with each block
trialData.blockStRegion = trialData.stRegion([ find(x0(1:end-1) ~= x0(2:end)) length(x0) ]);   %reward probabilities / rule associated with each block

end
