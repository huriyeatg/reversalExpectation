function [ sessionData, trialData ] = value_getSessionData_trainingStage( logData, phase )
% % value_getSessionData_trainingStage %
%PURPOSE: Retrieve basic info for Phase 1 and 2, which was used for animal
% lick shaping in bandit task training.This code is a modified/simpler
% version of value_getSessionData
%AUTHORS: H Atilgan and AC Kwan 30/10/2021
%
%INPUT ARGUMENTS
%   logdata:    Structure obtained with a call to parseLogfile().
%   phase:      The task for this logfile, so we know how to read out the
%               Code associated with the Event Type
%
%OUTPUT VARIABLES
%   sessionData:    Structure containing these fields:
%                   {subject, dateTime, nTrials, *lickTimes, *nSwitch}.
%                   * lickTimes([1 2]):=[left right] lick times.
%   trialData:      Fields:
%                   {startTimes, cueTimes, outcomeTimes, *cue, *response, *outcome}
%                   *cue, response, and outcome for each trial contains
%                   correspoinding eventCode from NBS Presentation

%% Parse the data extracted from logfile
%COPY FROM LOGDATA
sessionData.subject = logData.subject;
sessionData.dateTime = logData.dateTime;
trialData.presCodeSet = phase;

%SESSION DATA <<logData.header: 'Subject' 'Trial' 'Event Type' 'Code' 'Time'>>
% to infer what has occurred, will use both Event Type and Code
TYPE = logData.values{3};
CODE = str2double(logData.values{4});
TIME = str2double(logData.values{5});

%Is there any Event Type = Port? (this occcurs if INPUT is unchecked
%accidentally in the Presentation software during experiment), remove
%those entries
tempidx=strcmp(TYPE,'Port');
if any(tempidx)
    error('Port code exists - check this part of the code and then remove error');
    TYPE=TYPE(~tempidx);
    CODE=CODE(~tempidx);
    TIME=TIME(~tempidx);
end

% Report all the unique event codes in this log file - this is useful for diagnostic
tempidx=(strcmp(TYPE,'Nothing') | strcmp(TYPE,'Sound')); %do not consider RESPONSE or MANUAL
codeUsed = unique(CODE(tempidx));         %List the set of event codes found in this logfile

[ STIM, RESP, OUTCOME, RULE, EVENT ] = value_getPresentationCodes(trialData.presCodeSet);

% Get rid of any CODE before the first trial
ruleCodes = cell2mat(struct2cell(RULE));  %looking for the first Event, which is a Rule Event
firstCode = find(ismember(CODE,ruleCodes),1,'first');
TYPE = TYPE(firstCode:end);
CODE = CODE(firstCode:end);
TIME = TIME(firstCode:end);

% Get rid of any CODE beyond the last full trial
% FOR EARLY PHASE, GET THE LAST TRIAL STARTING CUE instead of waitcue which
% does not exist in early phase.
lastCode=find(ismember(CODE,ruleCodes),1,'last')-1;
TYPE = TYPE(1:lastCode);
CODE = CODE(1:lastCode);
TIME = TIME(1:lastCode);

% Set up the time axis, and identify lick times
t = TIME-TIME(1);           %time starts at zero
t = double(t)/10000;        %time as double in seconds
sessionData.lickTimes{1} = t(strcmp(TYPE,'Response') & CODE==RESP.LEFT);    %left licktimes
sessionData.lickTimes{2} = t(strcmp(TYPE,'Response') & CODE==RESP.RIGHT);   %right licktimes

% Stimlus trials - which occurred and when?
cueCodes = cell2mat(struct2cell(STIM)); %stimlus-associated codes as vector
trialData.cue =  CODE(strcmp(TYPE,'Sound') & ismember(CODE,cueCodes));
trialData.cueTimes = t(strcmp(TYPE,'Sound') & CODE==STIM.GO);

% Outcome trials - which occurred and when?
outcomeCodes = cell2mat(struct2cell(OUTCOME)); %outcome-associated codes as vector
trialData.outcome =  CODE((strcmp(TYPE,'Nothing') | strcmp(TYPE,'Sound')) & ismember(CODE,outcomeCodes));
trialData.outcomeTimes = t((strcmp(TYPE,'Nothing') | strcmp(TYPE,'Sound')) & ismember(CODE,outcomeCodes));

% Rule - which was specified and when?
ruleCodes = cell2mat(struct2cell(RULE)); %outcome-associated codes as vector
trialData.rule =  CODE(strcmp(TYPE,'Nothing') & ismember(CODE,ruleCodes));
trialData.ruleTimes =  t(strcmp(TYPE,'Nothing') & ismember(CODE,ruleCodes));

% How many trials?
sessionData.nTrials = min([numel(trialData.rule),numel(trialData.cue)]);
sessionData.nRules = numel(fieldnames(RULE));
sessionData.rule_labels = fieldnames(RULE);

%% ALL 'Check consistency' section was excluded as in training stage, we only
% want to check the basic of the session.

%% Responses - when?
respIdx = find(strcmp(TYPE,'Response') & (CODE==RESP.LEFT | CODE==RESP.RIGHT));   %only want lick responses, not experimenter's manual responses
respTimes = t(respIdx);

% What is the time of first lick after the cue
trialData.response = zeros(sessionData.nTrials,1,'uint32');   %Direction of the first lick
trialData.rt = nan(sessionData.nTrials,1);                    %Time of the first lick

idx = find(all([trialData.outcome~=OUTCOME.MISS trialData.outcome~=OUTCOME.REWARDMANUAL],2)); %Idx all non-miss trials
for i = 1:numel(idx)
    %Direction and time of the first lick
    temp = find(respTimes>trialData.cueTimes(idx(i)),1,'first');
    if ~isempty(temp)                          %if there were a lick (and there should be, given the outcome)
        trialData.response(idx(i)) = CODE(respIdx(temp));
        trialData.rt(idx(i)) = respTimes(temp)-trialData.cueTimes(idx(i));
    end
end

% What the lick times associated with each trial?
trialData.leftlickTimes = cell(sessionData.nTrials,1);
trialData.rightlickTimes = cell(sessionData.nTrials,1);
for i = 1:sessionData.nTrials
    %Times of all licks from prior trial to next trial
    if i>1
        time1=trialData.cueTimes(i-1);
    else
        time1=0;
    end
    
    if i<sessionData.nTrials
        time2=trialData.cueTimes(i+1);
    else
        time2=t(end);
    end
    
    temp=sessionData.lickTimes{1};
    temp=temp(temp>=time1 & temp<=time2); %save only those licks within range
    trialData.leftlickTimes{i}=temp'-trialData.cueTimes(i); %make lick time relative to go cue
    
    temp=sessionData.lickTimes{2};
    temp=temp(temp>=time1 & temp<=time2); %save only those licks within range
    trialData.rightlickTimes{i}=temp'-trialData.cueTimes(i); %make lick time relative to go cue
end

%inter-trial interval (from outcome to next cue)
if numel(trialData.cueTimes(2:end)) == numel(trialData.outcomeTimes(1:end-1))
    trialData.iti = [trialData.cueTimes(2:end)-trialData.outcomeTimes(1:end-1); NaN];
else
    trialData.iti = NaN;
end

end

