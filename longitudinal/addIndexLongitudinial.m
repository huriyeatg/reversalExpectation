function newDataIndex = addIndexLongitudinial(dataIndex)
% % determineBehCriteria %
%PURPOSE:   Check each session to see if it fulfills certain performance
%           criteria to be included for subsequent analyses
%AUTHORS:   H Atilgan and S Koc 211023
%
%INPUT ARGUMENTS
%   dataIndex:  a table of the data files
%
%OUTPUT ARGUMENTS
%   newDataIndex:  a table of the data files, now including information about
%                  longitudinal
%

%% Create criteria-related database index table

nFile = size(dataIndex,1);

critIndex = table(...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1)...
    );

critIndex.Properties.VariableNames = {...
    'trialNum',...   number of responsive trials in the session
    'switchNum',...  number of switches in the session
    'motorBias',...  absolute difference in response time for left versus right
    'sessionIndex',... nth session from the start of Phase 3
    'dayIndex',... nth day from the start of Phase 3
    'reverseSessionIndex',... nth session from the end of Phase 3
    'reverseDayIndex'... nth day from the end of Phase 3
    };
%% Calculate session performance relevant to the criteria

disp(['-----------------------------------------------------------']);
disp(['--- Determine behavioral criteria for ' int2str(size(dataIndex,1)) ' behavioral logfiles.']);
disp(['-----------------------------------------------------------']);

for i = 1:size(dataIndex,1)
    if ~isnan(dataIndex.BehCreated(i))
        
        disp(['Processing file #' int2str(i) '.']);
        
        load(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']));
        
        [ trials ] = value_getTrialMasks_longitudinal( trialData );
        
        % Calculate number of responsive trials in a session
        critIndex.trialNum(i) = sum(trials.left) + sum(trials.right);
        
        % Calculate number of switches in a session
        x0=trialData.rule';
        switchTrials = find(x0(1:end-1) ~= x0(2:end)) + 1;  %trial numbers, for very first trial after a switch
        critIndex.switchNum(i) = numel(switchTrials);  %number of switches
        
        % Calculate motor bias in terms of left vs right response times
        trialType={'go','left','right'};
        edges=[-1:0.05:2];
        valLabel='Response time (s)';
        respTime_trType=get_val_byTrialType(trialData.rt,trials,trialType,edges,valLabel);
        critIndex.motorBias(i) = abs(respTime_trType.valMedian{2} - respTime_trType.valMedian{3});
        
        % Calculate the previous sessions before this session
        % Points to consider: when the beh file is not created?        
        invar.animal = dataIndex.Animal{i};
        invar.phase = dataIndex.Phase(i);
        invar.fileName = dataIndex.LogFileName{i};
        
        if i > 1        
            invar.prevAnimal = dataIndex.Animal{i-1};            
            invar.prevPhase = dataIndex.Phase(i-1);            
        end
%             [sessionInd, dayInd] = getSessionInfo_longitudinal(invar);
        
        if invar.phase ~= 3
            sessionInd = NaN;
            dayInd = NaN;
        end
        
        if invar.phase == 3 && invar.prevPhase ~= 3  
            sessionInd = 1;
            dayInd = 1;

            fnameFirst_split = split(invar.fileName, '_');
            dateFirst = cell2mat(fnameFirst_split(4));
            dateFirst = int32(str2double(dateFirst(1:6)));
            dayFirst = datenum(num2str(dateFirst, '%d'), 'yymmdd');
            
        elseif invar.phase == 3 && invar.prevPhase == 3 && ...
                ~isequal(invar.animal, invar.prevAnimal) 
            sessionInd = 1;
            dayInd = 1;

            fnameFirst_split = split(invar.fileName, '_');
            dateFirst = cell2mat(fnameFirst_split(4));
            dateFirst = int32(str2double(dateFirst(1:6)));
            dayFirst = datenum(num2str(dateFirst, '%d'), 'yymmdd');

        elseif invar.phase == 3 && invar.prevPhase == 3 && ...
                isequal(invar.animal, invar.prevAnimal)
            sessionInd = critIndex.sessionIndex(i-1) + 1;

            fnameCurr_split = split(invar.fileName, '_');
            dateCurr = cell2mat(fnameCurr_split(4));
            dateCurr = int32(str2double(dateCurr(1:6)));
            dayCurr = datenum(num2str(dateCurr, '%d'), 'yymmdd');   
            dayInd = dayCurr - dayFirst + 1;
        end
        
        critIndex.sessionIndex(i) = sessionInd;
        critIndex.dayIndex(i) = dayInd; 
        

%     else
%         error(['Error in determineBehCriteria: Behavioral .mat file was not created for file #' int2str(i)]);
    end
end

        % calculate the reverse session and day indices        
        animals = unique(dataIndex.Animal);
        
        for k = 1:length(animals)
            totalPhase3 = length(find(str2double(dataIndex.Animal) == str2double(animals(k)) & ...
                dataIndex.Phase == 3));
            firstPhase3Ind = find(str2double(dataIndex.Animal) == str2double(animals(k)) & ...
                critIndex.sessionIndex == 1);
            lastPhase3Ind = totalPhase3 + firstPhase3Ind - 1;
            
            for j = 1:totalPhase3
                critIndex.reverseSessionIndex(lastPhase3Ind-j+1) = j;
            end
            
            for d = 1:totalPhase3
                critIndex.reverseDayIndex(lastPhase3Ind-d+1) = ...
                    critIndex.dayIndex(lastPhase3Ind) - critIndex.dayIndex(lastPhase3Ind-d+1) + 1;
            end
            
        end
                        
newDataIndex = [dataIndex critIndex];