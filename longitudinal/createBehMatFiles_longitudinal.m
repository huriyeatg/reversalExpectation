function newDataIndex = createBehMatFiles_longitudinal(dataIndex)
% % createBehMatFiles %
%PURPOSE:   Analyze each logfile specified in dataIndex and save the
%           results in a behavioral .mat file
%AUTHORS:   H Atilgan and AC Kwan 191127
%
%INPUT ARGUMENTS
%   dataIndex:  a table of the data files
%
%OUTPUT ARGUMENTS
%   newDataIndex:  a table of the data files, now with the BehFileCreated =
%                  true for those files that are processed
%

%% Create logfile-info related table

nFile = size(dataIndex,1);

behIndex = table(...
    cell(nFile,1),...
    cell(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1)...
    );

behIndex.Properties.VariableNames = {...
    'Animal',...
    'Experiment',... Name of the scenario file ran by NBS Presentation
    'Phase',...      Phase3=Reversal, Phase6=6 sets of reward prob, Phase8=same as phase 3, with pupil recording
    'DateNumber',... Date/Time e.g, 1806291321 = 2018 June 29 13:21
    'BehCreated'...  Has the behavioral .mat file been created
    };

%% parse and plot the logfiles specified in dataIndex

disp(['-----------------------------------------------------------']);
disp(['--- Detected: ' int2str(nFile) ' behavioral logfiles.']);
disp(['-----------------------------------------------------------']);

for i = 1:nFile
    
    % Is the analysis file already created?
    fn = dir(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']));
    if size(fn,1)>0
        behIndex.BehCreated(i) = 1;
    end
    try
        if isnan(behIndex.BehCreated(i))
            disp(['Parsing file #' int2str(i) '.']);
            disp(['    ' dataIndex.LogFileName{i}]);
            
            % extract data for each trial from raw logfile
            [ logData ] = parseLogfile(dataIndex.LogFilePath{i}, dataIndex.LogFileName{i});
            
            logfileData.Animal = logData.subject;
            logfileData.Experiment = logData.scenario;
            yr=num2str(logData.dateTime{1}(9:10));
            mo=num2str(logData.dateTime{1}(1:2));
            day=num2str(logData.dateTime{1}(4:5));
            hr=num2str(logData.dateTime{2}(1:2));
            min=num2str(logData.dateTime{2}(4:5));
            logfileData.DateNumber=str2num([yr mo day hr min]);
            
            logfileData.Experiment
            
            if strcmp(logfileData.Experiment,'Phase3R_71_NoCue')...
                    || strcmp(logfileData.Experiment,'Phase3_R71_NoCue')...
                    || strcmp(logfileData.Experiment,'Phase3_R71NoCue')...
                    || strcmp(logfileData.Experiment,'Phase3R_71')
                logfileData.Phase=3;
            elseif strcmp(logfileData.Experiment,'Phase6_Value')
                logfileData.Phase=6;
            elseif strcmp(logfileData.Experiment,'Phase8_R71NoCueWithPupil')
                logfileData.Phase=8;
            elseif strcmp(logfileData.Experiment,'Phase21_R71NoCueOpto')
                logfileData.Phase=21;
            elseif strcmp(logfileData.Experiment,'Phase22_R71NoCueOpto')
                logfileData.Phase=22;
            elseif strcmp(logfileData.Experiment, 'phase31_R71NoCueNM')...
                    || strcmp(logfileData.Experiment,'Phase31_R71NoCueWithPupil')
                logfileData.Phase = 31;
            elseif strcmp(logfileData.Experiment, 'Phase32_R71NoCueNM')
                logfileData.Phase = 32;
            elseif contains(logfileData.Experiment, 'Phase1')
                logfileData.Phase = 1;
            elseif contains(logfileData.Experiment, 'Phase2')
                logfileData.Phase = 2;
            elseif contains(logfileData.Experiment, 'Phase4')
                logfileData.Phase = 4;
            elseif contains(logfileData.Experiment, 'Phase5')
                logfileData.Phase = 5;
            elseif contains(logfileData.Experiment, 'Phase9')
                logfileData.Phase = 9;
            elseif contains(logfileData.Experiment, 'Phase10')
                logfileData.Phase = 10;
            elseif contains(logfileData.Experiment, 'Phase20')
                logfileData.Phase = 20;
            else
                logfileData.Phase = NaN;
            end           
        
        behIndex.Animal(i) = logfileData.Animal;
        behIndex.Experiment(i) = logfileData.Experiment;
        behIndex.Phase(i) = logfileData.Phase;
        behIndex.DateNumber(i) = logfileData.DateNumber;
        
            if  ~isnan (logfileData.Phase)
                if  logfileData.Phase >2 
                    [ sessionData, trialData ] = value_getSessionData(logData,logfileData.Phase);
                else
                    [ sessionData, trialData ] = value_getSessionData_trainingStage(logData,logfileData.Phase);
                end
                %save behavioral .mat file
                save(fullfile(dataIndex.BehPath{i}, [dataIndex.LogFileName{i}(1:end-4),'_beh'])',...
                    'sessionData','trialData','logfileData');
                
                behIndex.BehCreated(i) = 1;
            end
        
        elseif ~isnan(behIndex.BehCreated(i))  %else if the behavioral .mat file already created
            disp(['Loading file #' int2str(i) '.']);
            disp(['    ' dataIndex.LogFileName{i}]);
            
            load(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']));
        end
                    
        behIndex.Animal(i) = logfileData.Animal;
        behIndex.Experiment(i) = logfileData.Experiment;
        behIndex.Phase(i) = logfileData.Phase;
        behIndex.DateNumber(i) = logfileData.DateNumber;
        
        clearvars -except i dataIndex behIndex
    catch
        warning(['#### Beh file cannot be created for ', dataIndex.LogFileName{i}(1:end-4)]);
    end
    
end

%% Add the logfile-extracted information into the database index

newDataIndex = [dataIndex behIndex];

end
