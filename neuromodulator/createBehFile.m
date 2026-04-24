% This function creates beh.mat for each session.
disp ([ num2str(sum(isnan(dataIndex.BehCreated))), ' Session to calculate. Starting...'])
k = 1;
tic
for i = 1:size(dataIndex,1)
    if isnan(dataIndex.BehCreated(i))   % if not created before
        try
            disp ([num2str(round( k / sum(isnan(dataIndex.BehCreated))*100)), '%'])
            % Load and organize the log file of the session
            [ logData ] = parseLogfileMixStructure (dataIndex.LogFilePath{i}, dataIndex.LogFileName{i});
            % Get the information you need for each trial
            [ sessionData, trialData ] = value_getSessionData( logData,dataIndex.Phase(i) );
            % filename & save
            if ~exist(dataIndex.BehPath{i},'dir')
                mkdir(dataIndex.BehPath{i});
            end
            save(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']),...
                'trialData','sessionData');
        catch
            disp([ num2str(i), ' Failed - Try again'])
        end
        k = k+1;
    end
end
toc
disp (['All session calculated without a problem'])
