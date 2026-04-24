function newDataIndex = addIndexNeuromodulator(dataIndex,signal_path)
% % addIndexLesion %
%PURPOSE:   Append to the DataIndex information about the neuromodulator
%           neural data
%AUTHORS:   H Atilgan and AC Kwan 20092020
%
%INPUT ARGUMENTS
%   dataIndex:  a table of the data files
%
%OUTPUT ARGUMENTS
%   newDataIndex:  a table of the data files, now including information about
%                  neural data files
%

%% Create lesion-related table

nFile = size(dataIndex,1);

nmIndex = table(...
    NaN(nFile,1),...
    NaN(nFile,1),...
    NaN(nFile,1),...
    cell(nFile,1),... 
    cell(nFile,1),... 
    cell(nFile,1)... 
    );

nmIndex.Properties.VariableNames = {...
    'experiment',... 1 for norepinephrine signal, 2 for acetylcholine signal
    'cellCreated',...  NaN or 1 ( 1 means, data exported)
    'timeEventsCreated',...  NaN or 1 ( 1 means, data exported)
    'cellFilename',... 
    'timeEventsFilename',...
    'neuraldataPath',...
    };

%%  Get info about each logfile
for b = 1:nFile
    blockdate = num2str(dataIndex.DateNumber(b)); % get only date, exclude time. Time is sligtly different in recording file.
    nmfileName = ['M',dataIndex.Animal{b},'_Phase',num2str(dataIndex.Phase(b)),...
        '_',blockdate(1:6)];
    nmIndex.neuraldataPath(b) = {signal_path};
    % Add experiment based on animal ID
    if strcmp(dataIndex.Animal{b}(3),'3')
        nmIndex.experiment(b) = 1;
    elseif strcmp(dataIndex.Animal{b}(3),'4')
        nmIndex.experiment(b) = 2;
    else
        error('Animal is not defined, check the data')
    end
    
    % Two files we need one with dff signal "filename_cell"; one with
    % timeStamps "filename_timeEvents"
    filepath = dir(fullfile(signal_path,[nmfileName,'*_cell.csv']));
    if numel(filepath)>0
       nmIndex.cellCreated(b) = 1;  
       nmIndex.cellFilename(b) = {filepath.name};
    end
    filepath = dir(fullfile(signal_path,[nmfileName,'*_timeEvents.csv']));
    if numel(filepath)>0
       nmIndex.timeEventsCreated(b) = 1;  
       nmIndex.timeEventsFilename(b) = {filepath.name};
    end
     
end

%% Add the lesion information into the database index

newDataIndex = [dataIndex nmIndex];

end
