function dataIndex = makeBlockIndexPR_NE
% This was a modified version of makeBlockIndex - specific to data for PR Behaviour.

% First version: 7/22/18 ( ROI & Total cell missing!)

%% Create block index table
tic;
columnNames ={ ...
    'Index',...
    'Animal', ...
    'State',... Lesion or Not!
    'Experiment', ...
    'Phase',...
    'BlockNum',...
    'LogFileName',...
    'LogFilePath',...
    'BehCreated',... Beh, Xbeh exist
    'CriteriaPassed',... Beh
    'BehPath',...
    'RuleSwitch',...
    'MeanRuleSwitch',...
    'HR',...
    'NEData',...
    'NEDataPath','NEDataTimeEventPath',...
    'PupilData',...
    'PupilDataPath',...
    'Notes'};

path = 'A:\HuriyeAtilgan\NESensorData\LogFile'; % LogFile
AllLogfiles = dir ([path,'\1*']);

nB = size(AllLogfiles,1);

allData=table(((1:nB)'),NaN(nB,1), NaN(nB,1), cell(nB,1),zeros(nB,1),NaN(nB,1),... until BlockNum,
    cell(nB,1),cell(nB,1),NaN(nB,1),NaN(nB,1),...  LogFile, LogFilePath, BehCreated, Criteria
    cell(nB,1),... % Beh Path
    NaN(nB,1), NaN(nB,1),NaN(nB,1),... % ruleSw, meanruleSw,hr
    NaN(nB,1),cell(nB,1),cell(nB,1),...% NEData,NEDataPath,
    NaN(nB,1),cell(nB,1),... % VideoForPhase 8 (Pupil recording), PupilPath
    cell(nB,1));

allData.Properties.VariableNames = columnNames;
allData.Properties.VariableDescriptions{size(columnNames,2)} = [...
    'Index',...
    'Animal', ...
    'State',... Lesion or Not!
    'Experiment', ...
    'Phase',...
    'BlockNum',...
    'LogFileName',...
    'LogFilePath',...
    'Beh File Created',... Beh, Xbeh exist
    'CriteriaPassed',... Beh
    'Beh File Path',...
    'total Rule Switch',...
    'Mean Rule Switch',...
    'Mean Hit Rates',...
    'NEData','NEDataPath','NEDataTimeEventPath',...
    'PupilData','PupilDataPath',...
    'Notes'];

%% Specific Info about lesion dates - Manually written
disp ([' Total number of Animal Log File detected = ', num2str(nB)]);

%% For loop for each info about each block
for b = 1:nB
    [ind, ~] =regexp(AllLogfiles(b).name, '_');
    animal = str2double({AllLogfiles(b).name(1:ind(1)-1)});
    exp    = AllLogfiles(b).name(ind(2)+1:ind(3)-1);
    phase = AllLogfiles(b).name(ind(1)+1:ind(2)-1);
    blockNum = round(str2double(AllLogfiles(b).name(ind(end)+1:end-8)), 6);
    
    % Info about Logfile
    allData.LogFilePath(b) = {AllLogfiles(b).folder}; 
    allData.LogFileName(b) = {AllLogfiles(b).name};
    
    % Info about animal & session
    allData.Animal(b)      = animal;
    if (size(ind,2)>3 && AllLogfiles(b).name(ind(2)-1)=='3') || strcmp(exp, 'R71NoCue')
        exp ='ReversalBandit';end
    allData.Experiment(b)  = {exp};
    allData.Phase(b)       = str2double(phase(6:end));% AllLogfiles(b).name(ind(1)+6)
    allData.BlockNum(b)    = str2double({AllLogfiles(b).name(ind(end)+1:end-4)});
    
    % Check if there is a No LESION - all nans.

    % Check Beh File
    fn =dir(['A:\HuriyeAtilgan\NESensorData\BehFile\',...
        allData.LogFileName{b}(1:end-4),'_beh*']);
    allData.BehPath(b) = {'A:\HuriyeAtilgan\NESensorData\BehFile'};
    if size(fn,1)>0;  allData.BehCreated(b) = 1; end
    
    % Check NE data
    fn = dir(['A:\HuriyeAtilgan\NESensorData\Data\exported\M',...
        num2str(animal),'_Phase31_',num2str(blockNum),'*']);
    if size(fn,1)>0;  allData.NEData(b) = 1;
    allData.NEDataPath(b) = {fn(2).name};
    allData.NEDataTimeEventPath(b) = {fn(3).name};
    end
    
    % CheckPupil
     fn = dir(['A:\HuriyeAtilgan\Pupillometry\Data\extracted\M',...
        num2str(animal),'_phase8_R71NoCue_WithPupil_',num2str(blockNum),'*']);
    if size(fn,1)>0;  allData.PupilData(b) = 1; end
   
 

end
dataIndex = sortrows(allData,'Phase','ascend');

%% Add HitRates

for i = 1:size(dataIndex,1)
    if ~isnan(dataIndex.BehCreated(i))   % if not created before
        try
            load(fullfile(dataIndex.BehPath{i},[dataIndex.LogFileName{i}(1:end-4),'_beh.mat']));
            % Calculate #ruleSwitch
            dataIndex.RuleSwitch(i) = numel(sessionData.ruleSwitches);
            dataIndex.MeanRuleSwitch(i) = mean(diff(sessionData.ruleSwitches));
            
            % Calculate %HR
            if dataIndex.Phase(i) ==31 % rule41;response1 & rule42;response2
            dataIndex.HR(i) = sum(numel(find(trialData.rule==41 & trialData.response==2)) +...
                               numel(find(trialData.rule==42 & trialData.response==3)) ) /...
                               (sessionData.nTrials - numel(find(trialData.outcome==8)));
            end
            clear sessionData trialData
            
        catch
            disp([ 'Beh is not created for ind = ',num2str(i)])
        end
    end
end

toc;