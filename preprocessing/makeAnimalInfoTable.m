function InfoTable = makeAnimalInfoTable(dataIndex)
% % makeAnimalInfoTable %
%PURPOSE:   Make a table 
%AUTHORS:   H Atilgan 081420
%
%INPUT ARGUMENTS
%   dataIndex : outcome of "determineBehCriteria" - needs switchNum &
%               trialNum.
%
%OUTPUT ARGUMENTS
%   InfoTable:    a table of the infos
%



%% Per Animal
animalList = unique(dataIndex.Animal);
nAnimal = numel(animalList);

%% Create data index table

InfoTable = table(...
    cell(nAnimal,1),...
    nan(nAnimal,1),...
    nan(nAnimal,1),...
    nan(nAnimal,1)...
    );

InfoTable.Properties.VariableNames = {...
    'Animal',...
    'SessionNum',...
    'SwitchNum',...
    'TrialNum'...
    };

%%  Get info about each animal
for b = 1:nAnimal
    
   currAnimalSessions = ismember(dataIndex.Animal,animalList(b));
    % Animal ID
    ind = find(currAnimalSessions==1);
    InfoTable.Animal(b) = dataIndex.Animal(ind(1));
    
    % Total Session
    InfoTable.SessionNum(b) = sum(currAnimalSessions);
    
    % Total SwitchNum
    InfoTable.SwitchNum(b) = sum(dataIndex.switchNum(currAnimalSessions));
    
    % Total Trial
    InfoTable.TrialNum(b) = sum(dataIndex.trialNum(currAnimalSessions));
end

