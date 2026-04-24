function [ logData ] = parseLogfileMixStructure( data_dir, logfile )
% % modified version for parseLogfile created on 161209 by AK
% Can extract numeric as well as string for each column data for each line.
%
%INPUT ARGUMENTS
%   data_dir:    Path for logfile.   
%   logfile:     Filename for logfile.
%
%OUTPUT VARIABLES
%   logData:     Structure containing these fields:
%                {subject, dateTime, header, values}.
% HA, 31 Jan ,2019

fID=fopen(fullfile(data_dir,logfile));  %use fullfile to avoid backslash/slash difference in mac/pc

header{1} = textscan(fID,'%s %*[^\n]',1);
header{2} = textscan(fID,'%s %s %c %s %s',1);
header{3} = textscan(fID,'%s',5,'delimiter','\t');  %Presentation column labels 
header{4} = textscan(fID,'%s %s %s %s %s %s %s',1);
while ~strcmp(header{4}{3},'Nothing')                
    header{4} = textscan(fID,'%s %s %s %*[^\n]',1);
end

% Create a large matrix
temp = cell(1,5); % 5 column: %'Subject' 'Trial' 'Event Type' 'Code' 'Time'
k =1;
tline =  fgetl(fID);                     % Empty line before the data 
tline =  fgetl(fID);                     % First data line
while (~isempty(tline) ) 
    tline = strsplit(tline);
    temp{k,1} = str2double (tline{1});
    temp{k,2} = str2double (tline{2});
    temp{k,3} = tline(3);
    temp{k,4} = tline(4);
    temp{k,5} = str2double (tline{5});
    k = k+1;
    tline = fgetl(fID);
    if tline ==-1; tline =[]; end
end
fclose(fID);

logData.subject = header{4}{1};
logData.dateTime = [header{2}{4:5}];
logData.header = header{3}{1}(1:5)'; %'Subject' 'Trial' 'Event Type' 'Code' 'Time'
logData.values(1) = {cell2mat(temp(:,1))};
logData.values(2) = {cell2mat(temp(:,2))};
logData.values(3) = {temp(:,3)};
logData.values(4) = {temp(:,4)};
logData.values(5) = {cell2mat(temp(:,5))};

