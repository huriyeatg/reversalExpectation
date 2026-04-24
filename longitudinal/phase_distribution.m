function [sessionAfterTraining sessioninPhase3 ] = get_sessionInfo( animal_code, logfilepath)
% calculates the 
sessionAfterTraining =NaN;
sessioninPhase3 = NaN;

%%

animal_code = ['M', animal];
logfilepath = dataIndex.LogFilePath{i}

% read all subfolders within the specified path
AllLogFiles = dir(fullfile(logfilepath,'*.log'));

nFile = size(AllLogFiles,1);
disp(['Total number of behavioral logfiles detected = ', num2str(nFile)]);

% make an array including the log file names
for i = 1:nFile
    LogFileNames(i) = {AllLogFiles(i).name};
end
LogFileNames = LogFileNames';

% split the filenames
phaseInfo = cell(nFile, 4);
for j = 1:nFile
    fname_split = split(LogFileNames(j), '_');
    phaseInfo{j, 1} = fname_split(1);
    phaseInfo{j, 2} = fname_split(2);
    phaseInfo{j, 3} = fname_split(3);
    phaseInfo{j, 4} = fname_split(4);
end

% extract phase and date information
phases_dates = cell(nFile, 3);
sessions = [];
for d = 1:nFile
    phases_dates{d,1} = char(phaseInfo{d,2});
    if length(phases_dates{d,1}) == 6
        phases_dates{d,1} = phases_dates{d,1}(6);
    elseif length(phases_dates{d,1}) == 7
        phases_dates{d,1} = phases_dates{d,1}(6:7);
    end
    phases_dates{d,2} = char(phaseInfo{d,4});
    phases_dates{d,2} = phases_dates{d,2}(1:6);
    phases_dates{d,3} = [phases_dates{d,1}, phases_dates{d,2}];
    sessions(d,1) = str2double(phases_dates{d,3});
end

% count the sessions
days = unique(sessions);
disp(['Total number of unique days detected = ', num2str(length(days))]);
frequency = [days, histc(sessions(:),days)];

% split the phase number and date
freq = frequency(:, 2);
frequency(:, 3) = freq;
for s = 1:length(frequency)
    phaseStr = num2str(frequency(s, 1));
    if length(phaseStr) == 7
        frequency(s, 2) = str2double(phaseStr(2:7));
        frequency(s, 1) = str2double(phaseStr(1));
    elseif length(phaseStr) == 8
        frequency(s, 2) = str2double(phaseStr(3:8));
        frequency(s, 1) = str2double(phaseStr(1:2));
    end
end

% mark the lesion date for the lesioned animals
lesion = 1;
if strcmp(animal_code, 'M1806') == 1 || strcmp(animal_code, 'M1807') == 1
    lesion_date = datetime(num2str(180629), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M1808') == 1
    lesion_date = datetime(num2str(181003), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M18102') == 1 || ...
        strcmp(animal_code, 'M18103') == 1 || strcmp(animal_code, 'M18104') == 1
    lesion_date = datetime(num2str(190224), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M18106') == 1 || ...
        strcmp(animal_code, 'M18107') == 1 || strcmp(animal_code, 'M18109') == 1
    lesion_date = datetime(num2str(190214), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M19102') == 1 || strcmp(animal_code, 'M19106') == 1 || ...
        strcmp(animal_code, 'M19107') == 1 || strcmp(animal_code, 'M19109') == 1
    lesion_date = datetime(num2str(190520), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M19114') == 1
    lesion_date = datetime(num2str(190826), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
elseif strcmp(animal_code, 'M19116') == 1 || ...
        strcmp(animal_code, 'M19117') == 1 || strcmp(animal_code, 'M19118') == 1
    lesion_date = datetime(num2str(190722), 'InputFormat', 'yyMMdd',...
        'Format', 'dd/MM/yyyy');
end


% record the dates and number of sessions for each phase
phase1_dates = []; phase1_sessions = [];
phase2_dates = []; phase2_sessions = [];
phase3_dates = []; phase3_sessions = [];
phase4_dates = []; phase4_sessions = [];
phase5_dates = []; phase5_sessions = [];
phase6_dates = []; phase6_sessions = [];
phase8_dates = []; phase8_sessions = [];
phase9_dates = []; phase9_sessions = [];
phase10_dates = []; phase10_sessions = [];
phase20_dates = []; phase20_sessions = [];
phase21_dates = []; phase21_sessions = [];
phase22_dates = []; phase22_sessions = [];
phase31_dates = []; phase31_sessions = [];
phase32_dates = []; phase32_sessions = [];

for p = 1:length(frequency)
    date = num2str(frequency(p,2));
    switch frequency(p, 1) 
        case 1
            phase1_dates = [phase1_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase1_sessions = [phase1_sessions; frequency(p, 3)];
        case 2
            phase2_dates = [phase2_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase2_sessions = [phase2_sessions; frequency(p, 3)];
        case 3
            phase3_dates = [phase3_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase3_sessions = [phase3_sessions; frequency(p, 3)];    
        case 4
            phase4_dates = [phase4_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase4_sessions = [phase4_sessions; frequency(p, 3)];
        case 5
            phase5_dates = [phase5_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase5_sessions = [phase5_sessions; frequency(p, 3)];
        case 6
            phase6_dates = [phase6_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase6_sessions = [phase6_sessions; frequency(p, 3)];
        case 8
            phase8_dates = [phase8_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase8_sessions = [phase8_sessions; frequency(p, 3)];
        case 9
            phase9_dates = [phase9_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase9_sessions = [phase9_sessions; frequency(p, 3)];
        case 10
            phase10_dates = [phase10_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase10_sessions = [phase10_sessions; frequency(p, 3)];
        case 20
            phase20_dates = [phase20_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase20_sessions = [phase20_sessions; frequency(p, 3)];
        case 21
            phase21_dates = [phase21_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase21_sessions = [phase21_sessions; frequency(p, 3)];
        case 22
            phase22_dates = [phase22_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase22_sessions = [phase22_sessions; frequency(p, 3)];
        case 31
            phase31_dates = [phase31_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase31_sessions = [phase31_sessions; frequency(p, 3)];
        case 32
            phase32_dates = [phase32_dates; datetime(date, ...
                'InputFormat', 'yyMMdd', 'Format', 'dd/MM/yyyy')];
            phase32_sessions = [phase32_sessions; frequency(p, 3)];
    end
end

% plot the distribution of sessions by phase number and date
% the phases that exist for an animal need to be typed into 
% the legend function, by the order of phase no
% and 'Lesion' should also be added for the lesioned animals as the last
% legend
figure(1);
hold on;
bar(phase1_dates, phase1_sessions, 'b');
bar(phase2_dates, phase2_sessions, 'g');
bar(phase3_dates, phase3_sessions, 'r');

if isempty(phase4_dates) ~= 1
    bar(phase4_dates, phase4_sessions, 'c');
end

if isempty(phase5_dates) ~= 1
    bar(phase5_dates, phase5_sessions, 'm');
end

if isempty(phase6_dates) ~= 1
    bar(phase6_dates, phase6_sessions, 'y');
end

if isempty(phase8_dates) ~= 1
    bar(phase8_dates, phase8_sessions, 'FaceColor', [0.6350 0.0780 0.1840]);
end

if isempty(phase9_dates) ~= 1
    bar(phase9_dates, phase9_sessions, 'FaceColor', [0.8500 0.3250 0.0980]);
end

if isempty(phase10_dates) ~= 1
    bar(phase10_dates, phase10_sessions, 'FaceColor', [0.4940 0.1840 0.5560]);
end

if isempty(phase20_dates) ~= 1
    bar(phase20_dates, phase20_sessions, 'w');
end

if isempty(phase21_dates) ~= 1
    bar(phase21_dates, phase21_sessions, 'FaceColor', [0.4 0.4 0.4]);
end

if isempty(phase22_dates) ~= 1
    bar(phase22_dates, phase22_sessions, 'FaceColor', [0.8 0.8 0.8]);
end

if isempty(phase31_dates) ~= 1
    bar(phase31_dates, phase31_sessions, 'FaceColor', [0.3 0.3 0.3]);
end

if isempty(phase32_dates) ~= 1
    bar(phase32_dates, phase32_sessions, 'FaceColor', [0.7 0.7 0.7]);
end

if exist('lesion_date', 'var') == 1
    bar(lesion_date, lesion, 'k');
end

title([animal_code, ': Distribution of Phases']);
ylabel('session count');
yticks([1, 2]);
xlabel('session dates');
legend('Phase 1', 'Phase 2', 'Phase 3', 'Phase 31', 'Phase 32', ...
    'Location', 'NorthEast'); 
