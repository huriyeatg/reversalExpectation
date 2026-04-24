function newDataIndex = creatDffMatFiles_miniscope(dataIndex)
% %%creatDffMatFiles_miniscope(dataIndex)
%PURPOSE:   Analyze each logfile specified in dataIndex and save the
%           results in a neural dff .mat file
%AUTHORS:   H Atilgan and AC Kwan 200210
%
%INPUT ARGUMENTS
%   dataIndex:  a table of the data files
%
%OUTPUT ARGUMENTS
%   newDataIndex:  a table of the data files, now with the DffFileCreated =
%                  true for those files that are processed
%

%% Create logfile-info related table

nFile = size(dataIndex,1);

dffIndex = table(...
    NaN(nFile,1)...
    );

dffIndex.Properties.VariableNames = {...
    'dffCreated'...  Has the behavioral .mat file been created
    };

%%
fs = 20;
pre = 2; % 1 sec
for b =1:nFile
    disp(['Creating file #' int2str(b) '.']);
    disp(['    ' dataIndex.LogFileName{b}]);
    % Is the analysis file already created?
    fn = dir(fullfile(dataIndex.BehPath{b},[dataIndex.LogFileName{b}(1:end-4),'_dff.mat']));
    if size(fn,1)>0
        dffIndex.dffCreated(b) = 1;
    end
    
    if isnan(dffIndex.dffCreated(b))&& ~isnan(dataIndex.cellCreated(b)) && ~isnan(dataIndex.timeEventsCreated(b))
        try
            save_fig = ['E:\MATLAB\two-lickport-projects\figs\neuromodulator\',dataIndex.LogFileName{b}(1:end-4)];
            clear data
           
            % load beh file for trial timing
            load(fullfile(dataIndex.BehPath{b}, [dataIndex.LogFileName{b}(1:end-4),'_beh.mat']))
            % Get trial information
            trials = value_getTrialMasks(trialData);
            stats = value_getTrialStats(trials,sessionData.nRules);
            stats = value_getTrialStatsMore(stats);
            plot_session_task(stats,numel(stats.c),dataIndex.LogFileName{b}(1:end-4))
            print(gcf,'-dpng',[save_fig,'_session']);    %png format
            saveas(gcf, [save_fig,'_session'], 'fig');
            
            % Load miniscope cell extracted data
            fname = fullfile(dataIndex.neuraldataPath{b},dataIndex.cellFilename{b});
            temp = csvread(fname,2,0); % First two row is header - excluded.
            data.raw_signal = temp(:,2);
            data.signal_t   = temp(:,1);
            % add shutter time delay
            data.signal_t   = temp(:,1)+((1:numel(temp(:,1)))*0.00001)';
            
            % Load miniscope TDT pulse aligment data
            fname = fullfile(dataIndex.neuraldataPath{b},dataIndex.timeEventsFilename{b});
            temp = readtable(fname); % First two row is header - excluded.
            ind  = find(strcmp(temp.ChannelName,'IO1')); % Make a list of all datasources.
            datatime = [temp.Time_s_(ind),temp.Value(ind)];
            % Get timeStamps for each trial
            x0 = datatime(:,2)';
            trialStamps =datatime(find(x0(1:end-1) ~= x0(2:end))',1);
            trialStamps = trialStamps(2:2:end);
            data.tiffTimeStamps = trialStamps;
            
            % Check the pulse time recorded in miniscope vs pulse time sent
            % by behaviour computer/NBS
            check_imageTriggerTimes ( trialStamps,trialData.cueTimes, 'checkTimeStamps' );
            print(gcf,'-dpng',[save_fig,'_check-triggertiming']);    %png format
            saveas(gcf, [save_fig,'_check-triggertiming'], 'fig');
            
            % Clean flurosence signal for bleaching
            figure
            
            % Method1: try exp2 fitting
            %             subplot(3,1,1)
            %             ind_nans = ~isnan(data.raw_signal);
            %             expfit = fit(data.signal_t(ind_nans),data.raw_signal(ind_nans),'exp2');
            %             plot(expfit,data.signal_t(ind_nans),data.raw_signal(ind_nans))
            %             ylabel('Raw fluorescence and exponential fit')
            %             data.f   = feval(expfit,data.signal_t);
            
            % Method 2: try 2 mins average
            
            subplot(3,1,1)
            data.f   = movmean(data.raw_signal,fs*60*2); % 60 sec x 2 mins 
            plot(data.signal_t,data.raw_signal,'k'); hold on
            plot(data.signal_t,data.f,'r')
            ylabel('Raw fluorescence and exponential fit')
            
            % plot cleaned signal
            subplot(3,1,2)
            data.signal =(data.raw_signal-data.f)/nanmean(data.raw_signal);
            plot(data.signal_t,data.signal)
            ylabel('Detrended dff')

            % Exclude last trial
            for k = 1:(numel(trialStamps)-1)
                [~,stIndex]  = min(abs(data.signal_t(:,1)-data.tiffTimeStamps(k)));
                [~,endIndex] = min(abs(data.signal_t(:,1)-data.tiffTimeStamps(k+1)));
                ind = (stIndex-round(pre*fs)):endIndex-1;
                if ind(1)<=0 % for first trial, if we do not have enough pre-cue signal
                    ind = stIndex:endIndex-1;
                    data.dff(k,1:(numel(ind)+round(pre*fs)))  = [nan(round(pre*fs),1);data.signal(ind,1)];
                    data.dffN(k,1:(numel(ind)+round(pre*fs))) = [nan(round(pre*fs),1);data.signal(ind,1)];
                else
                    data.dff(k,1:numel(ind))  = data.signal(ind,1);
                    data.dffN(k,1:numel(ind)) = data.signal(ind,1)-mean(data.signal(ind(1):ind(round(pre*fs)),1));
                end
            end
            
            subplot(3,1,3)
            timeWindow = -2:1/fs:3;
            imagesc(timeWindow,1:(numel(trialStamps)-1),data.dffN(:,1:numel(timeWindow)))
            ylabel ('Trials')
            xlabel ( 'Time (sec)')
            print(gcf,'-dpng',[save_fig,'_signal']);    %png format
            saveas(gcf, [save_fig,'_signal'], 'fig');
            close all
            %save behavioral .mat file
            save(fullfile(dataIndex.BehPath{b}, [dataIndex.LogFileName{b}(1:end-4),'_dff'])',...
                'data');
        catch
        end
    end
end

%% Add the logfile-extracted information into the database index

newDataIndex = [dataIndex dffIndex];

end
