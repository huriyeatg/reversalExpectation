function compareParameters_M2stimulation(simPath,animalList)

if ~exist(simPath.fig,'dir') % create folders
    mkdir(simPath.fig);
end
%% load parameters for pre and post fittings
dat(1).input = load(fullfile(simPath.model_mat_M2stimulationPre,'belief_CK_stimulation_bilateral.mat'));
dat(1).label = 'PreStimulation';
dat(1).color = [ 0 0 0];
dat(2).input = load(fullfile(simPath.model_mat_M2stimulationPost,'belief_CK_stimulation_bilateral.mat'));
dat(2).label = 'PostStimulation';
dat(2).color = [0 0 1];

nAnimal= numel(animalList);
nParam = numel(dat(1).input.fitpar{1});
temp = nan(nParam,nAnimal);
temp2 = nan(nParam,nAnimal);

for k = 1:nAnimal
    temp (:,k) = dat(1).input.fitpar{k}; % for pre
    temp2 (:,k) = dat(2).input.fitpar{k}; % for post
end

%% Plot preStimulation
titleSt = [{'Hazard rate'};{'\alpha_C_K'}; {'\beta Sum'}; {'\beta Ratio'}];
figure
% violin plot - default params for figure
v_color  = [0 0 0]; % will change according to lesion side
v_alpha  = 0.8;
v_edge   = [0.5 0.5 0.5];
v_box    = [0.5 0.5 0.5];
v_median = [0.5 0.5 0.5];

temp_x = [ones(nAnimal,1),ones(nAnimal,1)*2,ones(nAnimal,1)*3 ]; % same for all plots

%% For pre-simulation
for k=1:numel(titleSt)
    if k==1 % hazard rate
        tempcontrol  = temp(1,:)';
        tempipsi     = temp(5,:)';
        tempcontra   = temp(7,:)';
    elseif k==2
         tempcontrol  = temp(3,:)';
         tempipsi     = temp(6,:)';
         tempcontra   = temp(8,:)';
    elseif k==3
        tempcontrol = temp(2,:)';
        tempipsi   = temp(9,:)';
    elseif k==4
        tempcontrol = temp(4,:)'./(temp(2,:)'+temp(4,:)');
        tempipsi   = temp(10,:)'./(temp(9,:)'+temp(10,:)');
    end
        
    subplot(2,4,k); hold on
    if k<3
        violinplot([tempcontrol; tempcontra;tempipsi],temp_x,'ViolinColor',v_color,'ViolinAlpha',v_alpha,...
            'EdgeColor',v_edge,'BoxColor',v_box,'MedianColor',v_median,'ShowData',true, 'ShowMean',true);hold on
        
        set(gca,'xtick',[1:3]);
        set(gca,'xticklabel',{'cnt','contra','ipsi'});
        title( [ 'pre - ', titleSt{k}])
    else
        violinplot([tempcontrol;tempipsi], temp_x(:,1:2),'ViolinColor',v_color,'ViolinAlpha',v_alpha,...
            'EdgeColor',v_edge,'BoxColor',v_box,'MedianColor',v_median,'ShowData',true, 'ShowMean',true);hold on
        
        set(gca,'xtick',[1:3]);
        set(gca,'xticklabel',{'cnt','stimulated'});
        title( titleSt{k})
    end
end

%% For post-simulation
for k=1:numel(titleSt)
    if k==1 % hazard rate
        tempcontrol  = temp2(1,:)';
        tempipsi     = temp2(5,:)';
        tempcontra   = temp2(7,:)';
    elseif k==2
         tempcontrol  = temp2(3,:)';
         tempipsi     = temp2(6,:)';
         tempcontra   = temp2(8,:)';
    elseif k==3
        tempcontrol = temp2(2,:)';
        tempipsi   = temp2(9,:)';
    elseif k==4
        tempcontrol = temp2(4,:)'./(temp2(2,:)'+temp2(4,:)');
        tempipsi   = temp2(10,:)'./(temp2(9,:)'+temp2(10,:)');
    end
    
    subplot(2,4,k+4); hold on
    if k<3 
        violinplot([tempcontrol; tempcontra;tempipsi],temp_x,'ViolinColor',v_color,'ViolinAlpha',v_alpha,...
            'EdgeColor',v_edge,'BoxColor',v_box,'MedianColor',v_median,'ShowData',true);hold on
        
        set(gca,'xtick',[1:3]);
        set(gca,'xticklabel',{'cnt','contra','ipsi'});
        title( [ 'post - ', titleSt{k}])
    else
        violinplot([tempcontrol;tempipsi], temp_x(:,1:2),'ViolinColor',v_color,'ViolinAlpha',v_alpha,...
            'EdgeColor',v_edge,'BoxColor',v_box,'MedianColor',v_median,'ShowData',true);hold on
        
        set(gca,'xtick',[1:3]);
        set(gca,'xticklabel',{'cnt','stimulated'});
        title( [ 'post - ', titleSt{k}])
    end
end

print(gcf,'-dpng',fullfile(simPath.fig ,['violin-per-animal']));
saveas(gcf, fullfile(simPath.fig ,['violin-per-animal']), 'fig');
