function bandit_simulateLateralization( model_type,save_path)
% %bandit_simulateLateralization( model_type,save_path) %%
if ~strcmp(model_type,'hybridFQ_RPE_CK')
    error ('Function only works for hybridFQ_RPE_CK')
end
save_path  = fullfile(save_path,'lateralSimulation');
if ~exist(save_path,'dir') % create folders
    mkdir(save_path);
end
%% Bandit task parameters
taskparams.p_pairs=[0.7 0.1; 0.1 0.7];
taskparams.rule_labels={'0.7:0.1','0.1:0.7'};
taskparams.crit_hit=10;       % switching criterion: number of times picked the high prob. side
taskparams.crit_geo=1/11;     % switching criterion: after hit, random number of trials defined by geometric dist. p=1/mean

player_sim.model_type=model_type;
n_stim =1e4 *35;
trials_back=10;
%% simulate with only one side hazard rate
dat = struct;
alphaR = [ 0.05 0.1 0.15 0.3 0.5];
for k=1:numel(alphaR)
    player_sim.params = [0.1259 alphaR(k) 0.4502 0.4502 2.2943 2.2361];
    stats = simBandit(player_sim,taskparams,n_stim);
    dat(k).sw_output  = choice_switch(stats,trials_back);
    dat(k).stats      = stats;
    dat(k).player_sim = player_sim;
    dat(k).legend = [ 'HL:0.1259; HR:' num2str(alphaR(k))];
end

plot_switch_simulationwithdifferentparameters(dat,stats.rule_labels);
%
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_hazardRonly_diffbilateralvalues',num2str(n_stim)]));
saveas(gcf, fullfile(save_path,[model_type,'_switches_hazardRonly_diffbilateralvalues',num2str(n_stim)]), 'fig');

save_name = fullfile(save_path,[model_type,'hazardRonly_diffbilateralvalues_stats.mat']);
save(save_name, 'stats', 'player_sim');
close all

%% simulate with BOTH side hazard rate
dat = struct;
alphaR = [ 0.05 0.1 0.15 0.3 0.5];
alphaL = [ 0.05 0.1 0.15 0.3 0.5];
j =1;
for kk = 1: numel(alphaL)
    for k=1:numel(alphaR)
        player_sim.params = [alphaL(k) alphaR(k) 0.4502 0.4502 2.2943 2.2361];
        stats = simBandit(player_sim,taskparams,n_stim);
        dat(j).sw_output  = choice_switch(stats,trials_back);
        dat(j).stats      = stats;
        dat(j).player_sim = player_sim;
        dat(j).legend = [ 'HL:', num2str(alphaL(k)),'; HR:' num2str(alphaR(k))];
        j = j+1;
    end
end

plot_switch_simulationwithdifferentparameters(dat,stats.rule_labels);
%
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_hazardBoth_diffbilateralvalues',num2str(n_stim)]));
saveas(gcf, fullfile(save_path,[model_type,'_switches_hazardBoth_diffbilateralvalues',num2str(n_stim)]), 'fig');

save_name = fullfile(save_path,[model_type,'hazardBoth_diffbilateralvalues_stats.mat']);
save(save_name, 'stats', 'player_sim');
close all

%% simulate with only one side choice kernel
dat = struct;
alphaR = [ 0.1 0.2 0.4 0.6 0.8];
for k=1:numel(alphaR)
    player_sim.params = [0.1259 0.1259 0.4502 alphaR(k) 2.2943 2.2361];
    stats = simBandit(player_sim,taskparams,n_stim);
    dat(k).sw_output  = choice_switch(stats,trials_back);
    dat(k).stats      = stats;
    dat(k).player_sim = player_sim;
    dat(k).legend = [ 'ckL:0.4502; ckR:' num2str(alphaR(k))];
end

plot_switch_simulationwithdifferentparameters(dat,stats.rule_labels);
%
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_ckRonly_diffbilateralvalues',num2str(n_stim)]));
saveas(gcf, fullfile(save_path,[model_type,'_switches_ckRonly_diffbilateralvalues',num2str(n_stim)]), 'fig');

save_name = fullfile(save_path,[model_type,'ckRonly_diffbilateralvalues_stats.mat']);
save(save_name, 'stats', 'player_sim');
close all

%% simulate BOTH side choice kernel
dat = struct;
alphaR = [ 0.1 0.2 0.4 0.6 0.8];
alphaL = [ 0.1 0.2 0.4 0.6 0.8];
j =1;
for kk = 1: numel(alphaL)
    for k=1:numel(alphaR)
        player_sim.params = [0.1259 0.1259 alphaL(k) alphaR(k)  2.2943 2.2361];
        stats = simBandit(player_sim,taskparams,n_stim);
        dat(j).sw_output  = choice_switch(stats,trials_back);
        dat(j).stats      = stats;
        dat(j).player_sim = player_sim;
        dat(j).legend = [ 'ckL:', num2str(alphaL(k)),'; ckR:' num2str(alphaR(k))];
        j = j+1;
    end
end
plot_switch_simulationwithdifferentparameters(dat,stats.rule_labels);
%
print(gcf,'-dsvg',fullfile(save_path,[model_type,'_switches_ckBoth_diffbilateralvalues',num2str(n_stim)]));
saveas(gcf, fullfile(save_path,[model_type,'_switches_ckBoth_diffbilateralvalues',num2str(n_stim)]), 'fig');

save_name = fullfile(save_path,[model_type,'ckBoth_diffbilateralvalues_stats.mat']);
save(save_name, 'stats', 'player_sim');
close all


%% simulate with both Hazardrate & one side choice kernel

%% simulate with one Hazardrate & both side choice kernel

%% simulate with both Hazardrate & both side choice kernel


