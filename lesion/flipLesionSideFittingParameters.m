function flipLesionSideFittingParameters(sim_path)
% flipLesionSideFittingParameters %%
% Only works for belief_CK_bilateral
model_type = 'belief_CK_varyingH_delta_bilateral';%'belief_CK_bilateral';
leftparamsIndex = [2 1 4 3 5 6];
leftLesionAnimals = [{'1806'}; {'1808'}; {'18102'}; {'18107'};{'19107'}];

%% FLIP PRE-LESION DATA - ANIMAL BASED FITTING
load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'.mat']));
nAnimal = numel(fitpar);

for k = 1:nAnimal
    curr_animal = animal(k);
    if ismember (curr_animal, leftLesionAnimals)  % Left side lesion- switch
        temp = fitpar{k};
        temp = temp(leftparamsIndex); % flip params accordingly
        fitpar{k}= temp;
    end
end

%save behavioral .mat file
save(fullfile(fullfile(sim_path,'mat_models'), [model_type,'_flipped.mat']),...
    'animal','fitpar','bic','nlike','negloglike');

%% FLIP LESION DATA - ANIMAL BASED FITTING
load(fullfile(fullfile(sim_path,'mat_models_lesion'),[model_type,'.mat']));

nAnimal = numel(fitpar);
for k = 1:nAnimal
    curr_animal = animal(k);
    if ismember (curr_animal, leftLesionAnimals)  % Left side lesion- switch
        temp = fitpar{k};
        temp = temp(leftparamsIndex); % flip params accordingly
        fitpar{k}= temp;
    end
end

%save behavioral .mat file
save(fullfile(fullfile(sim_path,'mat_models_lesion'), [model_type,'_flipped.mat']),...
    'animal','fitpar','bic','nlike','negloglike');

%% FLIP PRE/POST-LESION DATA - SESSION BASED FITTING - ALL SESSIONS INCLUDED
load(fullfile(fullfile(sim_path,'mat_models'),[model_type,'_perSession.mat']));
nSession = numel(fitpar);

for k = 1:nSession
    curr_session = session{k};
    curr_animal = curr_session(1:4);
    if ismember (curr_animal, leftLesionAnimals)  % Left side lesion- switch
        temp = fitpar{k};
        temp = temp(leftparamsIndex); % flip params accordingly
        fitpar{k}= temp;
    end
end

%save behavioral .mat file
save(fullfile(fullfile(sim_path,'mat_models'), [model_type,'_perSession_flipped.mat']),...
    'fitpar','bic','session');


