function plot_betaratio(model_type, model_path)
%  plot_betaratio%

%%
load(fullfile(model_path,model_type));
fitpar =cell2mat(fitpar');

switch model_type
    case {'belief_CK_bilateral'}
        beta = fitpar(:,5); % beta values
        betaCK = fitpar(:,6); % beta CK values
    case {'belief_CK'}
        beta = fitpar(:,2); % beta values
        betaCK = fitpar(:,4); % beta CK values
end

figure
subplot(2,2,1)
scatter(betaCK, beta, 100, 'dk','filled')

% yl =get(gca,'ylim'); 
% hold on
% plot(yl,yl,'k--','LineWidth',3)
% ylim( yl)
% xlim( yl)
axis('square')
ylabel( '\beta')
xlabel( '\beta_{CK}')
