% =========================================================================
% Script: plot_IQR.m
% Purpose: Generate publication-quality boxplots for TAC distributions 
%          under varying levels of stochastic disturbances.
% =========================================================================
clear; clc; close all;

% 1. Load inference data
try
    data_0  = readmatrix('Robustness_Matrix_Original_0pct.txt');
    data_5  = readmatrix('Robustness_Matrix_Light_5pct.txt');
    data_10 = readmatrix('Robustness_Matrix_Medium_10pct.txt');
    data_15 = readmatrix('Robustness_Matrix_Heavy_15pct.txt');
catch
    error('TXT data files not found. Please ensure the Monte Carlo results exist in the directory.');
end

% 2. Extract best TAC values for each sampled environment
best_TAC_0  = min(data_0(:, 2:end), [], 2);
best_TAC_5  = min(data_5(:, 2:end), [], 2);
best_TAC_10 = min(data_10(:, 2:end), [], 2);
best_TAC_15 = min(data_15(:, 2:end), [], 2);

% Remove NaNs uniformly based on the baseline mask
nan_mask = isnan(best_TAC_0);
best_TAC_0(nan_mask)  = [];
best_TAC_5(nan_mask)  = [];
best_TAC_10(nan_mask) = [];
best_TAC_15(nan_mask) = [];

% 3. Format data for boxplot
max_len = max([length(best_TAC_0), length(best_TAC_5), length(best_TAC_10), length(best_TAC_15)]);
plot_data = NaN(max_len, 4);
plot_data(1:length(best_TAC_0), 1)  = best_TAC_0;   
plot_data(1:length(best_TAC_5), 2)  = best_TAC_5;
plot_data(1:length(best_TAC_10), 3) = best_TAC_10;
plot_data(1:length(best_TAC_15), 4) = best_TAC_15;

% 4. Plot generation
figure('Position', [100, 100, 700, 500], 'Color', 'w');

positions = [1, 2, 3, 4];
labels = {'\pm0%', '\pm5%', '\pm10%', '\pm15%'};

h = boxplot(plot_data, 'Positions', positions, 'Labels', labels, ...
    'Colors', [0 0.4470 0.7410], 'Symbol', 'r+', 'Widths', 0.5);

set(h, 'LineWidth', 1.5);
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman', 'LineWidth', 1.2);
ylabel('Total Annual Cost (TAC) ($/yr)', 'FontSize', 14, 'FontName', 'Times New Roman');

grid on;
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.4);

% 5. Statistical Evaluation (Coefficient of Variation)
cv_0  = (std(best_TAC_0)  / mean(best_TAC_0))  * 100;
cv_5  = (std(best_TAC_5)  / mean(best_TAC_5))  * 100;
cv_10 = (std(best_TAC_10) / mean(best_TAC_10)) * 100;
cv_15 = (std(best_TAC_15) / mean(best_TAC_15)) * 100;

fprintf('\n=== Dispersion Metrics (For Manuscript) ===\n');
fprintf('±0%%  Disturbance -> CV: %.2f%%\n', cv_0);
fprintf('±5%%  Disturbance -> CV: %.2f%%\n', cv_5);
fprintf('±10%% Disturbance -> CV: %.2f%%\n', cv_10);
fprintf('±15%% Disturbance -> CV: %.2f%%\n', cv_15);