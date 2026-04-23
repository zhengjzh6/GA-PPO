% run_hen_drl.m
% Heat Exchanger Network (HEN) Synthesis using Deep Reinforcement Learning (PPO)
% Open-source MATLAB code for HEN-DRL optimization
%
% Copyright (c) 2025 [Your Name/Organization]
% Released under MIT License
% GitHub Repository: [Your GitHub URL]

clear; clc; close all;

%% 1. Data Preparation
% =========================================================
% Case 2: 8 Hot Streams + 7 Cold Streams (8H7C)
% =========================================================

% --- 1.1 Stream Data ---
stream_ids = {'H1'; 'H2'; 'H3'; 'H4'; 'H5'; 'H6'; 'H7'; 'H8'; ...
              'C1'; 'C2'; 'C3'; 'C4'; 'C5'; 'C6'; 'C7'};
stream_types = {'Hot'; 'Hot'; 'Hot'; 'Hot'; 'Hot'; 'Hot'; 'Hot'; 'Hot'; ...
                'Cold'; 'Cold'; 'Cold'; 'Cold'; 'Cold'; 'Cold'; 'Cold'};
                
T_in =  [180; 280; 180; 140; 220; 180; 200; 120; ...
          40; 100;  40;  50;  50;  90; 160];
T_out = [ 75; 120;  75;  40; 120;  55;  60;  40; ...
         230; 220; 190; 190; 250; 190; 250];
F_cp =  [ 30;  60;  30;  30;  50;  35;  30; 100; ...
          20;  60;  35;  30;  60;  50;  60];
h_coeff=[  2;   1;   2;   1;   1;   2; 0.4; 0.5; ...
           1;   1;   2;   2;   2;   1;   3];

streams = table(stream_ids, stream_types, T_in, T_out, F_cp, h_coeff, ...
    'VariableNames', {'ID', 'Type', 'Tin', 'Tout', 'F', 'h'});

% --- 1.2 Utility & Economic Parameters ---
utilities = struct();
utilities.HU = struct('Tin', 325, 'Tout', 325, 'UC', 80, 'h', 1.0); 
utilities.CU = struct('Tin', 25,  'Tout',  40, 'UC', 10, 'h', 2.0);     

% Annualization factor (AF = 1 for annualized capital cost)
Y = 1; 
r = 0;  
if r == 0
    AF = 1 / Y;  
else
    AF = (r * (1 + r)^Y) / ((1 + r)^Y - 1);  
end

cost_params = struct();
cost_params.delta_T_min = 10;     % Minimum approach temperature (EMAT)
cost_params.AF = AF;              
cost_params.FC = 8000;             % Heat exchanger fixed cost coefficient
cost_params.CC = 500;              % Heat exchanger area cost coefficient
cost_params.B = 0.75;              % Heat exchanger area exponent

I = sum(strcmp(stream_types,'Hot'));
J = sum(strcmp(stream_types,'Cold'));
cost_params.K_stages = max(I, J); 
K = cost_params.K_stages;

fprintf('System Dimension: %dH x %dC, Stages K=%d\n', I, J, K);

%% 2. Environment Initialization
env = HEN_Env_Discrete(streams, utilities, cost_params);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
numObs  = obsInfo.Dimension(1);
numActClasses = numel(actInfo.Elements); 
fprintf('Observation Dimension=%d, Discrete Actions=%d\n', numObs, numActClasses);

%% 3. Neural Networks (Discrete Multi-Classification)

% --- Critic Network (State Value Estimation V) ---
criticNet = [
    featureInputLayer(numObs, 'Normalization','none','Name','state')
    fullyConnectedLayer(256, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(1,   'Name','Value')
];
critic = rlValueFunction(criticNet, obsInfo, 'ObservationInputNames',{'state'});

% --- Actor Network (Softmax Probability Output) ---
actorNet = [
    featureInputLayer(numObs, 'Normalization','none','Name','state')
    fullyConnectedLayer(256, 'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128, 'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numActClasses, 'Name','ActionLogits')
    softmaxLayer('Name','ActionProb')
];

actor = rlDiscreteCategoricalActor(actorNet, obsInfo, actInfo, ...
    'ObservationInputNames', {'state'});

%% 4. PPO Agent Configuration
% Sequential decision making: Total steps = I * K
stepsPerEpisode = I * K;

actorOpts  = rlOptimizerOptions('LearnRate', 3e-4, 'GradientThreshold', 1.0);
criticOpts = rlOptimizerOptions('LearnRate', 1e-3, 'GradientThreshold', 1.0);

% Extended horizon & batch size for long episodes & sparse rewards
agentOpts = rlPPOAgentOptions( ...
    'SampleTime',               1, ...
    'ClipFactor',               0.2, ...
    'ExperienceHorizon',        stepsPerEpisode * 20, ...  % Collect 10 full episodes
    'MiniBatchSize',            256, ...
    'NumEpoch',                 4, ...
    'EntropyLossWeight',        0.02, ...  % Exploration weight
    'DiscountFactor',           0.99, ...
    'ActorOptimizerOptions',    actorOpts, ...
    'CriticOptimizerOptions',   criticOpts);

agent = rlPPOAgent(actor, critic, agentOpts);

%% 5. Training Process
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                15000, ...
    'MaxStepsPerEpisode',         stepsPerEpisode, ... 
    'ScoreAveragingWindowLength', 100, ...
    'Plots',                      'training-progress', ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          5, ...
    'SaveAgentCriteria',          'EpisodeReward');


fprintf('Starting training...\n');
trainingStats = train(agent, env, trainOpts);

%% 6. Validation & Evaluation
fprintf('\n=== Validation ===\n');
simOpts    = rlSimulationOptions('MaxSteps', stepsPerEpisode);
experience = sim(env, agent, simOpts);

final_Z   = env.Z_Buffer;
final_res = LP_DRL(streams, utilities, cost_params, final_Z);

if ~isempty(final_res) && ~isempty(final_res.process_hex)
    fprintf('\nTAC:     $%.2f\n', final_res.TAC);
    fprintf('Capital Cost: $%.2f\n', final_res.capital_cost);
    fprintf('Utility Cost: $%.2f\n', final_res.utility_cost);
    fprintf('HEX Count:   %d\n', height(final_res.process_hex));
else
    fprintf('[Warning] Infeasible HEN structure\n');
end

%% 7. Plot HEN Grid Diagram
if ~isempty(final_res) && ~isempty(final_res.process_hex)
    plot_hen_grid_DRL(final_res, streams);
end
