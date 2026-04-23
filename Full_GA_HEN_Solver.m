% run_hen_ga.m
% Heat Exchanger Network (HEN) Synthesis using Genetic Algorithm (GA)
% Open-source MATLAB code for HEN optimization
%
% Copyright (c) 2025 [Your Name/Organization]
% Released under MIT License
% GitHub Repository: [Your GitHub URL]

clear; clc; close all;

%% ========================================================================
%  1. Basic Data & Parameter Definition
%  Includes stream data, utility parameters, economic parameters, and hyperparameters
% =========================================================================
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

% --- 1.3 GA Hyperparameters ---
ga_params = struct();
ga_params.pop_size = 60;
ga_params.max_gen = 3000;
ga_params.stagnation_limit = 200;
ga_params.pc = 0.8;
ga_params.pm = 0.1;
ga_params.mutation_rate = 0.2;
ga_params.q_threshold = 50;
ga_params.diversity_ratio = 0.1;
ga_params.init_max_try_mul = 50;
ga_params.Penalty_DT = 1e7;
ga_params.Penalty_Energy = 1e8;
ga_params.use_repair = false;

cost_params.Penalty_DT = ga_params.Penalty_DT;
cost_params.Penalty_Energy = ga_params.Penalty_Energy;

% Precompute dimensions
hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
I = height(hot_streams);
J = height(cold_streams);
cost_params.K_stages = max(I, J);
K = cost_params.K_stages;
N_genes = I * J * K;

%% ========================================================================
%  2. Population Initialization
%  Hybrid strategy: heuristic generation + LP solution + random perturbation
% =========================================================================
fprintf('Initializing population (Heuristic + LP Hybrid)...\n');
population = init_population_heuristic(ga_params.pop_size, N_genes, streams, utilities, cost_params, ga_params);

% Initial evaluation
fitness = evaluate_population(population, streams, utilities, cost_params);
[best_fitness, min_idx] = min(fitness);
best_sol_chrom = population(min_idx, :);

fprintf('Initial Best TAC: %.2f\n', best_fitness);

%% ========================================================================
%  3. GA Main Loop
%  Elite preservation, tournament selection, arithmetic crossover, intelligent mutation
% =========================================================================
stagnation_count = 0;
last_best_fitness = best_fitness;
convergence_history = zeros(ga_params.max_gen, 1);

for gen = 1:ga_params.max_gen
    new_pop = zeros(size(population));
    
    % --- 3.1 Elite Strategy ---
    [~, sort_idx] = sort(fitness);
    new_pop(1, :) = population(sort_idx(1), :);
    start_idx = 2;
    
    % --- 3.2 Selection & Crossover ---
    while start_idx <= ga_params.pop_size
        p1 = tournament_selection(population, fitness, 4);
        p2 = tournament_selection(population, fitness, 4);
        
        child1 = p1;
        child2 = p2;
        
        if rand < ga_params.pc
            alpha = 0.1 + 0.8 * rand();
            child1 = alpha * p1 + (1-alpha) * p2;
            child2 = (1-alpha) * p1 + alpha * p2;
            
            if ga_params.use_repair
                child1 = repair_structure(child1, I, J, K);
                child2 = repair_structure(child2, I, J, K);
            end
        end
        
        new_pop(start_idx, :) = child1;
        if start_idx + 1 <= ga_params.pop_size
            new_pop(start_idx + 1, :) = child2;
        end
        start_idx = start_idx + 2;
    end
    
    % --- 3.3 Mutation ---
    for i = 2:ga_params.pop_size
        if rand < ga_params.pm
            original_chrom = new_pop(i, :);
            mutated_chrom = smart_mutation(original_chrom, streams, cost_params, ga_params.mutation_rate, ga_params.q_threshold);
            
            if ga_params.use_repair
                new_pop(i, :) = repair_structure(mutated_chrom, I, J, K);
            else
                new_pop(i, :) = mutated_chrom;
            end
        end
    end
    
    % --- 3.4 Update & Evaluation ---
    population = new_pop;
    fitness = evaluate_population(population, streams, utilities, cost_params);
    
    [current_min, current_idx] = min(fitness);
    if current_min < best_fitness
        best_fitness = current_min;
        best_sol_chrom = population(current_idx, :);
    end

    % Record convergence history
    convergence_history(gen) = best_fitness;
    
    % --- 3.5 Early Stopping ---
    if abs(best_fitness - last_best_fitness) < 1e-4
        stagnation_count = stagnation_count + 1;
    else
        stagnation_count = 0;
        last_best_fitness = best_fitness;
    end
    
    if mod(gen, 10) == 0
        fprintf('Gen %3d | Best TAC: %.2f\n', gen, best_fitness);
    end
    
    if stagnation_count >= ga_params.stagnation_limit
        fprintf('>>> Converged: No improvement for %d generations. Stopping early.\n', ga_params.stagnation_limit);
        convergence_history = convergence_history(1:gen);
        break;
    end
end

%% ========================================================================
%  4. Results Output & Plotting
% =========================================================================
fprintf('Optimization finished. Global Best TAC: %.2f\n', best_fitness);
final_results = reconstruct_final_results(best_sol_chrom, streams, utilities, cost_params);

% Export convergence curve
[filename, pathname] = uiputfile('Convergence_Curve.txt', 'Save Convergence Curve');
if ~isequal(filename, 0)
    fullpath = fullfile(pathname, filename);
    try
        convergence_col = convergence_history(:);
        gen_array = (1:length(convergence_col))';
        export_data = [gen_array, convergence_col];
        
        fid = fopen(fullpath, 'w');
        if fid ~= -1
            fprintf(fid, 'Generation\tTAC\n');
            fprintf(fid, '%d\t%.4f\n', export_data');
            fclose(fid);
            fprintf('>>> Convergence curve saved to: %s\n', fullpath);
        end
    catch ME
        warning('Error exporting data:');
    end
end

fprintf('Generating plot...\n');
plot_hen_grid_GA(final_results, streams);

%% ========================================================================
%  Subfunctions
% =========================================================================

%% --- Structure Repair Function ---
function chrom = repair_structure(chrom, I, J, K)
    for i = 1:I
        for j = 1:J
            base_idx = (i-1)*J*K + (j-1)*K;
            k_indices = base_idx + (1:K);
            q_values = chrom(k_indices);
            
            if sum(q_values > 1e-3) > 1
                [~, max_pos] = max(q_values);
                chrom(k_indices) = 0;
                chrom(k_indices(max_pos)) = q_values(max_pos);
            end
        end
    end
end

%% --- Population Initialization ---
function pop = init_population_heuristic(pop_size, N_genes, streams, utilities, params, ga_opts)
    pop = zeros(pop_size, N_genes);
    hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
    cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
    [I, J, K] = deal(height(hot_streams), height(cold_streams), params.K_stages);
    
    P_elite = 0.9; P_random = 0.1;
    valid_count = 0; try_count = 0;
    max_tries = pop_size * ga_opts.init_max_try_mul;
    
    while valid_count < pop_size && try_count < max_tries
        try_count = try_count + 1;
        Z_temp = zeros(I, J, K);
        
        for i = 1:I
            for j = 1:J
                is_temp = hot_streams.Tin(i) > cold_streams.Tout(j) + params.delta_T_min;
                fcp_diff = abs(hot_streams.F(i) - cold_streams.F(j)) / max(hot_streams.F(i), cold_streams.F(j));
                is_match = false;
                
                if is_temp && (fcp_diff < 0.6)
                    if rand() < P_elite, is_match = true; end
                else
                    if rand() < P_random, is_match = true; end
                end
                
                if is_match
                    Z_temp(i, j, randi(K)) = 1;
                end
            end
        end
        if nnz(Z_temp) == 0, Z_temp(randi(I), randi(J), randi(K)) = 1; end
        
        res = LP_GA(streams, utilities, params, Z_temp);
        
        if ~isempty(res) && ~isempty(res.process_hex)
            chrom = zeros(1, N_genes);
            for row = 1:height(res.process_hex)
                h_idx = find(strcmp(hot_streams.ID, res.process_hex.HotStream{row}));
                c_idx = find(strcmp(cold_streams.ID, res.process_hex.ColdStream{row}));
                k_stage = res.process_hex.Stage(row);
                idx = (h_idx-1)*J*K + (c_idx-1)*K + k_stage;
                chrom(idx) = res.process_hex.HeatLoad_q(row);
            end
            
            if valid_count >= round(pop_size * ga_opts.diversity_ratio)
                chrom = chrom .* (0.3 + 0.69 * rand(1, N_genes));
            end
            
            valid_count = valid_count + 1;
            pop(valid_count, :) = chrom;
        end
    end
    
    if valid_count < pop_size
        fprintf('Warning: Only %d valid individuals generated.\n', valid_count);
    else
        fprintf('Initialization complete: %d individuals (Tried %d times)\n', pop_size, try_count);
    end
end

%% --- Tournament Selection ---
function selected_chrom = tournament_selection(pop, fitness, k)
    pop_size = size(pop, 1);
    candidates_idx = randperm(pop_size, k);
    [~, best_loc] = min(fitness(candidates_idx));
    selected_chrom = pop(candidates_idx(best_loc), :);
end

%% --- Intelligent Mutation ---
function new_chrom = smart_mutation(chrom, streams, params, rate, threshold)
    new_chrom = chrom;
    N = length(chrom);
    hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
    cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
    I = height(hot_streams); J = height(cold_streams); K = params.K_stages;
    
    indices_nonzero = find(chrom > 1e-4);
    if ~isempty(indices_nonzero) && rand < 0.8
        idx = indices_nonzero(randi(length(indices_nonzero)));
    else
        idx = randi(N);
    end
    
    tmp = idx - 1;
    k = mod(tmp, K) + 1;
    tmp = floor(tmp / K);
    j = mod(tmp, J) + 1;
    i = floor(tmp / J) + 1;
    
    current_q = chrom(idx);
    Fi = hot_streams.F(i); Fj = cold_streams.F(j);
    
    q_sum_hot = 0;
    for k_p = 1:(k-1), for j_p = 1:J, q_sum_hot = q_sum_hot + chrom((i-1)*J*K + (j_p-1)*K + k_p); end, end
    for j_p = 1:(j-1), q_sum_hot = q_sum_hot + chrom((i-1)*J*K + (j_p-1)*K + k); end
    T_h_in = hot_streams.Tin(i) - q_sum_hot / Fi;
    
    q_sum_cold = 0;
    for k_p = (k+1):K, for i_p = 1:I, q_sum_cold = q_sum_cold + chrom((i_p-1)*J*K + (j-1)*K + k_p); end, end
    for i_p = (i+1):I, q_sum_cold = q_sum_cold + chrom((i_p-1)*J*K + (j-1)*K + k); end
    T_c_in = cold_streams.Tin(j) + q_sum_cold / Fj;
    
    available_dt = T_h_in - T_c_in - params.delta_T_min;
    limit_q = 0;
    if available_dt > 0
        limit_q = available_dt * min(Fi, Fj);
    end
    
    test_base = max(current_q, 10);
    if current_q < 1e-4, test_base = max(limit_q * 0.5, 10); end
    
    direction = 0;
    if current_q > limit_q * 0.95, direction = -1;
    elseif current_q < limit_q * 0.5, direction = 1; end
    
    if direction == 1, delta = rand * rate * test_base;
    elseif direction == -1, delta = -rand * rate * test_base;
    else, delta = (rand - 0.5) * 2 * rate * test_base; end
    
    new_q = current_q + delta;
    if new_q > limit_q, new_q = limit_q; end
    if new_q < threshold, new_q = 0; end
    
    new_chrom(idx) = new_q;
    
    idx_start_i = (i-1)*J*K + 1; idx_end_i = i*J*K;
    total_q_i = sum(new_chrom(idx_start_i : idx_end_i));
    max_q_i = hot_streams.F(i) * (hot_streams.Tin(i) - hot_streams.Tout(i));
    if total_q_i > max_q_i
        new_chrom(idx_start_i : idx_end_i) = new_chrom(idx_start_i : idx_end_i) * (max_q_i / total_q_i * 0.99);
    end
    
    idx_list_j = []; total_q_j = 0;
    for ii = 1:I, for kk = 1:K, idx_j = (ii-1)*J*K + (j-1)*K + kk; total_q_j = total_q_j + new_chrom(idx_j); idx_list_j = [idx_list_j, idx_j]; end, end
    max_q_j = cold_streams.F(j) * (cold_streams.Tout(j) - cold_streams.Tin(j));
    if total_q_j > max_q_j
        new_chrom(idx_list_j) = new_chrom(idx_list_j) * (max_q_j / total_q_j * 1.0);
    end
end

%% --- Population Evaluation ---
function fitness_scores = evaluate_population(pop, streams, utilities, params)
    [pop_size, ~] = size(pop);
    fitness_scores = zeros(pop_size, 1);
    
    hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
    cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
    I = height(hot_streams); J = height(cold_streams); K = params.K_stages;
    
    for p = 1:pop_size
        chrom = pop(p, :);
        total_capital = 0; total_utility = 0; penalty_val = 0;
        q_on_hot = zeros(I, 1); q_on_cold = zeros(J, 1);
        
        % Process HEX
        for k = 1:K, for i = 1:I, for j = 1:J
            idx = (i-1)*J*K + (j-1)*K + k;
            q = chrom(idx);
            if q > 1e-4
                Fi = hot_streams.F(i); Fj = cold_streams.F(j);
                q_sum_h = 0;
                for k_p = 1:k-1, for j_p = 1:J, q_sum_h = q_sum_h + chrom((i-1)*J*K + (j_p-1)*K + k_p); end, end
                for j_p = 1:j-1, q_sum_h = q_sum_h + chrom((i-1)*J*K + (j_p-1)*K + k); end
                T_h_in = hot_streams.Tin(i) - q_sum_h / Fi;
                
                q_sum_c = 0;
                for k_p = k+1:K, for i_p = 1:I, q_sum_c = q_sum_c + chrom((i_p-1)*J*K + (j-1)*K + k_p); end, end
                for i_p = i+1:I, q_sum_c = q_sum_c + chrom((i_p-1)*J*K + (j-1)*K + k); end
                T_c_in = cold_streams.Tin(j) + q_sum_c / Fj;
                
                available_dt = T_h_in - T_c_in - params.delta_T_min;
                limit_q = 0;
                if available_dt > 0, limit_q = available_dt * min(Fi, Fj); end
                
                effective_q = q;
                if q > limit_q + 1e-4, effective_q = max(0, limit_q); end
                
                q_on_hot(i) = q_on_hot(i) + effective_q;
                q_on_cold(j) = q_on_cold(j) + effective_q;
                
                if effective_q > 1e-3
                    T_h_out = T_h_in - effective_q / Fi;
                    T_c_out = T_c_in + effective_q / Fj;
                    d1 = T_h_in - T_c_out; d2 = T_h_out - T_c_in;
                    lmtd = 0;
                    if abs(d1-d2)<1e-4, lmtd=d1; elseif d1>0 && d2>0, lmtd=(d1-d2)/log(d1/d2); else, lmtd=1e-6; end
                    
                    if lmtd > 1e-5
                        U = 1 / (1/hot_streams.h(i) + 1/cold_streams.h(j));
                        area = effective_q / (U * lmtd);
                        cost = params.FC + params.CC * (area ^ params.B);
                        total_capital = total_capital + cost;
                    end
                end
            end, end, end
        end
        
        % Hot Utility
        for j = 1:J
            q_target = cold_streams.F(j) * (cold_streams.Tout(j) - cold_streams.Tin(j));
            q_hu = q_target - q_on_cold(j);
            if q_hu < -1e-3, penalty_val = penalty_val + abs(q_hu) * params.Penalty_Energy; q_hu = 0; end
            if q_hu > 1e-4
                total_utility = total_utility + q_hu * utilities.HU.UC;
                T_c_in = cold_streams.Tout(j) - q_hu / cold_streams.F(j);
                d1 = utilities.HU.Tin - cold_streams.Tout(j); d2 = utilities.HU.Tout - T_c_in;
                if d1 < params.delta_T_min, penalty_val = penalty_val + (params.delta_T_min - d1)*params.Penalty_DT; end
                if d2 < params.delta_T_min, penalty_val = penalty_val + (params.delta_T_min - d2)*params.Penalty_DT; end
                lmtd=1e-6; if d1>0 && d2>0, if abs(d1-d2)<1e-4, lmtd=d1; else, lmtd=(d1-d2)/log(d1/d2); end, end
                if lmtd > 1e-5
                    U = 1 / (1/cold_streams.h(j) + 1/utilities.HU.h);
                    area = q_hu / (U * lmtd);
                    total_capital = total_capital + (params.FC + params.CC * (area ^ params.B));
                end
            end
        end
        
        % Cold Utility
        for i = 1:I
            q_target = hot_streams.F(i) * (hot_streams.Tin(i) - hot_streams.Tout(i));
            q_cu = q_target - q_on_hot(i);
            if q_cu < -1e-3, penalty_val = penalty_val + abs(q_cu) * params.Penalty_Energy; q_cu = 0; end
            if q_cu > 1e-4
                total_utility = total_utility + q_cu * utilities.CU.UC;
                T_h_in = hot_streams.Tout(i) + q_cu / hot_streams.F(i);
                d1 = T_h_in - utilities.CU.Tout; d2 = hot_streams.Tout(i) - utilities.CU.Tin;
                if d1 < params.delta_T_min, penalty_val = penalty_val + (params.delta_T_min - d1)*params.Penalty_DT; end
                if d2 < params.delta_T_min, penalty_val = penalty_val + (params.delta_T_min - d2)*params.Penalty_DT; end
                lmtd=1e-6; if d1>0 && d2>0, if abs(d1-d2)<1e-4, lmtd=d1; else, lmtd=(d1-d2)/log(d1/d2); end, end
                if lmtd > 1e-5
                    U = 1 / (1/hot_streams.h(i) + 1/utilities.CU.h);
                    area = q_cu / (U * lmtd);
                    total_capital = total_capital + (params.FC + params.CC * (area ^ params.B));
                end
            end
        end
        
        fitness_scores(p) = total_capital * params.AF + total_utility + penalty_val;
    end
end

%% --- Final Results Reconstruction ---
function final_results = reconstruct_final_results(chrom, streams, utilities, params)
    final_results = struct();
    hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
    cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
    I = height(hot_streams); J = height(cold_streams); K = params.K_stages;
    
    proc_list = {};
    total_cap = 0; total_op = 0;
    q_on_hot = zeros(I, 1); q_on_cold = zeros(J, 1);
    
    % Process HEX
    for k = 1:K, for i = 1:I, for j = 1:J
        idx = (i-1)*J*K + (j-1)*K + k;
        q = chrom(idx);
        if q > 1e-4
            Fi = hot_streams.F(i); Fj = cold_streams.F(j);
            q_sum_h = 0;
            for k_p = 1:k-1, for j_p = 1:J, q_sum_h = q_sum_h + chrom((i-1)*J*K + (j_p-1)*K + k_p); end, end
            for j_p = 1:j-1, q_sum_h = q_sum_h + chrom((i-1)*J*K + (j_p-1)*K + k); end
            T_h_in = hot_streams.Tin(i) - q_sum_h / Fi;
            
            q_sum_c = 0;
            for k_p = k+1:K, for i_p = 1:I, q_sum_c = q_sum_c + chrom((i_p-1)*J*K + (j-1)*K + k_p); end, end
            for i_p = i+1:I, q_sum_c = q_sum_c + chrom((i_p-1)*J*K + (j-1)*K + k); end
            T_c_in = cold_streams.Tin(j) + q_sum_c / Fj;
            
            available_dt = T_h_in - T_c_in - params.delta_T_min;
            limit_q = 0;
            if available_dt > 0, limit_q = available_dt * min(Fi, Fj); end
            effective_q = q;
            if q > limit_q + 1e-4, effective_q = max(0, limit_q); end
            
            q_on_hot(i) = q_on_hot(i) + effective_q;
            q_on_cold(j) = q_on_cold(j) + effective_q;
            
            if effective_q > 1e-3
                T_h_out = T_h_in - effective_q / Fi;
                T_c_out = T_c_in + effective_q / Fj;
                d1 = T_h_in - T_c_out; d2 = T_h_out - T_c_in;
                lmtd = 0;
                if abs(d1-d2)<1e-4, lmtd=d1; elseif d1>0 && d2>0, lmtd=(d1-d2)/log(d1/d2); else, lmtd=1e-6; end
                
                if lmtd > 1e-5
                    U = 1 / (1/hot_streams.h(i) + 1/cold_streams.h(j));
                    area = effective_q / (U * lmtd);
                    cost = params.FC + params.CC * (area ^ params.B);
                    total_cap = total_cap + cost;
                    proc_list = [proc_list; {hot_streams.ID{i}, cold_streams.ID{j}, k, effective_q, area, lmtd, cost}];
                end
            end
        end
    end, end, end
    
    if ~isempty(proc_list)
        final_results.process_hex = cell2table(proc_list, 'VariableNames', {'HotStream', 'ColdStream', 'Stage', 'HeatLoad_q', 'Area', 'LMTD', 'Cost'});
    else
        final_results.process_hex = table();
    end
    
    % Hot Utility
    hu_list = {};
    for j = 1:J
        q_target = cold_streams.F(j) * (cold_streams.Tout(j) - cold_streams.Tin(j));
        q_hu = q_target - q_on_cold(j);
        if q_hu < -1e-3, q_hu = 0; end
        if q_hu > 1e-4
            op_cost = q_hu * utilities.HU.UC;
            total_op = total_op + op_cost;
            T_c_in = cold_streams.Tout(j) - q_hu / cold_streams.F(j);
            d1 = utilities.HU.Tin - cold_streams.Tout(j); d2 = utilities.HU.Tout - T_c_in;
            lmtd=1e-6; if d1>0 && d2>0, if abs(d1-d2)<1e-4, lmtd=d1; else, lmtd=(d1-d2)/log(d1/d2); end, end
            if lmtd > 1e-5
                U = 1 / (1/cold_streams.h(j) + 1/utilities.HU.h);
                area = q_hu / (U * lmtd);
                cap_cost = params.FC + params.CC * (area ^ params.B);
                total_cap = total_cap + cap_cost;
                hu_list = [hu_list; {cold_streams.ID{j}, q_hu, area, op_cost + cap_cost*params.AF}];
            end
        end
    end
    if ~isempty(hu_list)
        final_results.hot_utilities = cell2table(hu_list, 'VariableNames', {'ColdStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
    else
        final_results.hot_utilities = table();
    end
    
    % Cold Utility
    cu_list = {};
    for i = 1:I
        q_target = hot_streams.F(i) * (hot_streams.Tin(i) - hot_streams.Tout(i));
        q_cu = q_target - q_on_hot(i);
        if q_cu < -1e-3, q_cu = 0; end
        if q_cu > 1e-4
            op_cost = q_cu * utilities.CU.UC;
            total_op = total_op + op_cost;
            T_h_in = hot_streams.Tout(i) + q_cu / hot_streams.F(i);
            d1 = T_h_in - utilities.CU.Tout; d2 = hot_streams.Tout(i) - utilities.CU.Tin;
            lmtd=1e-6; if d1>0 && d2>0, if abs(d1-d2)<1e-4, lmtd=d1; else, lmtd=(d1-d2)/log(d1/d2); end, end
            if lmtd > 1e-5
                U = 1 / (1/hot_streams.h(i) + 1/utilities.CU.h);
                area = q_cu / (U * lmtd);
                cap_cost = params.FC + params.CC * (area ^ params.B);
                total_cap = total_cap + cap_cost;
                cu_list = [cu_list; {hot_streams.ID{i}, q_cu, area, op_cost + cap_cost*params.AF}];
            end
        end
    end
    if ~isempty(cu_list)
        final_results.cold_utilities = cell2table(cu_list, 'VariableNames', {'HotStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
    else
        final_results.cold_utilities = table();
    end
    
    final_results.capital_cost = total_cap * params.AF;
    final_results.utility_cost = total_op;
    final_results.TAC = final_results.capital_cost + final_results.utility_cost;
end