function results = Ternary_Search(stream_data, utilities_data, cost_params, Z_matrix, Q_bound)
% Ternary_Search - Ternary search + LP solver for fast nonlinear TAC optimization
% Optimizes total heat recovery to minimize Total Annual Cost (TAC)
% Uses Gurobi for linear feasibility and nonlinear area cost evaluation
%
% Copyright (c) 2025 [Your Name/Organization]
% Released under MIT License
% GitHub Repository: [Your GitHub URL]

    hot_streams = stream_data(strcmp(stream_data.Type, 'Hot'), :);
    cold_streams = stream_data(strcmp(stream_data.Type, 'Cold'), :);
    delta_T_min = cost_params.delta_T_min;
    [I, J, K] = size(Z_matrix); 

    %% 1. Extract active variables from Z_matrix
    active_indices = find(Z_matrix == 1);
    num_vars = length(active_indices);
    
    if num_vars == 0
        results.TAC = inf;
        results.process_hex = [];
        return;
    end
    [active_i, active_j, active_k] = ind2sub([I, J, K], active_indices);

    Q_hot_total = hot_streams.F .* (hot_streams.Tin - hot_streams.Tout);
    Q_cold_total = cold_streams.F .* (cold_streams.Tout - cold_streams.Tin);

    %% 2. Build base linear constraint matrix
    A_ub_base = [];
    b_ub_base = [];
    
    % Hot stream heat balance constraints
    for i = 1:I
        idx = find(active_i == i);
        if ~isempty(idx)
            row = zeros(1, num_vars);
            row(idx) = 1;
            A_ub_base = [A_ub_base; row];
            b_ub_base = [b_ub_base; Q_hot_total(i)];
        end
    end

    % Cold stream heat balance constraints
    for j = 1:J
        idx = find(active_j == j);
        if ~isempty(idx)
            row = zeros(1, num_vars);
            row(idx) = 1;
            A_ub_base = [A_ub_base; row];
            b_ub_base = [b_ub_base; Q_cold_total(j)];
        end
    end
    
    % Temperature feasibility constraints
    for v = 1:num_vars
        i = active_i(v);
        j = active_j(v);
        k = active_k(v);
        Fi = hot_streams.F(i);
        Fj = cold_streams.F(j);
        
        row_hot = zeros(1, num_vars);
        row_cold = zeros(1, num_vars);
        
        for v2 = 1:num_vars
            i2 = active_i(v2);
            j2 = active_j(v2);
            k2 = active_k(v2);
            
            % Hot side constraint
            if i2 == i && ((k2 < k) || (k2 == k && j2 < j))
                row_hot(v2) = 1/Fi;
            elseif j2 == j && ((k2 > k) || (k2 == k && i2 >= i))
                row_hot(v2) = 1/Fj;
            end
            
            % Cold side constraint
            if i2 == i && ((k2 < k) || (k2 == k && j2 <= j))
                row_cold(v2) = 1/Fi;
            elseif j2 == j && ((k2 > k) || (k2 == k && i2 > i))
                row_cold(v2) = 1/Fj;
            end
        end
        
        A_ub_base = [A_ub_base; row_hot; row_cold];
        b_val = hot_streams.Tin(i) - cold_streams.Tin(j) - delta_T_min;
        b_ub_base = [b_ub_base; b_val; b_val];
    end

    % Cold utility temperature constraints
    for i = 1:I
        idx = find(active_i == i);
        if ~isempty(idx)
            row = zeros(1, num_vars);
            row(idx) = 1/hot_streams.F(i);
            A_ub_base = [A_ub_base; row];
            b_ub_base = [b_ub_base; hot_streams.Tin(i) - utilities_data.CU.Tout - delta_T_min];
        end
    end

    % Hot utility temperature constraints
    for j = 1:J
        idx = find(active_j == j);
        if ~isempty(idx)
            row = zeros(1, num_vars);
            row(idx) = 1/cold_streams.F(j);
            A_ub_base = [A_ub_base; row];
            b_ub_base = [b_ub_base; utilities_data.HU.Tout - cold_streams.Tin(j) - delta_T_min];
        end
    end

    %% 3. Gurobi parameters
    params.OutputFlag = 0;
    params.Method = 1;

    %% 4. Ternary Search for optimal heat recovery
    a = 0;
    b = Q_bound;
    sense_eq = [repmat('<', length(b_ub_base), 1); '='];
    
    % Ternary search iterations (25 iterations for high precision)
    for iter = 1:25
        m1 = a + (b - a) / 3;
        m2 = b - (b - a) / 3;
        
        [tac_m1, ~, ~, ~] = evaluate_TAC(m1);
        [tac_m2, ~, ~, ~] = evaluate_TAC(m2);
        
        if tac_m1 < tac_m2
            b = m2;
        else
            a = m1;
        end
    end
    
    % Optimal solution at midpoint of converged interval
    results.Optimal_Q = (a + b) / 2;
    [results.TAC, results.process_hex, results.hot_utilities, results.cold_utilities] = evaluate_TAC(results.Optimal_Q);

    %% ================= Nested Function: TAC Evaluation =================
    function [tac_val, p_hex, h_ut, c_ut] = evaluate_TAC(target_Q)
        A_eq_row = ones(1, num_vars);
        model.A = sparse([A_ub_base; A_eq_row]);
        model.rhs = [b_ub_base; target_Q];
        model.sense = sense_eq;
        model.obj = zeros(num_vars, 1);
        
        res = gurobi(model, params);
        
        if ~strcmp(res.status, 'OPTIMAL')
            tac_val = inf;
            p_hex = table();
            h_ut = table();
            c_ut = table();
            return;
        end
        
        q_sol = res.x;
        
        % Initialize variables
        tac_val = 0;
        q_sum_hot = zeros(I, 1);
        q_sum_cold = zeros(J, 1);
        
        temp_proc_hex = {};
        temp_hu = {};
        temp_cu = {};
        
        % Process heat exchangers
        for idx_v = 1:num_vars
            q = q_sol(idx_v);
            if q < 1e-4
                continue;
            end
            
            curr_i = active_i(idx_v);
            curr_j = active_j(idx_v);
            curr_k = active_k(idx_v);
            
            q_sum_hot(curr_i) = q_sum_hot(curr_i) + q;
            q_sum_cold(curr_j) = q_sum_cold(curr_j) + q;
            
            % Calculate temperatures
            Th_in = hot_streams.Tin(curr_i) - sum(q_sol(active_i == curr_i & ((active_k < curr_k) | (active_k == curr_k & active_j < curr_j)))) / hot_streams.F(curr_i);
            Th_out = Th_in - q / hot_streams.F(curr_i);
            Tc_in = cold_streams.Tin(curr_j) + sum(q_sol(active_j == curr_j & ((active_k > curr_k) | (active_k == curr_k & active_i > curr_i)))) / cold_streams.F(curr_j);
            Tc_out = Tc_in + q / cold_streams.F(curr_j);
            
            dt1 = Th_in - Tc_out;
            dt2 = Th_out - Tc_in;
            
            if dt1 <= 0 || dt2 <= 0
                tac_val = inf;
                p_hex = table();
                h_ut = table();
                c_ut = table();
                return;
            end
            
            % Chen's approximation for LMTD
            lmtd = (dt1 * dt2 * 0.5 * (dt1 + dt2))^(1/3);
            U = 1 / (1/hot_streams.h(curr_i) + 1/cold_streams.h(curr_j));
            Area = q / (U * lmtd);
            
            cost_cap = cost_params.FC + cost_params.CC * (Area^cost_params.B);
            tac_val = tac_val + cost_cap * cost_params.AF;
            
            temp_proc_hex = [temp_proc_hex; {hot_streams.ID{curr_i}, cold_streams.ID{curr_j}, curr_k, q, Area, lmtd, cost_cap}];
        end
        
        % Cold utilities
        for curr_i = 1:I
            q_cu = Q_hot_total(curr_i) - q_sum_hot(curr_i);
            if q_cu > 1e-4
                Th_in = hot_streams.Tout(curr_i) + q_cu / hot_streams.F(curr_i);
                dt1 = Th_in - utilities_data.CU.Tout;
                dt2 = hot_streams.Tout(curr_i) - utilities_data.CU.Tin;
                
                if dt1 <= 0 || dt2 <= 0
                    tac_val = inf;
                    p_hex = table();
                    h_ut = table();
                    c_ut = table();
                    return;
                end
                
                lmtd = (dt1 * dt2 * 0.5 * (dt1 + dt2))^(1/3);
                U = 1 / (1/hot_streams.h(curr_i) + 1/utilities_data.CU.h);
                Area = q_cu / (U * lmtd);
                
                cost_cap = cost_params.FC + cost_params.CC * (Area^cost_params.B);
                cost_op = q_cu * utilities_data.CU.UC;
                tac_cu = cost_cap * cost_params.AF + cost_op;
                tac_val = tac_val + tac_cu;
                
                temp_cu = [temp_cu; {hot_streams.ID{curr_i}, q_cu, Area, tac_cu}];
            end
        end
        
        % Hot utilities
        for curr_j = 1:J
            q_hu = Q_cold_total(curr_j) - q_sum_cold(curr_j);
            if q_hu > 1e-4
                Tc_in = cold_streams.Tout(curr_j) - q_hu / cold_streams.F(curr_j);
                dt1 = utilities_data.HU.Tin - cold_streams.Tout(curr_j);
                dt2 = utilities_data.HU.Tout - Tc_in;
                
                if dt1 <= 0 || dt2 <= 0
                    tac_val = inf;
                    p_hex = table();
                    h_ut = table();
                    c_ut = table();
                    return;
                end
                
                lmtd = (dt1 * dt2 * 0.5 * (dt1 + dt2))^(1/3);
                U = 1 / (1/cold_streams.h(curr_j) + 1/utilities_data.HU.h);
                Area = q_hu / (U * lmtd);
                
                cost_cap = cost_params.FC + cost_params.CC * (Area^cost_params.B);
                cost_op = q_hu * utilities_data.HU.UC;
                tac_hu = cost_cap * cost_params.AF + cost_op;
                tac_val = tac_val + tac_hu;
                
                temp_hu = [temp_hu; {cold_streams.ID{curr_j}, q_hu, Area, tac_hu}];
            end
        end

        % Convert to tables for output
        if ~isempty(temp_proc_hex)
            p_hex = cell2table(temp_proc_hex, 'VariableNames', {'HotStream', 'ColdStream', 'Stage', 'HeatLoad_q', 'Area', 'LMTD', 'Cost'});
        else
            p_hex = table();
        end
        
        if ~isempty(temp_hu)
            h_ut = cell2table(temp_hu, 'VariableNames', {'ColdStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
        else
            h_ut = table();
        end
        
        if ~isempty(temp_cu)
            c_ut = cell2table(temp_cu, 'VariableNames', {'HotStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
        else
            c_ut = table();
        end
    end
end