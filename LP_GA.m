function results = LP_GA(stream_data, utilities_data, cost_params, Z_matrix)
% LP_GA - Linear Programming solver for HEN-GA optimization
% Solves maximum heat recovery problem with temperature feasibility constraints
% Uses sparse matrix (COO format) for high efficiency
%
% Copyright (c) 2025 [Your Name/Organization]
% Released under MIT License
% GitHub Repository: [Your GitHub URL]

%% --- 1. Data Preparation ---
hot_streams = stream_data(strcmp(stream_data.Type, 'Hot'), :);
cold_streams = stream_data(strcmp(stream_data.Type, 'Cold'), :);

delta_T_min = cost_params.delta_T_min;
[I, J, K] = size(Z_matrix); 

%% --- 2. Decision Variables ---
num_proc_hex = I * J * K;
total_vars = num_proc_hex;

% Objective: Maximize heat recovery (minimize -q)
f = -ones(total_vars, 1);

%% --- 3. Precompute Constraint Counts ---
num_Z_zero = nnz(Z_matrix == 0);  
num_hot_balance = I;               
num_cold_balance = J;              
num_Z_one = nnz(Z_matrix == 1);    
num_temp_diff = 2 * num_Z_one;     
num_HU = J;                        
num_CU = I;                        

% Total constraints
num_eq = num_Z_zero;
num_ub = num_hot_balance + num_cold_balance + num_temp_diff + num_HU + num_CU;

% Estimate non-zero elements
nnz_estimate = num_Z_zero + 2*I*J*K + num_temp_diff*K*max(I,J) + (I+J)*I*J*K;

%% --- 4. COO Format for Sparse Matrix Construction ---
max_nnz = ceil(nnz_estimate * 1.2);

% Equality constraints
eq_rows = zeros(max_nnz, 1);
eq_cols = zeros(max_nnz, 1);
eq_vals = zeros(max_nnz, 1);
eq_rhs = zeros(num_eq, 1);
eq_nnz = 0;

% Inequality constraints
ub_rows = zeros(max_nnz, 1);
ub_cols = zeros(max_nnz, 1);
ub_vals = zeros(max_nnz, 1);
ub_rhs = zeros(num_ub, 1);
ub_nnz = 0;

% Constraint indices
eq_row = 0;
ub_row = 0;

% Helper functions
    function idx = get_q_proc_idx(i, j, k)
        idx = (i-1)*J*K + (j-1)*K + k;
    end

    function add_eq_entry(row, col, val)
        eq_nnz = eq_nnz + 1;
        eq_rows(eq_nnz) = row;
        eq_cols(eq_nnz) = col;
        eq_vals(eq_nnz) = val;
    end

    function add_ub_entry(row, col, val)
        ub_nnz = ub_nnz + 1;
        ub_rows(ub_nnz) = row;
        ub_cols(ub_nnz) = col;
        ub_vals(ub_nnz) = val;
    end

%% --- 5. Constraint Construction ---

% --- 5.1 q = 0 for Z = 0 ---
for i = 1:I
    for j = 1:J
        for k = 1:K
            if Z_matrix(i,j,k) == 0
                eq_row = eq_row + 1;
                add_eq_entry(eq_row, get_q_proc_idx(i,j,k), 1);
                eq_rhs(eq_row) = 0;
            end
        end
    end
end

% --- 5.2 Hot Stream Heat Balance ---
for i = 1:I
    ub_row = ub_row + 1;
    for j = 1:J
        for k = 1:K
            add_ub_entry(ub_row, get_q_proc_idx(i, j, k), 1);
        end
    end
    ub_rhs(ub_row) = hot_streams.F(i) * (hot_streams.Tin(i) - hot_streams.Tout(i));
end

% --- 5.3 Cold Stream Heat Balance ---
for j = 1:J
    ub_row = ub_row + 1;
    for i = 1:I
        for k = 1:K
            add_ub_entry(ub_row, get_q_proc_idx(i, j, k), 1);
        end
    end
    ub_rhs(ub_row) = cold_streams.F(j) * (cold_streams.Tout(j) - cold_streams.Tin(j));
end

% --- 5.4 Temperature Difference Constraints (Z=1 only) ---
for k = 1:K
    for i = 1:I
        for j = 1:J
            if Z_matrix(i,j,k) == 1
                Fi = hot_streams.F(i);
                Fj = cold_streams.F(j);
                b_val = hot_streams.Tin(i) - cold_streams.Tin(j) - delta_T_min;
                
                % Hot side constraint
                ub_row = ub_row + 1;
                for k_p = 1:(k-1)
                    for j_p = 1:J
                        add_ub_entry(ub_row, get_q_proc_idx(i, j_p, k_p), 1/Fi);
                    end
                end
                for j_p = 1:(j-1)
                    add_ub_entry(ub_row, get_q_proc_idx(i, j_p, k), 1/Fi);
                end
                for k_p = (k+1):K
                    for i_p = 1:I
                        add_ub_entry(ub_row, get_q_proc_idx(i_p, j, k_p), 1/Fj);
                    end
                end
                for i_p = i:I
                    add_ub_entry(ub_row, get_q_proc_idx(i_p, j, k), 1/Fj);
                end
                ub_rhs(ub_row) = b_val;
                
                % Cold side constraint
                ub_row = ub_row + 1;
                for k_p = 1:(k-1)
                    for j_p = 1:J
                        add_ub_entry(ub_row, get_q_proc_idx(i, j_p, k_p), 1/Fi);
                    end
                end
                for j_p = 1:j
                    add_ub_entry(ub_row, get_q_proc_idx(i, j_p, k), 1/Fi);
                end
                for k_p = (k+1):K
                    for i_p = 1:I
                        add_ub_entry(ub_row, get_q_proc_idx(i_p, j, k_p), 1/Fj);
                    end
                end
                for i_p = (i+1):I
                    add_ub_entry(ub_row, get_q_proc_idx(i_p, j, k), 1/Fj);
                end
                ub_rhs(ub_row) = b_val;
            end
        end
    end
end

% --- 5.5 Hot Utility Constraints ---
for j = 1:J
    ub_row = ub_row + 1;
    for i=1:I
        for k=1:K
            add_ub_entry(ub_row, get_q_proc_idx(i,j,k), 1/cold_streams.F(j));
        end
    end
    ub_rhs(ub_row) = utilities_data.HU.Tout - cold_streams.Tin(j) - delta_T_min;
end

% --- 5.6 Cold Utility Constraints ---
for i = 1:I
    ub_row = ub_row + 1;
    for j=1:J
        for k=1:K
            add_ub_entry(ub_row, get_q_proc_idx(i,j,k), 1/hot_streams.F(i));
        end
    end
    ub_rhs(ub_row) = hot_streams.Tin(i) - utilities_data.CU.Tout - delta_T_min;
end

%% --- 6. Build Sparse Matrices ---
eq_rows = eq_rows(1:eq_nnz);
eq_cols = eq_cols(1:eq_nnz);
eq_vals = eq_vals(1:eq_nnz);

ub_rows = ub_rows(1:ub_nnz);
ub_cols = ub_cols(1:ub_nnz);
ub_vals = ub_vals(1:ub_nnz);

A_eq = sparse(eq_rows, eq_cols, eq_vals, num_eq, total_vars);
A_ub = sparse(ub_rows, ub_cols, ub_vals, num_ub, total_vars);

%% --- 7. Gurobi Solver ---
if ~exist('gurobi', 'file')
    error('Gurobi is not installed or not in MATLAB path!');
end

model.obj = f;
model.lb  = zeros(total_vars, 1);
model.modelsense = 'min';
model.A = [A_ub; A_eq]; 
model.rhs = [ub_rhs; eq_rhs];
model.sense = [repmat('<', num_ub, 1); repmat('=', num_eq, 1)];

params.OutputFlag = 0;
params.Method = 1;     
params.Presolve = 2;   
params.Threads = 1;

res = gurobi(model, params);

if ~strcmp(res.status, 'OPTIMAL')
    results = [];
    return;
end

q_v = res.x;

%% --- 8. Results Parsing ---
results = struct();
total_capital_cost = 0; 
total_utility_cost = 0;
results.process_hex = table(); 
results.hot_utilities = table(); 
results.cold_utilities = table();
temp_proc_hex = [];
q_values = q_v;

% Optional structure repair
if exist('structure_repair', 'file')
    q_values = structure_repair(q_v, I, J, K);
end

for k = 1:K
    for i = 1:I
        for j = 1:J
            if Z_matrix(i,j,k) == 1
                q = q_values(get_q_proc_idx(i, j, k));
                if q > 1e-4
                    Fi = hot_streams.F(i); 
                    Fj = cold_streams.F(j);

                    % Hot stream inlet temperature
                    q_sum_for_Th_in = 0;
                    for k_prime = 1:(k-1)
                        for j_prime = 1:J
                            q_sum_for_Th_in = q_sum_for_Th_in + q_values(get_q_proc_idx(i, j_prime, k_prime)); 
                        end
                    end
                    for j_prime = 1:(j-1)
                        q_sum_for_Th_in = q_sum_for_Th_in + q_values(get_q_proc_idx(i, j_prime, k)); 
                    end
                    T_h_in_ijk = hot_streams.Tin(i) - q_sum_for_Th_in / Fi;
                    T_h_out_ijk = T_h_in_ijk - q / Fi;
                    
                    % Cold stream inlet temperature
                    q_sum_for_Tc_in = 0;
                    for k_prime = (k+1):K
                        for i_prime = 1:I
                            q_sum_for_Tc_in = q_sum_for_Tc_in + q_values(get_q_proc_idx(i_prime, j, k_prime)); 
                        end
                    end
                    for i_prime = (i+1):I
                        q_sum_for_Tc_in = q_sum_for_Tc_in + q_values(get_q_proc_idx(i_prime, j, k)); 
                    end
                    T_c_in_ijk = cold_streams.Tin(j) + q_sum_for_Tc_in / Fj;
                    T_c_out_ijk = T_c_in_ijk + q / Fj;

                    % LMTD calculation
                    delta_T1 = T_h_in_ijk - T_c_out_ijk; 
                    delta_T2 = T_h_out_ijk - T_c_in_ijk;
                    if abs(delta_T1 - delta_T2) < 1e-6
                        lmtd = delta_T1; 
                    else
                        lmtd = (delta_T1 - delta_T2) / log(delta_T1 / delta_T2); 
                    end
                    if lmtd <= 0, continue; end

                    U = 1 / (1/hot_streams.h(i) + 1/cold_streams.h(j)); 
                    area = q / (U * lmtd);
                    cost = cost_params.FC + cost_params.CC * (area ^ cost_params.B);
                    total_capital_cost = total_capital_cost + cost;
                    hex_entry = {hot_streams.ID{i}, cold_streams.ID{j}, k, q, area, lmtd, cost};
                    temp_proc_hex = [temp_proc_hex; hex_entry];
                end
            end
        end
    end
end

if ~isempty(temp_proc_hex)
    results.process_hex = cell2table(temp_proc_hex, 'VariableNames', ...
        {'HotStream', 'ColdStream', 'Stage', 'HeatLoad_q', 'Area', 'LMTD', 'Cost'}); 
end

% Hot Utility
temp_hu = [];
for j = 1:J
    q_proc_total_for_cold_j = 0;
    for i = 1:I
        for k = 1:K
            q_proc_total_for_cold_j = q_proc_total_for_cold_j + q_values(get_q_proc_idx(i, j, k)); 
        end
    end
    
    q_hu_needed = cold_streams.F(j) * (cold_streams.Tout(j) - cold_streams.Tin(j)) - q_proc_total_for_cold_j;
    
    if q_hu_needed > 1e-4
        op_cost = q_hu_needed * utilities_data.HU.UC;
        total_utility_cost = total_utility_cost + op_cost;
        
        T_cold_in = cold_streams.Tout(j) - q_hu_needed / cold_streams.F(j); 
        T_cold_out = cold_streams.Tout(j);
        
        dt1 = utilities_data.HU.Tin - T_cold_out;
        dt2 = utilities_data.HU.Tout - T_cold_in;
        
        if abs(dt1 - dt2) < 1e-6
            lmtd = dt1;
        else
            lmtd = (dt1 - dt2) / log(dt1/dt2); 
        end
        
        U = 1 / (1/cold_streams.h(j) + 1/utilities_data.HU.h);
        area = q_hu_needed / (U * lmtd);
        cap_cost = cost_params.FC + cost_params.CC * (area ^ cost_params.B);
        total_capital_cost = total_capital_cost + cap_cost;
        
        hu_entry = {cold_streams.ID{j}, q_hu_needed, area, op_cost + cap_cost*cost_params.AF}; 
        temp_hu = [temp_hu; hu_entry];
    end
end

if ~isempty(temp_hu)
    results.hot_utilities = cell2table(temp_hu, 'VariableNames', ...
        {'ColdStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
end

% Cold Utility
temp_cu = [];
for i = 1:I
    q_proc_total_for_hot_i = 0;
    for j = 1:J
        for k = 1:K
            q_proc_total_for_hot_i = q_proc_total_for_hot_i + q_values(get_q_proc_idx(i, j, k)); 
        end
    end
    
    q_cu_needed = hot_streams.F(i) * (hot_streams.Tin(i) - hot_streams.Tout(i)) - q_proc_total_for_hot_i;
    
    if q_cu_needed > 1e-4
        op_cost = q_cu_needed * utilities_data.CU.UC;
        total_utility_cost = total_utility_cost + op_cost;
        
        T_hot_in = hot_streams.Tout(i) + q_cu_needed / hot_streams.F(i);
        T_hot_out = hot_streams.Tout(i);
        
        dt1 = T_hot_in - utilities_data.CU.Tout;
        dt2 = T_hot_out - utilities_data.CU.Tin;
        
        if abs(dt1 - dt2) < 1e-6
            lmtd = dt1; 
        else
            lmtd = (dt1 - dt2) / log(dt1/dt2); 
        end
        
        U = 1 / (1/hot_streams.h(i) + 1/utilities_data.CU.h);
        area = q_cu_needed / (U * lmtd);
        cap_cost = cost_params.FC + cost_params.CC * (area ^ cost_params.B);
        total_capital_cost = total_capital_cost + cap_cost;
        
        cu_entry = {hot_streams.ID{i}, q_cu_needed, area, op_cost + cap_cost*cost_params.AF};
        temp_cu = [temp_cu; cu_entry];
    end
end

if ~isempty(temp_cu)
    results.cold_utilities = cell2table(temp_cu, 'VariableNames', ...
        {'HotStream', 'HeatLoad_q', 'Area', 'TotalAnnualCost'});
end

annualized_capital_cost = total_capital_cost * cost_params.AF;
results.capital_cost = annualized_capital_cost;
results.utility_cost = total_utility_cost;
results.TAC = annualized_capital_cost + total_utility_cost;

end