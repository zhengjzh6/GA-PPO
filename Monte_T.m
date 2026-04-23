
clc; 
clearvars -except agent streams utilities cost_params; 
close all;

fprintf('=== Begin Disturbance ===\n');

disturbance_levels = [0, 0.05, 0.10, 0.15];
level_names = {'Original_0pct', 'Light_5pct', 'Medium_10pct', 'Heavy_15pct'};
num_envs = 100; 
num_infers = 20;

I = sum(strcmp(streams.Type, 'Hot'));
K = cost_params.K_stages;
stepsPerEpisode = I * K;

simOpts = rlSimulationOptions('MaxSteps', stepsPerEpisode, 'UseParallel', false);

for lvl_idx = 1:length(disturbance_levels)
    current_level = disturbance_levels(lvl_idx);
    fprintf('\n>>> Disturbance Level: %s (±%d%%)\n', level_names{lvl_idx}, current_level*100);
    
    tac_matrix = NaN(num_envs, num_infers);
    
    for env_idx = 1:num_envs
        streams_pert = streams;
        num_perturb = randi([3, 4]); 
        pert_idx = randperm(height(streams), num_perturb);
        
        for idx = pert_idx
            noise = current_level * randn(); 
               noise = max(min(noise, current_level*3), -current_level*3); 
            streams_pert.Tin(idx) = streams_pert.Tin(idx) * (1 + noise);
        end
        
        env_pert = HEN_Env_Discrete(streams_pert, utilities, cost_params);
        
        fprintf(' %3d/%d [SIMULATING]...', env_idx, num_envs);
        
        for inf_idx = 1:num_infers
            evalc('experience = sim(env_pert, agent, simOpts);'); 
            
            final_Z = env_pert.Z_Buffer;
            final_res = LP_DRL(streams_pert, utilities, cost_params, final_Z);
            
            if ~isempty(final_res) && ~isempty(final_res.process_hex)
                tac_matrix(env_idx, inf_idx) = final_res.TAC;
            end
        end
        
        best_tac_this_env = min(tac_matrix(env_idx, :));
        fprintf(' Finish! (Best TAC: %.2f)\n', best_tac_this_env);
    end
    
    filename = sprintf('Robustness_Matrix_%s.txt', level_names{lvl_idx});
    fid = fopen(filename, 'w');
    if fid ~= -1
        fprintf(fid, 'Env_ID');
        for k = 1:num_infers
            fprintf(fid, '\tInfer_%d', k);
        end
        fprintf(fid, '\n');
        
        for env_idx = 1:num_envs
            fprintf(fid, '%d', env_idx);
            for inf_idx = 1:num_infers
                if isnan(tac_matrix(env_idx, inf_idx))
                    fprintf(fid, '\tNaN');
                else
                    fprintf(fid, '\t%.2f', tac_matrix(env_idx, inf_idx));
                end
            end
            fprintf(fid, '\n');
        end
        fclose(fid);
        fprintf('%s %s\n', level_names{lvl_idx}, filename);
    else
        warning('Failed: %s', filename);
    end
end
