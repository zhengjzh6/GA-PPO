classdef HEN_Env_Discrete < rl.env.MATLABEnvironment
    % HEN_Env_Discrete: Custom Deep Reinforcement Learning Environment for HENS
    % Formulated as a finite Markov Decision Process (MDP).
    % Utilizes Action Masking to enforce thermodynamic constraints.
    
    properties
        StreamData      table
        Utilities       struct
        CostParams      struct
        
        NumHot          double
        NumCold         double
        NumStages       double
        
        % Row indices for quick property retrieval
        HotRowIdx       double
        ColdRowIdx      double
        
        % State tracking variables
        CurrentStage    double
        CurrentHotIdx   double      
        Z_Buffer        double      
        
        % Pseudo-heat tracking for sequential matching
        PseudoHot       double
        PseudoCold      double
        MaxHot          double
        MaxCold         double
        
        BaseTAC         double = 1.7e6
        TotalEpisodes   double = 0
        FailEpisodes    double = 0
    end
    
    methods
        function this = HEN_Env_Discrete(streams, utilities, cost_params)
            I = height(streams(strcmp(streams.Type,'Hot'),:));
            J = height(streams(strcmp(streams.Type,'Cold'),:));
            K = cost_params.K_stages;
            
            % Action Space: Choose which cold stream to match (1 to J) or None (J+1)
            actionInfo = rlFiniteSetSpec(1:(J + 1));
            actionInfo.Name = 'Match_Choice';
            
            % Observation Space: Normalized progress and remaining heat
            obs_dim = 2 + I + J; 
            observationInfo = rlNumericSpec([obs_dim 1]);
            observationInfo.Name = 'Dynamic_State';
            
            this = this@rl.env.MATLABEnvironment(observationInfo, actionInfo);
            
            this.StreamData = streams;
            this.Utilities = utilities;
            this.CostParams = cost_params;
            this.NumHot = I; this.NumCold = J; this.NumStages = K;
            
            this.HotRowIdx = find(strcmp(this.StreamData.Type, 'Hot'));
            this.ColdRowIdx = find(strcmp(this.StreamData.Type, 'Cold'));
            
            this.MaxHot = this.StreamData.F(this.HotRowIdx) .* ...
                          (this.StreamData.Tin(this.HotRowIdx) - this.StreamData.Tout(this.HotRowIdx));
            this.MaxCold = this.StreamData.F(this.ColdRowIdx) .* ...
                           (this.StreamData.Tout(this.ColdRowIdx) - this.StreamData.Tin(this.ColdRowIdx));
        end
        
        function InitialObservation = reset(this)
            this.CurrentStage = 1;
            this.CurrentHotIdx = 1;
            this.Z_Buffer = zeros(this.NumHot, this.NumCold, this.NumStages);
            
            this.PseudoHot = this.MaxHot;
            this.PseudoCold = this.MaxCold;
            
            InitialObservation = this.getObservation();
        end
        
        function [NextObs, Reward, IsDone, LoggedSignals] = step(this, Action)
            LoggedSignals = [];
            Reward = 0; 
            
            i = this.CurrentHotIdx;
            j_choice = Action; 
            
            % --- 1. Action Masking (Thermodynamic Feasibility Check) ---
            if j_choice <= this.NumCold
                isValid = true;
                
                real_h = this.HotRowIdx(i);
                real_c = this.ColdRowIdx(j_choice);
                Tin_h = this.StreamData.Tin(real_h);
                Tin_c = this.StreamData.Tin(real_c);
                
                % Check A: Minimum Temperature Approach (EMAT)
                if Tin_h < Tin_c + this.CostParams.delta_T_min
                    isValid = false;
                end
                
                % Check B: Remaining Enthalpy 
                if this.PseudoHot(i) <= 1e-3 || this.PseudoCold(j_choice) <= 1e-3
                    isValid = false;
                end
                
                if isValid
                    % Match approved: Update topology and remaining heat
                    this.Z_Buffer(i, j_choice, this.CurrentStage) = 1;
                    q_match = min(this.PseudoHot(i), this.PseudoCold(j_choice));
                    this.PseudoHot(i) = this.PseudoHot(i) - q_match;
                    this.PseudoCold(j_choice) = this.PseudoCold(j_choice) - q_match;
                else
                    % Match rejected: Apply penalty for invalid topological action
                    Reward = Reward - 0.2; 
                end
            end
            
            % --- 2. State Transition ---
            this.CurrentHotIdx = this.CurrentHotIdx + 1;
            if this.CurrentHotIdx > this.NumHot
                this.CurrentHotIdx = 1;
                this.CurrentStage = this.CurrentStage + 1;
            end
            
            % --- 3. Episode Termination and Inner-Level Evaluation ---
            if this.CurrentStage > this.NumStages
                IsDone = true;
                this.TotalEpisodes = this.TotalEpisodes + 1;
                
                try
                    % Solve inner-level LP for continuous heat loads
                    lp_results = LP_DRL(this.StreamData, this.Utilities, this.CostParams, this.Z_Buffer);
                    
                    if isempty(lp_results) || lp_results.TAC > 1e8
                        RealTAC = 1e10; 
                        Reward = Reward - 2.0; % Penalty for structural deadlock
                        this.FailEpisodes = this.FailEpisodes + 1;
                    else
                        RealTAC = lp_results.TAC;
                        
                        % Exponential reward scaling based on TAC improvement
                        reward_tac = exp((this.BaseTAC - RealTAC) / this.BaseTAC) * 5.0; 
                        Reward = Reward + reward_tac;
                        
                        % Structural sparsity penalty
                        num_Z_active = sum(this.Z_Buffer, 'all');
                        num_Q_active = height(lp_results.process_hex);
                        num_empty_shells = num_Z_active - num_Q_active;
                        Reward = Reward - num_empty_shells * 0.05 - num_Q_active * 0.02;
                    end
                catch
                    RealTAC = 1e10;
                    Reward = Reward - 2.0;
                end
                
                fail_rate = (this.FailEpisodes / this.TotalEpisodes) * 100;
                fprintf('Ep: %d | TAC: %.2e | Rw: %.2f | Fail: %.1f%% | Z: %d\n', ...
                    this.TotalEpisodes, RealTAC, Reward, fail_rate, sum(this.Z_Buffer,'all'));
            else
                IsDone = false;
            end
            
            NextObs = this.getObservation();
        end
        
        function obs = getObservation(this)
            progress_k = this.CurrentStage / this.NumStages;
            progress_i = this.CurrentHotIdx / this.NumHot;
            
            hot_norm = this.PseudoHot ./ this.MaxHot;
            cold_norm = this.PseudoCold ./ this.MaxCold;
            
            obs = [
                progress_k;
                progress_i;
                hot_norm;
                cold_norm
            ];
        end
    end
end