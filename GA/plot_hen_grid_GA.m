function plot_hen_grid_GA(results, streams)
% plot_hen_grid_GA - Plot HEN Grid Diagram for GA-based optimization
% Clean, full-screen, publication-quality plot
% Removes HU/CU text labels, keeps area annotations
%
% Copyright (c) 2025 [Your Name/Organization]
% Released under MIT License
% GitHub Repository: [Your GitHub URL]

if isempty(results)
    disp('Results are empty, cannot plot.');
    return;
end

% --- 0. Calculate temperature profiles locally ---
results = calculate_profiles_locally(results, streams);

if ~isfield(results, 'stream_temp_profiles')
    disp('Temperature profile calculation failed.');
    return;
end

hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
I = height(hot_streams);
J = height(cold_streams);
sm = results.process_hex; 

% --- 1. Plot parameters ---
y_spacing = 3.5;
x_spacing = 100;
font_name = 'Times New Roman';
font_size_T = 9;
font_size_ID = 13;
font_size_F = 11;
font_size_Area = 9;
marker_size = 8;
extension_len = 60;

% Text offsets
offset_area_y = 1.2;
offset_temp_y = 0.5;
offset_temp_x = 10;

% --- 2. Compress X coordinates ---
all_x_logical = [];
all_profiles = results.stream_temp_profiles;
for k = 1:length(all_profiles)
    if ~isempty(all_profiles{k})
        path = all_profiles{k};
        for n = 1:length(path)
            node = path(n);
            if contains(node.description, 'HEX') || contains(node.description, 'HU') || contains(node.description, 'CU')
                all_x_logical = [all_x_logical, node.x_pos];
            end
        end
    end
end
unique_x_logical = unique(all_x_logical);
x_map_plot = containers.Map('KeyType', 'double', 'ValueType', 'double');
for k = 1:length(unique_x_logical)
    x_map_plot(unique_x_logical(k)) = k - 1;
end
if isempty(unique_x_logical)
    x_max_draw = x_spacing;
else
    x_max_draw = (length(unique_x_logical) - 1) * x_spacing;
end

% --- 3. Create full-screen figure ---
figure('Name', 'Heat Exchanger Network Structure', 'Color', 'w', 'Units', 'normalized', 'OuterPosition', [0 0.02 1 0.99]);
hold on; ax = gca;
set(ax, 'YDir', 'reverse');
axis off;

set(gca, 'Position', [0.03 0.05 0.94 0.90]);

% Define Y positions
hot_y = (1:I) * y_spacing;
cold_y = (I + 1.2 : 1 : I + J + 0.2) * y_spacing;
stream_y_pos = [hot_y'; cold_y'];

% --- 4. Draw streams, IDs, and F values ---
for i = 1:I+J
    y = stream_y_pos(i);
    is_hot = i <= I;
    color = 'r'; if ~is_hot, color = 'b'; end
    
    plot([-extension_len, x_max_draw + extension_len], [y, y], '-', 'Color', color, 'LineWidth', 1.5);
    
    id_str = streams.ID{i};
    f_val = streams.F(i);
    
    if is_hot
        text(-extension_len - 15, y, id_str, 'Color', 'r', 'FontName', font_name, 'FontWeight', 'bold', 'FontSize', font_size_ID, 'HorizontalAlignment', 'right');
        text(x_max_draw + extension_len + 15, y, sprintf('%.1f', f_val), 'Color', 'k', 'FontName', font_name, 'FontAngle', 'italic', 'FontSize', font_size_F, 'HorizontalAlignment', 'left');
    else
        text(x_max_draw + extension_len + 15, y, id_str, 'Color', 'b', 'FontName', font_name, 'FontWeight', 'bold', 'FontSize', font_size_ID, 'HorizontalAlignment', 'left');
        text(-extension_len - 15, y, sprintf('%.1f', f_val), 'Color', 'k', 'FontName', font_name, 'FontAngle', 'italic', 'FontSize', font_size_F, 'HorizontalAlignment', 'right');
    end
    
    % Inlet / Outlet temperatures
    T_in = streams.Tin(i); T_out = streams.Tout(i);
    if is_hot
        text(-extension_len, y - offset_temp_y, [num2str(T_in) '°C'], 'Color', 'k', 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'left', 'FontWeight', 'bold');
        text(x_max_draw + extension_len, y - offset_temp_y, [num2str(T_out) '°C'], 'Color', 'k', 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'right', 'FontWeight', 'bold');
    else
        text(x_max_draw + extension_len, y - offset_temp_y, [num2str(T_in) '°C'], 'Color', 'k', 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'right', 'FontWeight', 'bold');
        text(-extension_len, y - offset_temp_y, [num2str(T_out) '°C'], 'Color', 'k', 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'left', 'FontWeight', 'bold');
    end
end

% --- 5. Draw intermediate temperature nodes ---
for i = 1:I+J
    path = all_profiles{i};
    y = stream_y_pos(i);
    is_hot = i <= I;
    
    for k = 1:length(path)
        node = path(k);
        if ~isKey(x_map_plot, node.x_pos), continue; end
        x = x_map_plot(node.x_pos) * x_spacing;
        
        desc = node.description;
        if contains(desc, 'Start') || contains(desc, 'End') || contains(desc, 'Extended') || ...
           contains(desc, 'After HU') || contains(desc, 'After CU')
            continue;
        end
        
        t_str = sprintf('%.1f°C', node.temp);
        if is_hot
            text(x + offset_temp_x, y - offset_temp_y, t_str, 'Color', [0.3 0.3 0.3], 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'left');
        else
            text(x - offset_temp_x, y - offset_temp_y, t_str, 'Color', [0.3 0.3 0.3], 'FontName', font_name, 'FontSize', font_size_T, 'HorizontalAlignment', 'right');
        end
    end
end

% --- 6. Draw process heat exchangers ---
if ~isempty(sm)
    for k = 1:height(sm)
        hex_info = sm(k, :);
        hot_idx = find(strcmp(hot_streams.ID, hex_info.HotStream));
        cold_idx = find(strcmp(cold_streams.ID, hex_info.ColdStream));
        
        hot_path = all_profiles{hot_idx};
        node_match = [];
        for p = 1:length(hot_path)
            if contains(hot_path(p).description, sprintf('Stage %d', hex_info.Stage)) && ...
               contains(hot_path(p).description, cold_streams.ID{cold_idx})
                node_match = hot_path(p);
                break;
            end
        end
        
        if isempty(node_match), continue; end
        logical_x = node_match(1).x_pos;
        if ~isKey(x_map_plot, logical_x), continue; end
        x = x_map_plot(logical_x) * x_spacing;
        
        y1 = hot_y(hot_idx); y2 = cold_y(cold_idx);
        plot([x, x], [y1, y2], 'k-', 'LineWidth', 1.2);
        plot(x, y1, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', marker_size, 'LineWidth', 1.2);
        plot(x, y2, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', marker_size, 'LineWidth', 1.2);
        
        text(x, (y1+y2)/2, num2str(k), 'Color', 'k', 'FontName', font_name, 'FontWeight', 'bold', 'FontSize', 12, 'HorizontalAlignment', 'center', 'BackgroundColor', 'w', 'Margin', 1);
        
        text(x, y2 + offset_area_y, sprintf('A=%.2f', hex_info.Area), 'HorizontalAlignment', 'center', 'FontName', font_name, 'FontSize', font_size_Area, 'Color', 'k', 'FontWeight', 'bold');
    end
end

% --- 7. Draw utilities (HU/CU text removed) ---
if ~isempty(results.cold_utilities)
    for k=1:height(results.cold_utilities)
        util = results.cold_utilities(k,:);
        hot_idx = find(strcmp(hot_streams.ID, util.HotStream));
        y = hot_y(hot_idx);
        hot_path = all_profiles{hot_idx};
        
        node_match = [];
        for p=1:length(hot_path), if contains(hot_path(p).description, 'After CU'), node_match = hot_path(p); break; end; end
        
        if ~isempty(node_match) && isKey(x_map_plot, node_match(1).x_pos)
            x = x_map_plot(node_match(1).x_pos) * x_spacing;
            plot(x, y, 'o', 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'c', 'MarkerSize', marker_size+3);
            
            text(x, y + offset_area_y, sprintf('A=%.2f', util.Area), 'Color', 'b', 'FontName', font_name, 'FontSize', font_size_Area, 'HorizontalAlignment', 'center');
        end
    end
end
if ~isempty(results.hot_utilities)
    if isKey(x_map_plot, 0)
        x = x_map_plot(0) * x_spacing;
        for k=1:height(results.hot_utilities)
            util = results.hot_utilities(k,:);
            cold_idx = find(strcmp(cold_streams.ID, util.ColdStream));
            y = cold_y(cold_idx);
            plot(x, y, 'o', 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'm', 'MarkerSize', marker_size+3);
            
            text(x, y + offset_area_y, sprintf('A=%.2f', util.Area), 'Color', 'r', 'FontName', font_name, 'FontSize', font_size_Area, 'HorizontalAlignment', 'center');
        end
    end
end

% --- 8. Annotations ---
text(-extension_len, -2.5, sprintf('Total Annual Cost (TAC): $%.2f /y', results.TAC), ...
    'FontName', font_name, 'FontSize', 11, 'FontWeight', 'bold', ...
    'Color', 'k', 'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'Margin', 5);

legend_x = x_max_draw * 0.75;
text(legend_x, -2.5, 'Under the node: Heat transfer area (A) / m^2', 'FontName', font_name, 'FontSize', 9);
text(legend_x, -1.5, 'At stream ends: Heat capacity flowrate (F) / kW·C^{-1}', 'FontName', font_name, 'FontSize', 9);

xlim([-extension_len - 10, x_max_draw + extension_len + 10]);
ylim([-4, max(cold_y) + 1]);
hold off;

end


%% --- Helper: Temperature profile calculation ---
function results = calculate_profiles_locally(results, streams)
hot_streams = streams(strcmp(streams.Type, 'Hot'), :);
cold_streams = streams(strcmp(streams.Type, 'Cold'), :);
I = height(hot_streams);
J = height(cold_streams);

if isempty(results.process_hex)
    sm_sorted = table();
else
    sm_sorted = sortrows(results.process_hex, {'Stage', 'HotStream', 'ColdStream'});
end

num_streams = I + J;
x_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
x_pos_counter = 1;

for i = 1:height(sm_sorted)
    hex_info = sm_sorted(i, :);
    hot_idx = find(strcmp(hot_streams.ID, hex_info.HotStream));
    cold_idx = find(strcmp(cold_streams.ID, hex_info.ColdStream));
    key = sprintf('S%02d-H%02d-C%02d', hex_info.Stage, hot_idx, cold_idx);
    x_map(key) = x_pos_counter;
    x_pos_counter = x_pos_counter + 1;
end

x_pos_hu = 0;
x_pos_cu = x_pos_counter;

stream_temp_profiles = cell(num_streams, 1);
for i = 1:I
    stream_temp_profiles{i} = struct('x_pos', -1, 'temp', hot_streams.Tin(i), 'description', 'Start');
end
for j = 1:J
    stream_temp_profiles{I+j} = struct('x_pos', x_pos_counter+1, 'temp', cold_streams.Tin(j), 'description', 'Start');
end

cold_stream_data = cell(J, 1);

for i = 1:height(sm_sorted)
    hex_info = sm_sorted(i, :);
    hot_idx = find(strcmp(hot_streams.ID, hex_info.HotStream));
    cold_idx = find(strcmp(cold_streams.ID, hex_info.ColdStream));
    q = hex_info.HeatLoad_q;
    stage = hex_info.Stage;

    current_key = sprintf('S%02d-H%02d-C%02d', hex_info.Stage, hot_idx, cold_idx);
    current_x_pos = x_map(current_key);
    
    last_temp_hot = stream_temp_profiles{hot_idx}(end).temp;
    new_temp_hot = last_temp_hot - q / hot_streams.F(hot_idx);
    desc_hot = sprintf('After HEX with %s (Stage %d)', cold_streams.ID{cold_idx}, stage);
    new_node_hot = struct('x_pos', current_x_pos, 'temp', new_temp_hot, 'description', desc_hot);
    stream_temp_profiles{hot_idx} = [stream_temp_profiles{hot_idx}; new_node_hot];
    
    desc_partner = sprintf('%s (Stage %d)', hot_streams.ID{hot_idx}, stage);
    cold_data_entry = struct('x_pos', current_x_pos, 'q', q, 'partner_id', desc_partner);
    cold_stream_data{cold_idx} = [cold_stream_data{cold_idx}; cold_data_entry];
end

for j = 1:J
    if ~isempty(cold_stream_data{j})
        data = cold_stream_data{j};
        x_positions = [data.x_pos];
        [~, sort_idx] = sort(x_positions, 'descend');
        data_sorted = data(sort_idx);
        
        current_temp = stream_temp_profiles{I+j}(end).temp;
        
        for k = 1:length(data_sorted)
            entry = data_sorted(k);
            q = entry.q;
            x = entry.x_pos;
            partner = entry.partner_id;
            
            new_temp = current_temp + q / cold_streams.F(j);
            new_node = struct('x_pos', x, 'temp', new_temp, 'description', ['After HEX with ' partner]);
            stream_temp_profiles{I+j} = [stream_temp_profiles{I+j}; new_node];
            current_temp = new_temp;
        end
    end
end

if ~isempty(results.cold_utilities)
    for i=1:height(results.cold_utilities)
        hot_idx = find(strcmp(hot_streams.ID, results.cold_utilities.HotStream{i}));
        target_T = hot_streams.Tout(hot_idx);
        new_node = struct('x_pos', x_pos_cu, 'temp', target_T, 'description', 'After CU');
        stream_temp_profiles{hot_idx} = [stream_temp_profiles{hot_idx}; new_node];
    end
end

if ~isempty(results.hot_utilities)
    for i=1:height(results.hot_utilities)
        cold_idx = find(strcmp(cold_streams.ID, results.hot_utilities.ColdStream{i}));
        target_T = cold_streams.Tout(cold_idx);
        new_node = struct('x_pos', x_pos_hu, 'temp', target_T, 'description', 'After HU');
        stream_temp_profiles{I+cold_idx} = [stream_temp_profiles{I+cold_idx}; new_node];
    end
end

for i = 1:I
    if stream_temp_profiles{i}(end).x_pos < x_pos_cu - 0.1
        stream_temp_profiles{i}(end+1) = struct('x_pos', x_pos_cu, 'temp', stream_temp_profiles{i}(end).temp, 'description', 'Extended End');
    end
end
for j = 1:J
    if stream_temp_profiles{I+j}(end).x_pos > x_pos_hu + 0.1
        stream_temp_profiles{I+j}(end+1) = struct('x_pos', x_pos_hu, 'temp', stream_temp_profiles{I+j}(end).temp, 'description', 'Extended End');
    end
end

results.stream_temp_profiles = stream_temp_profiles;
end