clc
clear all
% input_domain
i = input('Enter the domain number: ') - 1;
% input_edge
j = input(['Enter the edge number \n' ...
    '(1, 2, 3, 4 stand for bottom, right,\n top, left\n' ...
    'Note: The edge must be an interface \nor you will get error): ']) - 1;
% input_refinement
k = input('Enter the refinement level \n( from 0 meaning no refinement and \n4 the highest refinement): ') - 1;

data = readmatrix('interface_data/lambda'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');
data_residual = readmatrix('interface_data/residual'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');
data_true = readmatrix('interface_data/lambda_exact'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');

data_y = readmatrix('interface_data/lambda_y'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');
data_residual_y = readmatrix('interface_data/residual_y'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');
data_true_y = readmatrix('interface_data/lambda_exact_y'+string(i)+'_'+string(j)+'_'+string(k)+'.txt');

num_timesteps = size(data, 1);
num_values_per_timestep = size(data, 2);
min_val = min(data(:));
max_val = max(data(:));
min_true = min(data_true(:));
max_true = max(data_true(:));
min_val = min(min_true, min_val);
max_val = max(max_true, max_val);

min_val_y = min(data_y(:));
max_val_y = max(data_y(:));
min_true_y = min(data_true_y(:));
max_true_y = max(data_true_y(:));
min_val_y = min(min_true_y, min_val_y);
max_val_y = max(max_true_y, max_val_y);

% fprintf("num_timesteps = %d\n", num_timesteps);
% fprintf("num_values_per_timestep = %d\n", num_values_per_timestep);
% 
% fprintf("min_val = %f\n", min_val);
% fprintf("max_val = %f\n", max_val);
% 
% fprintf("min_val_y = %f\n", min_val_y);
% fprintf("max_val_y = %f\n", max_val_y);

data_x = data;
data_residual_x = data_residual;
data_true_x = data_true;

fprintf("\n Now you need to specify what plot you want to see:\n" + ...
    "1. Show the x component of Lambda\n" + ...
    "2. Show the y component of Lambda\n" + ...
    "3. Show the x component of the Residual\n" + ...
    "4. Show the y component of the Residual\n" + ...
    "Note: When all sides are Dirichlet pressure and hence stress will\n" + ...
    "be determined up to a constant and hence you might want to find this\n" + ...
    "constant by averaging on few of the interior nodes on the interface\n" + ...
    "and finding the correct approximation by adding this correction term.\n" + ...
    "But be careful if the interface values do not converge this will just\n" + ...
    "ouptut garbage\n" + ...
    "5.Show the x component of Lambda (dirichlet bdry correction)\n" + ...
    "6.Show the y component of Lambda (dirichlet bdry correction)\n")

input_plot = input("Enter input here (enter a number between 1-6):");

% Plot the x values of Lambda

switch input_plot
    case 1

clc
clear current_data current_data_true t minval maxval

min_val = min(data_x(:));
max_val = max(data_x(:));
min_true = min(data_true_x(:));
max_true = max(data_true_x(:));
min_val = min(min_true, min_val);
max_val = max(max_true, max_val);

numframes = num_timesteps;

% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot(src, data_x, data_true_x, min_val, max_val));

% Initial plot (optional)
updatePlot(slider, data_x, data_true_x, min_val, max_val);



% Plot the y values of Lambda
    case 2
clear current_data current_data_true minval maxval slider
min_val = min(data_y(:));
max_val = max(data_y(:));
min_true = min(data_true_y(:));
max_true = max(data_true_y(:));
min_val = min(min_true, min_val);
max_val = max(max_true, max_val);
numframes = num_timesteps;

% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot_y(src, data_y, data_true_y, min_val, max_val));

% Initial plot (optional)
updatePlot_y(slider, data_y, data_true_y, min_val, max_val);


% Plot the x values of residual
    case 3
clc
clear current_data current_data_true t minval maxval slider

min_val = min(data_residual_x(:));
max_val = max(data_residual_x(:));

numframes = num_timesteps;


% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot_r(src, data_residual_x, min_val, max_val));

% Initial plot (optional)
updatePlot_r(slider, data_residual_x, min_val, max_val);



% Plot the y values of residual
    case 4
clc
clear current_data current_data_true t minval maxval slider

min_val = min(data_residual_y(:));
max_val = max(data_residual_y(:));

numframes = num_timesteps;


% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot_ry(src, data_residual_y, min_val, max_val));

% Initial plot (optional)
updatePlot_ry(slider, data_residual_y, min_val, max_val);


% Plot the x values of Lambda (dirichlet bdry only)
    case 5
clc
clear current_data current_data_true t minval maxval

min_val = min(data_x(:));
max_val = max(data_x(:));
min_true = min(data_true_x(:));
max_true = max(data_true_x(:));
min_val = min(min_true, min_val);
max_val = max(max_true, max_val);

numframes = num_timesteps;

constant_x = mean(data_x(numframes,5:end-5) - data_true_x(numframes,5:end-5));

data_x = data_x - constant_x;

% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot_cx(src, data_x, data_true_x, min_val, max_val));

% Initial plot (optional)
updatePlot_cx(slider, data_x, data_true_x, min_val, max_val);



% Plot the y values of Lambda (4 subdomains, dirichlet bdry only)
    case 6
clear current_data current_data_true minval maxval slider
min_val = min(data_y(:));
max_val = max(data_y(:));
min_true = min(data_true_y(:));
max_true = max(data_true_y(:));
min_val = min(min_true, min_val);
max_val = max(max_true, max_val);
numframes = num_timesteps;

constant_y = mean(data_y(numframes,5:end-5) - data_true_y(numframes,5:end-5));

data_y = data_y - constant_y;

% Create slider
slider = uicontrol('Style', 'slider', ...
                   'Min', 1, 'Max', numframes, ...
                   'Value', 1, 'SliderStep', [1/(numframes-1) 1/(numframes-1)], ...
                   'Position', [150 10 300 20], ...
                   'Callback', @(src, event) updatePlot_cy(src, data_y, data_true_y, min_val, max_val));

% Initial plot (optional)
updatePlot_cy(slider, data_y, data_true_y, min_val, max_val);


    otherwise
        fprintf("Error: You did not enter a valid case!")
end


% Function to update plot based on slider value
function updatePlot(slider, data_x, data_true_x, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_x(t,:);
    current_data_true = data_true_x(t,:);
    
    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'approx');
    hold on;
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'true');
    title(['Time step ', num2str(t)]);
    legend;
    % ylim([min_val, max_val]);
    ylabel("bottom edge");
    hold off;
end


% Function to update plot based on slider value
function updatePlot_y(slider, data_y, data_true_y, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_y(t,:);
    current_data_true = data_true_y(t,:);

    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'approx');
    hold on;
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'true');
    title(['Time step ', num2str(t)]);
    legend;
    % ylim([asdfafdafdasdfaafdafdf, max_val]);
    ylabel("bottom edge");
    % xlim([0, num_values_per_timestep]);
    hold off;
end

% Function to update plot based on slider value
function updatePlot_r(slider, data_residual_x, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_residual_x(t,:);
    current_data_true = zeros(length(data_residual_x(t,:)),1);
    
    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'residual');
    hold on
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'y = 0');
    title(['Time step ', num2str(t)]);
    hold off
    legend;
    % ylim([min_val, max_val]);
    ylabel("bottom edge");
end

% Function to update plot based on slider value
function updatePlot_ry(slider, data_residual_y, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_residual_y(t,:);
    current_data_true = zeros(length(data_residual_y(t,:)),1);
    
    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'residual');
    hold on
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'y = 0');
    title(['Time step ', num2str(t)]);
    hold off
    legend;
    % ylim([min_val, max_val]);
    ylabel("bottom edge");
end

% Function to update plot based on slider value
function updatePlot_cx(slider, data_x, data_true_x, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_x(t,:);
    current_data_true = data_true_x(t,:);

    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'approx');
    hold on;
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'true');
    title(['Time step ', num2str(t)]);
    legend;
    % ylim([min_val, max_val]);
    ylabel("bottom edge");
    hold off;
end

% Function to update plot based on slider value
function updatePlot_cy(slider, data_y, data_true_y, min_val, max_val)
    t = round(get(slider, 'Value'));  % Get current frame from slider
    current_data = data_y(t,:);
    current_data_true = data_true_y(t,:);

    % Plot current data
    plot(current_data, '-*', 'Color', "blue", 'DisplayName', 'approx');
    hold on;
    plot(current_data_true, '-*', 'Color', "red", 'DisplayName', 'true');
    title(['Time step ', num2str(t)]);
    legend;
    ylim([min_val, max_val]);
    ylabel("bottom edge");
    % xlim([0, num_values_per_timestep]);
    hold off;
end